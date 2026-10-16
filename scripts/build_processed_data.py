"""Build the processed dataset and feature splits from raw CSVs.

Pipeline:
  data/raw/*.csv (untouched)
    -> load + downcast + normalize identity columns          (src/data/load.py)
    -> merge on TransactionID, add has_identity               (src/data/merge.py)
    -> save merged, pre-feature frame to data/processed/      (cache; avoids re-parsing 683MB CSV every run)
    -> chronological split (TRAIN/VAL/TEST)                   (src/data/splits.py)
    -> save split TransactionID lists + metadata               (data/processed/split_metadata.json)
    -> leakage-safe non-fit feature transforms, per split      (src/features/preprocessing.py)
    -> fit FrequencyEncoder on TRAIN only, transform all splits
    -> save feature frames to data/features/*.parquet
    -> save fitted encoder state to data/features/frequency_encoder.json

Run: .venv/Scripts/python.exe scripts/build_processed_data.py
"""
from __future__ import annotations

import json
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data.load import load_raw_train
from src.data.merge import merge_transaction_identity
from src.data.splits import split_frame, compute_day
from src.features.preprocessing import build_base_features, FrequencyEncoder, assert_features_exclude_blocklist


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    cfg.processed_dir.mkdir(parents=True, exist_ok=True)
    cfg.features_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading raw CSVs (downcast + identity column normalization) ===")
    txn, idn = load_raw_train(cfg)
    print(f"  train_transaction: {txn.shape[0]:,} rows x {txn.shape[1]} cols  "
          f"({txn.memory_usage(deep=True).sum()/1e6:.1f} MB in memory)")
    print(f"  train_identity   : {idn.shape[0]:,} rows x {idn.shape[1]} cols  "
          f"({idn.memory_usage(deep=True).sum()/1e6:.1f} MB in memory)")

    print("\n=== 2. Merging on TransactionID ===")
    merged = merge_transaction_identity(txn, idn)
    print(f"  merged: {merged.shape[0]:,} rows x {merged.shape[1]} cols "
          f"(has_identity=True: {int(merged['has_identity'].sum()):,}, "
          f"{100*merged['has_identity'].mean():.2f}%)")

    print("\n=== 3. Caching merged, pre-feature frame to data/processed/ ===")
    merged_path = cfg.processed_dir / "train_merged.parquet"
    merged.to_parquet(merged_path, index=False)
    print(f"  wrote {merged_path} ({merged_path.stat().st_size/1e6:.1f} MB)")

    print("\n=== 4. Chronological split ===")
    splits = split_frame(merged, cfg)
    n_unassigned = merged.shape[0] - sum(len(v) for v in splits.values())
    for name, df in splits.items():
        d = compute_day(df, cfg)
        print(f"  {name:>5}: {len(df):>7,} rows | days {int(d.min())}-{int(d.max())} | "
              f"fraud {int(df['isFraud'].sum()):>6,} ({100*df['isFraud'].mean():.3f}%)")
    if n_unassigned:
        print(f"  NOTE: {n_unassigned:,} rows fell outside all three configured day ranges and were excluded "
              f"(day range in data is 1-182; configured ranges cover all of it, so this should be 0).")

    print("\n=== 5. Saving split metadata (so every future script uses this exact split) ===")
    split_meta = {
        "config_path": "configs/data.yaml",
        "day_seconds": cfg.day_seconds,
        "split_bounds": {
            "train": [cfg.split.train_day_min, cfg.split.train_day_max],
            "val": [cfg.split.val_day_min, cfg.split.val_day_max],
            "test": [cfg.split.test_day_min, cfg.split.test_day_max],
        },
        "counts": {name: {"rows": int(len(df)), "fraud": int(df["isFraud"].sum())} for name, df in splits.items()},
        "transaction_ids": {name: df["TransactionID"].tolist() for name, df in splits.items()},
    }
    meta_path = cfg.processed_dir / "split_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(split_meta, fh)
    print(f"  wrote {meta_path} ({meta_path.stat().st_size/1e6:.1f} MB, includes per-split TransactionID lists)")

    print("\n=== 6. Leakage-safe non-fit feature transforms (per split) ===")
    feature_splits = {}
    build_log = {"d_columns_clipped_negative_count": {}}
    for name, df in splits.items():
        fdf, log = build_base_features(df, cfg)
        feature_splits[name] = fdf
        build_log["d_columns_clipped_negative_count"][name] = log["d_columns_clipped_negative_count"]
        clipped_total = sum(log["d_columns_clipped_negative_count"].values())
        print(f"  {name:>5}: added time/amount/missingness features; "
              f"{clipped_total} negative D-column values clipped to 0")

    print("\n=== 7. Fitting FrequencyEncoder on TRAIN only ===")
    encoder = FrequencyEncoder(columns=cfg.freq_encode_columns)
    encoder.fit(feature_splits["train"])
    for name in ("train", "val", "test"):
        feature_splits[name] = encoder.transform(feature_splits[name])
    n_unseen_val = int((feature_splits["val"][f"{cfg.freq_encode_columns[0]}_freq"] == 0.0).sum())
    print(f"  fit on {len(feature_splits['train']):,} TRAIN rows only; columns: {list(cfg.freq_encode_columns)}")

    encoder_path = cfg.features_dir / "frequency_encoder.json"
    encoder.save(encoder_path)
    print(f"  saved fitted encoder state to {encoder_path}")

    print("\n=== 8. Validating no blocklisted column leaked into feature columns ===")
    non_feature_cols = set(cfg.feature_blocklist) | {"has_identity"}
    for name, fdf in feature_splits.items():
        candidate_feature_cols = [c for c in fdf.columns if c not in non_feature_cols]
        assert_features_exclude_blocklist(candidate_feature_cols, cfg)
    print("  OK: TransactionID / TransactionDT / isFraud confirmed absent from candidate feature columns")

    print("\n=== 9. Saving feature frames to data/features/ ===")
    for name, fdf in feature_splits.items():
        out_path = cfg.features_dir / f"{name}_features.parquet"
        fdf.to_parquet(out_path, index=False)
        print(f"  wrote {out_path} ({out_path.stat().st_size/1e6:.1f} MB, {fdf.shape[1]} columns)")

    build_log_path = cfg.processed_dir / "build_log.json"
    with open(build_log_path, "w", encoding="utf-8") as fh:
        json.dump(build_log, fh, indent=1)
    print(f"  wrote {build_log_path} (data-quality diagnostics, e.g. D-column clip counts)")

    print(f"\nDone in {time.time()-t0:.1f}s. Raw files under {cfg.raw_dir} were not modified.")


if __name__ == "__main__":
    main()
