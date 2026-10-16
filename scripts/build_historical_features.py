"""Build the strictly historical, entity-level feature set (Phase 1's
"Level 3.5 blocklist baseline") and verify it is leakage-safe.

Computed over the FULL labeled frame (all three chronological splits) —
see src/features/historical.py's module docstring for why this is correct
and not a leakage violation. Writes data/features/historical_features.parquet,
a reusable artifact keyed by TransactionID.
"""
from __future__ import annotations

import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config
from src.features.historical import build_historical_features, historical_feature_columns, run_leakage_self_check


def main() -> None:
    t0 = time.time()
    cfg = load_config()
    merged = pd.read_parquet(cfg.processed_dir / "train_merged.parquet")

    print(f"=== Building historical features over {len(merged):,} labeled rows (all splits, chronological) ===")
    hist = build_historical_features(merged, cfg)
    cols = historical_feature_columns()
    print(f"  {len(cols)} historical features: {cols}")
    print(hist[cols].describe().T.round(4).to_string())

    print("\n=== Leakage self-check (independent recomputation using only strictly-prior rows) ===")
    run_leakage_self_check(merged, cfg, hist, n_samples=30, seed=0)
    print("  PASSED: prior_txn_count and prior_fraud_count independently reproduced for 30 sampled rows "
          "per entity, using only rows strictly earlier in the stable chronological order.")

    print("\n=== Sanity check against Phase 1's headline number ===")
    # Phase 1 (Section 10): among TEST-period (day 152-182) transactions on
    # accounts with prior history, prior_fraud=1 -> 59.86% vs prior_fraud=0 -> 0.08%.
    day = merged["TransactionDT"] // cfg.day_seconds
    test_mask = (day >= cfg.split.test_day_min) & (day <= cfg.split.test_day_max)
    check = merged.loc[test_mask, ["TransactionID", "isFraud"]].merge(hist, on="TransactionID")
    has_hist = check["account_has_prior_history"] == 1
    for flag_val, label in [(1, "prior_fraud=1"), (0, "prior_fraud=0")]:
        m = has_hist & (check["account_prior_fraud_flag"] == flag_val)
        rate = check.loc[m, "isFraud"].mean() if m.sum() else float("nan")
        print(f"  TEST, has history, {label}: n={int(m.sum()):,}  fraud_rate={100*rate:.2f}%")

    out_path = cfg.features_dir / "historical_features.parquet"
    hist.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")
    print(f"Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
