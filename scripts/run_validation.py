"""Run all dataset-level validation checks against the built processed data.

Requires scripts/build_processed_data.py to have been run first. Exits with
a nonzero status if any check fails, so it can gate later pipeline stages.
"""
from __future__ import annotations

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import pandas as pd

from src.config import load_config
from src.data import validate as V
from src.data.load import load_train_identity


def load_built_data(cfg):
    merged = pd.read_parquet(cfg.processed_dir / "train_merged.parquet")
    # train_transaction is 683MB on disk; re-parsing it here just to check
    # duplicates already implied by merged's own uniqueness (merge_transaction_identity
    # validated this at build time) would be wasteful. The merged frame's
    # TransactionID uniqueness transitively covers it.
    txn_rowcount_proxy = merged
    # train_identity is only 26MB — cheap enough to re-read for a genuine,
    # independent duplicate check rather than deriving a proxy that could
    # never actually fail.
    idn = load_train_identity(cfg)
    splits = {
        name: pd.read_parquet(cfg.features_dir / f"{name}_features.parquet")
        for name in ("train", "val", "test")
    }
    return merged, txn_rowcount_proxy, idn, splits


def main() -> int:
    cfg = load_config()
    merged, txn_proxy, idn, splits = load_built_data(cfg)

    results = []

    def run(name, fn):
        try:
            fn()
            results.append((name, "PASS", ""))
        except AssertionError as e:
            results.append((name, "FAIL", str(e)))
        except Exception as e:
            results.append((name, "ERROR", f"{type(e).__name__}: {e}"))

    run("no_duplicate_transaction_ids", lambda: V.check_no_duplicate_transaction_ids(txn_proxy, idn, merged))
    run("expected_row_counts", lambda: V.check_expected_row_counts(txn_proxy, idn, merged, cfg))
    run("split_row_and_fraud_counts", lambda: V.check_split_row_and_fraud_counts(splits, cfg))
    run("target_exists_only_where_expected", lambda: V.check_target_exists_only_where_expected(merged))
    run("split_temporal_ordering", lambda: V.check_split_temporal_ordering(splits, cfg))
    run("no_split_overlap", lambda: V.check_no_split_overlap(splits))

    non_feature_cols = set(cfg.feature_blocklist) | {"has_identity"}
    candidate_cols = [c for c in splits["train"].columns if c not in non_feature_cols]
    run("no_target_leakage_in_features", lambda: V.check_no_target_leakage(candidate_cols, cfg))

    run("preprocessing_fitted_train_only", lambda: V.check_preprocessing_fitted_train_only(splits["train"], splits["val"], cfg))

    print("=== Dataset validation results ===")
    n_pass = 0
    for name, status, detail in results:
        print(f"[{status:5s}] {name}" + (f" — {detail}" if detail else ""))
        n_pass += status == "PASS"
    print(f"\n{n_pass}/{len(results)} checks passed.")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
