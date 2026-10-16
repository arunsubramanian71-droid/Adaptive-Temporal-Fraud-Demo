"""Meta-tests: exercise the validators in src/data/validate.py against small
synthetic scenarios to confirm they actually detect the violations they
claim to detect (a validator that never fails is worse than no validator)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data import validate as V


def _expect_raises(fn, exc_types=(AssertionError, ValueError)):
    try:
        fn()
    except exc_types:
        return True
    return False


def test_check_no_split_overlap_detects_shared_transaction_id():
    train = pd.DataFrame({"TransactionID": [1, 2, 3]})
    val = pd.DataFrame({"TransactionID": [3, 4, 5]})  # 3 appears in both
    test = pd.DataFrame({"TransactionID": [6, 7]})
    assert _expect_raises(lambda: V.check_no_split_overlap({"train": train, "val": val, "test": test}))


def test_check_no_split_overlap_passes_when_disjoint():
    train = pd.DataFrame({"TransactionID": [1, 2, 3]})
    val = pd.DataFrame({"TransactionID": [4, 5]})
    test = pd.DataFrame({"TransactionID": [6, 7]})
    V.check_no_split_overlap({"train": train, "val": val, "test": test})  # must not raise


def test_check_split_temporal_ordering_detects_violation():
    cfg = load_config()
    day_sec = cfg.day_seconds
    train = pd.DataFrame({"TransactionDT": np.arange(1, 50) * day_sec})
    val = pd.DataFrame({"TransactionDT": np.arange(40, 60) * day_sec})  # overlaps train's tail
    test = pd.DataFrame({"TransactionDT": np.arange(70, 90) * day_sec})
    assert _expect_raises(lambda: V.check_split_temporal_ordering({"train": train, "val": val, "test": test}, cfg))


def test_check_split_temporal_ordering_passes_when_strictly_ordered():
    cfg = load_config()
    day_sec = cfg.day_seconds
    train = pd.DataFrame({"TransactionDT": np.arange(1, 50) * day_sec})
    val = pd.DataFrame({"TransactionDT": np.arange(51, 60) * day_sec})
    test = pd.DataFrame({"TransactionDT": np.arange(61, 90) * day_sec})
    V.check_split_temporal_ordering({"train": train, "val": val, "test": test}, cfg)  # must not raise


def test_check_no_target_leakage_detects_blocklisted_column():
    cfg = load_config()
    assert _expect_raises(lambda: V.check_no_target_leakage(["TransactionAmt", "TransactionID", "hour"], cfg))
    V.check_no_target_leakage(["TransactionAmt", "hour", "card1_freq"], cfg)  # must not raise


def test_check_target_exists_only_where_expected_detects_nulls():
    df = pd.DataFrame({"isFraud": [0, 1, None]})
    assert _expect_raises(lambda: V.check_target_exists_only_where_expected(df))


def test_check_target_exists_only_where_expected_detects_bad_values():
    df = pd.DataFrame({"isFraud": [0, 1, 2]})
    assert _expect_raises(lambda: V.check_target_exists_only_where_expected(df))


def test_check_target_exists_only_where_expected_passes_valid_target():
    df = pd.DataFrame({"isFraud": [0, 1, 0, 1, 0]})
    V.check_target_exists_only_where_expected(df)  # must not raise


def test_check_expected_row_counts_detects_mismatch():
    cfg = load_config()
    txn = pd.DataFrame({"a": range(cfg.expected_counts["train_transaction_rows"] - 1)})  # one row short
    idn = pd.DataFrame({"a": range(cfg.expected_counts["train_identity_rows"])})
    merged = pd.DataFrame({"a": range(cfg.expected_counts["merged_rows"] - 1)})
    assert _expect_raises(lambda: V.check_expected_row_counts(txn, idn, merged, cfg))


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return len(failed) == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
