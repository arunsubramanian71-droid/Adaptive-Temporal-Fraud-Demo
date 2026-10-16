"""Unit tests for src/data/splits.py, plus a regression test against the
built processed data (requires scripts/build_processed_data.py to have run)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import SplitBounds, load_config
from src.data.splits import assign_split, compute_day, split_frame


def test_split_bounds_rejects_overlapping_or_unordered_ranges():
    SplitBounds(1, 100, 101, 150, 151, 200)  # valid, must not raise
    for bad in [
        dict(train_day_min=1, train_day_max=100, val_day_min=90, val_day_max=150, test_day_min=151, test_day_max=200),  # overlap
        dict(train_day_min=1, train_day_max=100, val_day_min=101, val_day_max=150, test_day_min=140, test_day_max=200),  # overlap
        dict(train_day_min=100, train_day_max=1, val_day_min=101, val_day_max=150, test_day_min=151, test_day_max=200),  # inverted
    ]:
        try:
            SplitBounds(**bad)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"SplitBounds accepted an invalid configuration: {bad}"


def test_compute_day_derivation():
    cfg = load_config()
    df = pd.DataFrame({"TransactionDT": [86400, 86400 * 2 - 1, 86400 * 2]})
    day = compute_day(df, cfg)
    assert day.tolist() == [1, 1, 2]


def _synthetic_frame():
    # Spans days 1..200 so it crosses all three configured boundaries.
    dt = np.arange(1, 201) * 86400
    return pd.DataFrame({"TransactionDT": dt, "TransactionID": np.arange(len(dt))})


def test_assign_split_is_disjoint_and_covers_full_configured_range():
    cfg = load_config()
    df = _synthetic_frame()
    s = assign_split(df, cfg)
    day = compute_day(df, cfg)

    in_range = (
        ((day >= cfg.split.train_day_min) & (day <= cfg.split.test_day_max))
    )
    assert s[in_range].isna().sum() == 0, "some in-range rows were not assigned to any split"

    counts = s.value_counts()
    assert counts.get("train", 0) == cfg.split.train_day_max - cfg.split.train_day_min + 1
    assert counts.get("val", 0) == cfg.split.val_day_max - cfg.split.val_day_min + 1
    assert counts.get("test", 0) == cfg.split.test_day_max - cfg.split.test_day_min + 1


def test_split_frame_train_before_val_before_test():
    cfg = load_config()
    df = _synthetic_frame()
    splits = split_frame(df, cfg)
    train_max_day = compute_day(splits["train"], cfg).max()
    val_min_day, val_max_day = compute_day(splits["val"], cfg).min(), compute_day(splits["val"], cfg).max()
    test_min_day = compute_day(splits["test"], cfg).min()
    assert train_max_day < val_min_day
    assert val_max_day < test_min_day


def test_built_split_matches_expected_regression_counts():
    """Regression test against reports/phase1_reconnaissance.md Section 18.
    Requires scripts/build_processed_data.py to have already run."""
    cfg = load_config()
    path = cfg.features_dir / "train_features.parquet"
    if not path.exists():
        print("SKIP: data/features/*.parquet not found — run scripts/build_processed_data.py first")
        return
    exp_rows = cfg.expected_counts["split_rows"]
    exp_fraud = cfg.expected_counts["split_fraud"]
    for name in ("train", "val", "test"):
        df = pd.read_parquet(cfg.features_dir / f"{name}_features.parquet")
        assert len(df) == exp_rows[name], f"{name}: {len(df)} rows != expected {exp_rows[name]}"
        assert int(df["isFraud"].sum()) == exp_fraud[name], f"{name}: fraud count mismatch"
        d = compute_day(df, cfg)
        lo, hi = getattr(cfg.split, f"{name}_day_min"), getattr(cfg.split, f"{name}_day_max")
        assert d.min() >= lo and d.max() <= hi, f"{name}: day range [{d.min()},{d.max()}] outside configured [{lo},{hi}]"


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
