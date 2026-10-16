"""Unit tests for src/features/preprocessing.py.

Plain test_* functions, runnable directly or (if ever installed) via pytest.
Use scripts/run_tests.py to run all of tests/ without a test framework.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.features.preprocessing import (
    FrequencyEncoder,
    add_missingness_indicators,
    assert_expanding_feature_is_leakage_safe,
    clip_d_columns,
    derive_amount_features,
    derive_time_features,
)


def test_time_features_hour_matches_verified_formula():
    """hour must equal (TransactionDT // 3600) % 24 — the formula verified
    against D9 in Phase 1 (100% match on 74,926 real rows)."""
    cfg = load_config()
    df = pd.DataFrame({"TransactionDT": [86400, 86400 + 3600 * 7, 86400 + 3600 * 23 + 10]})
    out = derive_time_features(df, cfg)
    assert out["hour"].tolist() == [0, 7, 23]
    assert out["day_of_week"].tolist() == [1, 1, 1]  # all in day 1, day%7==1


def test_amount_decimal_flag():
    df = pd.DataFrame({"TransactionAmt": [68.50, 68.505, 100.00, 9.999]})
    out = derive_amount_features(df)
    assert out["amt_decimals_gt2"].tolist() == [False, True, False, True]


def test_d_column_clipping_negative_values():
    cfg = load_config()
    df = pd.DataFrame({"D1": [5.0, -3.0, 0.0], "D4": [-1.0, 2.0, np.nan]})
    out, counts = clip_d_columns(df, cfg)
    assert out["D1"].tolist() == [5.0, 0.0, 0.0]
    assert out["D4"].tolist()[0] == 0.0 and out["D4"].tolist()[1] == 2.0
    assert counts["D1"] == 1
    assert counts["D4"] == 1


def test_missingness_indicators_flag_correctly():
    df = pd.DataFrame({"addr1": [1.0, np.nan, 3.0], "unrelated": [1, 2, 3]})
    out = add_missingness_indicators(df, columns=["addr1"])
    assert out["addr1_isna"].tolist() == [False, True, False]
    assert "unrelated_isna" not in out.columns


def test_frequency_encoder_fit_train_only_and_unseen_category_maps_to_zero():
    train = pd.DataFrame({"cat": ["a", "a", "b", "b", "b"]})
    val = pd.DataFrame({"cat": ["a", "b", "c"]})  # "c" never seen in train

    enc = FrequencyEncoder(columns=("cat",)).fit(train)
    assert enc.freq_maps_["cat"]["a"] == 2 / 5
    assert enc.freq_maps_["cat"]["b"] == 3 / 5

    out_val = enc.transform(val)
    assert out_val.loc[out_val["cat"] == "a", "cat_freq"].iloc[0] == np.float32(2 / 5)
    assert out_val.loc[out_val["cat"] == "c", "cat_freq"].iloc[0] == 0.0, "unseen category must map to 0.0, not error or leak val's own frequency"

    # transform() must not mutate the fitted state
    state_before = dict(enc.freq_maps_["cat"])
    enc.transform(val)
    assert dict(enc.freq_maps_["cat"]) == state_before


def test_frequency_encoder_save_load_roundtrip(tmp_path=None):
    import tempfile
    train = pd.DataFrame({"cat": ["x", "y", "x"]})
    enc = FrequencyEncoder(columns=("cat",)).fit(train)
    with tempfile.TemporaryDirectory() as d:
        path = pathlib.Path(d) / "enc.json"
        enc.save(path)
        loaded = FrequencyEncoder.load(path)
        assert loaded.freq_maps_ == enc.freq_maps_
        assert loaded.fitted_


def test_leakage_checker_flags_a_leaky_feature():
    """A feature computed using the WHOLE dataset (including future rows)
    must be caught by assert_expanding_feature_is_leakage_safe."""
    df = pd.DataFrame({
        "t": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "entity": ["A"] * 10,
    })
    # Leaky: total count over ALL rows for the entity, regardless of time.
    df["leaky_count"] = df.groupby("entity")["entity"].transform("size")

    def recompute_count(history: pd.DataFrame, row: pd.Series) -> float:
        return float((history["entity"] == row["entity"]).sum())

    try:
        assert_expanding_feature_is_leakage_safe(df, "t", "leaky_count", recompute_count, n_samples=10, seed=0)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "leakage checker failed to flag a feature computed using the entire dataset"


def test_leakage_checker_passes_a_safe_feature():
    """A properly backward-looking expanding count must pass."""
    df = pd.DataFrame({
        "t": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "entity": ["A"] * 10,
    })
    df = df.sort_values("t").reset_index(drop=True)
    # Safe: strictly-prior count, i.e. cumulative count of earlier rows only.
    df["safe_count"] = df.groupby("entity").cumcount()

    def recompute_count(history: pd.DataFrame, row: pd.Series) -> float:
        return float((history["entity"] == row["entity"]).sum())

    assert_expanding_feature_is_leakage_safe(df, "t", "safe_count", recompute_count, n_samples=10, seed=0)


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
    import sys as _sys
    _sys.exit(0 if _run_all() else 1)
