"""Dataset validation checks.

Every function here either returns None (check passed) or raises
AssertionError/ValueError with a specific, actionable message. They are
plain functions rather than pytest tests so they can run with no test
framework installed (scripts/run_validation.py drives them directly), while
still being importable from tests/ for anyone who does have pytest.
"""
from __future__ import annotations

import pandas as pd

from src.config import Config
from src.data.splits import compute_day
from src.features.preprocessing import FrequencyEncoder, assert_features_exclude_blocklist


def check_no_duplicate_transaction_ids(txn: pd.DataFrame, idn: pd.DataFrame, merged: pd.DataFrame) -> None:
    assert not txn["TransactionID"].duplicated().any(), "train_transaction has duplicate TransactionID"
    assert not idn["TransactionID"].duplicated().any(), "train_identity has duplicate TransactionID"
    assert not merged["TransactionID"].duplicated().any(), "merged frame has duplicate TransactionID"


def check_expected_row_counts(txn: pd.DataFrame, idn: pd.DataFrame, merged: pd.DataFrame, cfg: Config) -> None:
    exp = cfg.expected_counts
    assert len(txn) == exp["train_transaction_rows"], f"train_transaction rows {len(txn)} != expected {exp['train_transaction_rows']}"
    assert len(idn) == exp["train_identity_rows"], f"train_identity rows {len(idn)} != expected {exp['train_identity_rows']}"
    assert len(merged) == exp["merged_rows"], f"merged rows {len(merged)} != expected {exp['merged_rows']} (merge must preserve transaction row count)"


def check_split_row_and_fraud_counts(splits: dict[str, pd.DataFrame], cfg: Config) -> None:
    exp_rows = cfg.expected_counts["split_rows"]
    exp_fraud = cfg.expected_counts["split_fraud"]
    for name, df in splits.items():
        assert len(df) == exp_rows[name], f"split '{name}' rows {len(df)} != expected {exp_rows[name]}"
        n_fraud = int(df["isFraud"].sum())
        assert n_fraud == exp_fraud[name], f"split '{name}' fraud count {n_fraud} != expected {exp_fraud[name]}"


def check_target_exists_only_where_expected(merged: pd.DataFrame) -> None:
    assert "isFraud" in merged.columns, "isFraud target column missing from merged training frame"
    assert merged["isFraud"].isna().sum() == 0, "isFraud has null values in labeled training data"
    vals = set(merged["isFraud"].unique().tolist())
    assert vals <= {0, 1}, f"isFraud contains values outside {{0, 1}}: {vals}"


def check_split_temporal_ordering(splits: dict[str, pd.DataFrame], cfg: Config) -> None:
    day = {name: compute_day(df, cfg) for name, df in splits.items()}
    train_max, val_min, val_max, test_min = (
        day["train"].max(), day["val"].min(), day["val"].max(), day["test"].min(),
    )
    assert train_max < val_min, f"TRAIN max day {train_max} is not strictly before VAL min day {val_min}"
    assert val_max < test_min, f"VAL max day {val_max} is not strictly before TEST min day {test_min}"


def check_no_split_overlap(splits: dict[str, pd.DataFrame]) -> None:
    ids = {name: set(df["TransactionID"]) for name, df in splits.items()}
    names = list(ids)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            shared = ids[names[i]] & ids[names[j]]
            assert not shared, f"{len(shared)} TransactionID(s) appear in both '{names[i]}' and '{names[j]}'"


def check_no_target_leakage(feature_columns: list[str], cfg: Config) -> None:
    """isFraud, TransactionID and raw TransactionDT must never reach a
    model's feature matrix (Phase 1 risks R1/R2)."""
    assert_features_exclude_blocklist(feature_columns, cfg)


def check_preprocessing_fitted_train_only(train_df: pd.DataFrame, val_df: pd.DataFrame, cfg: Config) -> None:
    """Demonstrates and checks the fit/transform separation (Phase 1 risk R4):
    fitting on TRAIN alone must give a different (and reproducible) result
    than fitting on TRAIN+VAL, and transforming VAL must not mutate the
    fitted state."""
    enc_train = FrequencyEncoder(columns=cfg.freq_encode_columns).fit(train_df)
    state_before = {k: dict(v) for k, v in enc_train.freq_maps_.items()}

    _ = enc_train.transform(val_df)  # must not refit / mutate state
    assert enc_train.freq_maps_ == state_before, "FrequencyEncoder state changed after transform() was called on VAL data"

    enc_train_val = FrequencyEncoder(columns=cfg.freq_encode_columns).fit(pd.concat([train_df, val_df], ignore_index=True))
    # If TRAIN and TRAIN+VAL produce identical frequency maps, fitting is not
    # actually sensitive to which rows are supplied — that would hide a bug
    # where the encoder secretly uses a global/cached table instead of the
    # data it was given.
    assert enc_train.freq_maps_ != enc_train_val.freq_maps_, (
        "Fitting on TRAIN alone produced identical frequencies to fitting on TRAIN+VAL; "
        "the encoder does not appear to be sensitive to its input, which would mask leakage"
    )


CHECKS = [
    "check_no_duplicate_transaction_ids",
    "check_expected_row_counts",
    "check_split_row_and_fraud_counts",
    "check_target_exists_only_where_expected",
    "check_split_temporal_ordering",
    "check_no_split_overlap",
    "check_no_target_leakage",
    "check_preprocessing_fitted_train_only",
]
