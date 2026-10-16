"""Strictly historical, entity-level features — the "Level 3.5 blocklist
baseline" identified in Phase 1 reconnaissance (reports/phase1_reconnaissance.md,
Section 10).

Purpose: isolate exactly how much predictive power comes from "this
account/card has a prior confirmed fraud" alone, so later graph-model gains
can be judged against it rather than mistaken for a novel discovery. Phase 1
found the effect is large: transactions on an account with prior confirmed
fraud ran 59.86% fraud vs. 0.08% for accounts with clean prior history.

LEAKAGE DESIGN — read before changing anything here:

Every feature is computed using `.shift(1)` / an inclusive-cumsum-minus-self
trick, over the FULL labeled dataset (all of train_merged.parquet, all three
of our internal chronological splits) sorted by TransactionDT. This is
DELIBERATE and different from the fit-on-TRAIN-only rule used for the
Phase 2/3 statistical encoders (FrequencyEncoder, MedianImputer, the LR
StandardScaler):

  - Encoders like FrequencyEncoder characterize a DISTRIBUTION (e.g. "how
    common is this ProductCD") — using VAL/TEST rows to compute that
    distribution would leak information about periods the model shouldn't
    know about yet, even in aggregate. So those are fit on TRAIN only.
  - A historical entity feature ("has this card had a prior confirmed
    fraud") is not a distributional statistic — it is a fact about the
    world that accumulates over real, elapsed time, exactly as it would in
    a live production system where a card's history keeps growing
    regardless of which artificial dataset partition a later transaction
    happens to fall into. Restricting it to "history within the same split
    only" would not make it safer, it would make it a different, weaker,
    and less realistic feature, and would defeat the purpose of this
    control baseline (whose entire point is to reproduce, as a classical
    ML feature, exactly the label-propagation effect Phase 1 measured for
    graph-adjacent entity structure).

Every feature here still strictly satisfies information_time < transaction_time
(`.shift(1)` and "inclusive minus self" both exclude the current row itself),
and the module ships `run_leakage_self_check` which re-derives each feature
independently for a sample of rows using only strictly-prior rows, so this is
verified, not just argued.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Config
from src.features.preprocessing import assert_expanding_feature_is_leakage_safe

ENTITY_DEFINITIONS = {
    # Simple entity: the anonymized card token alone. 100% coverage,
    # Phase 1 lift 1.80x.
    "card1": ["card1"],
    # Phase 1's key finding: card + billing address + inferred account
    # origin date. 88.7% coverage, 35.12x label-homophily lift — the
    # feature this whole control baseline exists to isolate.
    "account": ["card1", "addr1", "d1anchor"],
}


def _entity_key(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    parts = [df[c].astype("string").fillna("NA") for c in cols]
    key = parts[0]
    for p in parts[1:]:
        key = key + "|" + p
    return key


def build_historical_features(merged: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """merged: the full labeled frame (data/processed/train_merged.parquet,
    590,540 rows spanning all three of our chronological splits) — NOT a
    single split. Returns a frame keyed by TransactionID with only
    TransactionID + the historical feature columns, meant to be merged back
    onto each split afterward.
    """
    df = merged[["TransactionID", "TransactionDT", "isFraud", "card1", "addr1", "D1"]].copy()
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)  # stable sort: ties keep file order
    df["day"] = (df["TransactionDT"] // cfg.day_seconds).astype(np.int32)
    df["d1anchor"] = df["day"] - df["D1"].fillna(0)

    out = pd.DataFrame({"TransactionID": df["TransactionID"]})

    for name, cols in ENTITY_DEFINITIONS.items():
        key = _entity_key(df, cols)
        g = df.groupby(key, sort=False)

        prior_txn_count = g.cumcount()  # count of strictly-prior rows in this group, by construction
        inclusive_fraud_cumsum = g["isFraud"].cumsum()
        prior_fraud_count = inclusive_fraud_cumsum - df["isFraud"]  # subtract current row -> strictly prior
        prior_day = g["day"].shift(1)

        out[f"{name}_prior_txn_count"] = prior_txn_count.astype(np.int32)
        out[f"{name}_prior_fraud_count"] = prior_fraud_count.astype(np.int32)
        out[f"{name}_prior_fraud_flag"] = (prior_fraud_count > 0).astype(np.int8)
        out[f"{name}_has_prior_history"] = (prior_txn_count > 0).astype(np.int8)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate = prior_fraud_count / prior_txn_count.replace(0, np.nan)
        out[f"{name}_prior_fraud_rate"] = rate.fillna(0.0).astype(np.float32)
        days_since_prev = (df["day"] - prior_day)
        out[f"{name}_days_since_prev_txn"] = days_since_prev.fillna(-1).astype(np.int32)

    return out


def historical_feature_columns() -> list[str]:
    cols = []
    for name in ENTITY_DEFINITIONS:
        cols += [f"{name}_prior_txn_count", f"{name}_prior_fraud_count", f"{name}_prior_fraud_flag",
                 f"{name}_has_prior_history", f"{name}_prior_fraud_rate", f"{name}_days_since_prev_txn"]
    return cols


def run_leakage_self_check(merged: pd.DataFrame, cfg: Config, hist: pd.DataFrame, n_samples: int = 30, seed: int = 0) -> None:
    """Independently re-derives prior_txn_count and prior_fraud_count for a
    sample of rows using ONLY rows with TransactionDT strictly earlier, and
    raises if the vectorized computation above disagrees. This is the
    reusable checker built in Phase 2 (src/features/preprocessing.py),
    exercised here against a real feature for the first time.
    """
    df = merged[["TransactionID", "TransactionDT", "isFraud", "card1", "addr1", "D1"]].merge(hist, on="TransactionID")
    df = df.sort_values("TransactionDT", kind="mergesort").reset_index(drop=True)
    df["day"] = (df["TransactionDT"] // cfg.day_seconds).astype(np.int32)
    df["d1anchor"] = df["day"] - df["D1"].fillna(0)
    # ~17,191 rows share an exact TransactionDT with another row (Phase 1,
    # Section 9). A strict "< TransactionDT" comparison would treat same-
    # timestamp rows as simultaneous, which disagrees with cumcount/cumsum's
    # stable-sort tie-breaking (whichever came first in file order still
    # counts as "prior"). Check against that same tie-broken order — the
    # row's position in this stable sort — rather than raw TransactionDT.
    df["_seq"] = np.arange(len(df))

    for name, cols in ENTITY_DEFINITIONS.items():
        df["_key"] = _entity_key(df, cols)

        def recompute_count(history: pd.DataFrame, row: pd.Series) -> float:
            return float((history["_key"] == row["_key"]).sum())

        def recompute_fraud_count(history: pd.DataFrame, row: pd.Series) -> float:
            return float(history.loc[history["_key"] == row["_key"], "isFraud"].sum())

        assert_expanding_feature_is_leakage_safe(
            df, "_seq", f"{name}_prior_txn_count", recompute_count, n_samples=n_samples, seed=seed)
        assert_expanding_feature_is_leakage_safe(
            df, "_seq", f"{name}_prior_fraud_count", recompute_fraud_count, n_samples=n_samples, seed=seed)
