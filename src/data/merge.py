"""Merge train_transaction and train_identity on TransactionID.

The relationship is one-to-one, optional: every identity row joins to exactly
one transaction, but most transactions (75.6%) have no identity row at all
(Phase 1, Section 7). That absence is itself a strong signal (7.85% fraud
rate with identity vs. 2.09% without) so it must survive the merge as an
explicit column, not be lost as an implicit NaN pattern.
"""
from __future__ import annotations

import pandas as pd


def merge_transaction_identity(txn: pd.DataFrame, idn: pd.DataFrame) -> pd.DataFrame:
    if txn["TransactionID"].duplicated().any():
        raise ValueError("train_transaction has duplicate TransactionID; expected one row per transaction")
    if idn["TransactionID"].duplicated().any():
        raise ValueError("train_identity has duplicate TransactionID; expected at most one identity row per transaction")
    if not idn["TransactionID"].isin(txn["TransactionID"]).all():
        raise ValueError("train_identity contains TransactionID values not present in train_transaction")

    n_before = len(txn)
    merged = txn.merge(idn, on="TransactionID", how="left", validate="one_to_one")
    if len(merged) != n_before:
        raise ValueError(f"Merge changed row count: {n_before} -> {len(merged)} (expected a pure left join, one row per transaction)")

    # Identity presence is a feature in its own right (Phase 1, Section 7) —
    # make it explicit rather than relying on callers to re-derive it from
    # id_* column NaNs, which would silently break if the identity column set
    # ever changes. merge() already leaves the frame fragmented (434 columns
    # assembled from two sources); copy() once before adding a further column
    # rather than mutating the fragmented frame in place.
    merged = merged.copy()
    id_cols = [c for c in idn.columns if c != "TransactionID"]
    merged["has_identity"] = merged[id_cols].notna().any(axis=1) if id_cols else False

    return merged
