"""Chronological train/validation/test split.

Boundaries are read from config (configs/data.yaml), not hard-coded here, so
there is exactly one place that can change them — and any change is visible
in a diff of a data file, not buried in code. See reports/phase1_reconnaissance.md
Section 18 for how the boundaries were chosen.
"""
from __future__ import annotations

import enum

import numpy as np
import pandas as pd

from src.config import Config


class Split(str, enum.Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def compute_day(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """1-indexed day number derived from TransactionDT (verified semantics,
    Phase 1 Section 9: TransactionDT is seconds from an undisclosed reference
    instant aligned to local midnight)."""
    return (df["TransactionDT"] // cfg.day_seconds).astype(np.int32)


def assign_split(df: pd.DataFrame, cfg: Config) -> pd.Series:
    """Return a Split-valued Series aligned to df.index.

    Rows outside all three configured day ranges are left as NaN (object
    dtype) rather than silently dropped or misassigned — callers must decide
    what to do with them, and validation checks catch that case (see
    src/data/validate.py: rows must fall into exactly one split).
    """
    day = compute_day(df, cfg)
    s = cfg.split
    out = pd.Series(pd.NA, index=df.index, dtype="object")
    out[(day >= s.train_day_min) & (day <= s.train_day_max)] = Split.TRAIN.value
    out[(day >= s.val_day_min) & (day <= s.val_day_max)] = Split.VAL.value
    out[(day >= s.test_day_min) & (day <= s.test_day_max)] = Split.TEST.value
    return out


def split_frame(df: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Split df into {"train": ..., "val": ..., "test": ...} DataFrames,
    each still carrying TransactionID and TransactionDT for traceability
    (they are blocked from *modeling* via cfg.feature_blocklist, not deleted
    from the data)."""
    split_col = assign_split(df, cfg)
    out = {}
    for name in (Split.TRAIN, Split.VAL, Split.TEST):
        mask = split_col == name.value
        out[name.value] = df.loc[mask].copy()
    return out
