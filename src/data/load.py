"""Memory-conscious loading of the IEEE-CIS raw CSVs.

Raw files under data/raw/ are never modified or overwritten by this module.
Downcasting and column-name normalization happen only on in-memory copies /
on writes into data/processed/.
"""
from __future__ import annotations

import logging
import pathlib

import numpy as np
import pandas as pd

from src.config import Config

log = logging.getLogger(__name__)

# Safe int32 range check margin: TransactionID goes up to ~4.17M, TransactionDT
# up to ~34.2M (Kaggle test) — both far under 2^31-1 ~= 2.147e9. card1 tops out
# at 18,396. Verified against Phase 1 reconnaissance numbers; re-checked live
# below rather than trusted blindly, since a silent overflow would be a
# correctness bug, not just a performance one.
INT32_MAX = np.iinfo(np.int32).max
INT32_MIN = np.iinfo(np.int32).min


def _downcast(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Downcast float64->float32 and int64->int32 where numerically safe.

    Returns a new DataFrame; never mutates raw data (the caller only ever
    passes in a DataFrame already read into memory, not the file itself).
    """
    df = df.copy()
    for col in df.columns:
        dt = df[col].dtype
        if cfg.downcast_float64_to_float32 and dt == np.float64:
            df[col] = df[col].astype(np.float32)
        elif cfg.downcast_int64_to_int32 and dt == np.int64:
            col_min, col_max = df[col].min(), df[col].max()
            if col_min >= INT32_MIN and col_max <= INT32_MAX:
                df[col] = df[col].astype(np.int32)
            else:
                log.warning("Column %s exceeds int32 range (%s..%s); keeping int64", col, col_min, col_max)
    return df


def _normalize_identity_columns(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """train_identity uses id_01, test_identity uses id-01 (Phase 1, Section 6).

    Normalizing on load means downstream code never has to special-case which
    file a frame originally came from.
    """
    return df.rename(columns={c: c.replace(cfg.identity_from_sep, cfg.identity_to_sep) for c in df.columns})


def load_train_transaction(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_path("train_transaction")
    df = pd.read_csv(path, low_memory=False)
    df = _downcast(df, cfg)
    return df


def load_train_identity(cfg: Config) -> pd.DataFrame:
    path = cfg.raw_path("train_identity")
    df = pd.read_csv(path, low_memory=False)
    df = _normalize_identity_columns(df, cfg)
    df = _downcast(df, cfg)
    return df


def load_raw_train(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load both raw training tables, downcast, identity columns normalized.

    Raw CSVs on disk are read-only inputs here; nothing in this function
    writes to data/raw/.
    """
    txn = load_train_transaction(cfg)
    idn = load_train_identity(cfg)
    return txn, idn
