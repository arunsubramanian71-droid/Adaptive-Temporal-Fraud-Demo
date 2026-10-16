"""Leakage-safe, reusable preprocessing.

Scope for this foundational phase (see reports/phase1_reconnaissance.md):
  - verified time derivations (hour, day-of-week)
  - a cheap, leakage-free currency-conversion proxy (TransactionAmt decimals)
  - D-column negative-value clipping (Phase 1 risk R6)
  - missingness indicators for columns where absence itself carries signal
  - a frequency encoder demonstrating the fit-on-train-only pattern

Deliberately OUT of scope here: entity-level historical aggregates
(prior-transaction counts, prior-fraud flags, ...). Those are graph/model-
adjacent features (Phase 1 Section 10's "Level 3.5 blocklist baseline") and
belong to the modeling phase, not the foundational pipeline. The leakage
invariant checker below (`assert_expanding_feature_is_leakage_safe`) is
built now, ahead of that need, so those features are checked the moment
they're introduced.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from src.config import Config

# Columns where Phase 1 (Section 13) found missingness itself to be
# informative, not just an artifact of collection. Kept as an explicit list
# (not "all 380 columns") so every included indicator has a stated reason.
MISSINGNESS_INDICATOR_COLUMNS = [
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D10", "D11", "D12", "D13", "D14", "D15",
    "addr1", "addr2", "dist1", "dist2",
    "card2", "card3", "card5",
    "P_emaildomain", "R_emaildomain",
    "DeviceType", "DeviceInfo",
]


def assert_features_exclude_blocklist(columns, cfg: Config) -> None:
    """Guard against the single most damaging leakage mode (Phase 1 risk R1):
    TransactionID (r=0.998 with TransactionDT) or the raw target/timestamp
    ending up as model inputs."""
    hit = set(columns) & set(cfg.feature_blocklist)
    if hit:
        raise ValueError(f"Blocklisted column(s) present in feature set: {sorted(hit)}")


def derive_time_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """hour and day-of-week, derived from TransactionDT.

    hour = (TransactionDT // 3600) % 24 was independently verified against D9
    in Phase 1 (100.0000% match on 74,926 rows) — it is the true local hour,
    not a guess. Both derived columns are safe: they are coarse, bounded
    functions of the timestamp, not the timestamp itself.
    """
    df = df.copy()
    df["hour"] = ((df["TransactionDT"] // 3600) % 24).astype(np.int16)
    df["day_of_week"] = ((df["TransactionDT"] // cfg.day_seconds) % 7).astype(np.int16)
    return df


def derive_amount_features(df: pd.DataFrame) -> pd.DataFrame:
    """Flag TransactionAmt values with more than 2 decimal places.

    Phase 1 (Section 6): 10.49% of rows show this pattern (consistent with a
    foreign-currency conversion leaving sub-cent residue), with an 11.72% vs
    3.50% fraud-rate split. Purely a function of the current row's own
    amount — no leakage risk.
    """
    df = df.copy()
    cents = np.round(df["TransactionAmt"].astype(np.float64) * 1000).astype(np.int64) % 10
    df["amt_decimals_gt2"] = (cents != 0)
    return df


def clip_d_columns(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """Clip D1-D15 (excluding D9) at 0.

    Phase 1 (Section 9, risk R6): 45 rows total have small negative D values,
    which would mean a "days since" feature pointing into the future. Rare
    enough to clip rather than drop rows; the count is returned so it is
    logged, not silently absorbed.
    """
    df = df.copy()
    clipped_counts: dict[str, int] = {}
    for col in cfg.d_columns_to_clip:
        if col not in df.columns:
            continue
        n_neg = int((df[col] < 0).sum())
        if n_neg:
            df[col] = df[col].clip(lower=0)
        clipped_counts[col] = n_neg
    return df, clipped_counts


def add_missingness_indicators(df: pd.DataFrame, columns: list[str] | None = None) -> pd.DataFrame:
    """Explicit is-missing flags for columns where absence carries signal
    (Phase 1 Section 13: identity presence alone is a 3.75x fraud-rate
    signal). Never impute these columns without keeping this flag."""
    df = df.copy()
    cols = MISSINGNESS_INDICATOR_COLUMNS if columns is None else columns
    for col in cols:
        if col in df.columns:
            df[f"{col}_isna"] = df[col].isna()
    return df


@dataclass
class FrequencyEncoder:
    """Frequency-encodes categorical columns, fit on training data only.

    This exists specifically to demonstrate and enforce the fit/transform
    separation required by Phase 1 risk R4: every fitted statistic must come
    from TRAIN alone. `fit` stores nothing about val/test; `transform` maps
    unseen categories (present in val/test but never seen in train) to 0.0
    rather than erroring or silently leaking their own frequency.
    """
    columns: tuple[str, ...]
    freq_maps_: dict[str, dict] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "FrequencyEncoder":
        self.freq_maps_ = {}
        for col in self.columns:
            if col not in df.columns:
                continue
            vc = df[col].value_counts(normalize=True, dropna=True)
            # Keys are stringified so numeric-valued columns (e.g. card1,
            # addr1) survive a JSON save/load round trip: JSON object keys
            # are always strings, so an int key like 131 would silently come
            # back as "131" after load() and never match a numeric lookup
            # again. Stringifying consistently in both fit() and transform()
            # makes the encoder dtype-agnostic instead of relying on the
            # column happening to already be string-typed.
            self.freq_maps_[col] = {str(k): v for k, v in vc.to_dict().items()}
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("FrequencyEncoder.transform called before fit()")
        df = df.copy()
        for col in self.columns:
            if col not in df.columns or col not in self.freq_maps_:
                continue
            # NaNs were excluded from the fitted map (dropna=True above), so
            # a true missing value stringifies to something never present as
            # a key ("nan"/"<NA>") and correctly falls through to 0.0 via
            # fillna, same as an unseen category.
            df[f"{col}_freq"] = df[col].astype(str).map(self.freq_maps_[col]).fillna(0.0).astype(np.float32)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"columns": list(self.columns), "freq_maps": self.freq_maps_}, fh, indent=1)

    @classmethod
    def load(cls, path: pathlib.Path) -> "FrequencyEncoder":
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        enc = cls(columns=tuple(state["columns"]))
        enc.freq_maps_ = state["freq_maps"]
        enc.fitted_ = True
        return enc


@dataclass
class MedianImputer:
    """Median-imputes numeric columns, fit on training data only.

    Same rationale as FrequencyEncoder: the imputation value is a statistic
    of the data, so it must come from TRAIN alone (Phase 1 risk R4) or the
    validation/test distribution leaks into what "typical" looks like.
    """
    columns: tuple[str, ...]
    medians_: dict[str, float] = field(default_factory=dict)
    fitted_: bool = False

    def fit(self, df: pd.DataFrame) -> "MedianImputer":
        self.medians_ = {col: float(df[col].median()) for col in self.columns if col in df.columns}
        self.fitted_ = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("MedianImputer.transform called before fit()")
        df = df.copy()
        for col, med in self.medians_.items():
            df[col] = df[col].fillna(med)
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: pathlib.Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"columns": list(self.columns), "medians": self.medians_}, fh, indent=1)

    @classmethod
    def load(cls, path: pathlib.Path) -> "MedianImputer":
        with open(path, "r", encoding="utf-8") as fh:
            state = json.load(fh)
        imp = cls(columns=tuple(state["columns"]))
        imp.medians_ = state["medians"]
        imp.fitted_ = True
        return imp


def build_base_features(df: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict]:
    """Apply every leakage-safe transform that does not require fitting.

    Returns (transformed_df, log) where log records clip counts etc. for the
    EDA/validation report. Does NOT apply FrequencyEncoder — that must be
    fit on TRAIN and applied per-split by the caller (see scripts/build_processed_data.py).
    """
    log: dict = {}
    df = derive_time_features(df, cfg)
    df = derive_amount_features(df)
    df, clipped = clip_d_columns(df, cfg)
    log["d_columns_clipped_negative_count"] = clipped
    df = add_missingness_indicators(df)
    return df, log


def assert_expanding_feature_is_leakage_safe(
    df: pd.DataFrame,
    time_col: str,
    feature_col: str,
    recompute_fn: Callable[[pd.DataFrame, pd.Series], float],
    n_samples: int = 50,
    seed: int = 0,
    atol: float = 1e-6,
) -> None:
    """Reusable invariant check for any future backward-looking aggregate
    feature (Phase 1 risk R5): recomputing the feature for a sampled row
    using ONLY rows strictly earlier in time must reproduce the stored value.

    No aggregate features exist yet in this foundational phase — this
    checker is infrastructure, ready for when Level 3.5 / graph-entity
    features are introduced, and is exercised now against a synthetic
    leaky-vs-safe example in tests/test_preprocessing.py.

    recompute_fn(history_df, row) -> expected value, where history_df is
    already filtered to df[time_col] < row[time_col].
    """
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(df))
    sample_idx = rng.choice(df.index.to_numpy(), size=n, replace=False)
    mismatches = []
    for idx in sample_idx:
        row = df.loc[idx]
        history = df[df[time_col] < row[time_col]]
        expected = recompute_fn(history, row)
        actual = row[feature_col]
        if pd.isna(expected) and pd.isna(actual):
            continue
        if pd.isna(expected) or pd.isna(actual) or abs(float(expected) - float(actual)) > atol:
            mismatches.append((idx, expected, actual))
    if mismatches:
        raise AssertionError(
            f"Leakage check failed for feature '{feature_col}': {len(mismatches)}/{n} sampled rows "
            f"could not be reproduced using only rows with {time_col} strictly earlier. "
            f"First mismatch: index={mismatches[0][0]} expected={mismatches[0][1]} actual={mismatches[0][2]}"
        )
