"""Random Forest / XGBoost feature encoding.

Deliberately different from src/models/lr_features.py: tree models split on
thresholds, not distances, so they don't suffer a linear model's "raw
magnitude implies a false ordering" problem, and they don't need scaling.
This changes several choices relative to the LR baseline:

  - card1, card2, card3, card5, addr1, addr2: kept as RAW numeric (not
    frequency-encoded). This is standard, well-established practice for
    tree models on this exact dataset — unlike LR, a tree can use an
    anonymized numeric token's raw value productively via arbitrary
    threshold splits without implying it's actually ordinal.
  - TransactionAmt: kept raw (no log1p — trees are monotonic-transform
    invariant, so log1p buys nothing here).
  - hour, day_of_week: kept as raw integers (no sin/cos — a tree can
    approximate the wraparound with a few extra splits if it's useful, and
    usually isn't worth the extra encoding complexity for tree models).
  - Low-cardinality categoricals (25 columns, same list as LR's one-hot set):
    ordinal-encoded instead of one-hot, to avoid ~65 sparse dummy columns
    that gain trees nothing they can't get from a single integer-coded
    column split on repeatedly. Missing and "unseen in TRAIN" are two
    DISTINCT reserved integer codes (not NaN) — decided this way specifically
    so RandomForest (which cannot accept NaN at all) needs no separate
    imputation pass for these columns, and XGBoost sees an equally valid,
    explicit signal either way.
  - id_30, id_31, id_33, DeviceInfo (high-cardinality strings, up to 1,551
    levels): frequency-encoded, reusing the exact same encoder and column
    list as the LR baseline (src/models/lr_features.EXTRA_FREQ_COLUMNS) —
    per this project's explicit instruction not to blindly one-hot (or,
    for that matter, arbitrarily ordinal-encode, which would imply a
    meaningless order across ~100-1500 levels) extremely high-cardinality
    identifiers.

Missing-value handling — the one place RF and XGBoost genuinely diverge:
  - After the above, the only remaining real NaNs are in the continuous
    numeric block (C*, D*, V*, dist1/2, numeric id_*, card1/2/3/5, addr1/2,
    TransactionAmt). Frequency-encoded columns never contain NaN (the
    encoder maps missing/unseen to 0.0 by construction); ordinal columns
    never contain NaN (missing is its own reserved code); boolean flags
    never contain NaN.
  - XGBoost: NaN is left as-is and handled by XGBoost's native learned
    "default direction" for missing values at each split — this is a
    genuine strength of gradient-boosted trees and is used deliberately
    here, not worked around.
  - RandomForest (scikit-learn) cannot accept NaN at all, so the continuous
    block is median-imputed (fit on TRAIN only, reusing
    src/features/preprocessing.MedianImputer) for the RF matrix specifically.
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.config import Config
from src.features.preprocessing import FrequencyEncoder, MedianImputer
from src.models.lr_features import EXTRA_FREQ_COLUMNS, ONEHOT_COLUMNS as ORDINAL_COLUMNS, MISSING_SENTINEL

CONTINUOUS_PREFIXES = ("C", "D", "V")
CONTINUOUS_EXTRA = ["dist1", "dist2", "card1", "card2", "card3", "card5", "addr1", "addr2", "TransactionAmt"]
RAW_TIME_COLUMNS = ["hour", "day_of_week"]
UNSEEN_CODE_OFFSET = 1  # reserved code = len(train categories), i.e. one past the highest fitted code


def _continuous_columns(df: pd.DataFrame) -> list[str]:
    cols = [c for c in df.columns if c[0] in CONTINUOUS_PREFIXES and c[1:].isdigit()]
    id_numeric = [c for c in df.columns if c.startswith("id_") and pd.api.types.is_numeric_dtype(df[c])]
    extra = [c for c in CONTINUOUS_EXTRA if c in df.columns]
    return sorted(set(cols) | set(id_numeric) | set(extra))


def _bool_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]


@dataclass
class TreeFeatureBuilder:
    """Shared encoding for Random Forest and XGBoost. Produces one encoded
    frame; `to_matrix(..., for_model=...)` applies the one point of
    divergence (missing-value handling) at the end."""
    cfg: Config
    freq_encoder_: FrequencyEncoder = None
    imputer_: MedianImputer = None
    ordinal_maps_: dict[str, dict[str, int]] = field(default_factory=dict)
    ordinal_unseen_code_: dict[str, int] = field(default_factory=dict)
    continuous_cols_: list[str] = field(default_factory=list)
    bool_cols_: list[str] = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)
    fitted_: bool = False

    def _ordinal_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        cols = {}
        for col in ORDINAL_COLUMNS:
            filled = df[col].astype("object").where(df[col].notna(), MISSING_SENTINEL).astype(str)
            codes = filled.map(self.ordinal_maps_[col])
            codes = codes.fillna(self.ordinal_unseen_code_[col]).astype(np.int32)
            cols[f"{col}_ord"] = codes
        return pd.DataFrame(cols, index=df.index)

    def _freq_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        freq_df = self.freq_encoder_.transform(df)
        extra_cols = [f"{c}_freq" for c in EXTRA_FREQ_COLUMNS]
        phase2_cols = [c for c in df.columns if c.endswith("_freq") and c not in extra_cols]
        cols = {c: freq_df[c].astype(np.float32) for c in extra_cols + phase2_cols}
        return pd.DataFrame(cols, index=df.index)

    def _base_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Continuous (NaN preserved) + raw time + bool + freq + ordinal, all
        assembled with one concat (see Phase 2/3's earlier fragmentation-
        warning fix — same reasoning applies here with ~400+ columns)."""
        cols = {c: df[c].astype(np.float32) for c in self.continuous_cols_}
        for c in RAW_TIME_COLUMNS:
            cols[c] = df[c].astype(np.float32)
        for c in self.bool_cols_:
            cols[c] = df[c].astype(np.float32)
        base = pd.DataFrame(cols, index=df.index)
        return pd.concat([base, self._freq_frame(df), self._ordinal_frame(df)], axis=1)

    def fit(self, train_df: pd.DataFrame) -> "TreeFeatureBuilder":
        self.continuous_cols_ = _continuous_columns(train_df)
        self.bool_cols_ = _bool_columns(train_df)
        self.freq_encoder_ = FrequencyEncoder(columns=tuple(EXTRA_FREQ_COLUMNS)).fit(train_df)

        self.ordinal_maps_, self.ordinal_unseen_code_ = {}, {}
        for col in ORDINAL_COLUMNS:
            filled = train_df[col].astype("object").where(train_df[col].notna(), MISSING_SENTINEL).astype(str)
            cats = sorted(filled.unique())
            self.ordinal_maps_[col] = {cat: i for i, cat in enumerate(cats)}
            self.ordinal_unseen_code_[col] = len(cats)  # a value never seen in TRAIN maps here, distinct from MISSING

        full = self._base_frame(train_df)
        self.feature_names_ = list(full.columns)

        # Fit the RF-specific imputer even though XGBoost won't use it —
        # both matrices must be derivable from one fitted builder.
        self.imputer_ = MedianImputer(columns=tuple(self.continuous_cols_)).fit(full)

        self.fitted_ = True
        return self

    def to_matrix(self, df: pd.DataFrame, for_model: str) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TreeFeatureBuilder.to_matrix called before fit()")
        if for_model not in ("xgb", "rf"):
            raise ValueError(f"for_model must be 'xgb' or 'rf', got {for_model!r}")
        full = self._base_frame(df)[self.feature_names_]
        if for_model == "rf":
            full = self.imputer_.transform(full)
        return full.to_numpy(dtype=np.float32)

    def save(self, out_dir: pathlib.Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.freq_encoder_.save(out_dir / "extra_frequency_encoder.json")
        self.imputer_.save(out_dir / "median_imputer.json")
        with open(out_dir / "ordinal_maps.json", "w", encoding="utf-8") as fh:
            json.dump({"maps": self.ordinal_maps_, "unseen_codes": self.ordinal_unseen_code_}, fh, indent=1)
        with open(out_dir / "feature_names.json", "w", encoding="utf-8") as fh:
            json.dump(self.feature_names_, fh, indent=1)
        with open(out_dir / "column_roles.json", "w", encoding="utf-8") as fh:
            json.dump({"continuous_cols": self.continuous_cols_, "bool_cols": self.bool_cols_,
                       "ordinal_columns": ORDINAL_COLUMNS, "extra_freq_columns": EXTRA_FREQ_COLUMNS,
                       "raw_time_columns": RAW_TIME_COLUMNS}, fh, indent=1)

    @classmethod
    def load(cls, cfg: Config, in_dir: pathlib.Path) -> "TreeFeatureBuilder":
        b = cls(cfg=cfg)
        b.freq_encoder_ = FrequencyEncoder.load(in_dir / "extra_frequency_encoder.json")
        b.imputer_ = MedianImputer.load(in_dir / "median_imputer.json")
        with open(in_dir / "ordinal_maps.json", "r", encoding="utf-8") as fh:
            state = json.load(fh)
        b.ordinal_maps_, b.ordinal_unseen_code_ = state["maps"], state["unseen_codes"]
        with open(in_dir / "feature_names.json", "r", encoding="utf-8") as fh:
            b.feature_names_ = json.load(fh)
        with open(in_dir / "column_roles.json", "r", encoding="utf-8") as fh:
            roles = json.load(fh)
        b.continuous_cols_, b.bool_cols_ = roles["continuous_cols"], roles["bool_cols"]
        b.fitted_ = True
        return b
