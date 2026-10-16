"""Logistic-Regression-specific feature encoding.

This is deliberately kept separate from src/features/preprocessing.py (the
Phase 2 canonical, model-agnostic pipeline output in data/features/). Linear
models need choices tree models don't: bounded/scaled inputs, no raw
high-cardinality magnitudes, and cyclical time encoding. Random Forest and
XGBoost (later baselines) will consume data/features/*.parquet more directly.

Column classification (from data/features/train_features.parquet, 469 cols):
  - 1 target (isFraud), 3 blocklisted (TransactionID, TransactionDT — card1 is
    NOT blocklisted, see below)
  - 6 columns already frequency-encoded in Phase 2 (ProductCD, card4, card6,
    DeviceType, P_emaildomain, R_emaildomain -> *_freq)
  - 14 C*, 15 D*, 339 V*, 2 dist*, 23 numeric id_* -> kept as continuous,
    median-imputed (fit TRAIN only)
  - TransactionAmt -> log1p (heavy right skew, Phase 1 Section 6)
  - hour, day_of_week -> sin/cos (cyclical; raw integers imply a false
    ordinal distance for a *linear* model specifically — tree models don't
    have this problem and keep the raw columns)
  - 27 bool columns (has_identity, amt_decimals_gt2, 25 *_isna flags) -> used
    directly as 0/1
  - 25 low-cardinality string columns (<=5 levels: ProductCD, card4, card6,
    M1-M9, DeviceType, id_12/15/16/23/27/28/29/34/35/36/37/38) -> one-hot,
    with an explicit "missing" level rather than imputing (Phase 1: identity
    missingness is itself informative, so absence must survive as a level)
  - card1, card2, card3, card5, addr1, addr2, id_30, id_31, id_33, DeviceInfo
    -> frequency-encoded HERE (extending Phase 2's 6), not one-hot and not
    raw. Phase 1 Section 10 identifies these as anonymized entity/identifier
    tokens (Card, PseudoAccount components) whose raw magnitude has no
    linear meaning; per this project's explicit instruction, they are not
    blindly one-hot encoded either (up to 13,553 levels for card1). dist1/
    dist2 are excluded from this list — Kaggle documents them as literal
    distances, genuinely continuous, not identifier tokens.

Every fitted statistic (frequency maps, medians, one-hot categories, scaler
mean/std) is fit on the TRAIN split only and reused, unmodified, for VAL/TEST
(Phase 1 risk R4 / Phase 2 validation check "preprocessing_fitted_train_only").
"""
from __future__ import annotations

import json
import pathlib
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import Config
from src.features.preprocessing import FrequencyEncoder, MedianImputer

ONEHOT_COLUMNS = [
    "ProductCD", "card4", "card6", "DeviceType",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "id_12", "id_15", "id_16", "id_23", "id_27", "id_28", "id_29", "id_34",
    "id_35", "id_36", "id_37", "id_38",
]
EXTRA_FREQ_COLUMNS = ["card1", "card2", "card3", "card5", "addr1", "addr2", "id_30", "id_31", "id_33", "DeviceInfo"]
CONTINUOUS_PREFIXES = ("C", "D", "V")  # C1..C14, D1..D15, V1..V339
CONTINUOUS_EXTRA = ["dist1", "dist2"]
MISSING_SENTINEL = "__MISSING__"


def _continuous_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for c in df.columns:
        if c[0] in CONTINUOUS_PREFIXES and c[1:].isdigit():
            cols.append(c)
    id_numeric = [c for c in df.columns if c.startswith("id_") and pd.api.types.is_numeric_dtype(df[c])]
    return sorted(set(cols) | set(id_numeric) | set(c for c in CONTINUOUS_EXTRA if c in df.columns))


def _bool_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_bool_dtype(df[c])]


@dataclass
class LogisticRegressionFeatureBuilder:
    cfg: Config
    freq_encoder_: FrequencyEncoder = None
    imputer_: MedianImputer = None
    scaler_: StandardScaler = None
    onehot_categories_: dict[str, list[str]] = field(default_factory=dict)
    continuous_cols_: list[str] = field(default_factory=list)
    bool_cols_: list[str] = field(default_factory=list)
    feature_names_: list[str] = field(default_factory=list)
    fitted_: bool = False

    def _pre_numeric_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """Everything that does not depend on fitted state: log1p, sin/cos,
        frequency lookups (via the already-fitted encoder), one-hot lookups
        (via already-fitted category lists). Safe to call before or after fit.

        Builds all columns as a dict first and does a single pd.concat,
        rather than assigning one column at a time onto a growing frame —
        with ~380 continuous+bool columns, per-column assignment triggers
        pandas' internal fragmentation warning and is measurably slower.
        """
        cols = {
            "log1p_TransactionAmt": np.log1p(df["TransactionAmt"].astype(np.float64)).astype(np.float32),
            "hour_sin": np.sin(2 * np.pi * df["hour"].astype(np.float32) / 24),
            "hour_cos": np.cos(2 * np.pi * df["hour"].astype(np.float32) / 24),
            "dow_sin": np.sin(2 * np.pi * df["day_of_week"].astype(np.float32) / 7),
            "dow_cos": np.cos(2 * np.pi * df["day_of_week"].astype(np.float32) / 7),
        }
        for c in self.continuous_cols_:
            cols[c] = df[c].astype(np.float32)
        for c in self.bool_cols_:
            cols[c] = df[c].astype(np.float32)
        return pd.DataFrame(cols, index=df.index)

    def fit(self, train_df: pd.DataFrame) -> "LogisticRegressionFeatureBuilder":
        self.continuous_cols_ = _continuous_columns(train_df)
        self.bool_cols_ = _bool_columns(train_df)

        self.freq_encoder_ = FrequencyEncoder(columns=tuple(EXTRA_FREQ_COLUMNS)).fit(train_df)

        self.onehot_categories_ = {}
        for col in ONEHOT_COLUMNS:
            filled = train_df[col].astype("object").where(train_df[col].notna(), MISSING_SENTINEL)
            self.onehot_categories_[col] = sorted(str(v) for v in filled.unique())

        pre = self._pre_numeric_frame(train_df)
        pre = pd.concat([pre, self._freq_frame(train_df)], axis=1)

        self.imputer_ = MedianImputer(columns=tuple(self.continuous_cols_)).fit(pre)
        pre = self.imputer_.transform(pre)

        onehot = self._onehot_frame(train_df)
        full = pd.concat([pre, onehot], axis=1)
        self.feature_names_ = list(full.columns)

        self.scaler_ = StandardScaler()
        self.scaler_.fit(full.to_numpy(dtype=np.float32))

        self.fitted_ = True
        return self

    def _freq_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        """The 6 Phase-2 frequency columns (already computed, just selected)
        plus the 10 extra ones fit here (EXTRA_FREQ_COLUMNS), assembled with
        one concat rather than per-column assignment."""
        freq_df = self.freq_encoder_.transform(df)
        extra_cols = [f"{c}_freq" for c in EXTRA_FREQ_COLUMNS]
        phase2_cols = [c for c in df.columns if c.endswith("_freq") and c not in extra_cols]
        cols = {c: freq_df[c].astype(np.float32) for c in extra_cols + phase2_cols}
        return pd.DataFrame(cols, index=df.index)

    def _onehot_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = []
        for col in ONEHOT_COLUMNS:
            filled = df[col].astype("object").where(df[col].notna(), MISSING_SENTINEL).astype(str)
            cats = self.onehot_categories_[col]
            block = pd.DataFrame(
                {f"{col}__{cat}": (filled == cat).astype(np.float32) for cat in cats},
                index=df.index,
            )
            parts.append(block)
        return pd.concat(parts, axis=1)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("LogisticRegressionFeatureBuilder.transform called before fit()")
        pre = self._pre_numeric_frame(df)
        pre = pd.concat([pre, self._freq_frame(df)], axis=1)
        pre = self.imputer_.transform(pre)
        onehot = self._onehot_frame(df)
        full = pd.concat([pre, onehot], axis=1)[self.feature_names_]  # enforce identical column order
        return self.scaler_.transform(full.to_numpy(dtype=np.float32))

    def save(self, out_dir: pathlib.Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        self.freq_encoder_.save(out_dir / "extra_frequency_encoder.json")
        self.imputer_.save(out_dir / "median_imputer.json")
        joblib.dump(self.scaler_, out_dir / "standard_scaler.joblib")
        with open(out_dir / "onehot_categories.json", "w", encoding="utf-8") as fh:
            json.dump(self.onehot_categories_, fh, indent=1)
        with open(out_dir / "feature_names.json", "w", encoding="utf-8") as fh:
            json.dump(self.feature_names_, fh, indent=1)
        with open(out_dir / "column_roles.json", "w", encoding="utf-8") as fh:
            json.dump({"continuous_cols": self.continuous_cols_, "bool_cols": self.bool_cols_,
                       "onehot_columns": ONEHOT_COLUMNS, "extra_freq_columns": EXTRA_FREQ_COLUMNS}, fh, indent=1)

    @classmethod
    def load(cls, cfg: Config, in_dir: pathlib.Path) -> "LogisticRegressionFeatureBuilder":
        b = cls(cfg=cfg)
        b.freq_encoder_ = FrequencyEncoder.load(in_dir / "extra_frequency_encoder.json")
        b.imputer_ = MedianImputer.load(in_dir / "median_imputer.json")
        b.scaler_ = joblib.load(in_dir / "standard_scaler.joblib")
        with open(in_dir / "onehot_categories.json", "r", encoding="utf-8") as fh:
            b.onehot_categories_ = json.load(fh)
        with open(in_dir / "feature_names.json", "r", encoding="utf-8") as fh:
            b.feature_names_ = json.load(fh)
        with open(in_dir / "column_roles.json", "r", encoding="utf-8") as fh:
            roles = json.load(fh)
        b.continuous_cols_ = roles["continuous_cols"]
        b.bool_cols_ = roles["bool_cols"]
        b.fitted_ = True
        return b
