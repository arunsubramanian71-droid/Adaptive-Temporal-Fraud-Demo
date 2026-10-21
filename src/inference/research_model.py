"""Research-model inference: the recorded comparison-A XGBoost (experiments/xgboost) served exactly as it was evaluated.

The model file and the TRAIN-fit feature builder saved by scripts/train_xgboost.py are loaded unchanged; rows are encoded with the same
``TreeFeatureBuilder.to_matrix(..., for_model="xgb")`` call the evaluation used and scored with ``predict_proba``; the operating threshold is the
recorded validation best-F1 threshold (``metrics_val.json``). Nothing is re-fitted or re-tuned.

Input contract: one object per transaction in the project's processed-feature schema (the columns of ``data/features/*_features.parquet`` written by
scripts/build_processed_data.py; ``required_columns()`` lists them). A row is scored from its own columns only: the model has no history input, so no
other transaction (earlier or later) can influence its score. Label columns are dropped before encoding and never reach the model.
Explanations are the model's native per-feature contributions (TreeSHAP values computed by XGBoost), as in Phase 11; they describe the model's
output, not causes and not evidence of fraud.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np
import pandas as pd

from src.inference.scorer import refuse_sealed_path

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = "experiments/xgboost"
LABEL_COLUMNS = ("isFraud",)
STATEMENT = ("Research model: the recorded comparison-A XGBoost (trained on days 1-121, threshold = its recorded validation best-F1 threshold). "
             "A score is a model output on the posted row; it is not evidence of fraud, causation or coordination. All recorded results are validation-period results.")


class ResearchInputError(ValueError):
    """The posted rows do not match the model's input contract."""


class ResearchModel:
    def __init__(self, model_dir: str | pathlib.Path = DEFAULT_DIR, root: pathlib.Path = ROOT):
        import joblib
        from src.config import load_config
        from src.models.tree_features import TreeFeatureBuilder
        rel = pathlib.Path(model_dir)
        refuse_sealed_path(rel)
        d = rel if rel.is_absolute() else pathlib.Path(root) / rel
        model_path = d / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} does not exist: the trained research model is not distributed with the repository; "
                                    "regenerate it with scripts/train_xgboost.py on the IEEE-CIS data (README section 4)")
        self.dir = d
        self.model = joblib.load(model_path)
        self.builder = TreeFeatureBuilder.load(load_config(), d / "feature_builder")
        m = json.loads((d / "metrics_val.json").read_text(encoding="utf-8"))
        self.threshold = float(m["threshold"])
        self.validation_metrics = {k: m[k] for k in ("pr_auc", "roc_auc", "precision", "recall", "f1") if k in m}
        self.model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
        roles = json.loads((d / "feature_builder" / "column_roles.json").read_text(encoding="utf-8"))
        extra = {f"{c}_freq" for c in roles["extra_freq_columns"]}
        passthrough_freq = [c for c in self.builder.feature_names_ if c.endswith("_freq") and c not in extra]
        cols = list(roles["continuous_cols"]) + list(roles["raw_time_columns"]) + list(roles["bool_cols"]) + list(roles["extra_freq_columns"]) \
            + list(roles["ordinal_columns"]) + passthrough_freq
        self._required = list(dict.fromkeys(cols))

    def required_columns(self) -> list:
        return list(self._required)

    def info(self) -> dict:
        return {"model": "XGBoost, comparison-A recipe (cell A)", "model_dir": self.dir.relative_to(ROOT).as_posix() if self.dir.is_relative_to(ROOT) else str(self.dir),
                "model_sha256": self.model_sha256, "threshold": self.threshold, "threshold_source": "metrics_val.json (validation best-F1)",
                "recorded_validation_metrics": self.validation_metrics, "n_encoded_features": len(self.builder.feature_names_),
                "n_required_input_columns": len(self._required), "required_input_columns": self.required_columns(), "label_columns_ignored": list(LABEL_COLUMNS),
                "uses_history": False, "statement": STATEMENT}

    def matrix(self, df: pd.DataFrame) -> np.ndarray:
        """Encode rows exactly as the evaluation did (label columns removed first)."""
        df = df.drop(columns=[c for c in LABEL_COLUMNS if c in df.columns])
        missing = [c for c in self._required if c not in df.columns]
        if missing:
            raise ResearchInputError(f"{len(missing)} required input column(s) are missing, e.g. {missing[:10]}; rows must follow the processed-feature schema "
                                     "(GET /api/research/model lists all required columns)")
        try:
            return self.builder.to_matrix(df, for_model="xgb")
        except (ValueError, TypeError) as e:
            raise ResearchInputError(f"a column has a value of the wrong type: {type(e).__name__}: {e}") from None

    def score(self, df: pd.DataFrame, explain: bool = True, top_k: int = 5) -> pd.DataFrame:
        if len(df) == 0:
            raise ResearchInputError("no rows to score")
        X = self.matrix(df)
        p = self.model.predict_proba(X)[:, 1]
        out = pd.DataFrame({"row": np.arange(len(df)), "score": p.astype(np.float64), "alert": p >= self.threshold})
        out["TransactionID"] = df["TransactionID"].to_numpy() if "TransactionID" in df.columns else None
        tops = [None] * len(df)
        if explain and out["alert"].any():
            from src.explain.xgb_contrib import contributions, top_features
            idx = np.flatnonzero(out["alert"].to_numpy())
            contrib, _bias, _margin = contributions(self.model, X[idx])
            for j, i in enumerate(idx):
                tops[i] = top_features(contrib[j], self.builder.feature_names_, X[i], k=top_k)
        out["top_features"] = tops
        return out
