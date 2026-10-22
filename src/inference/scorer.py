"""Production-style scoring of new transactions from stored history (command-line / library API; no web service).

A ``Bundle`` holds an XGBoost model, the column schema, the operating threshold and the list of model columns. ``score`` takes a
HISTORY frame (earlier transactions) and a NEW frame, computes the label-free backward-looking graph features of
``src/features/graph_features.py`` over history + new (a row only ever sees rows with a strictly smaller time), and returns one score
per new transaction. Labels are never read when scoring; if present in the input they are ignored.

Data guard: ``refuse_sealed_path`` rejects TEST-named paths, mirroring the project rule that the held-out period is sealed.
Security note: bundles are pickled with joblib. Load only bundles you created or whose manifest hash you trust.
"""
from __future__ import annotations

import hashlib
import json
import math
import pathlib
import re
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.explain.xgb_contrib import contributions, top_features
from src.features import graph_features as G
from src.models.metrics import select_best_f1_threshold

DEFAULT_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.1, tree_method="hist", eval_metric="aucpr", early_stopping_rounds=20, n_jobs=1)


class ScoringError(ValueError):
    """Invalid input to training or scoring (message says what to fix)."""


class SealedDataError(PermissionError):
    """A TEST-named path was passed; the held-out period is sealed in this project."""


@dataclass(frozen=True)
class Schema:
    features: tuple                      # numeric tabular columns (NaN allowed)
    entities: dict = field(default_factory=lambda: {"pa": "account_key", "card": "card_key", "net": "net_key", "dev": "device_key"})
    id: str = "TransactionID"
    time: str = "TransactionDT"
    amount: str = "TransactionAmt"
    label: str = "isFraud"               # used by train_bundle only

    def required(self, need_label: bool) -> list:
        cols = [self.id, self.time, self.amount, *self.features, *self.entities.values()]
        return cols + ([self.label] if need_label else [])

    def to_json(self) -> dict:
        return {"features": list(self.features), "entities": dict(self.entities), "id": self.id, "time": self.time, "amount": self.amount, "label": self.label}

    @staticmethod
    def from_json(d: dict) -> "Schema":
        return Schema(features=tuple(d["features"]), entities=dict(d["entities"]), id=d["id"], time=d["time"], amount=d["amount"], label=d["label"])


@dataclass
class Bundle:
    model: xgb.XGBClassifier
    schema: Schema
    use_graph: bool
    threshold: float
    columns: list
    meta: dict


# ------------------------------------------------------------------------------------------------ guards and validation
def refuse_sealed_path(path) -> pathlib.Path:
    p = pathlib.Path(path)
    low = str(p).replace("\\", "/").lower().replace("latest", "")
    if re.search(r"(^|[^a-z])test([^a-z]|$)", low) or "test" in p.name.lower().replace("latest", ""):
        raise SealedDataError(f"refusing to read '{p}': the TEST period is sealed in this project (see reports/PROJECT_SYNTHESIS_REPORT.md section 6). "
                              "Rename the file only if it does not contain held-out data.")
    return p


def read_table(path) -> pd.DataFrame:
    p = refuse_sealed_path(path)
    if not p.exists():
        raise ScoringError(f"file not found: {p}")
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    if p.suffix in (".csv", ".txt"):
        return pd.read_csv(p)
    raise ScoringError(f"unsupported file type '{p.suffix}' for {p} (use .csv or .parquet)")


def integer_valued(values, allow_missing: bool = False) -> bool:
    """True if every value is an integer number (an int, or a float without a fractional part). Booleans, strings and other objects are not;
    missing values only if ``allow_missing``. Used for transaction identifiers, which are reported back to the caller and must not be rounded."""
    s = values if isinstance(values, pd.Series) else pd.Series(values, dtype="object")
    if pd.api.types.is_bool_dtype(s):
        return False
    if pd.api.types.is_integer_dtype(s):
        return True
    for v in s.tolist():
        if isinstance(v, (list, dict, tuple, set)):
            return False
        if v is None or v is pd.NA or (isinstance(v, (float, np.floating)) and math.isnan(v)):
            if allow_missing:
                continue
            return False
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, float, np.integer, np.floating)):
            return False
        if isinstance(v, (int, np.integer)):                       # exact, at any size (a float comparison would reject ints above 2**53)
            continue
        if not math.isfinite(v) or float(v) != math.floor(v):
            return False
    return True


def validate_frame(df: pd.DataFrame, schema: Schema, name: str, need_label: bool = False) -> None:
    if not isinstance(df, pd.DataFrame):
        raise ScoringError(f"{name} must be a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in schema.required(need_label) if c not in df.columns]
    if missing:
        raise ScoringError(f"{name} is missing required column(s): {missing}")
    if not integer_valued(df[schema.id]):
        raise ScoringError(f"{name}.{schema.id} must be integers without missing values (a fractional or text identifier cannot be reported back unchanged)")
    if df[schema.id].duplicated().any():
        raise ScoringError(f"{name} has duplicated {schema.id} values")
    if not pd.api.types.is_numeric_dtype(df[schema.time]) or df[schema.time].isna().any():
        raise ScoringError(f"{name}.{schema.time} must be numeric seconds without missing values")
    if (df[schema.time] < 0).any() or (df[schema.time] >= G.TIME_LIMIT).any():
        raise ScoringError(f"{name}.{schema.time} must lie in [0, {G.TIME_LIMIT}) seconds")
    for c in (schema.amount, *schema.features):
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ScoringError(f"{name}.{c} must be numeric")
    if not np.isfinite(df[schema.amount].to_numpy(dtype=np.float64)).all():
        raise ScoringError(f"{name}.{schema.amount} must be finite numbers without missing values (a missing or infinite amount would make the backward-looking "
                           "amount features of every later transaction of the same entity missing or infinite)")
    if need_label and not set(pd.unique(df[schema.label].dropna())) <= {0, 1}:
        raise ScoringError(f"{name}.{schema.label} must contain only 0/1")


# ------------------------------------------------------------------------------------------------ features
def _entity_ids(df: pd.DataFrame, schema: Schema) -> dict:
    out = {}
    for rel in G.RELATIONS:
        col = df[schema.entities[rel]]
        codes, _ = pd.factorize(col.astype("object").where(col.notna(), None), use_na_sentinel=True)
        out[rel] = codes.astype(np.int64)            # -1 where the key is missing
    return out


def model_matrix(df: pd.DataFrame, schema: Schema, use_graph: bool) -> tuple:
    """Return (X float32, column names) for a frame already ordered by time; graph features see only strictly earlier rows."""
    tab = df[list(schema.features)].to_numpy(dtype=np.float32)
    cols = list(schema.features)
    if not use_graph:
        return tab, cols
    gf = G.compute_graph_features(df[schema.time].to_numpy(dtype=np.int64), df[schema.amount].to_numpy(dtype=np.float64), _entity_ids(df, schema))
    return np.hstack([tab, gf.to_numpy(dtype=np.float32)]), cols + G.feature_names()


def _ordered(df: pd.DataFrame, schema: Schema) -> pd.DataFrame:
    return df.sort_values([schema.time, schema.id], kind="stable").reset_index(drop=True)


# ------------------------------------------------------------------------------------------------ training
def train_bundle(train: pd.DataFrame, valid: pd.DataFrame, schema: Schema, use_graph: bool = True, params: dict | None = None, seed: int = 42) -> Bundle:
    """Fit on ``train``; early stopping and the best-F1 threshold use ``valid`` (the project's TRAIN/VAL protocol). ``valid`` must
    be strictly later than ``train``."""
    validate_frame(train, schema, "train", need_label=True)
    validate_frame(valid, schema, "valid", need_label=True)
    if train[schema.time].max() >= valid[schema.time].min():
        raise ScoringError("valid must start strictly after train ends (chronological split)")
    if train[schema.label].sum() == 0 or train[schema.label].sum() == len(train):
        raise ScoringError("train needs both classes")
    allr = _ordered(pd.concat([train, valid], ignore_index=True), schema)
    X, cols = model_matrix(allr, schema, use_graph)
    is_train = allr[schema.id].isin(train[schema.id]).to_numpy()
    y = allr[schema.label].to_numpy()
    spw = float((y[is_train] == 0).sum() / max((y[is_train] == 1).sum(), 1))
    clf = xgb.XGBClassifier(**{**DEFAULT_PARAMS, **(params or {})}, scale_pos_weight=spw, random_state=seed)
    clf.fit(X[is_train], y[is_train], eval_set=[(X[~is_train], y[~is_train])], verbose=False)
    score = clf.predict_proba(X[~is_train])[:, 1]
    thr = select_best_f1_threshold(y[~is_train], score)
    meta = {"seed": seed, "n_train": int(is_train.sum()), "n_valid": int((~is_train).sum()), "best_iteration": int(clf.best_iteration),
            "threshold_selection": thr, "use_graph_features": use_graph, "xgboost": xgb.__version__}
    return Bundle(clf, schema, use_graph, float(thr["threshold"]), cols, meta)


# ------------------------------------------------------------------------------------------------ scoring
def score(bundle: Bundle, history: pd.DataFrame | None, new: pd.DataFrame, explain: bool = False, top_k: int = 5) -> pd.DataFrame:
    """Score ``new`` using ``history`` (earlier transactions; may be empty/None). Returns TransactionID, score, alert (score >= threshold)
    and, if ``explain``, the ``top_k`` model contributions for each ALERTED row as JSON text."""
    sch = bundle.schema
    validate_frame(new, sch, "new")
    if history is None or len(history) == 0:
        history = new.iloc[0:0]
    else:
        validate_frame(history, sch, "history")
        if history[sch.id].isin(new[sch.id]).any():
            raise ScoringError("history and new share TransactionID values")
        if len(new) and history[sch.time].max() > new[sch.time].min():
            raise ScoringError("history contains transactions later than the earliest new transaction; pass only earlier history "
                               "(scores would otherwise be computed on a timeline that includes the future of some new rows)")
    keep = [c for c in sch.required(False)]
    allr = _ordered(pd.concat([history[keep], new[keep]], ignore_index=True), sch)
    X, cols = model_matrix(allr, sch, bundle.use_graph)
    if cols != bundle.columns:
        raise ScoringError("feature columns do not match the bundle (schema changed since training)")
    is_new = allr[sch.id].isin(new[sch.id]).to_numpy()
    Xn = X[is_new]
    proba = bundle.model.predict_proba(Xn)[:, 1]
    out = pd.DataFrame({sch.id: allr.loc[is_new, sch.id].to_numpy(), "score": proba, "alert": proba >= bundle.threshold})
    out = out.set_index(sch.id).loc[new[sch.id]].reset_index()          # return in the caller's row order
    if explain:
        order = {i: k for k, i in enumerate(allr.loc[is_new, sch.id].to_numpy())}
        contrib, _, _ = contributions(bundle.model, Xn)
        txt = []
        for tid, a in zip(out[sch.id], out["alert"]):
            if not a:
                txt.append(None)
                continue
            k = order[tid]
            txt.append(json.dumps(top_features(contrib[k], cols, Xn[k], k=top_k)))
        out["top_features"] = txt
    return out


# ------------------------------------------------------------------------------------------------ persistence
def save_bundle(bundle: Bundle, directory) -> dict:
    d = pathlib.Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": bundle.model, "columns": bundle.columns}, d / "bundle.joblib")
    h = hashlib.sha256((d / "bundle.joblib").read_bytes()).hexdigest()
    manifest = {"schema": bundle.schema.to_json(), "use_graph": bundle.use_graph, "threshold": bundle.threshold, "columns": bundle.columns,
                "meta": bundle.meta, "bundle_sha256": h}
    (d / "bundle_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def load_bundle(directory) -> Bundle:
    d = pathlib.Path(directory)
    refuse_sealed_path(d)
    try:
        manifest = json.loads((d / "bundle_manifest.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ScoringError(f"no bundle_manifest.json in {d}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ScoringError(f"bundle_manifest.json in {d} is damaged (not valid JSON: {e}); rebuild the bundle with scripts/score_transactions.py train") from None
    if not isinstance(manifest, dict) or not {"bundle_sha256", "schema", "use_graph", "threshold"} <= set(manifest):
        raise ScoringError(f"bundle_manifest.json in {d} lacks required fields; rebuild the bundle with scripts/score_transactions.py train")
    h = hashlib.sha256((d / "bundle.joblib").read_bytes()).hexdigest()
    if h != manifest["bundle_sha256"]:
        raise ScoringError("bundle.joblib does not match the hash in bundle_manifest.json; refusing to load it")
    obj = joblib.load(d / "bundle.joblib")
    return Bundle(obj["model"], Schema.from_json(manifest["schema"]), manifest["use_graph"], manifest["threshold"], obj["columns"], manifest["meta"])
