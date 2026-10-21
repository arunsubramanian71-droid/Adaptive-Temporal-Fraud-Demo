"""Phase 11 first pass: native XGBoost contributions and the label-free transaction selection.

Scope record: experiments/phase11_explainable_evidence/SCOPE_AND_ACCEPTANCE.md (P1-P4). No new dependency: contributions come from the
booster's own ``pred_contribs`` (Tree SHAP in log-odds margin space). Contributions explain a model's output; they are not evidence of
fraud, causality or coordination. Nothing here reads, uses or needs a fraud label: the loaders project the label column away and the
selection function receives only ids, days and model scores.
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb

VAL_DAYS = (122, 151)           # TRAIN 1-121, VAL 122-151, TEST (152-182) sealed (src/config.py, frozen registry)
SELECTION_SEED = 40042          # 40000 + 42, the Phase 9 case-study sampling convention
N_TOP, N_DRAW = 5, 5
DAY_SECONDS = 86400
_NEVER_LOADED = ("isFraud",)    # columns excluded from every read


# ------------------------------------------------------------------------------------------------ contributions
def _iteration_range(model):
    try:
        return (0, int(model.best_iteration) + 1)      # what predict_proba uses after early stopping
    except (AttributeError, TypeError):
        return (0, 0)                                   # all trees


def contributions(model, X):
    """Return (contribs [n, F], bias [n], margin [n]) for a fitted ``XGBClassifier`` and a float32 matrix (NaN stays native).

    ``contribs.sum(axis=1) + bias == margin`` up to float32 rounding (acceptance test A1).
    """
    X = np.asarray(X, dtype=np.float32)
    booster = model.get_booster()
    ir = _iteration_range(model)
    dm = xgb.DMatrix(X)
    raw = booster.predict(dm, pred_contribs=True, iteration_range=ir)
    margin = booster.predict(dm, output_margin=True, iteration_range=ir)
    return raw[:, :-1], raw[:, -1], margin


def sigmoid(m):
    m = np.asarray(m, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-m))


def top_features(contrib_row, feature_names, values_row, k=10):
    """The k features with the largest absolute contribution (ties: lower column index first)."""
    order = np.lexsort((np.arange(len(contrib_row)), -np.abs(contrib_row)))[:k]
    return [{"feature": feature_names[i], "value": None if np.isnan(values_row[i]) else float(values_row[i]),
             "contribution": float(contrib_row[i])} for i in order]


# ------------------------------------------------------------------------------------------------ selection (label-free)
def select_transactions(transaction_id, day, score, seed=SELECTION_SEED, n_top=N_TOP, n_draw=N_DRAW, day_range=VAL_DAYS):
    """Per day: the n_top highest model scores (ties: lower TransactionID first) plus n_draw uniform draws without replacement from the
    day's remaining transactions, ``default_rng(SeedSequence([seed, day]))``. The groups are disjoint by construction.

    Inputs are ids, days and model scores only: there is no label argument.
    """
    ids = np.asarray(transaction_id, dtype=np.int64)
    days = np.asarray(day, dtype=np.int64)
    sc = np.asarray(score, dtype=np.float64)
    if not (len(ids) == len(days) == len(sc)):
        raise ValueError("ids, days and scores must have equal length")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("TransactionID must be unique")
    if not np.all(np.isfinite(sc)):
        raise ValueError("scores must be finite")
    lo, hi = day_range
    if days.size and (days.min() < lo or days.max() > hi):
        raise ValueError(f"days outside {lo}-{hi} are not allowed (TEST is sealed)")
    rows = []
    for d in range(lo, hi + 1):
        m = days == d
        if m.sum() < n_top + n_draw:
            raise ValueError(f"day {d} has fewer than {n_top + n_draw} transactions")
        idd, ss = ids[m], sc[m]
        order = np.lexsort((idd, -ss))
        top = idd[order[:n_top]]
        score_of = dict(zip(idd.tolist(), ss.tolist()))
        for r, t in enumerate(top.tolist(), start=1):
            rows.append((t, d, "top_score", r, score_of[t]))
        rest = np.sort(idd[~np.isin(idd, top)])
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), d]))
        pick = np.sort(rest[rng.choice(len(rest), size=n_draw, replace=False)])
        for r, t in enumerate(pick.tolist(), start=1):
            rows.append((t, d, "uniform_draw", r, score_of[t]))
    return pd.DataFrame(rows, columns=["TransactionID", "day", "group", "rank_in_group", "y_score"])


# ------------------------------------------------------------------------------------------------ loaders (label never loaded)
def _guard_path(path):
    p = pathlib.Path(path)
    if "test" in p.name.lower():
        raise ValueError(f"refusing to open a TEST-named file: {p.name}")
    return p


def load_ids_days(features_path, day_seconds=DAY_SECONDS, day_range=VAL_DAYS):
    """TransactionID and day (``TransactionDT // day_seconds``) for the VAL file; any day outside the VAL range is an error."""
    t = pq.read_table(_guard_path(features_path), columns=["TransactionID", "TransactionDT"]).to_pandas()
    t["day"] = (t["TransactionDT"] // day_seconds).astype(np.int64)
    if t["day"].min() < day_range[0] or t["day"].max() > day_range[1]:
        raise ValueError("features file contains days outside the VAL window")
    return t[["TransactionID", "day"]]


def load_scores(pred_path):
    """Stored model scores only (TransactionID, y_score); the stored label and prediction columns are not read."""
    return pq.read_table(_guard_path(pred_path), columns=["TransactionID", "y_score"]).to_pandas()


def load_feature_rows(features_path, ids):
    """Full feature rows for the selected ids, with the label column projected away. Returned in the order of ``ids``."""
    p = _guard_path(features_path)
    names = [n for n in pq.read_schema(p).names if n not in _NEVER_LOADED]
    ids = [int(i) for i in ids]
    t = pq.read_table(p, columns=names, filters=[("TransactionID", "in", ids)]).to_pandas()
    t = t.set_index("TransactionID").loc[ids].reset_index()
    return t
