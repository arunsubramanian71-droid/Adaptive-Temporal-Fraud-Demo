"""Relationship-level evidence for the graph-feature XGBoost model B (experiments/comparison_B_graph_features/GRAPH_EVIDENCE_PREREG.md).

Pure functions on a fitted booster, a feature matrix and its column names. No label is an argument anywhere. The perturbation is a
feature-level counterfactual (the relation's features are set to missing), not an edit of the underlying graph.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import xgboost as xgb

from src.explain.xgb_contrib import _iteration_range
from src.features.graph_features import RELATIONS

_REL = re.compile(r"^g_(pa|card|net|dev)_")


def relation_of_feature(name: str):
    """The relation a graph feature belongs to (its own entity's relation), or None for a tabular column."""
    m = _REL.match(name)
    return m.group(1) if m else None


def relation_columns(columns) -> dict:
    out = {r: [] for r in RELATIONS}
    for i, c in enumerate(columns):
        r = relation_of_feature(c)
        if r:
            out[r].append(i)
    return out


def relation_influence(contrib: np.ndarray, columns) -> pd.DataFrame:
    """Signed summed contribution per relation plus the tabular block; columns: pa, card, net, dev, tabular."""
    rc = relation_columns(columns)
    out = {r: contrib[:, idx].sum(axis=1) if idx else np.zeros(len(contrib)) for r, idx in rc.items()}
    graph_idx = {i for idx in rc.values() for i in idx}
    tab = [i for i in range(len(columns)) if i not in graph_idx]
    out["tabular"] = contrib[:, tab].sum(axis=1)
    return pd.DataFrame(out)


def present_relations(X: np.ndarray, columns) -> np.ndarray:
    """Boolean [n, 4] (order of RELATIONS): the row has an edge for the relation."""
    idx = [list(columns).index(f"g_{r}_has_edge") for r in RELATIONS]
    return X[:, idx] > 0.5


def perturb_relation(X: np.ndarray, columns, rel: str) -> np.ndarray:
    """Copy of X where the relation looks absent: has_edge = 0 and every feature of the relation missing (NaN)."""
    if rel not in RELATIONS:
        raise ValueError(f"unknown relation '{rel}'")
    Xp = np.array(X, dtype=np.float32, copy=True)
    cols = list(columns)
    for i in relation_columns(cols)[rel]:
        Xp[:, i] = 0.0 if cols[i] == f"g_{rel}_has_edge" else np.nan
    return Xp


def margin(model, X: np.ndarray) -> np.ndarray:
    return model.get_booster().predict(xgb.DMatrix(np.asarray(X, dtype=np.float32)), output_margin=True, iteration_range=_iteration_range(model))


def perturbation_effects(model, X: np.ndarray, columns) -> pd.DataFrame:
    """Margin change (perturbed - original) when each relation is removed, one column per relation; NaN where the row has no such edge."""
    base = margin(model, X)
    present = present_relations(X, columns)
    out = {}
    for k, r in enumerate(RELATIONS):
        d = margin(model, perturb_relation(X, columns, r)) - base
        out[r] = np.where(present[:, k], d, np.nan)
    return pd.DataFrame(out)


def pick_relations(ids, present: np.ndarray, abs_influence: pd.DataFrame, seed: int = 40042):
    """Per row with >= 1 present relation: the present relation with the largest absolute summed contribution (ties: RELATIONS order) and
    one drawn uniformly at random among the present relations (``default_rng(SeedSequence([seed, id]))``). Rows without any present
    relation get None."""
    top, rnd = [], []
    for k, tid in enumerate(np.asarray(ids, dtype=np.int64)):
        pres = [j for j in range(len(RELATIONS)) if present[k, j]]
        if not pres:
            top.append(None)
            rnd.append(None)
            continue
        vals = [abs_influence.iloc[k][RELATIONS[j]] for j in pres]
        top.append(RELATIONS[pres[int(np.argmax(vals))]])
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(tid)]))
        rnd.append(RELATIONS[pres[int(rng.integers(0, len(pres)))]])
    return top, rnd


def perturbation_sanity(effects: pd.DataFrame, present: np.ndarray, top, rnd) -> dict:
    """GE4: among rows with >= 2 present relations, mean |effect| of removing the top-ranked relation vs the randomly drawn one."""
    multi = present.sum(axis=1) >= 2
    a = [abs(effects.iloc[k][t]) for k in np.nonzero(multi)[0] for t in [top[k]]]
    b = [abs(effects.iloc[k][r]) for k in np.nonzero(multi)[0] for r in [rnd[k]]]
    n = len(a)
    ma, mb = (float(np.mean(a)), float(np.mean(b))) if n else (float("nan"), float("nan"))
    return {"n_rows_with_two_or_more_relations": n, "mean_abs_effect_top_relation": ma, "mean_abs_effect_random_relation": mb,
            "top_at_least_random": bool(n > 0 and ma >= mb)}
