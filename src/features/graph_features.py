"""Handcrafted, label-free, strictly backward-looking graph features (comparison B).

Specification: experiments/comparison_B_graph_features/PREREGISTRATION_B.md, section 4. For a row with time ``t`` a feature may
use only rows of the SAME entity whose time is strictly smaller than ``t``. The function takes no label, score or prediction.

Implementation: for each relation the rows with an edge are keyed ``entity * 2**25 + time`` and sorted; every "prior" quantity
is then a prefix of the entity's block (``searchsorted(side='left')`` makes ties in time count as not prior), so counts, windows,
amount sums and distinct-neighbour counts are O(n log n) lookups on cumulative arrays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RELATIONS = ("pa", "card", "net", "dev")
DAY = 86400
WEEK = 7 * DAY
TIME_BITS = 25                       # times must be < 2**25 s (about 388 days); the data spans 182 days
TIME_LIMIT = 1 << TIME_BITS
TIME_WINDOW_STEMS = ("prior_n_1d", "prior_n_7d", "secs_since_last")


def feature_names() -> list:
    names = []
    for r in RELATIONS:
        names += [f"g_{r}_has_edge", f"g_{r}_prior_n", f"g_{r}_prior_n_1d", f"g_{r}_prior_n_7d", f"g_{r}_secs_since_last",
                  f"g_{r}_prior_amt_logmean", f"g_{r}_amt_dev"]
        names += [f"g_{r}_prior_distinct_{s}" for s in RELATIONS if s != r]
    return names


def ablation_groups() -> dict:
    """Feature names removed by each pre-registered ablation (PREREGISTRATION_B.md section 4)."""
    names = feature_names()
    groups = {}
    for r in RELATIONS:
        groups[f"drop_{r}"] = [n for n in names if n.startswith(f"g_{r}_") or n.endswith(f"_prior_distinct_{r}")]
    groups["drop_time_windows"] = [n for n in names if any(n.endswith(stem) for stem in TIME_WINDOW_STEMS)]
    return groups


def _check_inputs(time, amount, entities):
    n = len(time)
    if len(amount) != n:
        raise ValueError(f"amount has {len(amount)} rows, time has {n}")
    if set(entities) != set(RELATIONS):
        raise ValueError(f"entities must have exactly the keys {RELATIONS}, got {sorted(entities)}")
    for k, a in entities.items():
        if len(a) != n:
            raise ValueError(f"entity array '{k}' has {len(a)} rows, time has {n}")
    if n and (time.min() < 0 or time.max() >= TIME_LIMIT):
        raise ValueError(f"times must lie in [0, {TIME_LIMIT}); got [{time.min()}, {time.max()}]")


def compute_graph_features(time, amount, entities: dict) -> pd.DataFrame:
    """Return the 40 features, one row per input row, in input order.

    time      : integer seconds (TransactionDT)
    amount    : transaction amount (TransactionAmt); only log1p(amount) is used
    entities  : {relation: int array}, -1 where the row has no edge for that relation
    """
    time = np.asarray(time, dtype=np.int64)
    amount = np.asarray(amount, dtype=np.float64)
    entities = {k: np.asarray(v, dtype=np.int64) for k, v in entities.items()}
    _check_inputs(time, amount, entities)
    n = len(time)
    logamt = np.log1p(np.maximum(amount, 0.0))
    out = {}
    for r in RELATIONS:
        e = entities[r]
        valid = np.nonzero(e >= 0)[0]
        has = np.zeros(n)
        has[valid] = 1.0
        out[f"g_{r}_has_edge"] = has
        cols = {k: np.full(n, np.nan) for k in ("prior_n", "prior_n_1d", "prior_n_7d", "secs_since_last", "prior_amt_logmean", "amt_dev")}
        dist = {s: np.full(n, np.nan) for s in RELATIONS if s != r}
        if len(valid):
            ev, tv = e[valid], time[valid]
            key = ev * TIME_LIMIT + tv
            # ties in (entity, time) are ordered by amount so the result does not depend on the input row order
            order = np.lexsort((logamt[valid], tv, ev))
            skey, sidx = key[order], valid[order]            # sorted keys and the original row of each
            se, st = e[sidx], time[sidx]
            pos = np.searchsorted(skey, key, side="left")     # rows with a smaller (entity, time) key: prefix incl. other entities
            start = np.searchsorted(skey, ev * TIME_LIMIT, side="left")
            prior = pos - start
            lo1 = np.searchsorted(skey, ev * TIME_LIMIT + np.maximum(tv - DAY, 0), side="left")
            lo7 = np.searchsorted(skey, ev * TIME_LIMIT + np.maximum(tv - WEEK, 0), side="left")
            cols["prior_n"][valid] = prior
            cols["prior_n_1d"][valid] = pos - lo1
            cols["prior_n_7d"][valid] = pos - lo7
            has_prior = prior > 0
            last = np.where(has_prior, st[np.maximum(pos - 1, 0)], 0)
            secs = np.where(has_prior, tv - last, np.nan)
            cols["secs_since_last"][valid] = secs
            # running sum inside each entity's block: it depends only on earlier rows of that entity (bit-exact under truncation)
            cs = pd.Series(logamt[sidx]).groupby(se).cumsum().to_numpy()
            psum = np.where(has_prior, cs[np.maximum(pos - 1, 0)], 0.0)
            mean = np.where(has_prior, psum / np.maximum(prior, 1), np.nan)
            cols["prior_amt_logmean"][valid] = mean
            cols["amt_dev"][valid] = logamt[valid] - mean
            for s in dist:
                o = entities[s][sidx]
                ok = o >= 0
                width = int(entities[s].max()) + 2
                # rows with no ``s`` edge get unique negative keys so they can never collide with (or hide) a real pair
                pair = pd.Series(np.where(ok, se * width + o, -1 - np.arange(len(se))))
                first = (~pair.duplicated().to_numpy()) & ok  # first occurrence of this (entity, other) pair in sorted order
                cf = np.concatenate([[0], np.cumsum(first)])
                dist[s][valid] = (cf[pos] - cf[start]).astype(float)
        for k, v in cols.items():
            out[f"g_{r}_{k}"] = v
        for s, v in dist.items():
            out[f"g_{r}_prior_distinct_{s}"] = v
    return pd.DataFrame(out)[feature_names()]
