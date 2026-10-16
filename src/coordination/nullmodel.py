"""Null-calibrated percentiles for S and T (design 7.1, 7.2): built once on TRAIN background-only windows, then locked.

S null: degree-preserving rewiring (same-day infrastructure endpoint shuffles) -> identical pipeline -> Q of communities found.
T null: time-shuffled (day labels permuted among edges of the same infrastructure entity) -> T of the real window's candidates.
K_null = 10 rewirings x 8 sampled TRAIN windows per (W, gamma) configuration (frozen). Percentiles are size-bucketed
(cards 3-4, 5-8, 9-16, 17+); a bucket with fewer than MIN_NULL samples falls back to the pooled samples of that candidate kind.
Calibration uses the real (clean) TRAIN data only; seed-independent (rewiring seed 30000+42, Louvain seed 42 - consistent with the
720-run count of design section 18).
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field

import numpy as np

from src.coordination import frozen as F
from src.coordination.data import GraphView
from src.coordination.graphs import aggregate, louvain_candidates, rewire, star_candidates, window_rows
from src.coordination.scoring import ArrivalTable, arrivals_time_shuffled, t_raw_for

MIN_NULL = 30


def calibration_days() -> list[int]:
    rng = np.random.default_rng(F.NULL_SEED_BASE + 42)
    return sorted(int(x) for x in rng.choice(np.arange(40, 122), size=F.N_NULL_WINDOWS, replace=False))


@dataclass
class NullTables:
    W: int | None
    gamma: float
    s: dict = field(default_factory=dict)     # (kind, bucket) -> sorted np.ndarray
    t: dict = field(default_factory=dict)

    def _lookup(self, store, kind, bucket):
        a = store.get((kind, bucket))
        if a is not None and len(a) >= MIN_NULL:
            return a
        # sparse bucket: nearest bucket (larger first on ties) with >= MIN_NULL samples (decided before any dev/eval result existed)
        for d in range(1, len(F.SIZE_BUCKETS)):
            for b in (bucket + d, bucket - d):
                c = store.get((kind, b))
                if c is not None and len(c) >= MIN_NULL:
                    return c
        pooled = [x for (k, b), x in store.items() if k == kind]
        return np.sort(np.concatenate(pooled)) if pooled else np.zeros(1)

    def u_s(self, kind: str, n_cards: int, q: float) -> float:
        a = self._lookup(self.s, kind, F.size_bucket(n_cards))
        return float(np.searchsorted(a, q, side="right")) / len(a)

    def u_t(self, kind: str, n_cards: int, t_raw: float) -> float:
        a = self._lookup(self.t, kind, F.size_bucket(n_cards))
        return float(np.searchsorted(a, t_raw, side="right")) / len(a)

    def summary(self) -> dict:
        return {"W": self.W, "gamma": self.gamma,
                "s_samples": {f"{k}|{b}": int(len(v)) for (k, b), v in self.s.items()},
                "t_samples": {f"{k}|{b}": int(len(v)) for (k, b), v in self.t.items()}}


def calibrate(v: GraphView, table: ArrivalTable, W: int | None, gamma: float, days: list[int], k_null: int = F.K_NULL) -> NullTables:
    acc_s: dict = {}
    acc_t: dict = {}

    def add(store, kind, cards_n, vals):
        for n, x in zip(cards_n, vals):
            store.setdefault((kind, F.size_bucket(int(n))), []).append(float(x))

    for t in days:
        e = window_rows(v, t, W)
        g = aggregate(e)
        real_comm = louvain_candidates(g, gamma, seed=42)
        real_star = star_candidates(g)
        for k in range(k_null):
            rng = np.random.default_rng(np.random.SeedSequence([F.NULL_SEED_BASE + 42, 0 if W is None else W, int(round(gamma * 10)), t, k]))
            g2 = aggregate(rewire(e, rng))
            c2 = louvain_candidates(g2, gamma, seed=42)
            s2 = star_candidates(g2)
            add(acc_s, "comm", [len(c.cards) for c in c2], [c.q for c in c2])
            add(acc_s, "star", [len(c.cards) for c in s2], [c.q for c in s2])
            M_s, b_s = arrivals_time_shuffled(table, v, t, rng)
            if real_comm:
                add(acc_t, "comm", [len(c.cards) for c in real_comm], t_raw_for([c.infra for c in real_comm], M_s, b_s, table.kind, t))
            if real_star:
                add(acc_t, "star", [len(c.cards) for c in real_star], t_raw_for([c.infra for c in real_star], M_s, b_s, table.kind, t))
    nt = NullTables(W, gamma)
    nt.s = {k: np.sort(np.array(x)) for k, x in acc_s.items()}
    nt.t = {k: np.sort(np.array(x)) for k, x in acc_t.items()}
    return nt


def save_tables(tables: dict, path) -> None:
    with open(path, "wb") as fh:
        pickle.dump(tables, fh)


def load_tables(path) -> dict:
    with open(path, "rb") as fh:
        return pickle.load(fh)
