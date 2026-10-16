"""Daily discovery pipeline (design sections 5-8): windowed graph -> candidate sets -> component scores -> shared tracker.

Inputs: a label-free GraphView, an ArrivalTable, null tables, and (optionally) RiskData (model outputs only). No labels.
Candidate sets produced per day:
  comm      Louvain communities (proposed method; ablations A1-A7, baseline B2 = A1, B3b = A4)
  star      per-infrastructure stars (B1 and ablation A8)
  randpart  random re-partition of the comm pool preserving (n_cards, n_infra) per community (ablation A9)
  rand      size-matched random card sets from the day's active cards (baseline B0)
  riskfirst top-1%-risk transactions grouped through shared infrastructure (baseline B3a)
All records are append-only per day; the tracker assigns track ids with the same rule for every set.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from src.coordination import frozen as F
from src.coordination.data import GraphView
from src.coordination.graphs import (Candidate, aggregate, louvain_candidates, q_of_nodes, star_candidates, window_rows)
from src.coordination.nullmodel import NullTables
from src.coordination.scoring import ArrivalTable, card_recent_stats, rh_scores, rp_scores, t_raw_for
from src.coordination.tracking import Tracker


@dataclass
class RiskData:
    pct: np.ndarray       # per-transaction within-day risk rank percentile in (0, 1], aligned with the arrays
    logit: np.ndarray
    snaps: dict           # day -> (n_card, 64) float32 card states at end of that day


def _rec(day, idx, cand_cards, cand_infra, q=np.nan):
    return {"day": int(day), "idx": int(idx), "cards": np.asarray(cand_cards, dtype=np.int64),
            "infra": np.asarray(cand_infra, dtype=np.int64), "size": int(len(cand_cards)), "q": float(q)}


def _score_s_t(recs, kind, tables: NullTables, table: ArrivalTable, t):
    if not recs:
        return
    t_raw = t_raw_for([r["infra"] for r in recs], table.M, table.birth, table.kind, t)
    for r, tr in zip(recs, t_raw):
        r["u_s"] = tables.u_s(kind, r["size"], r["q"])
        r["t_raw"] = float(tr)
        r["u_t"] = tables.u_t(kind, r["size"], float(tr))


def _score_risk(recs, risk: RiskData, card_stat, t, rng):
    if not recs:
        return
    cards = [r["cards"] for r in recs]
    rp_raw, u_rp = rp_scores(cards, card_stat, rng)
    S_t = risk.snaps.get(t)
    S_prev = risk.snaps.get(t - F.W_SHORT)
    rh = rh_scores(cards, S_t, S_prev, rng) if S_t is not None else None
    for j, r in enumerate(recs):
        r["rp_raw"] = float(rp_raw[j])
        r["u_rp"] = float(u_rp[j])
        if rh is not None:
            r["rh_coh"], r["rh_shift"], r["u_rh"] = float(rh["coh"][j]), float(rh["shift"][j]), float(rh["u"][j])
        else:
            r["u_rh"] = 0.5


def _riskfirst(v: GraphView, risk: RiskData, t: int, W: int):
    lo = t - W
    m = (v.day > lo) & (v.day <= t) & (v.card >= 0) & (risk.pct > 0.99)
    idx = np.nonzero(m)[0]
    cards, infra, pct = [], [], []
    for arr, off in ((v.net, 0), (v.dev, v.n_net)):
        a = arr[idx]
        k = a >= 0
        cards.append(v.card[idx][k])
        infra.append(a[k] + off)
        pct.append(risk.pct[idx][k])
    c = np.concatenate(cards)
    i = np.concatenate(infra)
    p = np.concatenate(pct)
    if len(c) == 0:
        return []
    uc, ci = np.unique(c, return_inverse=True)
    ui, ii = np.unique(i, return_inverse=True)
    n = len(uc) + len(ui)
    A = coo_matrix((np.ones(len(c)), (ci, len(uc) + ii)), shape=(n, n))
    _, lab = connected_components(A, directed=False)
    out = []
    for comp in np.unique(lab):
        nodes = np.nonzero(lab == comp)[0]
        cc = uc[nodes[nodes < len(uc)]]
        ii_ = ui[nodes[nodes >= len(uc)] - len(uc)]
        if len(cc) >= F.M_MIN and len(ii_) >= 1:
            sel = np.isin(c, cc)
            out.append((np.sort(cc), np.sort(ii_), float(p[sel].mean())))
    out.sort(key=lambda x: int(x[0][0]))
    return out


def discover(v: GraphView, table: ArrivalTable, tables: NullTables, W: int | None, gamma: float, seed: int, days: list[int],
             risk: RiskData | None = None, sets=("comm", "star"), extra_seed: int = 0) -> dict:
    n_card = int(v.card.max()) + 1
    trackers = {k: Tracker() for k in sets}
    out = {k: [] for k in sets}
    for t in days:
        rng = np.random.default_rng(np.random.SeedSequence([seed, extra_seed, 0 if W is None else W, int(round(gamma * 10)), t]))
        e = window_rows(v, t, W)
        g = aggregate(e)
        comm = louvain_candidates(g, gamma, seed=seed)
        card_stat = None
        if risk is not None:
            card_stat, _ = card_recent_stats(v.day, v.card, risk.pct, t, n_card)
        per = {}
        if "comm" in sets or "randpart" in sets:
            recs = [_rec(t, i, c.cards, c.infra, c.q) for i, c in enumerate(comm)]
            _score_s_t(recs, "comm", tables, table, t)
            if risk is not None:
                _score_risk(recs, risk, card_stat, t, rng)
            per["comm"] = recs
        if "star" in sets:
            stars = star_candidates(g)
            recs = [_rec(t, i, c.cards, c.infra, c.q) for i, c in enumerate(stars)]
            _score_s_t(recs, "star", tables, table, t)
            if risk is not None:
                _score_risk(recs, risk, card_stat, t, rng)
            per["star"] = recs
        if "randpart" in sets and comm:
            all_c = np.concatenate([c.cards for c in comm])
            all_i = np.concatenate([c.infra for c in comm])
            pc, pi = rng.permutation(all_c), rng.permutation(all_i)
            recs, oc, oi = [], 0, 0
            for j, c in enumerate(comm):
                cc, ii = np.sort(pc[oc:oc + len(c.cards)]), np.sort(pi[oi:oi + len(c.infra)])
                oc += len(c.cards)
                oi += len(c.infra)
                recs.append(_rec(t, j, cc, ii, q_of_nodes(g, cc, ii)))
            _score_s_t(recs, "comm", tables, table, t)
            per["randpart"] = recs
        elif "randpart" in sets:
            per["randpart"] = []
        if "rand" in sets:
            m = (v.day > t - F.W_SHORT) & (v.day <= t) & (v.card >= 0)
            active = np.unique(v.card[m])
            recs = []
            for j, c in enumerate(comm):
                if len(active) >= len(c.cards):
                    cc = np.sort(rng.choice(active, size=len(c.cards), replace=False))
                    r = _rec(t, j, cc, [], np.nan)
                    r["u_rand"] = float(rng.random())
                    recs.append(r)
            per["rand"] = recs
        if "riskfirst" in sets and risk is not None and W is not None:
            recs = []
            for j, (cc, ii, sc) in enumerate(_riskfirst(v, risk, t, W)):
                r = _rec(t, j, cc, ii, np.nan)
                r["score_rf"] = sc
                recs.append(r)
            per["riskfirst"] = recs
        for k in sets:
            recs = per.get(k, [])
            tids = trackers[k].step(t, [r["cards"] for r in recs])
            for r, tid in zip(recs, tids):
                r["track"] = int(tid)
            out[k].extend(recs)
    out["_trackers"] = trackers
    return out


# ------------------------------------------------------------------ method definitions (design sections 13-14)
METHODS = {
    "A1_S": ("comm", ("s",)),
    "A2_ST": ("comm", ("s", "t")),
    "A3_SRp": ("comm", ("s", "rp")),
    "A4_Rp": ("comm", ("rp",)),
    "A5_STRp": ("comm", ("s", "t", "rp")),
    "A6_STRh": ("comm", ("s", "t", "rh")),
    "A7_STRpRh": ("comm", ("s", "t", "rp", "rh")),
    "A8_star_STRp": ("star", ("s", "t", "rp")),
    "A9_randpart_ST": ("randpart", ("s", "t")),
    "B0_random": ("rand", ("rand",)),
    "B1_naive": ("star", ("size",)),
    "B3a_riskfirst": ("riskfirst", ("rf",)),
}
ALIASES = {"B2_S_only": "A1_S", "B3b_Rp_on_candidates": "A4_Rp"}


def record_score(rec: dict, comps) -> float:
    vals = []
    for c in comps:
        if c == "size":
            return float(rec["size"])
        if c == "rand":
            return float(rec["u_rand"])
        if c == "rf":
            return float(rec["score_rf"])
        x = rec.get("u_" + c, np.nan)
        vals.append(0.5 if x is None or np.isnan(x) else x)
    return float(np.mean(vals))
