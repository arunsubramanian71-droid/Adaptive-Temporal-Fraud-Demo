"""Label-free descriptors of discovered candidate communities (Tier 0, design sections 6 step 3, 9, 10, 12).

Inputs are the label-free GraphView, the day's records from `run_discovery`, and array-level segment / observability codes
(pseudo-account history only). Descriptors reported per community-day record:
  hub_share     fraction of the community's internal link mass carried by infrastructure entities whose window degree (number of
                distinct card tokens in the window) exceeds the 90th percentile over all infrastructure entities active in the window
                (design 6 step 3 diagnostic; percentile uses numpy's default linear interpolation, strict '>' comparison)
  density       unique internal card-infrastructure edges / (n_cards * n_infra)
  conductance   cut weight / min(vol(C), 2m - vol(C)), vol = weighted degree sum of the community's nodes (cards + infrastructure)
  composition   member transactions = transactions of member card tokens inside the window (t-W, t]; counts per
                segment (cold / warm / no-edge, never merged) x infra_observed (>= 1 netproxy/device edge)
  q_check       recomputed bipartite modularity contribution (must equal the pipeline's Q)
"""
from __future__ import annotations

import numpy as np

from src.coordination.data import GraphView
from src.coordination.graphs import aggregate, window_rows
from src.coordination.segments_array import COLD, NOEDGE, WARM, segments_from_arrays

SEG_NAMES = (COLD, WARM, NOEDGE)


def segment_codes(day: np.ndarray, pa_key: np.ndarray) -> np.ndarray:
    seg = segments_from_arrays(day, pa_key)
    codes = np.zeros(len(day), dtype=np.int8)
    codes[seg == WARM] = 1
    codes[seg == NOEDGE] = 2
    return codes


def infra_observed(v: GraphView) -> np.ndarray:
    return (v.net >= 0) | (v.dev >= 0)


def describe_day(v: GraphView, t: int, W: int, recs: list, seg_code: np.ndarray, infra_obs: np.ndarray) -> list[dict]:
    if not recs:
        return []
    e = window_rows(v, t, W)
    g = aggregate(e)
    if len(g.card) == 0:
        return []
    n_card = int(v.card.max()) + 1
    n_inf = v.n_net + v.n_dev
    m = g.m
    kc = np.bincount(g.card, weights=g.w, minlength=n_card)
    ki = np.bincount(g.infra, weights=g.w, minlength=n_inf)
    inf_ids, ndist = np.unique(g.infra, return_counts=True)       # unique (card, infra) edges -> distinct cards per infra
    p90 = float(np.percentile(ndist, 90))
    hub = np.zeros(n_inf, dtype=bool)
    hub[inf_ids[ndist > p90]] = True
    comm_card = np.full(n_card, -1, dtype=np.int64)
    comm_inf = np.full(n_inf, -1, dtype=np.int64)
    for j, r in enumerate(recs):
        comm_card[r["cards"]] = j
        comm_inf[r["infra"]] = j
    cm, im = comm_card[g.card], comm_inf[g.infra]
    inside = (cm >= 0) & (cm == im)
    nr = len(recs)
    w_in = np.bincount(cm[inside], weights=g.w[inside], minlength=nr)
    e_in = np.bincount(cm[inside], minlength=nr)
    hub_in = inside & hub[g.infra]
    w_hub = np.bincount(cm[hub_in], weights=g.w[hub_in], minlength=nr)
    rows = np.nonzero((v.day > t - W) & (v.day <= t) & (v.card >= 0))[0]
    cid = comm_card[v.card[rows]]
    k = cid >= 0
    comp = np.zeros((nr, 3, 2), dtype=np.int64)
    np.add.at(comp, (cid[k], seg_code[rows[k]].astype(np.int64), infra_obs[rows[k]].astype(np.int64)), 1)
    out = []
    for j, r in enumerate(recs):
        KC, KI = float(kc[r["cards"]].sum()), float(ki[r["infra"]].sum())
        vol = KC + KI
        cut = vol - 2.0 * w_in[j]
        den = min(vol, 2.0 * m - vol)
        n_c, n_i = len(r["cards"]), len(r["infra"])
        q_re = float(w_in[j]) - KC * KI / (2.0 * m) if m > 0 else 0.0
        d = {"day": int(t), "track": int(r["track"]), "n_cards": int(n_c), "n_infra": int(n_i), "q": float(r["q"]), "q_recomputed": q_re,
             "w_in": float(w_in[j]), "edges_in": int(e_in[j]), "density": float(e_in[j]) / float(n_c * n_i),
             "hub_share": float(w_hub[j] / w_in[j]) if w_in[j] > 0 else float("nan"),
             "conductance": float(cut / den) if den > 0 else float("nan"),
             "u_s": float(r["u_s"]), "t_raw": float(r["t_raw"]), "u_t": float(r["u_t"]), "score_st": 0.5 * (float(r["u_s"]) + float(r["u_t"])),
             "n_member_txn": int(comp[j].sum()), "p90_window_degree": p90}
        for si, sn in enumerate(SEG_NAMES):
            for oi, on in enumerate(("noinfra", "infra")):
                d[f"txn_{sn}__{on}"] = int(comp[j, si, oi])
        out.append(d)
    return out
