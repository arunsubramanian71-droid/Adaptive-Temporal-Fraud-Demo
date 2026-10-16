"""Windowed bipartite card-infrastructure graphs, Louvain communities, star candidates (design sections 5-7).

No labels and no model output enter this module. A window at day t uses only rows with t-W < day <= t.
Infrastructure ids are global: netproxy id i -> i; device id j -> n_net + j. Graph node ids: card c -> c;
infrastructure g -> INFRA_OFF + g, so cards and infrastructure never collide and ordering is deterministic.
"""
from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import numpy as np

from src.coordination.frozen import EDGE_WEIGHT_CAP, M_MIN
from src.coordination.data import GraphView

INFRA_OFF = 10_000_000


@dataclass
class WindowEdges:
    """Transaction-level (card, infra, day) rows inside the window with an infrastructure edge."""
    card: np.ndarray
    infra: np.ndarray     # global infra id
    day: np.ndarray
    kind: np.ndarray      # 0 = netproxy, 1 = device


def window_rows(v: GraphView, t: int, W: int | None) -> WindowEdges:
    lo = 0 if W is None else t - W
    m = (v.day > lo) & (v.day <= t) & (v.card >= 0)
    idx = np.nonzero(m)[0]
    cards, infra, days, kinds = [], [], [], []
    for kind, arr, off in ((0, v.net, 0), (1, v.dev, v.n_net)):
        a = arr[idx]
        k = a >= 0
        cards.append(v.card[idx][k])
        infra.append(a[k] + off)
        days.append(v.day[idx][k])
        kinds.append(np.full(int(k.sum()), kind, dtype=np.int8))
    return WindowEdges(np.concatenate(cards), np.concatenate(infra), np.concatenate(days), np.concatenate(kinds))


@dataclass
class WindowGraph:
    card: np.ndarray      # unique edge list (sorted by card, infra)
    infra: np.ndarray
    w: np.ndarray         # capped weights
    kc: dict              # weighted degree per card
    ki: dict              # weighted degree per infra
    m: float              # total weight


def aggregate(e: WindowEdges) -> WindowGraph:
    if len(e.card) == 0:
        z = np.zeros(0, dtype=np.int64)
        return WindowGraph(z, z, np.zeros(0), {}, {}, 0.0)
    key = e.card.astype(np.int64) * np.int64(1 << 24) + e.infra.astype(np.int64)
    uk, cnt = np.unique(key, return_counts=True)
    c = (uk >> 24).astype(np.int64)
    i = (uk & ((1 << 24) - 1)).astype(np.int64)
    w = np.minimum(cnt, EDGE_WEIGHT_CAP).astype(np.float64)
    kc = {}
    ki = {}
    uc, inv = np.unique(c, return_inverse=True)
    for cc, s in zip(uc.tolist(), np.bincount(inv, weights=w).tolist()):
        kc[cc] = s
    ui, inv2 = np.unique(i, return_inverse=True)
    for ii, s in zip(ui.tolist(), np.bincount(inv2, weights=w).tolist()):
        ki[ii] = s
    return WindowGraph(c, i, w, kc, ki, float(w.sum()))


@dataclass
class Candidate:
    cards: np.ndarray     # sorted card ids
    infra: np.ndarray     # sorted global infra ids
    q: float              # unnormalised bipartite modularity contribution


def _q(w_in: float, KC: float, KI: float, m: float) -> float:
    return w_in - KC * KI / (2.0 * m) if m > 0 else 0.0


def build_nx(g: WindowGraph) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from(sorted(set(g.card.tolist()) | {INFRA_OFF + x for x in g.infra.tolist()}))
    G.add_weighted_edges_from(zip(g.card.tolist(), (g.infra + INFRA_OFF).tolist(), g.w.tolist()))
    return G


def louvain_candidates(g: WindowGraph, gamma: float, seed: int, m_min: int = M_MIN) -> list[Candidate]:
    """Louvain (networkx, weighted, fixed seed) on the bipartite window graph; split disconnected communities into connected
    components; keep components with >= m_min distinct cards and >= 1 infrastructure entity (design section 6, steps 4-5)."""
    if g.m <= 0:
        return []
    G = build_nx(g)
    comms = nx.community.louvain_communities(G, weight="weight", resolution=gamma, seed=seed)
    out: list[Candidate] = []
    for comm in comms:
        sub = G.subgraph(comm)
        for comp in nx.connected_components(sub):
            cards = sorted(x for x in comp if x < INFRA_OFF)
            infra = sorted(x - INFRA_OFF for x in comp if x >= INFRA_OFF)
            if len(cards) < m_min or not infra:
                continue
            w_in = sum(d["weight"] for _, _, d in sub.subgraph(comp).edges(data=True))
            KC = sum(g.kc[c] for c in cards)
            KI = sum(g.ki[i] for i in infra)
            out.append(Candidate(np.array(cards, dtype=np.int64), np.array(infra, dtype=np.int64), _q(w_in, KC, KI, g.m)))
    out.sort(key=lambda c: (int(c.cards[0]), len(c.cards)))
    return out


def star_candidates(g: WindowGraph, m_min: int = M_MIN) -> list[Candidate]:
    """Per-infrastructure star groups: every infrastructure entity with >= m_min distinct cards in the window (baseline B1 /
    ablation A8 candidate pool); Q is the single-infrastructure bipartite modularity contribution."""
    if g.m <= 0:
        return []
    order = np.argsort(g.infra, kind="stable")
    inf, car, w = g.infra[order], g.card[order], g.w[order]
    starts = np.nonzero(np.r_[True, inf[1:] != inf[:-1]])[0]
    ends = np.r_[starts[1:], len(inf)]
    out = []
    for s, e in zip(starts, ends):
        if e - s < m_min:
            continue
        cards = np.sort(car[s:e])
        i = int(inf[s])
        KC = sum(g.kc[c] for c in cards.tolist())
        out.append(Candidate(cards.astype(np.int64), np.array([i], dtype=np.int64), _q(float(w[s:e].sum()), KC, g.ki[i], g.m)))
    return out


def q_of_nodes(g: WindowGraph, cards: np.ndarray, infra: np.ndarray) -> float:
    """Q for an arbitrary node set (used by the random-partition control A9)."""
    if g.m <= 0 or len(cards) == 0 or len(infra) == 0:
        return 0.0
    cs, is_ = set(cards.tolist()), set(infra.tolist())
    mask = np.array([(c in cs) and (i in is_) for c, i in zip(g.card.tolist(), g.infra.tolist())]) if len(g.card) else np.zeros(0, bool)
    w_in = float(g.w[mask].sum()) if len(g.w) else 0.0
    KC = sum(g.kc.get(c, 0.0) for c in cs)
    KI = sum(g.ki.get(i, 0.0) for i in is_)
    return _q(w_in, KC, KI, g.m)


def rewire(e: WindowEdges, rng: np.random.Generator) -> WindowEdges:
    """Degree-preserving null (design 7.1): within each (day, infrastructure kind) group, shuffle the infrastructure endpoint
    among that day's transaction edges (equivalent to random same-day endpoint swaps; per-day degree sequences preserved)."""
    infra = e.infra.copy()
    key = e.day.astype(np.int64) * 2 + e.kind
    order = np.argsort(key, kind="stable")
    ks = key[order]
    starts = np.nonzero(np.r_[True, ks[1:] != ks[:-1]])[0]
    ends = np.r_[starts[1:], len(ks)]
    for s, t in zip(starts, ends):
        idx = order[s:t]
        infra[idx] = infra[idx][rng.permutation(len(idx))]
    return WindowEdges(e.card, infra, e.day, e.kind)
