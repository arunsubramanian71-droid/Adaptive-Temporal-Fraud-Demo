"""Community scoring components (design section 7): S (structure), T (temporal synchrony), R_p / R_h (learned-model risk).

Discovery/scoring never receives labels. S and T are risk-free. R components consume only model outputs (within-day risk rank
percentiles; frozen-model card states) for days <= t.

Implementation constants not listed in section 15.1 (documented, fixed before any evaluation): empirical-Bayes prior strength
EB_PRIOR_DAYS = 7 pseudo-days; random-card-set nulls use K_RANDOM = 200 draws (with replacement from the day's active cards,
negligible bias for set sizes << active-card count); neutral percentile 0.5 when a component is undefined.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import poisson

from src.coordination.data import GraphView
from src.coordination.frozen import MAX_DAY, T_BASELINE_DAYS, W_SHORT
from src.coordination.graphs import window_rows

EB_PRIOR_DAYS = 7.0
K_RANDOM = 200
NEUTRAL = 0.5
RATE_FLOOR = 1e-3


class ArrivalTable:
    """M[i, d] = number of distinct cards FIRST seen on infrastructure i on day d (first-seen = earliest day over all data; for
    d <= t this equals the first-seen computed from data <= t, so reading M[:, <= t] is causal). birth[i] = first day with an arrival."""

    def __init__(self, v: GraphView, n_infra: int | None = None):
        e = window_rows(v, MAX_DAY, None)
        self.n_infra = v.n_net + v.n_dev if n_infra is None else n_infra
        key = e.infra.astype(np.int64) * (1 << 24) + e.card.astype(np.int64)
        order = np.lexsort((e.day, key))
        ks, ds = key[order], e.day[order]
        first = np.r_[True, ks[1:] != ks[:-1]]
        fk, fd = ks[first], ds[first]
        self.pair_first = dict(zip(fk.tolist(), fd.tolist()))
        self.pair_infra = (fk >> 24).astype(np.int64)
        self.pair_card = (fk & ((1 << 24) - 1)).astype(np.int64)
        self.pair_day = fd.astype(np.int64)
        self.kind = np.zeros(self.n_infra, dtype=np.int8)
        self.kind[v.n_net:] = 1
        self.M = np.zeros((self.n_infra, MAX_DAY + 2), dtype=np.int32)
        np.add.at(self.M, (self.pair_infra, self.pair_day), 1)
        self.birth = self._birth(self.M)

    @staticmethod
    def _birth(M):
        pos = M > 0
        b = np.where(pos.any(1), pos.argmax(1), 10**6)
        return b


def t_raw_for(infra_sets, M, birth, kind, t: int) -> np.ndarray:
    """T_raw per candidate: max over its infrastructure entities of -log10 of the Poisson upper-tail probability of the number of
    distinct new cards seen on the entity in the last W_SHORT days, given an empirical-Bayes baseline rate estimated from the
    preceding T_BASELINE_DAYS days (strictly before the short window), shrunk toward the pooled rate of that infrastructure kind."""
    lo_new = max(t - W_SHORT + 1, 0)
    b_hi = t - W_SHORT                       # baseline days: [b_lo, b_hi]
    b_lo = max(b_hi - T_BASELINE_DAYS + 1, 1)
    n_new = M[:, lo_new:t + 1].sum(1).astype(np.float64)
    base = M[:, b_lo:b_hi + 1].sum(1).astype(np.float64) if b_hi >= b_lo else np.zeros(M.shape[0])
    exposure = np.clip(b_hi - np.minimum(birth, 10**5) + 1, 0, T_BASELINE_DAYS).astype(np.float64)
    exposure = np.minimum(exposure, max(b_hi - b_lo + 1, 0))
    lam = {}
    for k in (0, 1):
        m = (kind == k) & (exposure > 0)
        lam[k] = max(base[m].sum() / exposure[m].sum(), RATE_FLOOR) if m.any() else RATE_FLOOR
    lam_i = np.where(kind == 0, lam[0], lam[1])
    rate = (EB_PRIOR_DAYS * lam_i + base) / (EB_PRIOR_DAYS + exposure)
    mu = np.maximum(W_SHORT * rate, RATE_FLOOR)
    p = poisson.sf(n_new - 1, mu)
    nl = -np.log10(np.clip(p, 1e-300, 1.0))
    out = np.zeros(len(infra_sets))
    for j, inf in enumerate(infra_sets):
        out[j] = nl[inf].max() if len(inf) else 0.0
    return out


def arrivals_time_shuffled(table: ArrivalTable, v: GraphView, t: int, rng: np.random.Generator):
    """Time-shuffled null (design 7.2): within each infrastructure entity, permute the day labels among that entity's transaction
    edges in the last (T_BASELINE_DAYS + W_SHORT) days, then recompute first-seen arrivals for pairs first seen inside that span.
    Returns (M_null, birth_null) with columns outside the span taken unchanged from the real table (they are not read by T)."""
    span_lo = t - (T_BASELINE_DAYS + W_SHORT) + 1
    e = window_rows(v, t, T_BASELINE_DAYS + W_SHORT)
    M = table.M.copy()
    M[:, max(span_lo, 0):t + 1] = 0
    if len(e.card):
        key = e.infra.astype(np.int64) * (1 << 24) + e.card.astype(np.int64)
        old = np.fromiter((table.pair_first.get(k, 10**6) < span_lo for k in key.tolist()), dtype=bool, count=len(key))
        days = e.day.copy()
        order = np.argsort(e.infra, kind="stable")
        si = e.infra[order]
        starts = np.nonzero(np.r_[True, si[1:] != si[:-1]])[0]
        ends = np.r_[starts[1:], len(si)]
        for s, en in zip(starts, ends):
            idx = order[s:en]
            days[idx] = days[idx][rng.permutation(len(idx))]
        keep = ~old
        if keep.any():
            kk, dd = key[keep], days[keep]
            o = np.lexsort((dd, kk))
            kk, dd = kk[o], dd[o]
            first = np.r_[True, kk[1:] != kk[:-1]]
            np.add.at(M, ((kk[first] >> 24).astype(np.int64), dd[first]), 1)
    return M, ArrivalTable._birth(M)


def card_recent_stats(day, card, pct, t: int, n_card: int):
    """Card-level mean risk percentile over days (t-W_SHORT, t]; NaN where the card has no transaction in that span."""
    m = (day > t - W_SHORT) & (day <= t) & (card >= 0)
    c = card[m]
    s = np.bincount(c, weights=pct[m], minlength=n_card)
    n = np.bincount(c, minlength=n_card)
    out = np.full(n_card, np.nan)
    nz = n > 0
    out[nz] = s[nz] / n[nz]
    return out, n


def _cdf(null_sorted: np.ndarray, x: float) -> float:
    return float(np.searchsorted(null_sorted, x, side="right")) / max(len(null_sorted), 1)


def rp_scores(cands_cards, card_stat: np.ndarray, rng: np.random.Generator):
    """R_p per candidate: mean over active member cards of the card-level risk percentile, and its percentile against
    size-matched random card sets drawn from the day's active cards."""
    active = np.nonzero(~np.isnan(card_stat))[0]
    vals = card_stat[active]
    raw = np.full(len(cands_cards), np.nan)
    u = np.full(len(cands_cards), NEUTRAL)
    cache = {}
    for j, cards in enumerate(cands_cards):
        s = card_stat[cards]
        s = s[~np.isnan(s)]
        if len(s) == 0 or len(active) == 0:
            continue
        raw[j] = s.mean()
        n = len(s)
        if n not in cache:
            idx = rng.integers(0, len(active), size=(K_RANDOM, n))
            cache[n] = np.sort(vals[idx].mean(1))
        u[j] = _cdf(cache[n], raw[j])
    return raw, u


def rh_scores(cands_cards, S_t: np.ndarray, S_prev: np.ndarray | None, rng: np.random.Generator):
    """R_h per candidate from frozen-model card states: cohesion (mean pairwise cosine of member states) and shift (mean cosine
    distance between S(t) and S(t-3)), each as a percentile against size-matched random card sets; u_Rh = mean of the two
    (cohesion alone if shift is undefined for fresh members)."""
    nrm = np.linalg.norm(S_t, axis=1)
    ok = nrm > 0
    Z = np.zeros_like(S_t)
    Z[ok] = S_t[ok] / nrm[ok, None]
    active = np.nonzero(ok)[0]
    shift = np.full(len(S_t), np.nan)
    if S_prev is not None:
        n0 = np.linalg.norm(S_prev, axis=1)
        both = ok & (n0 > 0)
        shift[both] = 1.0 - (Z[both] * (S_prev[both] / n0[both, None])).sum(1)

    def coh_of(idx):  # idx (K, n) member indices
        z = Z[idx].sum(1)
        n = idx.shape[1]
        return ((z * z).sum(1) - n) / (n * (n - 1))

    n_c = len(cands_cards)
    coh_raw = np.full(n_c, np.nan)
    shf_raw = np.full(n_c, np.nan)
    u = np.full(n_c, NEUTRAL)
    ucoh = np.full(n_c, NEUTRAL)
    ushf = np.full(n_c, np.nan)
    cache_c, cache_s = {}, {}
    for j, cards in enumerate(cands_cards):
        mem = cards[ok[cards]]
        if len(mem) < 2 or len(active) < 2:
            continue
        n = len(mem)
        coh_raw[j] = float(coh_of(mem[None, :])[0])
        if n not in cache_c:
            idx = active[rng.integers(0, len(active), size=(K_RANDOM, n))]
            same = (idx[:, :, None] == idx[:, None, :]).sum((1, 2)) > n  # drop draws with repeated members (self-cosine inflates cohesion)
            cc = coh_of(idx)
            cache_c[n] = np.sort(cc[~same]) if (~same).any() else np.sort(cc)
        ucoh[j] = _cdf(cache_c[n], coh_raw[j])
        sv = shift[mem]
        sv = sv[~np.isnan(sv)]
        if len(sv):
            shf_raw[j] = sv.mean()
            sa = np.nonzero(~np.isnan(shift))[0]
            if len(sa):
                m_ = len(sv)
                if m_ not in cache_s:
                    cache_s[m_] = np.sort(shift[sa[rng.integers(0, len(sa), size=(K_RANDOM, m_))]].mean(1))
                ushf[j] = _cdf(cache_s[m_], shf_raw[j])
        u[j] = ucoh[j] if np.isnan(ushf[j]) else 0.5 * (ucoh[j] + ushf[j])
    return {"coh": coh_raw, "shift": shf_raw, "u_coh": ucoh, "u_shift": ushf, "u": u}
