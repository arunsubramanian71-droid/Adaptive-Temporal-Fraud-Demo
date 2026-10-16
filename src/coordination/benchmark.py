"""Synthetic planted-group benchmark generator (design section 11). The ONLY module on the discovery side that touches labels, and
only to (a) build feature pools from real transactions strictly before the planted day and (b) set planted `isFraud`
(coordinated groups 1, decoy groups 0). Detector code never imports this module or the manifest (static check in leakage.py).

Planted constructs are called "coordinated groups" and "decoy groups". Patterns: P1 shared-infrastructure burst group, P2
account-cycling group, P3 dormant-then-escalate group; decoys D1 steady shared-egress, D2 synchronous-on-hub, D3 small household.

Implementation clarifications (design leaves them open; fixed before any evaluation):
  * P3 groups are counted in camouflage regime alpha = 0.5 (time-averaged overtness ~0.67), so regime alpha = 0 contains only
    P1/P2: eval per regime of 12 -> alpha 0: 6 P1 + 6 P2; alpha 0.5: 4 P1 + 4 P2 + 4 P3; alpha 1: 6 P1 + 6 P2. Dev (20 per
    regime): 10 P1 + 10 P2 / 7 P1 + 7 P2 + 6 P3 / 10 P1 + 10 P2. Decoys split evenly over D1-D3.
  * Eval parameter ranges follow section 11.6 (n in {3,6,12,24}, delta = 7, duration 4-20); dev: n in {3,4,6,8,12,16,24,32}, delta in {1,3,7}.
  * The <= 5% cap is enforced by random thinning of non-first planted transactions (each member keeps its first transaction); if
    still above the cap, D1 decoy members are trimmed. The scale factor is recorded in the manifest.
  * Infrastructure identity inside the generator is (kind, local id): kind 0 = netproxy, 1 = device; synthetic ids are appended
    after the real ids of each kind.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from src.coordination import frozen as F
from src.coordination.data import Arrays
from src.coordination.scoring import ArrivalTable

SYN_TXN_BASE = 10_000_000
DEV_SIZES = (3, 4, 6, 8, 12, 16, 24, 32)
EVAL_SIZES = (3, 6, 12, 24)
Q_OBS = (1.0, 0.5, 0.25)
NOVELTY = ("fresh", "rare", "popular")
COMPOSITION = ("fresh", "hijacked", "mixed")


def _regime_plan(counts: dict) -> list[tuple[str, float]]:
    g = counts["groups"]
    if g // 3 == 12:
        plan = [("P1", 0.0)] * 6 + [("P2", 0.0)] * 6 + [("P1", 0.5)] * 4 + [("P2", 0.5)] * 4 + [("P3", 0.5)] * 4 + [("P1", 1.0)] * 6 + [("P2", 1.0)] * 6
    else:
        plan = [("P1", 0.0)] * 10 + [("P2", 0.0)] * 10 + [("P1", 0.5)] * 7 + [("P2", 0.5)] * 7 + [("P3", 0.5)] * 6 + [("P1", 1.0)] * 10 + [("P2", 1.0)] * 10
    assert len(plan) == g
    return plan


def plant(real: Arrays, period: tuple[int, int], counts: dict, gen_seed: int, is_eval: bool, pool_min_day: int, cap: float = F.PLANT_CAP):
    """Returns (augmented Arrays WITHOUT X, x_src into real.X for every augmented row, manifest dict, is_planted mask, row_group id per row (-1 real))."""
    rng = np.random.default_rng(gen_seed)
    lo, hi = period
    n_real = len(real.day)
    n_pa, n_card = real.n_entities["pseudo_account"], real.n_entities["card"]
    n_net, n_dev = real.n_entities["netproxy"], real.n_entities["device_fingerprint"]
    day, y = real.day, real.y
    n_period = int(((day >= lo) & (day <= hi)).sum())
    cap_n = int(cap * n_period)

    # ---- real statistics: cumulative infrastructure degree, arrival table, card first-seen, card->pseudo-account, pools
    n_inf0 = n_net + n_dev
    inf_all = np.concatenate([real.net, np.where(real.dev >= 0, real.dev + n_net, -1)])
    day_all = np.concatenate([day, day])
    kk = inf_all >= 0
    daily = np.zeros((n_inf0, F.MAX_DAY + 2), dtype=np.int32)
    np.add.at(daily, (inf_all[kk], day_all[kk]), 1)
    cum_deg = np.cumsum(daily, axis=1)
    arr_tab = ArrivalTable(real.view())
    card_first = np.full(n_card, 10**6, dtype=np.int64)
    ok = real.card >= 0
    np.minimum.at(card_first, real.card[ok], day[ok])
    pa_rows = np.nonzero(real.pa >= 0)[0]
    dfp = pd.DataFrame({"c": real.card[pa_rows], "p": real.pa[pa_rows], "d": day[pa_rows]}).groupby(["c", "p"], as_index=False)["d"].min()
    card_pa = {int(c): (g["p"].to_numpy(), g["d"].to_numpy()) for c, g in dfp.groupby("c")}
    fraud_idx = np.nonzero((y == 1) & (day >= pool_min_day))[0]
    nonf_idx = np.nonzero((y == 0) & (day >= pool_min_day))[0]
    fraud_days, nonf_days = day[fraud_idx], day[nonf_idx]

    used_infra: set = set()
    used_cards: set = set()
    state = {"next_card": n_card, "next_pa": n_pa, "syn_net": 0, "syn_dev": 0}
    groups, rows = [], []

    def new_infra(kind):
        if kind == 0:
            state["syn_net"] += 1
            return (0, n_net + state["syn_net"] - 1)
        state["syn_dev"] += 1
        return (1, n_dev + state["syn_dev"] - 1)

    def pick_real_infra(kind, f, cls):
        d = max(f - 1, 1)
        deg = cum_deg[:, d]
        lo_i, hi_i = (0, n_net) if kind == 0 else (n_net, n_inf0)
        cand = np.arange(lo_i, hi_i)
        cand = cand[deg[cand] > 0]
        if len(cand) == 0:
            return None
        dd = deg[cand]
        p25, p90 = np.percentile(dd, 25), np.percentile(dd, 90)
        sel = cand[dd <= p25] if cls == "rare" else cand[dd >= p90]
        sel = [int(i) for i in sel if (kind, int(i) - (0 if kind == 0 else n_net)) not in used_infra]
        if not sel:
            return None
        i = int(rng.choice(sel))
        return (kind, i - (0 if kind == 0 else n_net))

    def pick_hub_arrival(f):
        d = max(f - 1, 1)
        arr = arr_tab.M[:, max(d - 28, 0):d + 1].sum(1)
        cand = [int(i) for i in np.nonzero(arr > 0)[0]
                if ((0, int(i)) if i < n_net else (1, int(i) - n_net)) not in used_infra]
        thr = np.percentile(arr[cand], 90)
        sel = [i for i in cand if arr[i] >= thr]
        i = int(rng.choice(sel))
        return (0, i) if i < n_net else (1, i - n_net)

    def alloc_infra(k, novelty, f):
        out = []
        for _ in range(k):
            kind = int(rng.integers(0, 2))
            got = None
            if novelty in ("rare", "popular"):
                got = pick_real_infra(kind, f, novelty) or pick_real_infra(1 - kind, f, novelty)
            if got is None:
                got, nov = new_infra(kind), "fresh"
            else:
                nov = novelty
            used_infra.add(got)
            out.append((got, nov))
        return out

    def alloc_cards(n, comp, f):
        cards, pas, hij = [], [], []
        n_h = {"fresh": 0, "hijacked": n, "mixed": n // 2}[comp]
        elig = np.array([int(c) for c in np.nonzero(card_first <= f - 14)[0] if int(c) not in used_cards and int(c) in card_pa], dtype=np.int64)
        for j in range(n):
            if j < n_h and len(elig) > 0:
                c = int(rng.choice(elig))
                elig = elig[elig != c]
                used_cards.add(c)
                pa_ids, pa_days = card_pa[c]
                okp = pa_ids[pa_days <= f - 1]
                if len(okp):
                    p = int(rng.choice(okp))
                else:
                    p = state["next_pa"]
                    state["next_pa"] += 1
                cards.append(c), pas.append(p), hij.append(True)
            else:
                cards.append(state["next_card"]), pas.append(state["next_pa"]), hij.append(False)
                state["next_card"] += 1
                state["next_pa"] += 1
        return cards, pas, hij

    def add_rows(gid, cards, pas, infra_ids, q_obs, arrivals, spans, rates, alpha_fn, label):
        for m, c in enumerate(cards):
            a = arrivals[m]
            for d in range(a, min(a + spans[m], hi + 1)):
                for j in range(int(rates[m])):
                    inf = infra_ids[int(rng.integers(0, len(infra_ids)))] if (rng.random() < q_obs and infra_ids) else None
                    rows.append((gid, m, c, pas[m], d, inf, alpha_fn(d), label, d == a and j == 0))

    def infra_json(lst):
        return [dict(kind="netproxy" if k == 0 else "device", local=int(loc), novelty=nov) for (k, loc), nov in lst]

    plan = _regime_plan(counts)
    gid = 0
    for pattern, alpha in plan:
        n = int(rng.choice(EVAL_SIZES if is_eval else DEV_SIZES))
        k = int(rng.integers(1, 4))
        novelty = str(rng.choice(NOVELTY))
        delta = 7 if is_eval else int(rng.choice((1, 3, 7)))
        dur = int(rng.integers(4, 21))
        comp = str(rng.choice(COMPOSITION))
        q = float(rng.choice(Q_OBS))
        f = int(rng.integers(lo, (146 if is_eval else hi - 3) + 1))
        infra = alloc_infra(k, novelty, f)
        cards, pas, hij = alloc_cards(n, comp, f)
        end = min(f + dur - 1, hi)
        if pattern in ("P1", "P3"):
            arrivals = [f + int(rng.integers(0, max(min(delta, dur), 1))) for _ in cards]
            spans = [max(end - a + 1, 1) for a in arrivals]
        else:                                   # P2: staggered arrivals over the whole duration, each member briefly active
            arrivals = [min(f + (int(rng.integers(0, 2)) if j < max(n // 3, 1) else int(rng.integers(0, dur))), end) for j in range(n)]
            spans = [int(rng.integers(1, 4)) for _ in cards]
        rates = [int(rng.integers(1, 5)) for _ in cards]
        esc = f + max(dur // 3, 1) if pattern == "P3" else None
        alpha_fn = (lambda d, esc=esc: 1.0 if d >= esc else 0.0) if pattern == "P3" else (lambda d, a=alpha: a)
        add_rows(gid, cards, pas, [i for i, _ in infra], q, arrivals, spans, rates, alpha_fn, 1)
        groups.append(dict(group_id=gid, kind="coordinated", pattern=pattern, regime_alpha=alpha, n_planned=n, composition=comp, q_obs=q,
                           infra=infra_json(infra), infra_novelty=novelty, delta=delta if pattern != "P2" else None, duration=dur,
                           formation_day=f, planned_end=end, escalation_day=esc, censored=bool(f + dur - 1 > hi),
                           cards=cards, pas=pas, hijacked=hij))
        gid += 1

    nd = counts["decoys"]
    for dk in ["D1"] * (nd // 3) + ["D2"] * (nd // 3) + ["D3"] * (nd - 2 * (nd // 3)):
        delta = None
        if dk == "D1":
            n = int(rng.integers(20, 41 if is_eval else 61))
            ff = lo
            infra = alloc_infra(1, "popular", ff)
            cards, pas, hij = alloc_cards(n, "fresh", ff)
            arrivals = [int(rng.integers(lo, hi - 1)) for _ in cards]
            spans = [int(rng.integers(1, 11)) for _ in cards]
            rates = [1] * n
            dur = hi - lo + 1
            add_rows(gid, cards, pas, [i for i, _ in infra], 1.0, arrivals, spans, rates, lambda d: 0.0, 0)
        elif dk == "D2":
            n = int(rng.integers(8, 31))
            ff = int(rng.integers(lo, (146 if is_eval else hi - 5) + 1))
            hub = pick_hub_arrival(ff)
            used_infra.add(hub)
            infra = [(hub, "hub_high_arrival")]
            cards, pas, hij = alloc_cards(n, "fresh", ff)
            delta = int(rng.choice((1, 3)))
            arrivals = [ff + int(rng.integers(0, delta)) for _ in cards]
            spans = [int(rng.integers(2, 6)) for _ in cards]
            rates = [int(rng.integers(1, 3)) for _ in cards]
            dur = 6
            add_rows(gid, cards, pas, [i for i, _ in infra], 1.0, arrivals, spans, rates, lambda d: 0.0, 0)
        else:
            n, ff = 2, lo
            infra = alloc_infra(1, str(rng.choice(("fresh", "rare"))), ff)
            cards, pas, hij = alloc_cards(n, "fresh", ff)
            dur = hi - lo + 1
            for m, c in enumerate(cards):
                for d in range(ff, hi + 1, 3):
                    rows.append((gid, m, c, pas[m], d, infra[0][0], 0.0, 0, d == ff))
        groups.append(dict(group_id=gid, kind="decoy", pattern=dk, regime_alpha=0.0, n_planned=n, composition="fresh", q_obs=1.0,
                           infra=infra_json(infra), infra_novelty=infra[0][1], delta=delta, duration=dur, formation_day=ff,
                           planned_end=hi, escalation_day=None, censored=False, cards=cards, pas=pas, hijacked=hij))
        gid += 1

    # ---- cap enforcement by thinning (members keep their first transaction)
    total0 = len(rows)
    is_first = np.array([r[8] for r in rows], dtype=bool)
    keep = np.ones(total0, dtype=bool)
    if total0 > cap_n:
        nonfirst = np.nonzero(~is_first)[0]
        need = total0 - cap_n
        if need <= len(nonfirst):
            keep[rng.choice(nonfirst, size=need, replace=False)] = False
        else:
            keep[nonfirst] = False
            d1 = [g["group_id"] for g in groups if g["pattern"] == "D1"]
            gid_arr = np.array([r[0] for r in rows])
            m_arr = np.array([r[1] for r in rows])
            while keep.sum() > cap_n and d1:
                g = int(rng.choice(d1))
                alive = np.unique(m_arr[(gid_arr == g) & keep])
                if len(alive) <= 3:
                    d1.remove(g)
                    continue
                keep[(gid_arr == g) & (m_arr == int(rng.choice(alive)))] = False
            assert keep.sum() <= cap_n, "planting cap cannot be met"
    rows = [r for r, kp in zip(rows, keep) if kp]
    n_pl = len(rows)

    # ---- features: whole real feature vectors resampled from pools strictly before the planted day
    pl_day = np.array([r[4] for r in rows], dtype=np.int64)
    pl_label = np.array([r[7] for r in rows], dtype=np.float32)
    use_f = rng.random(n_pl) < np.array([r[6] for r in rows], dtype=float)
    x_src_pl = np.empty(n_pl, dtype=np.int64)
    for i in range(n_pl):
        pool, pdays = (fraud_idx, fraud_days) if use_f[i] else (nonf_idx, nonf_days)
        cut = int(np.searchsorted(pdays, pl_day[i], side="left"))
        if cut == 0:
            pool, pdays = nonf_idx, nonf_days
            cut = int(np.searchsorted(pdays, pl_day[i], side="left"))
        x_src_pl[i] = pool[int(rng.integers(0, cut))]
    pl_card = np.array([r[2] for r in rows], dtype=np.int64)
    pl_pa = np.array([r[3] for r in rows], dtype=np.int64)
    pl_net = np.full(n_pl, -1, dtype=np.int64)
    pl_dev = np.full(n_pl, -1, dtype=np.int64)
    for i, r in enumerate(rows):
        if r[5] is not None:
            (pl_net if r[5][0] == 0 else pl_dev)[i] = r[5][1]

    aug_day = np.concatenate([day, pl_day])
    order = np.argsort(aug_day, kind="stable")          # real rows first within a day (documented within-day order)
    txn_pl = SYN_TXN_BASE + np.arange(n_pl, dtype=np.int64)
    aug = Arrays(
        day=aug_day[order], txn_id=np.concatenate([real.txn_id, txn_pl])[order], card=np.concatenate([real.card, pl_card])[order],
        pa=np.concatenate([real.pa, pl_pa])[order], net=np.concatenate([real.net, pl_net])[order],
        dev=np.concatenate([real.dev, pl_dev])[order],
        n_entities={"pseudo_account": state["next_pa"], "card": state["next_card"], "netproxy": n_net + state["syn_net"],
                    "device_fingerprint": n_dev + state["syn_dev"]},
        y=np.concatenate([real.y, pl_label])[order])
    x_src = np.concatenate([np.arange(n_real, dtype=np.int64), x_src_pl])[order]
    is_planted = np.concatenate([np.zeros(n_real, bool), np.ones(n_pl, bool)])[order]
    gid_pl = np.array([r[0] for r in rows])
    row_group = np.concatenate([np.full(n_real, -1, dtype=np.int64), gid_pl.astype(np.int64)])[order]

    for g in groups:
        sel = gid_pl == g["group_id"]
        cards_alive = sorted(set(pl_card[sel].tolist()))
        g["cards_final"] = cards_alive
        g["card_first_day"] = {int(c): int(pl_day[sel & (pl_card == c)].min()) for c in cards_alive}
        g["active_end"] = int(pl_day[sel].max()) if sel.any() else g["formation_day"]
        g["n_transactions"] = int(sel.sum())
        g["n_with_infra_edge"] = int(((pl_net[sel] >= 0) | (pl_dev[sel] >= 0)).sum())
        g["cards"] = [int(c) for c in g["cards"]]
        g["pas"] = [int(c) for c in g["pas"]]
        g["hijacked"] = [bool(c) for c in g["hijacked"]]
    manifest = dict(
        design="experiments/phase9_coordinated_fraud/DESIGN_BEFORE_EXECUTION.md section 11", gen_seed=int(gen_seed), is_eval=bool(is_eval),
        period=[lo, hi], pool_min_day=int(pool_min_day), n_real_period=n_period, cap=cap, cap_n=cap_n,
        n_planted_before_thinning=int(total0), n_planted=int(n_pl), scale_factor=float(n_pl / max(total0, 1)),
        planted_fraction=float(n_pl / max(n_period, 1)), n_real_net=int(n_net), n_real_dev=int(n_dev), n_syn_net=int(state["syn_net"]),
        n_syn_dev=int(state["syn_dev"]), n_real_card=int(n_card),
        n_coordinated=sum(1 for g in groups if g["kind"] == "coordinated"), n_decoy=sum(1 for g in groups if g["kind"] == "decoy"),
        groups=groups)
    return aug, x_src, manifest, is_planted, row_group


def manifest_json(m: dict) -> str:
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(type(o))
    return json.dumps(m, indent=1, default=default)


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def check_manifest(aug: Arrays, manifest: dict, row_group: np.ndarray) -> dict:
    """Gate G5: every manifest member exists in the augmented arrays with the specified first planted day; ids in range."""
    bad = []
    n_card = manifest["n_real_card"]
    for g in manifest["groups"]:
        rows = np.nonzero(row_group == g["group_id"])[0]
        if len(rows) != g["n_transactions"]:
            bad.append((g["group_id"], "n_transactions"))
        for c in g["cards_final"]:
            r = rows[aug.card[rows] == c]
            if len(r) == 0 or int(aug.day[r].min()) != g["card_first_day"][c]:
                bad.append((g["group_id"], f"card {c}"))
        for c, h in zip(g["cards"], g["hijacked"]):
            if (c < n_card) != h:
                bad.append((g["group_id"], f"hijack flag {c}"))
        if not (g["formation_day"] <= g["active_end"]):
            bad.append((g["group_id"], "interval"))
    return {"passed": not bad, "n_problems": len(bad), "problems": bad[:10]}
