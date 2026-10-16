"""Phase 9 unit tests: graph windows/masks, null-model invariants, T causality, tracker, matching/AP, planting invariants,
terminology lint, static no-label checks. Toy/synthetic data only (no real-data run)."""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination import frozen as F
from src.coordination.data import Arrays, GraphView
from src.coordination.graphs import aggregate, louvain_candidates, q_of_nodes, rewire, star_candidates, window_rows
from src.coordination.lint import lint_identifiers, lint_text
from src.coordination.metrics import GroupIndex, annotate, track_ap
from src.coordination.scoring import ArrivalTable, t_raw_for
from src.coordination.tracking import Tracker


def _toy_view(n_days=40, per_day=60, n_card=80, n_net=10, n_dev=10, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(1, n_days + 1), per_day)
    n = len(day)
    card = rng.integers(0, n_card, n)
    net = np.where(rng.random(n) < 0.4, rng.integers(0, n_net, n), -1)
    dev = np.where(rng.random(n) < 0.3, rng.integers(0, n_dev, n), -1)
    return GraphView(day, card, net, dev, n_net, n_dev)


def test_window_rows_never_include_future_or_older_than_window():
    v = _toy_view()
    for t, W in ((10, 3), (25, 7), (30, 14), (40, None)):
        e = window_rows(v, t, W)
        assert e.day.max() <= t
        if W is not None:
            assert e.day.min() > t - W


def test_aggregate_caps_weights():
    v = _toy_view(per_day=400, n_card=3)
    g = aggregate(window_rows(v, 20, 14))
    assert g.w.max() <= F.EDGE_WEIGHT_CAP


def test_rewire_preserves_per_day_degree_sequences():
    v = _toy_view()
    e = window_rows(v, 20, 14)
    r = rewire(e, np.random.default_rng(1))
    for kind in (0, 1):
        for d in np.unique(e.day):
            a = (e.day == d) & (e.kind == kind)
            assert sorted(e.infra[a].tolist()) == sorted(r.infra[(r.day == d) & (r.kind == kind)].tolist())
    assert (e.card == r.card).all() and (e.day == r.day).all()
    assert (e.infra != r.infra).any()


def test_louvain_candidates_eligibility_and_determinism():
    v = _toy_view()
    g = aggregate(window_rows(v, 30, 14))
    a = louvain_candidates(g, 1.0, seed=7)
    b = louvain_candidates(g, 1.0, seed=7)
    assert [c.cards.tolist() for c in a] == [c.cards.tolist() for c in b]
    for c in a:
        assert len(c.cards) >= F.M_MIN and len(c.infra) >= 1


def test_q_of_nodes_matches_candidate_q():
    v = _toy_view()
    g = aggregate(window_rows(v, 30, 14))
    for c in louvain_candidates(g, 1.0, seed=3)[:5]:
        assert abs(q_of_nodes(g, c.cards, c.infra) - c.q) < 1e-9


def test_star_candidates_have_min_cards():
    v = _toy_view()
    g = aggregate(window_rows(v, 30, 14))
    for c in star_candidates(g):
        assert len(c.cards) >= F.M_MIN and len(c.infra) == 1


def test_t_score_is_causal_and_increases_with_burst():
    v = _toy_view(n_days=60)
    tab = ArrivalTable(v)
    base = t_raw_for([np.array([0, 1, 2])], tab.M, tab.birth, tab.kind, 50)[0]
    # corrupt everything after day 50: values for t = 50 must not change
    day2 = v.day.copy()
    card2 = v.card.copy()
    card2[day2 > 50] = (card2[day2 > 50] * 7 + 3) % 80
    v2 = GraphView(day2, card2, v.net.copy(), v.dev.copy(), v.n_net, v.n_dev)
    tab2 = ArrivalTable(v2)
    assert t_raw_for([np.array([0, 1, 2])], tab2.M, tab2.birth, tab2.kind, 50)[0] == base
    # inject a burst of 25 new cards on infra 0 during days 49-50
    extra = 25
    day3 = np.concatenate([v.day, np.full(extra, 50)])
    card3 = np.concatenate([v.card, 1000 + np.arange(extra)])
    net3 = np.concatenate([v.net, np.zeros(extra, dtype=np.int64)])
    dev3 = np.concatenate([v.dev, -np.ones(extra, dtype=np.int64)])
    o = np.argsort(day3, kind="stable")
    v3 = GraphView(day3[o], card3[o], net3[o], dev3[o], v.n_net, v.n_dev)
    tab3 = ArrivalTable(v3)
    assert t_raw_for([np.array([0])], tab3.M, tab3.birth, tab3.kind, 50)[0] > t_raw_for([np.array([0])], tab.M, tab.birth, tab.kind, 50)[0] + 3


def test_tracker_continue_split_merge():
    tr = Tracker()
    a = tr.step(1, [np.array([1, 2, 3, 4]), np.array([10, 11, 12])])
    b = tr.step(2, [np.array([1, 2, 3, 4, 5]), np.array([10, 11, 12])])
    assert b == a                                               # both continue
    c = tr.step(3, [np.array([1, 2, 3]), np.array([4, 5, 90])])  # split of track a: largest overlap continues
    assert c[0] == a[0] and c[1] != a[0]
    d = tr.step(5, [np.array([1, 2, 3])])                        # gap day -> no continuation (matching uses t-1 only)
    assert d[0] not in a


def _toy_manifest():
    g = [dict(group_id=0, kind="coordinated", pattern="P1", regime_alpha=0.0, cards_final=[1, 2, 3, 4], card_first_day={1: 5, 2: 5, 3: 6, 4: 8},
              formation_day=5, active_end=12),
         dict(group_id=1, kind="decoy", pattern="D1", regime_alpha=0.0, cards_final=[20, 21, 22], card_first_day={20: 5, 21: 5, 22: 5},
              formation_day=5, active_end=12)]
    return {"groups": g}


def test_matching_uses_observable_ground_truth_at_t():
    gi = GroupIndex(_toy_manifest())
    m = gi.best_match(np.array([1, 2]), 5)          # G_5 = {1,2}: perfect match
    assert m[0][0] == 0 and abs(m[0][1] - 1.0) < 1e-12
    m = gi.best_match(np.array([1, 2, 3]), 5)       # card 3 not yet observable at day 5 -> J = 2/3
    assert abs(m[0][1] - 2 / 3) < 1e-12
    assert gi.best_match(np.array([1, 2]), 13) == []  # group no longer active


def test_track_ap_toy():
    gi = GroupIndex(_toy_manifest())
    recs = [dict(day=6, idx=0, cards=np.array([1, 2, 3]), track=0, _score=0.9),
            dict(day=6, idx=1, cards=np.array([50, 51, 52]), track=1, _score=0.8),
            dict(day=6, idx=2, cards=np.array([20, 21, 22]), track=2, _score=0.7)]
    annotate(recs, gi)
    r = track_ap(recs, gi, regime=0.0, thr=0.5)
    assert r["n_recovered"] == 1 and abs(r["ap"] - 1.0) < 1e-12 and r["fp"] == 2


def test_planting_invariants_small():
    from src.coordination.benchmark import SYN_TXN_BASE, plant
    rng = np.random.default_rng(5)
    days = np.repeat(np.arange(1, 81), 2000)
    n = len(days)
    real = Arrays(day=days, txn_id=np.arange(n) + 1000, card=rng.integers(0, 2000, n), pa=rng.integers(0, 9000, n),
                  net=np.where(rng.random(n) < 0.25, rng.integers(0, 100, n), -1),
                  dev=np.where(rng.random(n) < 0.15, rng.integers(0, 120, n), -1),
                  n_entities={"pseudo_account": 9000, "card": 2000, "netproxy": 100, "device_fingerprint": 120},
                  y=(rng.random(n) < 0.05).astype(np.float32))
    aug, xs, man, is_pl, rg = plant(real, (40, 79), dict(groups=60, decoys=30), gen_seed=123, is_eval=False, pool_min_day=1)
    assert man["planted_fraction"] <= F.PLANT_CAP + 1e-9
    assert (np.diff(aug.day) >= 0).all() and len(aug.day) == n + man["n_planted"]
    assert aug.txn_id[is_pl].min() >= SYN_TXN_BASE and len(np.unique(aug.txn_id)) == len(aug.txn_id)
    assert (aug.y[is_pl] == np.array([1.0 if g["kind"] == "coordinated" else 0.0 for g in man["groups"]]).max()).any()
    cards = {c for g in man["groups"] for c in g["cards_final"]}
    assert cards <= set(aug.card.tolist())
    assert man["n_coordinated"] == 60 and man["n_decoy"] == 30
    aug2, _, man2, _, _ = plant(real, (40, 79), dict(groups=60, decoys=30), gen_seed=123, is_eval=False, pool_min_day=1)
    assert (aug.day == aug2.day).all() and (aug.card == aug2.card).all() and man["n_planted"] == man2["n_planted"]


def test_discovery_modules_do_not_reference_labels_or_manifest():
    for mod in ("graphs", "scoring", "pipeline", "tracking", "nullmodel", "risk", "segments_array"):
        src = (ROOT / "src" / "coordination" / f"{mod}.py").read_text(encoding="utf-8")
        code = re.sub(r'""".*?"""', "", src, flags=re.S)
        assert "isFraud" not in code and ".y[" not in code and "arrays.y" not in code and "real.y" not in code, mod
        assert "benchmark" not in code.replace("benchmark-specific", ""), mod
        assert "import metrics" not in code and "from src.coordination.metrics" not in code, mod


def test_real_data_runner_never_imports_recovery_metrics_or_benchmark():
    p = ROOT / "scripts" / "run_phase9_realdata.py"
    if p.exists():
        src = p.read_text(encoding="utf-8")
        assert "coordination.metrics" not in src and "coordination.benchmark" not in src


def test_terminology_lint():
    assert not lint_text("This is a candidate coordinated community; not a confirmed fraud ring.")
    assert not lint_text("This does not establish that IEEE-CIS contains fraud rings.")
    assert lint_text("We detected a fraud ring of 5 cards")
    assert lint_text("a criminal network")
    assert lint_text("members of the gang")
    assert lint_text("a mule ring")
    assert not lint_text("candidate suspicious coordinated community")
    assert lint_identifiers("ring_id = 3") and lint_identifiers("n_rings")
    assert not lint_identifiers("rewiring scoring string during ensuring")


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
