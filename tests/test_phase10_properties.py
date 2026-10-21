"""Phase 10 core: accounting ledgers, invariants I1-I9, property tests against an independent oracle, determinism, exact arithmetic.
Fixture parameters are test inputs, not research-selected thresholds. Random generators are toy/synthetic only."""
from __future__ import annotations

import ast
import pathlib
import random
import re
import sys
from fractions import Fraction as Fr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests.phase10_helpers import (CO, ED, EN, N, P, R, ComponentRecord, EdgeRecord, EvolutionTracker, LineageEnd, NodeRecord, expect,
                                   kinds, rel, run, record_sort_key)

ROOT = pathlib.Path(__file__).resolve().parent.parent


# ------------------------------------------------------------------------------------------------ pinned accounting ledgers
def _led(row, **kw):
    for k, v in kw.items():
        assert getattr(row, k) == v, f"day {row.day}: {k} = {getattr(row, k)} expected {v}\n{row}"


def test_ledger_D_c_and_later_day_declarations_g_r0_g_d3_event_and_compat():
    seq = [[R(1, 10)], [], [], [], []]
    for mode in ("event", "compat"):
        tr, rr = run(seq, mode=mode, g_r=0, g_d=3)
        L = tr.ledger
        _led(L[0], V_pre=0, b=1, N=1, V_post=1, Lam=1)
        _led(L[1], V_pre=1, A=1, D_r=0, D_c=0, z=1, N=0, d=0, V_post=1)
        _led(L[2], V_pre=1, A=0, D_r=0, D_c=1, d=0, V_post=1)
        _led(L[3], V_pre=1, A=0, D_r=0, D_c=1, d=1, d_from_z=0, d_from_Dr=0, d_from_Dc=1, V_post=0, X=1, E=0)
        _led(L[4], V_pre=0, D_c=0, d=0, V_post=0, X=1, Lam=1)
        assert [r for r in rr[3]] == [EN(0, 4, "disappeared")]
        assert rr[1] == rr[2] == rr[4] == []


def test_ledger_declaration_source_classes_z_Dr_Dc():
    seq = [[R(1, 10)], [], [], [], []]
    tr, rr = run(seq, g_r=1, g_d=2)                          # primary g_d = g_r + 1: declared from D_r
    L = tr.ledger
    _led(L[1], A=1, z=1, d=0, V_post=1)
    _led(L[2], A=0, D_r=1, D_c=0, d=1, d_from_Dr=1, d_from_z=0, d_from_Dc=0, V_post=0, X=1)
    assert rr[2] == [EN(0, 3, "disappeared")]
    tr, rr = run(seq, g_r=0, g_d=1)                          # g_d = 1: declared from z (no dormancy)
    L = tr.ledger
    _led(L[1], A=1, z=1, d=1, d_from_z=1, d_from_Dr=0, d_from_Dc=0, V_post=0, X=1)
    assert rr[1] == [EN(0, 2, "disappeared")]
    tr, rr = run(seq, g_r=1, g_d=3)                          # non-primary g_d > g_r + 1 in event mode: D_c exists
    L = tr.ledger
    _led(L[2], D_r=1, D_c=0, d=0, V_post=1)
    _led(L[3], D_r=0, D_c=1, d=1, d_from_Dc=1, V_post=0, X=1)


def test_ledger_simultaneous_relink_and_declaration_on_day_L_plus_g_d():
    tr, rr = run([[R(1, 10), R(21, 30)], [], [], [R(1, 10)]])
    L = tr.ledger
    _led(L[0], b=2, N=2, V_post=2, Lam=2)
    _led(L[1], V_pre=2, A=2, z=2, N=0, V_post=2)
    _led(L[2], V_pre=2, A=0, D_r=2, V_post=2)
    _led(L[3], V_pre=2, A=0, D_r=2, D_c=0, s=0, e=0, z=0, r=1, b=0, N=1, d=1, d_from_Dr=1, V_post=1, E=0, X=1, Lam=2)
    expect(rr[3], [N((4, 1), 10, 0, "relinked", edge=("relink", (1, 1), (4, 1)), gap=2, pred=(1, 1), sc="stable"),
                   ED("relink", (1, 1), (4, 1), 10, 10, 10, "1", "1", "1", "ID F D", True), EN(1, 4, "disappeared")])


def test_ledger_mixed_day_with_event_relink_birth_and_declaration():
    seq = [[R(1, 10), R(21, 30), R(61, 80)], [R(61, 80)], [R(61, 80)], [R(1, 10), R(61, 110)]]
    tr, rr = run(seq)
    L = tr.ledger
    _led(L[0], b=3, N=3, V_post=3, Lam=3)
    _led(L[1], V_pre=3, A=3, s=1, e=0, z=2, N=1, V_post=3)
    _led(L[2], V_pre=3, A=1, D_r=2, s=1, z=0, N=1, V_post=3)
    _led(L[3], V_pre=3, A=1, D_r=2, D_c=0, s=0, e=1, z=0, r=1, b=1, N=2, d=1, V_post=2, E=1, X=1, Lam=4)
    E = rel((3, 61), (4, 61))
    expect(rr[3], [
        N((4, 1), 10, 0, "relinked", edge=("relink", (1, 1), (4, 1)), gap=2, pred=(1, 1), sc="stable"),
        N((4, 61), 50, 3, "born", origin="born_succession", comp=(4, (4, 61)), role="unselected_member", parents=[((3, 61), 2, E)]),
        ED("relink", (1, 1), (4, 1), 10, 10, 10, "1", "1", "1", "ID F D", True),
        ED("relation", (3, 61), (4, 61), 20, 20, 50, "2/5", "1", "2/5", "F", False),
        CO((4, (4, 61)), "one_to_one", [2], [(4, 61)], [E], [], 0, 1, 1, [(2, (4, 61), 20, 20, 50)]),
        EN(1, 4, "disappeared"), EN(2, 4, "succession", [(4, 61)], [3], [E])])
    assert [c.status for c in tr.classification()] == ["active", "disappeared", "ended_by_event", "active"]


def test_ledger_T33prime_extension_under_R_A():
    X, P_ = R(6, 10), {1, 2, 3, 4, 5, 8, 9, 10}
    tr, rr = run([[X], [P_], [R(1, 5), {8, 9, 10}], []])
    L = tr.ledger
    _led(L[0], b=1, V_post=1)
    _led(L[1], V_pre=1, A=1, s=0, e=0, z=1, r=0, b=1, N=1, V_post=2)
    _led(L[2], V_pre=2, A=1, D_r=1, s=1, e=0, z=0, r=0, b=1, N=2, d=0, V_post=3)
    _led(L[3], V_pre=3, A=2, D_r=1, z=2, s=0, N=0, d=1, d_from_Dr=1, V_post=2, X=1)


def test_ledger_empty_days_all_empty_and_between_data():
    tr, rr = run([[], [], []])
    assert all(r.V_pre == r.V_post == r.N == r.b == r.d == 0 for r in tr.ledger)
    tr, rr = run([[R(1, 10)], [], [R(1, 10)]])
    _led(tr.ledger[1], N=0, s=0, e=0, r=0, b=0, z=1)


# ------------------------------------------------------------------------------------------------ independent oracle
def _oracle_day(t, canon, lins, prm):
    th_id, th_ev, g_r, g_d = prm["theta_id"], prm["theta_ev"], prm["g_r"], prm["g_d"]
    nodes = [(t, min(s)) for s in canon]
    active = sorted(l for l, v in lins.items() if v["status"] == "live" and v["last_day"] == t - 1)
    pairs = {}
    for lid in active:
        ls = lins[lid]["last_set"]
        for r, s in enumerate(canon):
            x = len(ls & s)
            if x:
                a, b = len(ls), len(s)
                J, kf, kb = Fr(x, a + b - x), Fr(x, a), Fr(x, b)
                pairs[(lid, r)] = (x, a, b, J, kf, kb, tuple(f for f, on in (("ID", J >= th_id), ("F", kf >= th_ev), ("D", kb >= th_ev)) if on))
    sel, used_l, used_r = {}, set(), set()
    for k in sorted((k for k, v in pairs.items() if "ID" in v[6]), key=lambda k: (-pairs[k][3], -pairs[k][0], k[0], k[1])):
        if k[0] not in used_l and k[1] not in used_r:
            sel[k] = True
            used_l.add(k[0])
            used_r.add(k[1])
    Q = {k: v for k, v in pairs.items() if v[6]}
    # connected components by repeated merging
    comps = []
    for k in Q:
        hit = [c for c in comps if k[0] in c["l"] or k[1] in c["r"]]
        merged = {"l": {k[0]}, "r": {k[1]}, "e": {k}}
        for c in hit:
            merged["l"] |= c["l"]
            merged["r"] |= c["r"]
            merged["e"] |= c["e"]
            comps.remove(c)
        comps.append(merged)
    shape_of = lambda c: ("one_to_one" if len(c["l"]) == 1 and len(c["r"]) == 1 else "one_to_many" if len(c["l"]) == 1
                          else "many_to_one" if len(c["r"]) == 1 else "many_to_many")
    born_o = {"one_to_one": "born_succession", "one_to_many": "born_split_child", "many_to_one": "born_merge_product",
              "many_to_many": "born_reorganization_child"}
    end_r = {"one_to_one": "succession", "one_to_many": "dissolved", "many_to_one": "merged", "many_to_many": "reorganized"}
    in_comp_r = {r for c in comps for r in c["r"]}
    origin, ends = {}, {}
    for c in comps:
        for r in c["r"]:
            if not any(k[1] == r for k in sel):
                origin[r] = born_o[shape_of(c)]
        for l in c["l"]:
            if not any(k[0] == l for k in sel):
                ends[l] = end_r[shape_of(c)]
    sel_r = {k[1] for k in sel}
    dormant = sorted(l for l, v in lins.items() if v["status"] == "live" and 2 <= t - v["last_day"] <= g_r + 1)
    elig = [r for r in range(len(canon)) if r not in in_comp_r and r not in sel_r]
    cand = {}
    for l in dormant:
        for r in elig:
            x = len(lins[l]["last_set"] & canon[r])
            if x:
                a, b = len(lins[l]["last_set"]), len(canon[r])
                J = Fr(x, a + b - x)
                if J >= th_id:
                    cand[(l, r)] = (x, J, t - lins[l]["last_day"] - 1)
    relink, ul, ur = {}, set(), set()
    for k in sorted(cand, key=lambda k: (-cand[k][1], -cand[k][0], cand[k][2], k[0], k[1])):
        if k[0] not in ul and k[1] not in ur:
            relink[k] = True
            ul.add(k[0])
            ur.add(k[1])
    return dict(nodes=nodes, active=active, pairs=pairs, Q=Q, sel=sel, comps=comps, shape_of=shape_of, origin=origin, ends=ends,
                relink=relink, dormant=dormant)


def _check_run(seq, prm, check_perm=False):
    """Run the tracker on ``seq`` and compare every day against the independent oracle; check I1-I9 and the ledger equations."""
    tr = EvolutionTracker(**prm)
    lins = {}
    created = 0
    node_sets, all_recs, ended_ids = {}, [], set()
    prev_ledger = None
    prev_nodes = []
    for t, comms in enumerate(seq, start=1):
        recs = tr.step(t, comms)
        all_recs.append(recs)
        canon = sorted((frozenset(c) for c in comms), key=min)
        o = _oracle_day(t, canon, lins, prm)
        nodes = [r for r in recs if isinstance(r, NodeRecord)]
        edges = [r for r in recs if isinstance(r, EdgeRecord)]
        comps = [r for r in recs if isinstance(r, ComponentRecord)]
        ends = [r for r in recs if isinstance(r, LineageEnd)]
        # I5 ordering and causality of references
        assert recs == sorted(recs, key=record_sort_key)
        assert all(e.available_at >= e.day for e in ends)
        assert [n.node_id for n in nodes] == o["nodes"] and [n.size for n in nodes] == [len(s) for s in canon]
        for n, s in zip(nodes, canon):
            node_sets[n.node_id] = s
        # edges: soundness and completeness of Q, exact arithmetic recomputed from the raw sets
        rel_edges = {(e.edge_id[1], e.edge_id[2]): e for e in edges if e.kind == "relation"}
        assert len(rel_edges) == len(o["Q"])
        for (lid, r), v in o["Q"].items():
            e = rel_edges[(lins[lid]["last_node"], o["nodes"][r])]
            assert (e.x, e.a, e.b, e.J, e.k_f, e.k_b, e.flags) == v, (t, lid, r)
            assert e.selected == ((lid, r) in o["sel"]) and e.day == t
        for e in edges:
            assert e.edge_id[1][0] < e.edge_id[2][0] == t
            x = len(node_sets[e.edge_id[1]] & node_sets[e.edge_id[2]])
            assert e.x == x and e.a == len(node_sets[e.edge_id[1]]) and e.b == len(node_sets[e.edge_id[2]])
        # identity selection (stage 1) equals the oracle's greedy; no blocking pair
        sel_mod = {(n.identity.lineage_id, i) for i, n in enumerate(nodes) if n.identity.basis == "selected_identity"}
        assert sel_mod == set(o["sel"])
        key = lambda k: (-o["pairs"][k][3], -o["pairs"][k][0], k[0], k[1])
        for k, v in o["pairs"].items():
            if "ID" in v[6] and k not in o["sel"]:
                pl = next((s for s in o["sel"] if s[0] == k[0]), None)
                pr = next((s for s in o["sel"] if s[1] == k[1]), None)
                assert not ((pl is None or key(pl) > key(k)) and (pr is None or key(pr) > key(k))), ("blocking pair", t, k)
        # components partition the relation graph (I7) with the right shapes, counts and one event kind per component
        assert len(comps) == len(o["comps"])
        seen_edges = set()
        for c in comps:
            oc = next(x for x in o["comps"] if {("relation", lins[k[0]]["last_node"], o["nodes"][k[1]]) for k in x["e"]} == set(c.edge_ids))
            assert c.shape == o["shape_of"](oc) and set(c.lineages) == oc["l"]
            assert set(c.nodes) == {o["nodes"][r] for r in oc["r"]}
            assert not (set(c.edge_ids) & seen_edges)
            seen_edges |= set(c.edge_ids)
            assert c.s == len(set(c.selected_edge_ids)) == sum(1 for k in oc["e"] if k in o["sel"])
            assert c.born_count == sum(1 for r in oc["r"] if r in o["origin"]) and c.relinked_count == 0
            assert c.unselected_count == c.born_count + c.relinked_count
            assert c.ends == sum(1 for l in oc["l"] if l in o["ends"])
            assert all(0 <= v <= 1 for _, v in c.parent_coverage + c.child_coverage)
        assert seen_edges == {e.edge_id for e in edges if e.kind == "relation"}
        # nodes: origins, roles, relink eligibility (I3), identity fields
        rl_mod = {(n.identity.lineage_id, i) for i, n in enumerate(nodes) if n.identity.basis == "relinked"}
        assert rl_mod == set(o["relink"])
        new_ids = []
        for i, n in enumerate(nodes):
            idn = n.identity
            if idn.basis == "born":
                assert n.origin == o["origin"].get(i, "pure") and idn.edge_id is None and idn.gap is None and n.identity_predecessor is None
                new_ids.append(idn.lineage_id)
            elif idn.basis == "selected_identity":
                assert idn.gap == 0 and n.origin is None and "ID" in rel_edges[(n.identity_predecessor, n.node_id)].flags
                assert n.component_ref is not None and n.component_role == "selected_member"
            else:
                lv = lins[idn.lineage_id]
                assert n.component_ref is None and n.component_role == "none" and n.relation_parents == ()
                assert 2 <= t - lv["last_day"] <= prm["g_r"] + 1 and lv["status"] == "live" and idn.gap == t - lv["last_day"] - 1
                assert n.identity_predecessor == lv["last_node"]
            if idn.basis != "born":
                assert idn.lineage_id not in ended_ids
            assert (n.component_ref is None) == (n.component_role == "none")
        assert new_ids == list(range(created, created + len(new_ids)))
        created += len(new_ids)
        ids_today = [n.identity.lineage_id for n in nodes]
        assert len(ids_today) == len(set(ids_today))                                            # I1
        # I8 / I9 / I2: selected edges form a matching; relation selected edges carry ID; no relation-only selection
        selE = [e for e in edges if e.selected]
        assert len({e.edge_id[1] for e in selE}) == len(selE) and len({e.edge_id[2] for e in selE}) == len(selE)
        assert all(("ID" in e.flags) for e in selE) and all(e.selected for e in edges if e.kind == "relink")
        # lineage ends: reasons, adjacency (E2-A), never reappear (I4)
        em = {e.lineage_id: e for e in ends if e.reason != "disappeared"}
        assert {l: e.reason for l, e in em.items()} == o["ends"]
        id_of_rank = {n.node_id: n.identity.lineage_id for n in nodes}
        for lid, e in em.items():
            own = sorted(ed.edge_id for ed in edges if ed.kind == "relation" and ed.edge_id[1] == lins[lid]["last_node"])
            assert list(e.via_edges) == own and list(e.target_nodes) == [x[2] for x in own]
            assert list(e.target_lineages) == [id_of_rank[x[2]] for x in own] and len(own) >= 1
        dec = [e for e in ends if e.reason == "disappeared"]
        for e in dec:
            assert e.target_nodes == e.target_lineages == e.via_edges == () and t - lins[e.lineage_id]["last_day"] == prm["g_d"]
        ended_ids |= {e.lineage_id for e in ends}
        # oracle state update, then ledger equations
        for lid, l in list(lins.items()):
            if lid in em:
                l["status"] = "ended"
        for n, s in zip(nodes, canon):
            lid = n.identity.lineage_id
            if n.identity.basis == "born":
                lins[lid] = dict(last_day=t, last_set=s, last_node=n.node_id, status="live")
            else:
                lins[lid].update(last_day=t, last_set=s, last_node=n.node_id)
        for e in dec:
            lins[e.lineage_id]["status"] = "declared"
        row = tr.ledger[-1]
        A = len(prev_nodes)
        s_, r_, b_ = len(sel_mod), len(rl_mod), len(new_ids)
        e_, d_ = len(em), len(dec)
        assert (row.A, row.s, row.r, row.b, row.N, row.e, row.d) == (A, s_, r_, b_, len(nodes), e_, d_)
        assert row.A == row.s + row.e + row.z and row.N == row.s + row.r + row.b
        assert row.V_post == row.V_pre + row.b - row.e - row.d == row.N + row.z + row.D_r + row.D_c - row.r - row.d
        assert row.Lam == row.V_post + row.E + row.X and row.d == row.d_from_z + row.d_from_Dr + row.d_from_Dc
        if prev_ledger is not None:
            assert row.V_pre == prev_ledger.V_post and row.Lam == prev_ledger.Lam + row.b
            assert row.E == prev_ledger.E + row.e and row.X == prev_ledger.X + row.d
        assert row.V_pre == row.A + row.D_r + row.D_c
        if prm["g_d"] == prm["g_r"] + 1:
            assert row.D_c == 0
        prev_ledger, prev_nodes = row, nodes
    # classification equals the arithmetic definition at the horizon (errata 2, E2-C)
    T = len(seq)
    cls = {c.lineage_id: c for c in tr.classification()}
    assert set(cls) == set(range(created))
    declared_ids = {e.lineage_id for rr in all_recs for e in rr if isinstance(e, LineageEnd) and e.reason == "disappeared"}
    ev_ids = {e.lineage_id for rr in all_recs for e in rr if isinstance(e, LineageEnd) and e.reason != "disappeared"}
    for lid, c in cls.items():
        L = lins[lid]["last_day"]
        if lid in ev_ids:
            assert c.status == "ended_by_event"
        elif L == T:
            assert c.status == "active"
        elif L + prm["g_d"] <= T:
            assert c.status == "disappeared" and c.declared_day == L + prm["g_d"]
        else:
            assert c.status == "undetermined" and c.known_at_if_unseen == L + prm["g_d"]
        assert (c.status == "disappeared") == (lid in declared_ids)
    if check_perm:
        rng = random.Random(len(seq))
        shuffled = [[set(rng.sample(sorted(c), len(c))) for c in rng.sample(day, len(day))] for day in seq]
        tr2 = EvolutionTracker(**prm)
        assert [tr2.step(i, d) for i, d in enumerate(shuffled, start=1)] == all_recs
    return all_recs


# ------------------------------------------------------------------------------------------------ random sequences
def gen_seq(rng, n_days, universe=40, p_empty=0.12):
    history, seq, prev = [], [], []
    for _ in range(n_days):
        if rng.random() < p_empty:
            day = []
        else:
            pool = []
            for c in prev:
                if rng.random() < 0.15:
                    continue
                c = {x for x in c if rng.random() > 0.15} | {rng.randrange(1, universe + 1) for _ in range(rng.choice([0, 0, 1, 2, 5]))}
                if len(c) >= 2 and rng.random() < 0.15:
                    cl = sorted(c)
                    rng.shuffle(cl)
                    pool += [set(cl[:len(cl) // 2]), set(cl[len(cl) // 2:])]
                else:
                    pool.append(c)
            if len(pool) >= 2 and rng.random() < 0.2:
                a, b = rng.sample(range(len(pool)), 2)
                pool = [c for i, c in enumerate(pool) if i not in (a, b)] + [pool[a] | pool[b]]
            if history and rng.random() < 0.3:
                pool.append(set(rng.choice(history[-8:])))
            if rng.random() < 0.4:
                pool.append({rng.randrange(1, universe + 1) for _ in range(rng.randint(2, 8))})
            rng.shuffle(pool)
            taken, day = set(), []
            for c in pool:
                c = {x for x in c if x not in taken}
                if c:
                    taken |= c
                    day.append(c)
            history += day
        seq.append(day)
        prev = day
    return seq


_THETA_ID = (Fr(1, 2), Fr(1, 3), Fr(2, 3), Fr(1, 5))
_THETA_EV = (Fr(1, 2), Fr(2, 3), Fr(3, 4), Fr(1, 3))
_DELTA = (Fr(1, 2), Fr(1, 3), Fr(1))
_GAPS = ((0, 1), (0, 2), (1, 2), (2, 3), (1, 3), (2, 4), (3, 4), (0, 3))


def test_property_random_sequences_match_oracle_and_satisfy_invariants():
    n_runs = 0
    relinks = births_by_origin = ends_by_reason = 0
    seen_origins, seen_reasons, seen_dc = set(), set(), False
    for seed in range(120):
        rng = random.Random(1000 + seed)
        prm = P(theta_id=rng.choice(_THETA_ID), theta_ev=rng.choice(_THETA_EV), delta=rng.choice(_DELTA))
        prm["g_r"], prm["g_d"] = rng.choice(_GAPS)
        recs = _check_run(gen_seq(rng, rng.randint(8, 24)), prm, check_perm=(seed % 6 == 0))
        n_runs += 1
        for rr in recs:
            for r in rr:
                if isinstance(r, NodeRecord):
                    seen_origins.add(r.origin)
                    relinks += r.identity.basis == "relinked"
                if isinstance(r, LineageEnd):
                    seen_reasons.add(r.reason)
        seen_dc |= prm["g_d"] > prm["g_r"] + 1
    # coverage of the generator: the property run must actually exercise every record kind (otherwise the test proves little)
    assert relinks > 20, relinks
    assert {"pure", "born_split_child", "born_merge_product", "born_reorganization_child", "born_succession"} <= seen_origins, seen_origins
    assert {"succession", "dissolved", "merged", "reorganized", "disappeared"} <= seen_reasons, seen_reasons
    assert seen_dc and n_runs == 120


def test_property_repeat_runs_are_identical():
    rng = random.Random(5)
    seq = gen_seq(rng, 20)
    assert run(seq)[1] == run(seq)[1]


# ------------------------------------------------------------------------------------------------ compat-mode accounting identities
def test_compat_mode_accounting_is_the_same_equations_with_Q_empty():
    for seed in range(40):
        rng = random.Random(77 + seed)
        seq = gen_seq(rng, rng.randint(8, 20))
        g_d = rng.choice((1, 2, 3, 4))
        tr, rr = run(seq, mode="compat", theta_id=rng.choice((Fr(1, 2), Fr(1, 3))), g_r=0, g_d=g_d)
        for row in tr.ledger:
            assert row.e == 0 and row.r == 0 and row.N == row.s + row.b and row.z == row.A - row.s
            assert row.V_post == row.V_pre + row.b - row.d
            assert row.V_pre == row.A + row.D_r + row.D_c and row.D_r == 0
        for recs in rr:
            assert not kinds(recs, EdgeRecord) and not kinds(recs, ComponentRecord)
            assert all(n.component_ref is None and n.component_role == "none" and n.relation_parents == () for n in kinds(recs, NodeRecord))
            assert all(e.reason == "disappeared" for e in kinds(recs, LineageEnd))


# ------------------------------------------------------------------------------------------------ exact arithmetic / wording
def test_module_source_has_no_floating_point():
    src = (ROOT / "src" / "evolution" / "tracker.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in ast.walk(tree):
        assert not (isinstance(node, ast.Constant) and isinstance(node.value, float)), "float literal"
        assert not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)), "true division"
        assert not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "float"), "float()"
        assert not (isinstance(node, (ast.Import, ast.ImportFrom)) and "math" in ast.dump(node)), "math import"


def test_new_files_do_not_use_the_prohibited_wording():
    pat = re.compile(r"fraud\s+rings?", re.I)
    files = list((ROOT / "src" / "evolution").glob("*.py")) + list((ROOT / "tests").glob("*phase10*.py"))
    assert len(files) >= 5
    for f in files:
        assert not pat.search(f.read_text(encoding="utf-8")), f


def test_new_package_does_not_import_phase9_code():
    # the new package must not import from the Phase 9 coordination package (only the compatibility tests import it, read-only)
    for f in (ROOT / "src" / "evolution").glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert "coordination" not in (node.module or ""), f
            if isinstance(node, ast.Import):
                assert all("coordination" not in a.name for a in node.names), f


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
