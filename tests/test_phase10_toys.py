"""Phase 10 core: pinned toy cases (hand-derived expectations from the approved specification; fixture values are test inputs,
not research-selected thresholds). Expected records are written independently of the module's computations."""
from __future__ import annotations

import itertools
import pathlib
import random
import sys
from fractions import Fraction as Fr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from tests.phase10_helpers import (CO, ED, EN, F, N, P, R, ComponentRecord, EvolutionTracker, InvalidInputError, expect, kinds,
                                   rel, run)
from src.evolution import ClassificationRow, EdgeRecord, LineageEnd, NodeRecord


def U(*parts):
    out = set()
    for p in parts:
        out |= set(p)
    return out


# --------------------------------------------------------------------------------------------------- continuation family
def _cont(sa, sb, x, J, kf, kb, flags, sc):
    tr, rr = run([[sa], [sb]])
    pa, ca = (1, min(sa)), (2, min(sb))
    a, b = len(sa), len(sb)
    E, cid = rel(pa, ca), (2, ca)
    expect(rr[0], [N(pa, a, 0)])
    expect(rr[1], [
        N(ca, b, 0, "selected_identity", edge=E, gap=0, comp=cid, role="selected_member", parents=[(pa, 0, E)], pred=pa, sc=sc),
        ED("relation", pa, ca, x, a, b, J, kf, kb, flags, True),
        CO(cid, "one_to_one", [0], [ca], [E], [E], 1, 0, 0, [(0, ca, x, a, b)])])
    assert [c.status for c in tr.classification()] == ["active"]


def test_T1_identical_sets_continue():
    _cont(R(1, 10), R(1, 10), 10, "1", "1", "1", "ID F D", "stable")


def test_T2a_growth_below_boundary_is_stable():
    _cont(R(1, 10), R(1, 12), 10, "5/6", "1", "5/6", "ID F D", "stable")


def test_T2b_growth_boundary_exact_is_grow():
    _cont(R(1, 10), R(1, 15), 10, "2/3", "1", "2/3", "ID F D", "grow")


def test_T4a_contraction_boundary_exact_is_contract():
    _cont(R(1, 15), R(1, 10), 10, "2/3", "2/3", "1", "ID F D", "contract")


def test_T16_swap_is_stable_with_counts():
    _cont(R(1, 10), R(3, 12), 8, "2/3", "4/5", "4/5", "ID F D", "stable")
    tr, rr = run([[R(1, 10)], [R(3, 12)]])
    e = kinds(rr[1], EdgeRecord)[0]
    assert (e.x, e.a - e.x, e.b - e.x) == (8, 2, 2)          # retained / left / joined counts; 1 - J is the churn
    assert 1 - e.J == Fr(1, 3)


# --------------------------------------------------------------------------------------------------- break_with_link
def test_T3_relation_only_edge_ends_lineage_as_succession():
    tr, rr = run([[R(1, 10)], [R(1, 25)]])
    pa, ca = (1, 1), (2, 1)
    E, cid = rel(pa, ca), (2, (2, 1))
    expect(rr[1], [
        N(ca, 25, 1, "born", origin="born_succession", comp=cid, role="unselected_member", parents=[(pa, 0, E)]),
        ED("relation", pa, ca, 10, 10, 25, "2/5", "1", "2/5", "F", False),
        CO(cid, "one_to_one", [0], [ca], [E], [], 0, 1, 1, [(0, ca, 10, 10, 25)]),
        EN(0, 2, "succession", [ca], [1], [E])])
    comp = kinds(rr[1], ComponentRecord)[0]
    assert dict(comp.parent_coverage) == {0: Fr(1)} and dict(comp.child_coverage) == {ca: Fr(2, 5)}
    assert (comp.component_parent_coverage, comp.component_child_coverage) == (Fr(1), Fr(2, 5))


def test_T3_compat_only_row_no_relation_dormant_then_declared():
    tr, rr = run([[R(1, 10)], [R(1, 25)], [R(1, 25)], [R(1, 25)]], mode="compat", g_r=0, g_d=3)
    expect(rr[1], [N((2, 1), 25, 1)])
    expect(rr[2], [N((3, 1), 25, 1, "selected_identity", gap=0, pred=(2, 1), sc="stable")])
    expect(rr[3], [N((4, 1), 25, 1, "selected_identity", gap=0, pred=(3, 1), sc="stable"), EN(0, 4, "disappeared")])


def test_T4b_contraction_to_core_relation_only():
    tr, rr = run([[R(1, 20)], [R(1, 8)]])
    pa, ca = (1, 1), (2, 1)
    E, cid = rel(pa, ca), (2, (2, 1))
    expect(rr[1], [
        N(ca, 8, 1, "born", origin="born_succession", comp=cid, role="unselected_member", parents=[(pa, 0, E)]),
        ED("relation", pa, ca, 8, 20, 8, "2/5", "2/5", "1", "D", False),
        CO(cid, "one_to_one", [0], [ca], [E], [], 0, 1, 1, [(0, ca, 8, 20, 8)]),
        EN(0, 2, "succession", [ca], [1], [E])])


# --------------------------------------------------------------------------------------------------- split family
def test_T5_tie_goes_to_lower_rank_other_half_is_split_child():
    tr, rr = run([[R(1, 10)], [R(1, 5), R(6, 10)]])
    pa, c1, c2 = (1, 1), (2, 1), (2, 6)
    E1, E2, cid = rel(pa, c1), rel(pa, c2), (2, (2, 1))
    expect(rr[1], [
        N(c1, 5, 0, "selected_identity", edge=E1, gap=0, comp=cid, role="selected_member", parents=[(pa, 0, E1)], pred=pa, sc="contract"),
        N(c2, 5, 1, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(pa, 0, E2)]),
        ED("relation", pa, c1, 5, 10, 5, "1/2", "1/2", "1", "ID D", True),
        ED("relation", pa, c2, 5, 10, 5, "1/2", "1/2", "1", "ID D", False),
        CO(cid, "one_to_many", [0], [c1, c2], [E1, E2], [E1], 1, 1, 0, [(0, c1, 5, 10, 5), (0, c2, 5, 10, 5)])])
    comp = kinds(rr[1], ComponentRecord)[0]
    assert dict(comp.parent_coverage) == {0: Fr(1)} and dict(comp.child_coverage) == {c1: Fr(1), c2: Fr(1)}
    e2 = [e for e in kinds(rr[1], EdgeRecord) if e.edge_id == E2][0]
    assert (e2.k_f, e2.k_b) == (Fr(1, 2), Fr(1))             # share and child_fraction of the split child


def test_T6_split_with_dominant_part():
    tr, rr = run([[R(1, 10)], [R(1, 7), R(8, 10)]])
    pa, c1, c2 = (1, 1), (2, 1), (2, 8)
    E1, E2, cid = rel(pa, c1), rel(pa, c2), (2, (2, 1))
    expect(rr[1], [
        N(c1, 7, 0, "selected_identity", edge=E1, gap=0, comp=cid, role="selected_member", parents=[(pa, 0, E1)], pred=pa, sc="stable"),
        N(c2, 3, 1, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(pa, 0, E2)]),
        ED("relation", pa, c1, 7, 10, 7, "7/10", "7/10", "1", "ID F D", True),
        ED("relation", pa, c2, 3, 10, 3, "3/10", "3/10", "1", "D", False),
        CO(cid, "one_to_many", [0], [c1, c2], [E1, E2], [E1], 1, 1, 0, [(0, c1, 7, 10, 7), (0, c2, 3, 10, 3)])])


def test_T7_n_way_split():
    tr, rr = run([[R(1, 15)], [R(1, 8), R(9, 12), R(13, 15)]])
    pa, c1, c2, c3 = (1, 1), (2, 1), (2, 9), (2, 13)
    E1, E2, E3, cid = rel(pa, c1), rel(pa, c2), rel(pa, c3), (2, (2, 1))
    expect(rr[1], [
        N(c1, 8, 0, "selected_identity", edge=E1, gap=0, comp=cid, role="selected_member", parents=[(pa, 0, E1)], pred=pa, sc="contract"),
        N(c2, 4, 1, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(pa, 0, E2)]),
        N(c3, 3, 2, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(pa, 0, E3)]),
        ED("relation", pa, c1, 8, 15, 8, "8/15", "8/15", "1", "ID D", True),
        ED("relation", pa, c2, 4, 15, 4, "4/15", "4/15", "1", "D", False),
        ED("relation", pa, c3, 3, 15, 3, "1/5", "1/5", "1", "D", False),
        CO(cid, "one_to_many", [0], [c1, c2, c3], [E1, E2, E3], [E1], 1, 2, 0,
           [(0, c1, 8, 15, 8), (0, c2, 4, 15, 4), (0, c3, 3, 15, 3)])])


def test_T25_hub_fragmentation_is_dissolved_with_three_split_children():
    tr, rr = run([[R(1, 60)], [R(1, 20), R(21, 40), R(41, 60)]])
    pa = (1, 1)
    cs = [(2, 1), (2, 21), (2, 41)]
    Es = [rel(pa, c) for c in cs]
    cid = (2, (2, 1))
    exp = [N(c, 20, i + 1, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(pa, 0, E)])
           for i, (c, E) in enumerate(zip(cs, Es))]
    exp += [ED("relation", pa, c, 20, 60, 20, "1/3", "1/3", "1", "D", False) for c in cs]
    exp += [CO(cid, "one_to_many", [0], cs, Es, [], 0, 3, 1, [(0, c, 20, 60, 20) for c in cs]),
            EN(0, 2, "dissolved", cs, [1, 2, 3], Es)]
    expect(rr[1], exp)


# --------------------------------------------------------------------------------------------------- merge family
def test_T8_merge_unselected_parent_ends_merged():
    tr, rr = run([[R(1, 7), R(8, 10)], [R(1, 10)]])
    p0, p1, c = (1, 1), (1, 8), (2, 1)
    E0, E1, cid = rel(p0, c), rel(p1, c), (2, (2, 1))
    expect(rr[0], [N(p0, 7, 0), N(p1, 3, 1)])
    expect(rr[1], [
        N(c, 10, 0, "selected_identity", edge=E0, gap=0, comp=cid, role="selected_member", parents=[(p0, 0, E0), (p1, 1, E1)], pred=p0, sc="stable"),
        ED("relation", p0, c, 7, 7, 10, "7/10", "1", "7/10", "ID F D", True),
        ED("relation", p1, c, 3, 3, 10, "3/10", "1", "3/10", "F", False),
        CO(cid, "many_to_one", [0, 1], [c], [E0, E1], [E0], 1, 0, 1, [(0, c, 7, 7, 10), (1, c, 3, 3, 10)]),
        EN(1, 2, "merged", [c], [0], [E1])])


def test_T9_n_way_merge():
    tr, rr = run([[R(1, 8), R(9, 12), R(13, 15)], [R(1, 15)]])
    ps = [(1, 1), (1, 9), (1, 13)]
    c = (2, 1)
    Es = [rel(p, c) for p in ps]
    cid = (2, (2, 1))
    expect(rr[1], [
        N(c, 15, 0, "selected_identity", edge=Es[0], gap=0, comp=cid, role="selected_member",
          parents=[(ps[0], 0, Es[0]), (ps[1], 1, Es[1]), (ps[2], 2, Es[2])], pred=ps[0], sc="grow"),
        ED("relation", ps[0], c, 8, 8, 15, "8/15", "1", "8/15", "ID F", True),
        ED("relation", ps[1], c, 4, 4, 15, "4/15", "1", "4/15", "F", False),
        ED("relation", ps[2], c, 3, 3, 15, "1/5", "1", "1/5", "F", False),
        CO(cid, "many_to_one", [0, 1, 2], [c], Es, [Es[0]], 1, 0, 2, [(0, c, 8, 8, 15), (1, c, 4, 4, 15), (2, c, 3, 3, 15)]),
        EN(1, 2, "merged", [c], [0], [Es[1]]), EN(2, 2, "merged", [c], [0], [Es[2]])])


def _lineage_graph_has_cycle(recs_by_day):
    edges = set()
    for rr in recs_by_day:
        for r in rr:
            if isinstance(r, NodeRecord):
                for rp in r.relation_parents:
                    if rp.parent_lineage_before != r.identity.lineage_id:
                        edges.add((rp.parent_lineage_before, r.identity.lineage_id))
            if isinstance(r, LineageEnd):
                for tl in r.target_lineages:
                    if tl != r.lineage_id:
                        edges.add((r.lineage_id, tl))
    graph = {}
    for a, b in edges:
        graph.setdefault(a, set()).add(b)
    state = {}

    def dfs(u):
        state[u] = 1
        for v in graph.get(u, ()):
            if state.get(v) == 1 or (v not in state and dfs(v)):
                return True
        state[u] = 2
        return False
    return any(dfs(u) for u in list(graph) if u not in state), edges


def test_T20_split_then_remerge_cd_acyclic_lineage_graph_cyclic():
    tr, rr = run([[R(1, 10)], [R(1, 5), R(6, 10)], [R(1, 10)]])
    p0, p1, c = (2, 1), (2, 6), (3, 1)
    E0, E1, cid = rel(p0, c), rel(p1, c), (3, (3, 1))
    expect(rr[2], [
        N(c, 10, 0, "selected_identity", edge=E0, gap=0, comp=cid, role="selected_member", parents=[(p0, 0, E0), (p1, 1, E1)], pred=p0, sc="grow"),
        ED("relation", p0, c, 5, 5, 10, "1/2", "1", "1/2", "ID F", True),
        ED("relation", p1, c, 5, 5, 10, "1/2", "1", "1/2", "ID F", False),
        CO(cid, "many_to_one", [0, 1], [c], [E0, E1], [E0], 1, 0, 1, [(0, c, 5, 5, 10), (1, c, 5, 5, 10)]),
        EN(1, 3, "merged", [c], [0], [E1])])
    cyc, edges = _lineage_graph_has_cycle(rr)
    assert cyc and {(0, 1), (1, 0)} <= edges
    # community-day graph: every reference points to an earlier day, so it is acyclic
    for day_recs in rr:
        for r in day_recs:
            if isinstance(r, NodeRecord):
                for rp in r.relation_parents:
                    assert rp.parent_node[0] < r.node_id[0]


def test_T21_tie_with_k_f_equality_boundary():
    tr, rr = run([[R(1, 6), R(7, 12)], [U(R(1, 4), R(7, 10))]], theta_id=Fr(1, 3))
    p0, p1, c = (1, 1), (1, 7), (2, 1)
    E0, E1, cid = rel(p0, c), rel(p1, c), (2, (2, 1))
    expect(rr[1], [
        N(c, 8, 0, "selected_identity", edge=E0, gap=0, comp=cid, role="selected_member", parents=[(p0, 0, E0), (p1, 1, E1)], pred=p0, sc="stable"),
        ED("relation", p0, c, 4, 6, 8, "2/5", "2/3", "1/2", "ID F", True),
        ED("relation", p1, c, 4, 6, 8, "2/5", "2/3", "1/2", "ID F", False),
        CO(cid, "many_to_one", [0, 1], [c], [E0, E1], [E0], 1, 0, 1, [(0, c, 4, 6, 8), (1, c, 4, 6, 8)]),
        EN(1, 2, "merged", [c], [0], [E1])])


def test_T24_equal_halves_merge():
    tr, rr = run([[R(1, 40), R(41, 80)], [R(1, 80)]])
    p0, p1, c = (1, 1), (1, 41), (2, 1)
    E0, E1, cid = rel(p0, c), rel(p1, c), (2, (2, 1))
    expect(rr[1], [
        N(c, 80, 0, "selected_identity", edge=E0, gap=0, comp=cid, role="selected_member", parents=[(p0, 0, E0), (p1, 1, E1)], pred=p0, sc="grow"),
        ED("relation", p0, c, 40, 40, 80, "1/2", "1", "1/2", "ID F", True),
        ED("relation", p1, c, 40, 40, 80, "1/2", "1", "1/2", "ID F", False),
        CO(cid, "many_to_one", [0, 1], [c], [E0, E1], [E0], 1, 0, 1, [(0, c, 40, 40, 80), (1, c, 40, 40, 80)]),
        EN(1, 2, "merged", [c], [0], [E1])])


# --------------------------------------------------------------------------------------------------- reorganization
def _t15_sets():
    return [[R(1, 10), R(11, 20)], [U(R(1, 5), R(11, 15)), U(R(6, 10), R(16, 20))]]


def test_T15_crossing_reorganization_at_theta_ev_half():
    tr, rr = run(_t15_sets(), theta_ev=Fr(1, 2))
    pa, pb, c1, c2 = (1, 1), (1, 11), (2, 1), (2, 6)
    Es = {(p, c): rel(p, c) for p in (pa, pb) for c in (c1, c2)}
    cid = (2, (2, 1))
    xab = [(0, c1, 5, 10, 10), (0, c2, 5, 10, 10), (1, c1, 5, 10, 10), (1, c2, 5, 10, 10)]
    exp = [
        N(c1, 10, 2, "born", origin="born_reorganization_child", comp=cid, role="unselected_member",
          parents=[(pa, 0, Es[(pa, c1)]), (pb, 1, Es[(pb, c1)])]),
        N(c2, 10, 3, "born", origin="born_reorganization_child", comp=cid, role="unselected_member",
          parents=[(pa, 0, Es[(pa, c2)]), (pb, 1, Es[(pb, c2)])]),
        CO(cid, "many_to_many", [0, 1], [c1, c2], list(Es.values()), [], 0, 2, 2, xab),
        EN(0, 2, "reorganized", [c1, c2], [2, 3], [Es[(pa, c1)], Es[(pa, c2)]]),
        EN(1, 2, "reorganized", [c1, c2], [2, 3], [Es[(pb, c1)], Es[(pb, c2)]])]
    exp += [ED("relation", p, c, 5, 10, 10, "1/3", "1/2", "1/2", "F D", False) for (p, c) in Es]
    expect(rr[1], exp)


def test_T15_crossing_no_edges_at_theta_ev_two_thirds():
    tr, rr = run(_t15_sets())
    expect(rr[1], [N((2, 1), 10, 2), N((2, 6), 10, 3)])
    st = {c.lineage_id: c.status for c in tr.classification()}
    assert st == {0: "undetermined", 1: "undetermined", 2: "active", 3: "active"}


def test_T22_greedy_selection_and_reorganized_target_is_adjacent_only():
    ca = U(R(1, 6), R(11, 14))
    cb = U(R(7, 10), R(21, 26))
    tr, rr = run([[R(1, 10), R(11, 20)], [ca, cb]], theta_id=Fr(1, 5))
    l0, l1, na, nb = (1, 1), (1, 11), (2, 1), (2, 7)
    Eoa, Eob, E1a, cid = rel(l0, na), rel(l0, nb), rel(l1, na), (2, (2, 1))
    expect(rr[1], [
        N(na, 10, 0, "selected_identity", edge=Eoa, gap=0, comp=cid, role="selected_member", parents=[(l0, 0, Eoa), (l1, 1, E1a)], pred=l0, sc="stable"),
        N(nb, 10, 2, "born", origin="born_reorganization_child", comp=cid, role="unselected_member", parents=[(l0, 0, Eob)]),
        ED("relation", l0, na, 6, 10, 10, "3/7", "3/5", "3/5", "ID", True),
        ED("relation", l0, nb, 4, 10, 10, "1/4", "2/5", "2/5", "ID", False),
        ED("relation", l1, na, 4, 10, 10, "1/4", "2/5", "2/5", "ID", False),
        CO(cid, "many_to_many", [0, 1], [na, nb], [Eoa, Eob, E1a], [Eoa], 1, 1, 1,
           [(0, na, 6, 10, 10), (0, nb, 4, 10, 10), (1, na, 4, 10, 10)]),
        EN(1, 2, "reorganized", [na], [0], [E1a])])
    comp = kinds(rr[1], ComponentRecord)[0]
    assert dict(comp.parent_coverage) == {0: Fr(1), 1: Fr(2, 5)} and dict(comp.child_coverage) == {na: Fr(1), nb: Fr(2, 5)}
    assert (comp.component_parent_coverage, comp.component_child_coverage) == (Fr(7, 10), Fr(7, 10))
    # greedy is not maximum-weight: the alternative matching {(L0,Cb),(L1,Ca)} weighs 1/2 > 3/7 and has a blocking pair
    J = {("L0", "Ca"): Fr(3, 7), ("L0", "Cb"): Fr(1, 4), ("L1", "Ca"): Fr(1, 4)}
    edges = list(J)
    best = max((m for k in range(len(edges) + 1) for m in itertools.combinations(edges, k)
                if len({e[0] for e in m}) == len(m) and len({e[1] for e in m}) == len(m)), key=lambda m: sum(J[e] for e in m))
    greedy = (("L0", "Ca"),)
    assert sum(J[e] for e in best) == Fr(1, 2) and sum(J[e] for e in greedy) == Fr(3, 7) and set(best) != set(greedy)
    assert ("L0", "Ca") not in best                      # (L0, Ca) blocks the max-weight matching (both prefer it)


# --------------------------------------------------------------------------------------------------- threshold sensitivity
def _t26_sets():
    return [[R(1, 10)], [U(R(1, 6), R(11, 40)), R(7, 10)]]


def test_T26_theta_ev_half_split_without_carrier():
    tr, rr = run(_t26_sets(), theta_ev=Fr(1, 2))
    p, a, b = (1, 1), (2, 1), (2, 7)
    EA, EB, cid = rel(p, a), rel(p, b), (2, (2, 1))
    expect(rr[1], [
        N(a, 36, 1, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(p, 0, EA)]),
        N(b, 4, 2, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(p, 0, EB)]),
        ED("relation", p, a, 6, 10, 36, "3/20", "3/5", "1/6", "F", False),
        ED("relation", p, b, 4, 10, 4, "2/5", "2/5", "1", "D", False),
        CO(cid, "one_to_many", [0], [a, b], [EA, EB], [], 0, 2, 1, [(0, a, 6, 10, 36), (0, b, 4, 10, 4)]),
        EN(0, 2, "dissolved", [a, b], [1, 2], [EA, EB])])


def test_T26_theta_ev_two_thirds_succession_and_pure_birth():
    tr, rr = run(_t26_sets())
    p, a, b = (1, 1), (2, 1), (2, 7)
    EB, cid = rel(p, b), (2, (2, 7))
    expect(rr[1], [
        N(a, 36, 1),
        N(b, 4, 2, "born", origin="born_succession", comp=cid, role="unselected_member", parents=[(p, 0, EB)]),
        ED("relation", p, b, 4, 10, 4, "2/5", "2/5", "1", "D", False),
        CO(cid, "one_to_one", [0], [b], [EB], [], 0, 1, 1, [(0, b, 4, 10, 4)]),
        EN(0, 2, "succession", [b], [2], [EB])])


def test_T27_k_f_equality_boundary_and_variant_without_edge():
    tr, rr = run([[R(1, 12)], [U(R(1, 8), R(21, 40))]])
    p, a = (1, 1), (2, 1)
    E, cid = rel(p, a), (2, (2, 1))
    expect(rr[1], [
        N(a, 28, 1, "born", origin="born_succession", comp=cid, role="unselected_member", parents=[(p, 0, E)]),
        ED("relation", p, a, 8, 12, 28, "1/4", "2/3", "2/7", "F", False),
        CO(cid, "one_to_one", [0], [a], [E], [], 0, 1, 1, [(0, a, 8, 12, 28)]),
        EN(0, 2, "succession", [a], [1], [E])])
    tr, rr = run([[R(1, 12)], [U(R(1, 7), R(21, 41))]])
    expect(rr[1], [N((2, 1), 28, 1)])


def test_T28_float_free_boundaries():
    over = dict(theta_id=Fr(1, 3), delta=Fr(1, 3))
    tr, rr = run([[R(1, 4)], [R(3, 6)]], **over)                       # J = 2/6 matches
    e = kinds(rr[1], EdgeRecord)[0]
    assert e.J == Fr(1, 3) and e.flags == ("ID",) and e.selected
    tr, rr = run([[R(1, 4)], [R(3, 7)]], **over)                       # J = 2/7 does not
    assert kinds(rr[1], EdgeRecord) == [] and kinds(rr[1], NodeRecord)[0].identity.basis == "born"
    for a, b, want in ((9, 12, "grow"), (9, 11, "stable"), (12, 9, "contract"), (12, 10, "stable")):
        tr, rr = run([[R(1, a)], [R(1, b)]], **over)
        node = kinds(rr[1], NodeRecord)[0]
        assert node.identity.basis == "selected_identity" and node.size_class == want, (a, b, node.size_class)


# --------------------------------------------------------------------------------------------------- tie-key
def test_T30_jaccard_first_picks_B():
    A, B = U(R(1, 6), R(11, 16)), R(7, 10)
    tr, rr = run([[R(1, 10)], [A, B]], theta_id=Fr(1, 3))
    p, a, b = (1, 1), (2, 1), (2, 7)
    EA, EB, cid = rel(p, a), rel(p, b), (2, (2, 1))
    expect(rr[1], [
        N(a, 12, 1, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(p, 0, EA)]),
        N(b, 4, 0, "selected_identity", edge=EB, gap=0, comp=cid, role="selected_member", parents=[(p, 0, EB)], pred=p, sc="contract"),
        ED("relation", p, a, 6, 10, 12, "3/8", "3/5", "1/2", "ID", False),
        ED("relation", p, b, 4, 10, 4, "2/5", "2/5", "1", "ID D", True),
        CO(cid, "one_to_many", [0], [a, b], [EA, EB], [EB], 1, 1, 0, [(0, a, 6, 10, 12), (0, b, 4, 10, 4)])])


def _t30b_seq():
    return [[R(1, 5)], [{1, 2, 11}, {3, 4, 5, 12, 13, 14, 15}]]


def test_T30b_primary_key_uses_overlap_after_jaccard_tie():
    tr, rr = run(_t30b_seq(), theta_id=Fr(1, 3))
    p, c2, c1 = (1, 1), (2, 1), (2, 3)
    E2, E1, cid = rel(p, c2), rel(p, c1), (2, (2, 1))
    expect(rr[1], [
        N(c2, 3, 1, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(p, 0, E2)]),
        N(c1, 7, 0, "selected_identity", edge=E1, gap=0, comp=cid, role="selected_member", parents=[(p, 0, E1)], pred=p, sc="stable"),
        ED("relation", p, c2, 2, 5, 3, "1/3", "2/5", "2/3", "ID D", False),
        ED("relation", p, c1, 3, 5, 7, "1/3", "3/5", "3/7", "ID", True),
        CO(cid, "one_to_many", [0], [c2, c1], [E2, E1], [E1], 1, 1, 0, [(0, c2, 2, 5, 3), (0, c1, 3, 5, 7)])])


def test_T30b_legacy_key_in_compat_mode_picks_lower_rank():
    tr, rr = run(_t30b_seq(), mode="compat", theta_id=Fr(1, 3), g_r=0, g_d=3)
    nodes = {n.node_id: n for n in kinds(rr[1], NodeRecord)}
    assert nodes[(2, 1)].identity.lineage_id == 0 and nodes[(2, 1)].identity.basis == "selected_identity"
    assert nodes[(2, 3)].identity.basis == "born" and nodes[(2, 3)].identity.lineage_id == 1


# --------------------------------------------------------------------------------------------------- dormancy, gaps, declarations
def test_T10_relink_across_one_absent_day_for_g_pairs_2_3_and_1_2():
    for g_r, g_d in ((2, 3), (1, 2)):
        tr, rr = run([[R(1, 10)], [], [R(1, 10)]], g_r=g_r, g_d=g_d)
        assert rr[1] == []
        E = ("relink", (1, 1), (3, 1))
        expect(rr[2], [N((3, 1), 10, 0, "relinked", edge=E, gap=1, pred=(1, 1), sc="stable"),
                       ED("relink", (1, 1), (3, 1), 10, 10, 10, "1", "1", "1", "ID F D", True)])


def test_T10_no_dormancy_when_g_d_is_one():
    tr, rr = run([[R(1, 10)], [], [R(1, 10)]], g_r=0, g_d=1)
    expect(rr[1], [EN(0, 2, "disappeared")])
    expect(rr[2], [N((3, 1), 10, 1)])


def test_T11_boundaries_between_relink_and_declaration():
    # (2,3): present d1, d4 -> relink with gap 2
    tr, rr = run([[R(1, 10)], [], [], [R(1, 10)]])
    expect(rr[3], [N((4, 1), 10, 0, "relinked", edge=("relink", (1, 1), (4, 1)), gap=2, pred=(1, 1), sc="stable"),
                   ED("relink", (1, 1), (4, 1), 10, 10, 10, "1", "1", "1", "ID F D", True)])
    # (2,3): present d1, d5 -> declared at end of d4, d5 is a new lineage
    tr, rr = run([[R(1, 10)], [], [], [], [R(1, 10)]])
    expect(rr[3], [EN(0, 4, "disappeared")])
    expect(rr[4], [N((5, 1), 10, 1)])
    # (1,2): present d1, d4 -> declared at end of d3
    tr, rr = run([[R(1, 10)], [], [], [R(1, 10)]], g_r=1, g_d=2)
    expect(rr[2], [EN(0, 3, "disappeared")])
    expect(rr[3], [N((4, 1), 10, 1)])
    tr, rr = run([[R(1, 10)], [], [], [], [R(1, 10)]], g_r=1, g_d=2)
    expect(rr[2], [EN(0, 3, "disappeared")])
    expect(rr[4], [N((5, 1), 10, 1)])


def test_T14_declaration_timing_and_censoring_progression():
    tr = EvolutionTracker(**P())
    seq = [[R(1, 10)], [R(1, 10)], [], [], []]
    want = {2: ("active", None, None), 3: ("undetermined", None, 5), 4: ("undetermined", None, 5), 5: ("disappeared", 5, None)}
    for day, comms in enumerate(seq, start=1):
        recs = tr.step(day, comms)
        if day == 5:
            expect(recs, [EN(0, 5, "disappeared")])
        elif day >= 3:
            assert recs == []
        if day >= 2:
            st, dd, kn = want[day]
            assert tr.classification() == [ClassificationRow(0, 2, st, dd, kn, None)]


def test_T32_right_censoring_boundary_at_horizon():
    for last, status in ((7, "disappeared"), (8, "undetermined"), (10, "active")):
        tr = EvolutionTracker(**P())
        emitted = []
        for day in range(1, 11):
            emitted += tr.step(day, [R(1, 10)] if day <= last else [])
        row = tr.classification()[0]
        assert row.status == status, (last, row)
        assert (any(isinstance(r, LineageEnd) and r.reason == "disappeared" for r in emitted)) == (status == "disappeared")


# --------------------------------------------------------------------------------------------------- B1: re-link key (errata 1)
def _relink_contest(l1_set, c_set, theta_id):
    """L0 = [1-10] on day 1, L1 = l1_set on day 2 (weakly related to L0, so no relation edge), day 3 empty, day 4 = c_set."""
    tr, rr = run([[R(1, 10)], [l1_set], [], [c_set]], theta_id=theta_id)
    assert rr[1] == [N((2, min(l1_set)), len(l1_set), 1)]                    # L1 is a pure birth, L0 untouched
    return tr, rr


def test_B1_relink_tie_on_jaccard_and_overlap_goes_to_shorter_gap_before_lineage_id():
    tr, rr = _relink_contest(R(10, 19), R(5, 15), Fr(1, 3))                  # J = 6/15 and x = 6 for both
    expect(rr[3], [
        N((4, 5), 11, 1, "relinked", edge=("relink", (2, 10), (4, 5)), gap=1, pred=(2, 10), sc="stable"),
        ED("relink", (2, 10), (4, 5), 6, 10, 11, "2/5", "3/5", "6/11", "ID", True),
        EN(0, 4, "disappeared")])                                            # L0 (gap 2) loses and is declared at L + g_d = 4


def test_B1_relink_jaccard_comes_first_even_against_a_shorter_gap():
    tr, rr = _relink_contest(R(10, 19), R(4, 14), Fr(1, 5))                  # J(L0) = 1/2 (gap 2) vs J(L1) = 5/16 (gap 1)
    expect(rr[3], [
        N((4, 4), 11, 0, "relinked", edge=("relink", (1, 1), (4, 4)), gap=2, pred=(1, 1), sc="stable"),
        ED("relink", (1, 1), (4, 4), 7, 10, 11, "1/2", "7/10", "7/11", "ID F", True)])
    assert tr.classification()[1].status == "undetermined"


def test_B1_relink_overlap_comes_before_gap():
    tr, rr = _relink_contest(R(10, 15), {7, 8, 9, 10, 11, 12}, Fr(1, 3))      # J = 1/3 for both, x = 4 (L0, gap 2) vs 3 (L1, gap 1)
    expect(rr[3], [
        N((4, 7), 6, 0, "relinked", edge=("relink", (1, 1), (4, 7)), gap=2, pred=(1, 1), sc="contract"),
        ED("relink", (1, 1), (4, 7), 4, 10, 6, "1/3", "2/5", "2/3", "ID D", True)])


# --------------------------------------------------------------------------------------------------- R-A, dormancy interplay
def test_T23prime_active_priority_and_R_A_leaves_dormant_lineage_unrelinked():
    A = U(R(1, 9), {21})
    D = U(R(4, 10), {31, 32, 33})
    C = R(1, 10)
    tr, rr = run([[A], [D], [C], [C]])
    assert rr[1] == [N((2, 4), 10, 1)]                             # no flags between A and D: no edge, pure birth
    E, cid = rel((2, 4), (3, 1)), (3, (3, 1))
    expect(rr[2], [
        N((3, 1), 10, 1, "selected_identity", edge=E, gap=0, comp=cid, role="selected_member", parents=[((2, 4), 1, E)], pred=(2, 4), sc="stable"),
        ED("relation", (2, 4), (3, 1), 7, 10, 10, "7/13", "7/10", "7/10", "ID F D", True),
        CO(cid, "one_to_one", [1], [(3, 1)], [E], [E], 1, 0, 0, [(1, (3, 1), 7, 10, 10)])])
    E4 = rel((3, 1), (4, 1))
    expect(rr[3], [
        N((4, 1), 10, 1, "selected_identity", edge=E4, gap=0, comp=(4, (4, 1)), role="selected_member", parents=[((3, 1), 1, E4)], pred=(3, 1), sc="stable"),
        ED("relation", (3, 1), (4, 1), 10, 10, 10, "1", "1", "1", "ID F D", True),
        CO((4, (4, 1)), "one_to_one", [1], [(4, 1)], [E4], [E4], 1, 0, 0, [(1, (4, 1), 10, 10, 10)]),
        EN(0, 4, "disappeared")])


def test_T33prime_R_A_gives_split_child_and_leaves_old_lineage_dormant():
    X = R(6, 10)
    P_ = U(R(1, 5), {8, 9, 10})
    A, B = R(1, 5), {8, 9, 10}
    tr, rr = run([[X], [P_], [A, B]])
    assert rr[1] == [N((2, 1), 8, 1)]
    p, a, b = (2, 1), (3, 1), (3, 8)
    EA, EB, cid = rel(p, a), rel(p, b), (3, (3, 1))
    expect(rr[2], [
        N(a, 5, 1, "selected_identity", edge=EA, gap=0, comp=cid, role="selected_member", parents=[(p, 1, EA)], pred=p, sc="contract"),
        N(b, 3, 2, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(p, 1, EB)]),
        ED("relation", p, a, 5, 8, 5, "5/8", "5/8", "1", "ID D", True),
        ED("relation", p, b, 3, 8, 3, "3/8", "3/8", "1", "D", False),
        CO(cid, "one_to_many", [1], [a, b], [EA, EB], [EA], 1, 1, 0, [(1, a, 5, 8, 5), (1, b, 3, 8, 3)])])
    assert tr.classification()[0].status == "undetermined"        # horizon day 3: lineage 0 stays dormant under R-A
    expect(tr.step(4, []), [EN(0, 4, "disappeared")])             # declared at the end of day L + g_d = 4
    assert tr.classification()[0].status == "disappeared"


# --------------------------------------------------------------------------------------------------- newcomer rule
def _newcomer(fs, size_cards=None):
    tr = EvolutionTracker(**P())
    for d in range(1, 10):
        assert tr.step(d, []) == []
    recs = tr.step(10, [R(1, 8)], fs)
    return kinds(recs, NodeRecord)[0]


def test_T12_T13_newcomer_majority_is_strict():
    assert _newcomer({1: 8, 2: 9, 3: 9, 4: 10, 5: 10, 6: 1, 7: 1, 8: 1}).birth_subtype == "newcomer_dominated"    # 5 of 8
    assert _newcomer({1: 8, 2: 9, 3: 9, 4: 10, 5: 1, 6: 1, 7: 1, 8: 1}).birth_subtype == "regrouped"                # 4 of 8
    assert _newcomer({1: 8, 2: 9, 3: 9, 4: 10, 5: 7, 6: 1, 7: 1, 8: 1}).birth_subtype == "regrouped"                # first_seen = 7 is not new
    assert _newcomer({1: 8, 2: 9, 3: 9, 4: 10, 5: 10, 6: 1, 7: 1}).birth_subtype is None                            # card 8 missing
    assert _newcomer(None).birth_subtype is None
    n = _newcomer({c: 10 for c in range(1, 9)})
    assert n.origin == "pure" and n.identity.basis == "born"


def test_birth_subtype_only_for_pure_origin():
    tr = EvolutionTracker(**P())
    tr.step(1, [R(1, 10)])
    recs = tr.step(2, [R(1, 25)], {c: 2 for c in range(1, 26)})
    n = kinds(recs, NodeRecord)[0]
    assert n.origin == "born_succession" and n.birth_subtype is None


# --------------------------------------------------------------------------------------------------- S1 satellite descriptors
def test_T34_satellite_reported_with_exact_descriptors_no_cutoff():
    K, S = R(1, 97), R(98, 100)
    tr, rr = run([[R(1, 100)], [K, S]])
    p, k, s = (1, 1), (2, 1), (2, 98)
    EK, ES, cid = rel(p, k), rel(p, s), (2, (2, 1))
    expect(rr[1], [
        N(k, 97, 0, "selected_identity", edge=EK, gap=0, comp=cid, role="selected_member", parents=[(p, 0, EK)], pred=p, sc="stable"),
        N(s, 3, 1, "born", origin="born_split_child", comp=cid, role="unselected_member", parents=[(p, 0, ES)]),
        ED("relation", p, k, 97, 100, 97, "97/100", "97/100", "1", "ID F D", True),
        ED("relation", p, s, 3, 100, 3, "3/100", "3/100", "1", "D", False),
        CO(cid, "one_to_many", [0], [k, s], [EK, ES], [EK], 1, 1, 0, [(0, k, 97, 100, 97), (0, s, 3, 100, 3)])])
    assert not hasattr(kinds(rr[1], NodeRecord)[0], "headline_eligible")


# --------------------------------------------------------------------------------------------------- empty days, determinism, validation
def test_T17_empty_days_do_not_error_and_emit_nothing():
    tr = EvolutionTracker(**P())
    for d in (1, 2, 3):
        assert tr.step(d, []) == []
    assert tr.classification() == [] and all(r.N == 0 and r.V_post == 0 for r in tr.ledger)
    assert EvolutionTracker(**P()).classification() == []


def _perm_sequences():
    yield [[R(1, 10)], [R(1, 5), R(6, 10)]], {}
    yield _t15_sets(), dict(theta_ev=Fr(1, 2))
    yield [[R(1, 10), R(11, 20)], [U(R(1, 6), R(11, 14)), U(R(7, 10), R(21, 26))]], dict(theta_id=Fr(1, 5))
    yield [[R(1, 8), R(9, 12), R(13, 15)], [R(1, 15)], [R(1, 8), R(9, 15)]], {}


def test_T18_T19_permutation_and_repeat_determinism():
    rng = random.Random(7)
    for seq, over in _perm_sequences():
        base = run(seq, **over)[1]
        assert run(seq, **over)[1] == base
        for _ in range(5):
            shuffled = [rng.sample(day, len(day)) for day in seq]
            shuffled = [[set(rng.sample(sorted(c), len(c))) for c in day] for day in shuffled]
            assert run(shuffled, **over)[1] == base


def _bad_calls():
    return [
        ("empty set", dict(day=2, communities=[R(1, 3), set()])),
        ("overlap", dict(day=2, communities=[R(1, 5), R(5, 9)])),
        ("first_seen later than day", dict(day=2, communities=[R(1, 5)], first_seen={1: 3})),
        ("first_seen malformed", dict(day=2, communities=[R(1, 5)], first_seen=[(1, 1)])),
        ("first_seen non-integer day", dict(day=2, communities=[R(1, 5)], first_seen={1: 1.5})),
        ("skipped day", dict(day=3, communities=[R(1, 5)])),
        ("repeated day", dict(day=1, communities=[R(1, 5)])),
        ("non-integer day", dict(day=2.0, communities=[R(1, 5)])),
        ("boolean day", dict(day=True, communities=[R(1, 5)])),
        ("communities not a list", dict(day=2, communities={1, 2, 3})),
        ("non-integer card", dict(day=2, communities=[[1, 2, 2.5]])),
        ("string card", dict(day=2, communities=[["a", "b"]])),
        ("community is a string", dict(day=2, communities=["abc"])),
    ]


def test_T29_invalid_inputs_raise_and_leave_state_unchanged():
    for label, kw in _bad_calls():
        tr, twin = EvolutionTracker(**P()), EvolutionTracker(**P())
        tr.step(1, [R(1, 10)])
        twin.step(1, [R(1, 10)])
        try:
            tr.step(**kw)
        except InvalidInputError:
            pass
        else:
            raise AssertionError(f"no error for: {label}")
        # state unchanged: the same valid day can be resubmitted and gives exactly what an untouched twin gives
        good = [R(1, 5), R(6, 10)]
        assert tr.step(2, good, {1: 1}) == twin.step(2, good, {1: 1}), label
        assert tr.classification() == twin.classification() and tr.ledger == twin.ledger


def test_interface_start_day_is_free_and_first_seen_is_per_call():
    for start in (0, 1, 57, -3):
        tr = EvolutionTracker(**P())
        tr.step(start, [R(1, 4)])
        tr.step(start + 1, [R(1, 4)], {1: start - 2})
        try:
            tr.step(start + 3, [R(1, 4)])
        except InvalidInputError:
            pass
        else:
            raise AssertionError("skipped day accepted")
    tr = EvolutionTracker(**P())
    tr.step(1, [R(1, 4)], {99: 1})                                  # entries for cards in no community are allowed
    try:
        tr.step(2, [R(1, 4)], {99: 3})                              # ... but are validated
    except InvalidInputError:
        pass
    else:
        raise AssertionError("first_seen later than day accepted")


def test_numpy_integer_card_ids_are_accepted_and_floats_rejected_in_parameters():
    import numpy as np
    tr = EvolutionTracker(**P())
    recs = tr.step(1, [np.array([1, 2, 3]), [4, 5, 6]])
    assert [r.node_id for r in recs] == [(1, 1), (1, 4)]
    for bad in (dict(theta_id=0.5), dict(delta=0.5), dict(g_d=1.0), dict(g_r=-1), dict(g_d=2, g_r=2), dict(theta_ev=Fr(3, 2)),
                dict(mode="compat", g_r=1)):
        try:
            EvolutionTracker(**P(**bad))
        except ValueError:
            pass
        else:
            raise AssertionError(f"parameters accepted: {bad}")


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
