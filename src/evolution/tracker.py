"""Phase 10 core: candidate-community evolution tracker (event typing mode and Phase 9 compatibility mode).

Governing specification (all in experiments/phase10_community_evolution/):
DESIGN_ADDENDUM_D12_approved_decisions.md, DESIGN_ADDENDUM_ERRATA_1_B1_B5.md, DESIGN_ADDENDUM_ERRATA_2_targets_interface_censoring.md.

Exact rational arithmetic only (fractions.Fraction, integer cross-multiplication): there is no floating point anywhere in this module.
The objects are candidate communities. Nothing produced here is evidence of fraud or coordination.

Input contract (errata 2, E2-B): ``step(day, communities, first_seen=None)``; days consecutive, empty list = observed day without communities,
violations raise ``InvalidInputError`` before any state change.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from numbers import Integral
from typing import ClassVar

MODES = ("event", "compat")


class InvalidInputError(ValueError):
    """Raised for any violation of the input contract; the tracker state is left unchanged."""


# ------------------------------------------------------------------------------------------------ records
@dataclass(frozen=True)
class Identity:
    lineage_id: int
    basis: str                      # selected_identity | relinked | born  (selected_extended is reserved, never emitted)
    edge_id: tuple | None
    gap: int | None


@dataclass(frozen=True)
class RelationParent:
    parent_node: tuple
    parent_lineage_before: int
    edge_id: tuple


@dataclass(frozen=True)
class NodeRecord:
    type_rank: ClassVar[int] = 0
    node_id: tuple
    size: int
    identity: Identity
    origin: str | None
    component_ref: tuple | None
    component_role: str             # selected_member | unselected_member | none
    relation_parents: tuple
    identity_predecessor: tuple | None
    size_class: str | None
    birth_subtype: str | None

    @property
    def day(self) -> int:
        return self.node_id[0]

    @property
    def primary_id(self):
        return self.node_id


@dataclass(frozen=True)
class EdgeRecord:
    type_rank: ClassVar[int] = 1
    edge_id: tuple                  # (kind, parent_node, child_node)
    kind: str                       # relation | relink
    x: int
    a: int
    b: int
    J: Fraction
    k_f: Fraction
    k_b: Fraction
    flags: tuple                    # subset of ("ID", "F", "D")
    selected: bool
    day: int

    @property
    def primary_id(self):
        return self.edge_id


@dataclass(frozen=True)
class ComponentRecord:
    type_rank: ClassVar[int] = 2
    component_id: tuple             # (day, smallest child node id)
    shape: str                      # one_to_one | one_to_many | many_to_one | many_to_many
    lineages: tuple
    nodes: tuple
    edge_ids: tuple
    selected_edge_ids: tuple
    s: int
    born_count: int
    relinked_count: int
    ends: int
    unselected_count: int
    parent_coverage: tuple          # ((lineage_id, Fraction), ...)
    child_coverage: tuple           # ((node_id, Fraction), ...)
    component_parent_coverage: Fraction
    component_child_coverage: Fraction

    @property
    def day(self) -> int:
        return self.component_id[0]

    @property
    def primary_id(self):
        return self.component_id


@dataclass(frozen=True)
class LineageEnd:
    type_rank: ClassVar[int] = 3
    lineage_id: int
    day: int
    reason: str                     # succession | dissolved | merged | reorganized | disappeared
    target_nodes: tuple
    target_lineages: tuple
    via_edges: tuple
    available_at: int

    @property
    def primary_id(self):
        return self.lineage_id


@dataclass(frozen=True)
class ClassificationRow:
    lineage_id: int
    last_day: int
    status: str                     # active | undetermined | disappeared | ended_by_event
    declared_day: int | None
    known_at_if_unseen: int | None
    end_reason: str | None
    ended: bool | None = None                       # compat mode only (legacy Phase 9 semantics)
    disappearance_known_at: int | None = None       # compat mode only (legacy Phase 9 semantics)


@dataclass(frozen=True)
class LedgerRow:
    day: int
    V_pre: int
    A: int
    D_r: int
    D_c: int
    s: int
    e: int
    z: int
    r: int
    b: int
    N: int
    d: int
    d_from_z: int
    d_from_Dr: int
    d_from_Dc: int
    V_post: int
    E: int
    X: int
    Lam: int


def record_sort_key(rec):
    """Invariant I5 ordering: (day, record type, primary id)."""
    return (rec.day, rec.type_rank, rec.primary_id)


@dataclass
class _Lineage:
    lid: int
    first_day: int
    last_day: int
    last_node: tuple
    last_set: frozenset
    last_size: int
    status: str = "live"            # live | ended | declared
    end_reason: str | None = None
    parent: int | None = None       # legacy parent pointer (compat mode only)
    flags: list = field(default_factory=list)
    days: list = field(default_factory=list)


_SHAPE_BORN = {"one_to_one": "born_succession", "one_to_many": "born_split_child",
               "many_to_one": "born_merge_product", "many_to_many": "born_reorganization_child"}
_SHAPE_END = {"one_to_one": "succession", "one_to_many": "dissolved",
              "many_to_one": "merged", "many_to_many": "reorganized"}


def _is_int(v) -> bool:
    return isinstance(v, Integral) and not isinstance(v, bool)


def _overlaps(lins, sets):
    """x = |last_set(lineage) & set| for every (lineage id, index) with x > 0."""
    inv: dict = {}
    for l in lins:
        for c in l.last_set:
            inv.setdefault(c, []).append(l.lid)
    out: dict = {}
    for i, s in enumerate(sets):
        cnt: dict = {}
        for c in s:
            for lid in inv.get(c, ()):
                cnt[lid] = cnt.get(lid, 0) + 1
        for lid, x in cnt.items():
            out[(lid, i)] = x
    return out


class EvolutionTracker:
    """Incremental tracker. Parameters have no defaults and are fixed at construction."""

    def __init__(self, mode, theta_id, theta_ev, delta, g_r, g_d, w):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.theta_id = self._frac("theta_id", theta_id)
        self.theta_ev = self._frac("theta_ev", theta_ev)
        self.delta = self._frac("delta", delta)
        if not (0 < self.theta_id <= 1 and 0 < self.theta_ev <= 1):
            raise ValueError("theta_id and theta_ev must lie in (0, 1]")
        if not self.delta > 0:
            raise ValueError("delta must be positive")
        for name, v in (("g_r", g_r), ("g_d", g_d), ("w", w)):
            if not _is_int(v):
                raise ValueError(f"{name} must be an integer")
        if g_r < 0 or g_d < g_r + 1 or w < 1:
            raise ValueError("require g_r >= 0, g_d >= g_r + 1, w >= 1")
        if mode == "compat" and g_r != 0:
            raise ValueError("compatibility mode has no re-link: g_r must be 0")
        self.g_r, self.g_d, self.w = int(g_r), int(g_d), int(w)
        self._lins: list[_Lineage] = []
        self._next = 0
        self._first_day: int | None = None
        self._last_day: int | None = None
        self._ledger: list[LedgerRow] = []

    @staticmethod
    def _frac(name, v) -> Fraction:
        if isinstance(v, Fraction):
            return v
        if _is_int(v):
            return Fraction(int(v))
        raise ValueError(f"{name} must be a Fraction or an integer (floating point is not accepted)")

    # ------------------------------------------------------------------------------------------ validation
    def _validate(self, day, communities, first_seen):
        if not _is_int(day):
            raise InvalidInputError("day must be an integer")
        t = int(day)
        if self._last_day is not None and t != self._last_day + 1:
            raise InvalidInputError(f"day must equal the previous accepted day + 1 ({self._last_day + 1}), got {t}")
        if not isinstance(communities, (list, tuple)):
            raise InvalidInputError("communities must be a list or tuple")
        sets = []
        for comm in communities:
            if isinstance(comm, (str, bytes, Mapping)) or not hasattr(comm, "__iter__"):
                raise InvalidInputError("each community must be a collection of integer card ids")
            cards = []
            for c in comm:
                if not _is_int(c):
                    raise InvalidInputError("card ids must be integers")
                cards.append(int(c))
            fs = frozenset(cards)
            if not fs:
                raise InvalidInputError("empty community")
            sets.append(fs)
        if sets and sum(len(s) for s in sets) != len(frozenset().union(*sets)):
            raise InvalidInputError("communities of one day must be pairwise disjoint")
        fsn = None
        if first_seen is not None:
            if not isinstance(first_seen, Mapping):
                raise InvalidInputError("first_seen must be a mapping from card id to day")
            fsn = {}
            for k, v in first_seen.items():
                if not _is_int(k) or not _is_int(v):
                    raise InvalidInputError("first_seen keys and values must be integers")
                if int(v) > t:
                    raise InvalidInputError("first_seen contains a day later than the current day")
                fsn[int(k)] = int(v)
        sets.sort(key=min)
        return t, sets, fsn

    # ------------------------------------------------------------------------------------------ arithmetic
    def _size_class(self, a: int, b: int) -> str:
        p, q = self.delta.numerator, self.delta.denominator
        if b * q >= a * (q + p):
            return "grow"
        if b * (q + p) <= a * q:
            return "contract"
        return "stable"

    # ------------------------------------------------------------------------------------------ step
    def step(self, day, communities, first_seen=None):
        t, sets, fsn = self._validate(day, communities, first_seen)
        event = self.mode == "event"
        n = len(sets)
        sizes = [len(s) for s in sets]
        nodes = [(t, min(s)) for s in sets]
        live = [l for l in self._lins if l.status == "live"]
        by_lid = {l.lid: l for l in live}
        active = [l for l in live if l.last_day == t - 1]
        dormant = [l for l in live if 2 <= t - l.last_day <= self.g_r + 1]
        closed = [l for l in live if t - l.last_day > self.g_r + 1]

        # ---- stage 1: pairs with active lineages, identity selection
        pairs: dict = {}
        for (lid, r), x in _overlaps(active, sets).items():
            a, b = by_lid[lid].last_size, sizes[r]
            J, kf, kb = Fraction(x, a + b - x), Fraction(x, a), Fraction(x, b)
            pairs[(lid, r)] = (x, a, b, J, kf, kb, J >= self.theta_id, kf >= self.theta_ev, kb >= self.theta_ev)
        id_edges = [k for k, v in pairs.items() if v[6]]
        if event:
            def key(k):
                return (-pairs[k][3], -pairs[k][0], k[0], k[1])      # Jaccard, overlap, lineage id, canonical rank
        else:
            def key(k):
                return (-pairs[k][3], k[0], k[1])                    # legacy Phase 9 key
        sel: dict = {}      # community rank -> lineage id
        sel_l: dict = {}    # lineage id -> community rank
        for k in sorted(id_edges, key=key):
            if k[0] in sel_l or k[1] in sel:
                continue
            sel[k[1]] = k[0]
            sel_l[k[0]] = k[1]
        q_edges = [k for k, v in pairs.items() if v[6] or v[7] or v[8]] if event else []

        # ---- components of the relation graph Q (only components with at least one relation edge exist)
        uf: dict = {}

        def find(u):
            while uf.setdefault(u, u) != u:
                uf[u] = uf[uf[u]]
                u = uf[u]
            return u

        for lid, r in q_edges:
            uf[find(("L", lid))] = find(("C", r))
        groups: dict = {}
        for e in q_edges:
            groups.setdefault(find(("L", e[0])), []).append(e)
        comp_of_rank, comp_of_lid, born_origin, end_plan, comp_info = {}, {}, {}, {}, []
        for g in groups.values():
            lids = sorted({e[0] for e in g})
            ranks = sorted({e[1] for e in g})
            shape = ("one_to_one" if len(lids) == 1 and len(ranks) == 1 else "one_to_many" if len(lids) == 1
                     else "many_to_one" if len(ranks) == 1 else "many_to_many")
            cid = (t, nodes[ranks[0]])
            for r in ranks:
                comp_of_rank[r] = cid
                if r not in sel:
                    born_origin[r] = _SHAPE_BORN[shape]
            for lid in lids:
                comp_of_lid[lid] = cid
                if lid not in sel_l:
                    end_plan[lid] = _SHAPE_END[shape]
            comp_info.append((cid, shape, lids, ranks, sorted(g)))

        # ---- stage 2: re-link of dormant lineages (R-A: only to communities outside every component)
        relink: dict = {}   # community rank -> lineage id
        relink_pair: dict = {}
        if event and dormant:
            elig = [r for r in range(n) if r not in comp_of_rank and r not in sel]
            cand = {}
            for (lid, i), x in _overlaps(dormant, [sets[r] for r in elig]).items():
                r = elig[i]
                l = by_lid[lid]
                a, b = l.last_size, sizes[r]
                J = Fraction(x, a + b - x)
                if J >= self.theta_id:
                    cand[(lid, r)] = (x, a, b, J, Fraction(x, a), Fraction(x, b), t - l.last_day - 1)
            used_l = set()
            for k in sorted(cand, key=lambda k: (-cand[k][3], -cand[k][0], cand[k][6], k[0], k[1])):
                if k[0] in used_l or k[1] in relink:
                    continue
                relink[k[1]] = k[0]
                used_l.add(k[0])
                relink_pair[k[1]] = cand[k]

        # ---- births (numbered after re-links, canonical order) and lineage lookup
        lid_of_rank = dict(sel)
        lid_of_rank.update(relink)
        nxt = self._next
        birth_ranks = []
        for r in range(n):
            if r not in lid_of_rank:
                lid_of_rank[r] = nxt
                nxt += 1
                birth_ranks.append(r)

        # ---- legacy flags / parent pointers (compat mode: exact Phase 9 semantics)
        flag_appends, legacy_parent = [], {}
        if not event:
            ordered = sorted(id_edges, key=key)
            first_of_rank: dict = {}
            for lid, r in ordered:
                first_of_rank.setdefault(r, lid)
            for r in range(n):
                if r not in sel and r in first_of_rank:
                    legacy_parent[r] = first_of_rank[r]
                    flag_appends.append((first_of_rank[r], ("split", t)))
            for lid, r in ordered:
                if lid not in sel_l:
                    flag_appends.append((lid, ("merged", t)))

        # ---- records
        recs: list = []
        edge_id = {e: ("relation", by_lid[e[0]].last_node, nodes[e[1]]) for e in q_edges}
        for r in range(n):
            lid = lid_of_rank[r]
            cref = comp_of_rank.get(r)
            parents = tuple(RelationParent(by_lid[e[0]].last_node, e[0], edge_id[e])
                            for e in sorted((e for e in q_edges if e[1] == r), key=lambda e: edge_id[e]))
            if r in sel:
                l = by_lid[lid]
                ident = Identity(lid, "selected_identity", edge_id[(lid, r)] if event else None, 0)
                rec = NodeRecord(nodes[r], sizes[r], ident, None, cref, "selected_member" if cref else "none", parents,
                                 l.last_node, self._size_class(l.last_size, sizes[r]), None)
            elif r in relink:
                l = by_lid[lid]
                ident = Identity(lid, "relinked", ("relink", l.last_node, nodes[r]), t - l.last_day - 1)
                rec = NodeRecord(nodes[r], sizes[r], ident, None, None, "none", (), l.last_node,
                                 self._size_class(l.last_size, sizes[r]), None)
            else:
                origin = born_origin.get(r, "pure")
                sub = None
                if origin == "pure" and fsn is not None and all(c in fsn for c in sets[r]):
                    n_new = sum(1 for c in sets[r] if t - self.w < fsn[c] <= t)
                    sub = "newcomer_dominated" if 2 * n_new > sizes[r] else "regrouped"
                rec = NodeRecord(nodes[r], sizes[r], Identity(lid, "born", None, None), origin, cref,
                                 "unselected_member" if cref else "none", parents, None, None, sub)
            recs.append(rec)
        for e in q_edges:
            x, a, b, J, kf, kb, fi, ff, fd = pairs[e]
            recs.append(EdgeRecord(edge_id[e], "relation", x, a, b, J, kf, kb,
                                   tuple(f for f, on in (("ID", fi), ("F", ff), ("D", fd)) if on),
                                   sel.get(e[1]) == e[0], t))
        for r, (x, a, b, J, kf, kb, gap) in relink_pair.items():
            recs.append(EdgeRecord(("relink", by_lid[relink[r]].last_node, nodes[r]), "relink", x, a, b, J, kf, kb,
                                   tuple(f for f, on in (("ID", True), ("F", kf >= self.theta_ev), ("D", kb >= self.theta_ev)) if on),
                                   True, t))
        for cid, shape, lids, ranks, g in comp_info:
            sx = sum(pairs[e][0] for e in g)
            sa = sum(by_lid[l].last_size for l in lids)
            sb = sum(sizes[r] for r in ranks)
            born = sum(1 for r in ranks if r not in sel)
            sel_ids = tuple(sorted(edge_id[e] for e in g if sel.get(e[1]) == e[0]))
            pc = tuple((l, Fraction(sum(pairs[e][0] for e in g if e[0] == l), by_lid[l].last_size)) for l in lids)
            cc = tuple((nodes[r], Fraction(sum(pairs[e][0] for e in g if e[1] == r), sizes[r])) for r in ranks)
            recs.append(ComponentRecord(cid, shape, tuple(lids), tuple(nodes[r] for r in ranks),
                                        tuple(sorted(edge_id[e] for e in g)), sel_ids, len(sel_ids), born, 0,
                                        sum(1 for l in lids if l not in sel_l), born + 0, pc, cc,
                                        Fraction(sx, sa), Fraction(sx, sb)))
        for lid, reason in end_plan.items():
            own = sorted((e for e in q_edges if e[0] == lid), key=lambda e: edge_id[e])
            recs.append(LineageEnd(lid, t, reason, tuple(nodes[e[1]] for e in own), tuple(lid_of_rank[e[1]] for e in own),
                                   tuple(edge_id[e] for e in own), t))
        relinked_l = set(relink.values())
        declared = [l for l in live if l.lid not in end_plan and l.lid not in sel_l and l.lid not in relinked_l
                    and t - l.last_day == self.g_d]
        for l in declared:
            recs.append(LineageEnd(l.lid, t, "disappeared", (), (), (), t))
        recs.sort(key=record_sort_key)

        # ---- ledger (counts derived from the plan, V_post counted directly from lineage states below)
        z = sum(1 for l in active if l.lid not in comp_of_lid and l.lid not in sel_l)

        def cls(l):
            u = t - l.last_day
            return "z" if u == 1 else "Dr" if u <= self.g_r + 1 else "Dc"
        dc = [cls(l) for l in declared]

        # ---- commit (no state was modified before this point)
        for lid, rr in sel_l.items():
            l = by_lid[lid]
            l.last_day, l.last_node, l.last_set, l.last_size = t, nodes[rr], sets[rr], sizes[rr]
            l.days.append((t, sizes[rr]))
        for rr, lid in relink.items():
            l = by_lid[lid]
            l.last_day, l.last_node, l.last_set, l.last_size = t, nodes[rr], sets[rr], sizes[rr]
            l.days.append((t, sizes[rr]))
        for lid, reason in end_plan.items():
            by_lid[lid].status, by_lid[lid].end_reason = "ended", reason
        for l in declared:
            l.status, l.end_reason = "declared", "disappeared"
        for lid, flag in flag_appends:
            by_lid[lid].flags.append(flag)
        for r in birth_ranks:
            self._lins.append(_Lineage(lid_of_rank[r], t, t, nodes[r], sets[r], sizes[r], parent=legacy_parent.get(r),
                                       days=[(t, sizes[r])]))
        self._next = nxt
        if self._first_day is None:
            self._first_day = t
        self._last_day = t
        v_post = sum(1 for l in self._lins if l.status == "live")
        self._ledger.append(LedgerRow(
            day=t, V_pre=len(live), A=len(active), D_r=len(dormant), D_c=len(closed), s=len(sel), e=len(end_plan), z=z,
            r=len(relink), b=len(birth_ranks), N=n, d=len(declared), d_from_z=dc.count("z"), d_from_Dr=dc.count("Dr"),
            d_from_Dc=dc.count("Dc"), V_post=v_post, E=sum(1 for l in self._lins if l.status == "ended"),
            X=sum(1 for l in self._lins if l.status == "declared"), Lam=len(self._lins)))
        return recs

    # ------------------------------------------------------------------------------------------ queries
    @property
    def ledger(self):
        return list(self._ledger)

    def classification(self):
        """Read-only status of every lineage at the horizon T = last accepted day (errata 2, E2-C)."""
        T = self._last_day
        if T is None:
            return []
        rows = []
        for l in sorted(self._lins, key=lambda l: l.lid):
            if l.status == "ended":
                st, dd, kn, er = "ended_by_event", None, None, l.end_reason
            elif l.status == "declared":
                st, dd, kn, er = "disappeared", l.last_day + self.g_d, None, None
            elif l.last_day == T:
                st, dd, kn, er = "active", None, None, None
            else:
                st, dd, kn, er = "undetermined", None, l.last_day + self.g_d, None
            ended = disc = None
            if self.mode == "compat":
                ended = l.last_day < T
                disc = l.last_day + self.g_d if ended else None
            rows.append(ClassificationRow(l.lid, l.last_day, st, dd, kn, er, ended, disc))
        return rows

    def legacy_lifecycle(self):
        """Phase 9 ``lifecycle_table`` fields reproduced in compatibility mode (ended = last_day < T, no censoring adjustment)."""
        if self.mode != "compat":
            raise InvalidInputError("legacy lifecycle descriptors exist only in compatibility mode")
        T = self._last_day
        rows = []
        for l in sorted(self._lins, key=lambda l: l.lid):
            ended = l.last_day < T
            rows.append({"track": l.lid, "first_day": l.first_day, "last_day": l.last_day, "age_days": l.last_day - l.first_day + 1,
                         "size_first": l.days[0][1], "size_last": l.days[-1][1], "size_max": max(s for _, s in l.days),
                         "ended": bool(ended), "disappearance_known_at": (l.last_day + self.g_d) if ended else None,
                         "parent": l.parent, "flags": list(l.flags)})
        return rows
