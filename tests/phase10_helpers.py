"""Shared helpers for the Phase 10 core tests. Expected-value builders are independent of the module's own computations."""
from __future__ import annotations

import pathlib
import sys
from fractions import Fraction as Fr

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evolution import (  # noqa: E402
    ComponentRecord, EdgeRecord, EvolutionTracker, Identity, InvalidInputError, LineageEnd, NodeRecord, RelationParent,
    record_sort_key,
)

# Fixture F (test inputs, NOT research-selected thresholds)
F = dict(mode="event", theta_id=Fr(1, 2), theta_ev=Fr(2, 3), delta=Fr(1, 2), g_r=2, g_d=3, w=3)


def P(**over):
    d = dict(F)
    d.update(over)
    return d


def R(a, b):
    """Inclusive card range [a-b]."""
    return set(range(a, b + 1))


def run(seq, first_seen=None, **over):
    """Step a tracker through ``seq`` (list of per-day community lists, days 1..); returns (tracker, [records per day])."""
    tr = EvolutionTracker(**P(**over))
    out = []
    for i, comms in enumerate(seq, start=1):
        out.append(tr.step(i, comms, None if first_seen is None else first_seen.get(i)))
    return tr, out


def fr(v):
    return Fr(v) if not isinstance(v, Fr) else v


def N(node, size, lid, basis="born", origin="default", edge=None, gap=None, comp=None, role="none", parents=(), pred=None,
      sc=None, sub=None):
    if origin == "default":
        origin = "pure" if basis == "born" else None
    return NodeRecord(node, size, Identity(lid, basis, edge, gap), origin, comp, role,
                      tuple(RelationParent(*p) for p in parents), pred, sc, sub)


def ED(kind, pn, cn, x, a, b, J, kf, kb, flags, sel, day=None):
    return EdgeRecord((kind, pn, cn), kind, x, a, b, fr(J), fr(kf), fr(kb), tuple(flags.split()), sel,
                      cn[0] if day is None else day)


def COV(xab):
    """Independent coverage formulas (errata 1 B3). xab: list of (parent_lid, child_node, x, a, b)."""
    lids = sorted({e[0] for e in xab})
    kids = sorted({e[1] for e in xab})
    a_of = {e[0]: e[3] for e in xab}
    b_of = {e[1]: e[4] for e in xab}
    sx = sum(e[2] for e in xab)
    pc = tuple((l, Fr(sum(e[2] for e in xab if e[0] == l), a_of[l])) for l in lids)
    cc = tuple((k, Fr(sum(e[2] for e in xab if e[1] == k), b_of[k])) for k in kids)
    return pc, cc, Fr(sx, sum(a_of.values())), Fr(sx, sum(b_of.values()))


def CO(cid, shape, lids, nodes, edges, sel, s, born, ends, xab):
    pc, cc, cpc, ccc = COV(xab)
    return ComponentRecord(cid, shape, tuple(lids), tuple(nodes), tuple(sorted(edges)), tuple(sorted(sel)), s, born, 0, ends,
                           born, pc, cc, cpc, ccc)


def EN(lid, day, reason, tn=(), tl=(), via=()):
    return LineageEnd(lid, day, reason, tuple(tn), tuple(tl), tuple(via), day)


def rel(pn, cn):
    return ("relation", pn, cn)


def expect(recs, expected, label=""):
    exp = sorted(expected, key=record_sort_key)
    if recs != exp:
        lines = [f"records differ {label}: got {len(recs)} expected {len(exp)}"]
        for i in range(max(len(recs), len(exp))):
            g = recs[i] if i < len(recs) else None
            e = exp[i] if i < len(exp) else None
            if g != e:
                lines.append(f"  [{i}] got      {g}\n      expected {e}")
        raise AssertionError("\n".join(lines))
    assert recs == sorted(recs, key=record_sort_key)   # I5 ordering


def kinds(recs, cls):
    return [r for r in recs if isinstance(r, cls)]
