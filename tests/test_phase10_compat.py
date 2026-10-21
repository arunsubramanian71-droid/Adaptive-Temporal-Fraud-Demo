"""Phase 10 core: compatibility mode against the frozen Phase 9 ``Tracker`` (imported read-only; Phase 9 is never modified).

T31: seeded random canonical sequences, per-day track ids and the full ``lifecycle_table`` descriptors at several horizons.
T35: characterization of Phase 9 behaviour on inputs the new module rejects. Toy/synthetic data only."""
from __future__ import annotations

import pathlib
import random
import sys
from fractions import Fraction as Fr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.coordination.frozen import G_DISAPPEAR  # noqa: E402  (read-only import of the frozen constant)
from src.coordination.tracking import Tracker, lifecycle_table  # noqa: E402
from tests.phase10_helpers import R, EvolutionTracker, InvalidInputError, NodeRecord, kinds  # noqa: E402
from tests.test_phase10_properties import gen_seq  # noqa: E402


def U(*parts):
    out = set()
    for p in parts:
        out |= set(p)
    return out


def canon(day):
    return sorted((set(c) for c in day), key=min)


def new_compat(theta, g_d=G_DISAPPEAR):
    return EvolutionTracker(mode="compat", theta_id=theta, theta_ev=Fr(2, 3), delta=Fr(1, 2), g_r=0, g_d=g_d, w=3)


def step_both(old, new, t, day):
    lists = canon(day)
    out_old = old.step(t, [set(c) for c in lists])
    recs = new.step(t, [set(c) for c in lists])
    out_new = [n.identity.lineage_id for n in kinds(recs, NodeRecord)]
    return out_old, out_new


def compare_lifecycle(old, new, T):
    """Phase 9 lifecycle_table rows must be reproduced field by field; the three-way status is a derived view (errata 2, E2-C)."""
    rows_old = lifecycle_table(old, T)
    assert new.legacy_lifecycle() == rows_old, (T, rows_old, new.legacy_lifecycle())
    cls = {c.lineage_id: c for c in new.classification()}
    assert set(cls) == {r["track"] for r in rows_old}
    for r in rows_old:
        c = cls[r["track"]]
        assert (c.ended, c.disappearance_known_at) == (r["ended"], r["disappearance_known_at"])      # legacy fields unchanged
        if not r["ended"]:
            want = "active"
        elif r["disappearance_known_at"] > T:
            want = "undetermined"                                        # ended in the legacy sense but not yet declarable
        else:
            want = "disappeared"
        assert c.status == want, (r, c)
        assert (c.status == "disappeared") == (r["ended"] and r["last_day"] <= T - G_DISAPPEAR)    # Tier 0 'observable disappearance'


def run_pair(seq, theta, extra_empty=4, g_d=G_DISAPPEAR):
    old, new = Tracker(jaccard=float(theta)), new_compat(theta, g_d)
    t = 0
    for t, day in enumerate(seq, start=1):
        a, b = step_both(old, new, t, day)
        assert a == b, (t, a, b)
    compare_lifecycle(old, new, t)
    for k in range(1, extra_empty + 1):                                   # horizons beyond the data = stepping empty days on both sides
        a, b = step_both(old, new, t + k, [])
        assert a == b == []
        compare_lifecycle(old, new, t + k)
    return old, new


# --------------------------------------------------------------------------------------------------- fixed sequences (hand-derived legacy values)
A_ = U(R(1, 9), {21})
D_ = U(R(4, 10), {31, 32, 33})
FIXED = {
    # name: (sequence, theta, expected tids per day, expected parents {tid: parent}, expected flags {tid: [...]})
    "T5": ([[R(1, 10)], [R(1, 5), R(6, 10)]], Fr(1, 2), [[0], [0, 1]], {1: 0}, {0: [("split", 2)]}),
    "T6": ([[R(1, 10)], [R(1, 7), R(8, 10)]], Fr(1, 2), [[0], [0, 1]], {}, {}),
    "T8": ([[R(1, 7), R(8, 10)], [R(1, 10)]], Fr(1, 2), [[0, 1], [0]], {}, {}),
    "T20": ([[R(1, 10)], [R(1, 5), R(6, 10)], [R(1, 10)]], Fr(1, 2), [[0], [0, 1], [0]], {1: 0}, {0: [("split", 2)], 1: [("merged", 3)]}),
    "T21": ([[R(1, 6), R(7, 12)], [U(R(1, 4), R(7, 10))]], Fr(1, 3), [[0, 1], [0]], {}, {1: [("merged", 2)]}),
    "T24": ([[R(1, 40), R(41, 80)], [R(1, 80)]], Fr(1, 2), [[0, 1], [0]], {}, {1: [("merged", 2)]}),
    "T22": ([[R(1, 10), R(11, 20)], [U(R(1, 6), R(11, 14)), U(R(7, 10), R(21, 26))]], Fr(1, 5), [[0, 1], [0, 2]], {2: 0},
            {0: [("split", 2)], 1: [("merged", 2)]}),
    "T30": ([[R(1, 10)], [U(R(1, 6), R(11, 16)), R(7, 10)]], Fr(1, 3), [[0], [1, 0]], {1: 0}, {0: [("split", 2)]}),
    "T30b": ([[R(1, 5)], [{1, 2, 11}, {3, 4, 5, 12, 13, 14, 15}]], Fr(1, 3), [[0], [0, 1]], {1: 0}, {0: [("split", 2)]}),
    "T23prime": ([[A_], [D_], [R(1, 10)]], Fr(1, 2), [[0], [1], [1]], {}, {}),
    "T33prime": ([[R(6, 10)], [U(R(1, 5), {8, 9, 10})], [R(1, 5), {8, 9, 10}]], Fr(1, 2), [[0], [1], [1, 2]], {}, {}),
    # one unselected lineage with TWO losing pairs gets two legacy "merged" flags (the flag is per pair, not per lineage)
    "merged_flag_per_pair": ([[R(1, 8), R(9, 12), R(13, 16)], [U(R(1, 4), R(9, 12)), U(R(5, 8), R(13, 16))]], Fr(1, 3),
                             [[0, 1, 2], [1, 2]], {}, {0: [("merged", 2), ("merged", 2)]}),
    "T3_compat_row": ([[R(1, 10)], [R(1, 25)]], Fr(1, 2), [[0], [1]], {}, {}),
    "T15": ([[R(1, 10), R(11, 20)], [U(R(1, 5), R(11, 15)), U(R(6, 10), R(16, 20))]], Fr(1, 2), [[0, 1], [2, 3]], {}, {}),
}


def test_fixed_sequences_phase9_characterization_and_compat_equivalence():
    for name, (seq, theta, tids, parents, flags) in FIXED.items():
        old, new = run_pair(seq, theta)
        # (a) hand-derived expectation versus the frozen Phase 9 tracker (read-only characterization)
        rows = {r["track"]: r for r in lifecycle_table(old, len(seq))}
        for tid, p in parents.items():
            assert rows[tid]["parent"] == p, (name, tid)
        for tid, r in rows.items():
            assert r["parent"] == parents.get(tid), (name, tid, r["parent"])
            assert r["flags"] == flags.get(tid, []), (name, tid, r["flags"])
        # (b) the tids per day from a fresh pair of trackers
        o2, n2 = Tracker(jaccard=float(theta)), new_compat(theta)
        got = []
        for t, day in enumerate(seq, start=1):
            a, b = step_both(o2, n2, t, day)
            assert a == b
            got.append(a)
        assert got == tids, (name, got)


def test_equality_boundaries_are_exact_in_compat_mode():
    # J exactly equal to the threshold: inclusive in both implementations (float division is correctly rounded)
    for theta, seq in ((Fr(1, 2), [[R(1, 10)], [R(1, 5), R(6, 10)]]), (Fr(1, 3), [[R(1, 6), R(7, 12)], [U(R(1, 4), R(7, 10))]]),
                       (Fr(1, 3), [[R(1, 4)], [R(3, 6)]]), (Fr(2, 3), [[R(1, 15)], [R(1, 10)]])):
        run_pair(seq, theta)


# --------------------------------------------------------------------------------------------------- T31: seeded random equivalence
def test_T31_random_canonical_sequences_match_phase9_ids_flags_and_lifecycle():
    n = 0
    flagged = 0
    for theta in (Fr(1, 2), Fr(1, 3)):
        for seed in range(250):
            rng = random.Random(9000 + seed)
            seq = gen_seq(rng, rng.randint(6, 24), universe=rng.choice((20, 30, 40)))
            old, new = run_pair(seq, theta)
            n += 1
            flagged += sum(1 for r in lifecycle_table(old, len(seq)) if r["flags"])
    assert n == 500
    assert flagged > 100, flagged            # the generator must actually produce splits and merges, otherwise the test proves little


def test_T31_other_declaration_delays_are_the_g_d_parameter():
    # Phase 9 hard-codes known_at = last_day + 3; the module's compat g_d reproduces it at 3 and is otherwise a pure parameter
    rng = random.Random(3)
    seq = gen_seq(rng, 14)
    new = new_compat(Fr(1, 2), g_d=2)
    old = Tracker(jaccard=0.5)
    for t, day in enumerate(seq, start=1):
        step_both(old, new, t, day)
    for r, mine in zip(lifecycle_table(old, len(seq)), new.legacy_lifecycle()):
        assert mine["disappearance_known_at"] == (None if not r["ended"] else r["last_day"] + 2)


# --------------------------------------------------------------------------------------------------- T35: inputs Phase 9 accepts and the new module rejects
def test_T35_phase9_accepts_inputs_the_new_module_rejects():
    # Phase 9 performs no validation (characterization of the frozen behaviour, run read-only):
    tr = Tracker()
    assert tr.step(1, [R(1, 5)]) == [0]
    assert tr.step(3, [R(1, 5)]) == [1]                       # skipped day: previous list silently ignored, new track
    assert Tracker().step(1, [set()]) == [0]                  # empty set becomes a size-0 track
    t2 = Tracker()
    assert t2.step(1, [{1, 2, 3}, {3, 4, 5}]) == [0, 1]       # overlapping sets accepted
    t3 = Tracker()
    assert t3.step(1, [{5, 6}, {1, 2}]) == [0, 1]             # list order decides ids (not canonical)
    mine = {n.node_id: n.identity.lineage_id for n in kinds(new_compat(Fr(1, 2)).step(1, [{5, 6}, {1, 2}]), NodeRecord)}
    assert mine == {(1, 1): 0, (1, 5): 1}                       # new module: canonical order, so {1, 2} is lineage 0 (Phase 9: {5, 6} is 0)
    # the new module rejects each of them
    for t_first, bad_day, comms in ((1, 3, [R(1, 5)]), (1, 2, [set()]), (1, 2, [{1, 2, 3}, {3, 4, 5}])):
        n = new_compat(Fr(1, 2))
        n.step(t_first, [R(1, 5)])
        try:
            n.step(bad_day, comms)
        except InvalidInputError:
            pass
        else:
            raise AssertionError(("accepted", bad_day, comms))


def test_phase9_frozen_constants_used_by_compat_are_as_documented():
    from src.coordination import frozen
    assert frozen.G_DISAPPEAR == 3 and frozen.TRACK_JACCARD == 0.5


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
