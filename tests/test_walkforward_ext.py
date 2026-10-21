"""Walk-forward with state burn-in: tests on a small synthetic dataset and an untrained Temporal GraphSAGE (S3 of the Phase 8 supplement pre-registration).
No real data, no checkpoint."""
from __future__ import annotations

import copy
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.adaptive.update_loop import run_walk_forward  # noqa: E402
from src.coordination.lint import lint_files  # noqa: E402
from src.graph.temporal import RELATIONS, DayRel, TemporalDataset  # noqa: E402
from src.heldout import walkforward as W  # noqa: E402
from src.models.temporal_graphsage import TemporalGraphSAGE  # noqa: E402

N_DAYS, PER_DAY = 14, 4
WINDOWS = [(9, 11), (12, 14)]               # burn-in streams days 1..8; "train" days for updates: 1..8, then everything up to each boundary


def build(seed=0):
    rng = np.random.default_rng(seed)
    n = N_DAYS * PER_DAY
    x = torch.tensor(rng.normal(size=(n, 4)), dtype=torch.float32)
    y = torch.tensor((rng.random(n) < 0.3).astype(np.float32))
    ents = {"pseudo_account": 5, "card": 6, "netproxy": 4, "device_fingerprint": 3}
    rel = {}
    for d in range(1, N_DAYS + 1):
        per = {}
        for r, e in RELATIONS.items():
            src = [i for i in range(PER_DAY) if rng.random() < 0.7]
            dst = [int(rng.integers(0, ents[e])) for _ in src]
            ids, inv = torch.unique(torch.tensor(dst, dtype=torch.long), return_inverse=True) if dst else (torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long))
            per[r] = DayRel(torch.tensor(src, dtype=torch.long), inv, ids)
        rel[d] = per
    day = np.repeat(np.arange(1, N_DAYS + 1), PER_DAY)
    ptr = {d: ((d - 1) * PER_DAY, d * PER_DAY) for d in range(1, N_DAYS + 1)}
    return TemporalDataset(x, y, np.arange(n), day, ptr, rel, ents, N_DAYS)


def model_pair():
    torch.manual_seed(1)
    m1 = TemporalGraphSAGE(4)
    m2 = TemporalGraphSAGE(4)
    m2.load_state_dict(copy.deepcopy(m1.state_dict()))
    return m1, m2


def same(a, b):
    return all(np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1]) and np.array_equal(x[2], y[2]) for x, y in zip(a, b))


def test_without_burn_in_it_reproduces_the_phase_8_loop_for_every_arm():
    ds = build()
    for arm in ("F", "Ad", "E"):
        m1, m2 = model_pair()
        kw = {}
        if arm != "F":
            kw = dict(pos_weight=torch.tensor(3.0), fine_tune_epochs=2, base_seed=7, train_day_min=1, train_day_max=8)
        o1 = torch.optim.Adam(m1.parameters(), lr=1e-2) if arm != "F" else None
        o2 = torch.optim.Adam(m2.parameters(), lr=1e-2) if arm != "F" else None
        torch.manual_seed(11)                      # dropout during the update epochs draws from the global generator: seed both runs identically
        a = run_walk_forward(arm, m1, ds, WINDOWS, opt=o1, **kw)
        torch.manual_seed(11)
        b = W.run_walk_forward_ext(arm, m2, ds, WINDOWS, opt=o2, **kw)
        assert same(a, b), arm


def test_burn_in_streams_exactly_the_days_before_the_first_window_and_equals_a_manual_stream():
    ds = build()
    m1, m2 = model_pair()
    res = W.run_walk_forward_ext("F", m1, ds, WINDOWS, burn_in_through=8)
    state = m2.init_state(ds)
    with torch.no_grad():
        m2.eval()
        for d in range(1, 9):
            m2.step(state, ds, d, predict=False)
    from src.adaptive.update_loop import score_window
    manual = [score_window(m2, state, ds, w) for w in WINDOWS]
    assert same(res, manual)
    plain = W.run_walk_forward_ext("F", model_pair()[0], ds, WINDOWS)
    assert not np.array_equal(res[0][1], plain[0][1]), "burn-in must change the first window's scores"
    seen = []
    orig = ds.day_slice

    def spy(d):
        seen.append(d)
        return orig(d)
    ds.day_slice = spy
    try:
        W.burn_in(model_pair()[0], ds, model_pair()[0].init_state(ds), 8)
    finally:
        ds.day_slice = orig
    assert seen == list(range(1, 9))


def test_burn_in_day_must_be_the_day_before_the_first_window_and_arguments_are_validated():
    ds = build()
    m, _ = model_pair()
    for bad in (7, 9, 0):
        try:
            W.run_walk_forward_ext("F", m, ds, WINDOWS, burn_in_through=bad)
        except ValueError as e:
            assert "day before the first window" in str(e)
        else:
            raise AssertionError(bad)
    for fn in (lambda: W.run_walk_forward_ext("X", m, ds, WINDOWS), lambda: W.run_walk_forward_ext("Ad", m, ds, WINDOWS)):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("invalid arguments accepted")


def test_labels_are_requested_only_after_the_window_is_scored_and_never_during_burn_in():
    ds = build()
    events = []
    orig_labels, orig_slice = ds.labels, ds.day_slice
    ds.labels = lambda d: (events.append(("labels", d)), orig_labels(d))[1]
    ds.day_slice = lambda d: (events.append(("slice", d)), orig_slice(d))[1]
    m, _ = model_pair()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    try:
        W.run_walk_forward_ext("Ad", m, ds, WINDOWS, opt=opt, pos_weight=torch.tensor(2.0), fine_tune_epochs=1, burn_in_through=8, train_day_min=1, train_day_max=8)
    finally:
        ds.labels, ds.day_slice = orig_labels, orig_slice
    first_label = next(i for i, e in enumerate(events) if e[0] == "labels")
    burn = [e for e in events[:first_label] if e[0] == "slice"]
    assert [d for _, d in burn[:8]] == list(range(1, 9)), "the burn-in must stream days 1..8 first"
    scored_before_first_update = {d for _, d in events[8:first_label]}
    assert scored_before_first_update >= {9, 10, 11}, "window 1 must be scored before any label is requested"
    assert max(d for k, d in events if k == "labels") <= 11, "labels beyond the last window boundary at an update were requested"
    assert not any(e[0] == "labels" and e[1] > 11 for e in events), "labels of the last window were requested (no update follows it)"


def test_replay_arm_draws_only_train_days_and_matches_the_adaptive_day_count():
    ds = build()
    m, _ = model_pair()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    pools = []
    orig = W.train_epoch_on_day_pool
    W.train_epoch_on_day_pool = lambda model, ds_, opt_, pool, pw, window=7: (pools.append(list(pool)), 0.0)[1]
    try:
        W.run_walk_forward_ext("E", m, ds, WINDOWS, opt=opt, pos_weight=torch.tensor(2.0), fine_tune_epochs=1, base_seed=3, train_day_min=1, train_day_max=8,
                               burn_in_through=8, replay_windows=[(9, 11), (12, 14)])
    finally:
        W.train_epoch_on_day_pool = orig
    assert len(pools) == 1 and all(1 <= d <= 8 for d in pools[0]) and len(pools[0]) == 8 + 3, pools


def test_lint_clean():
    bad = {str(k): v for k, v in lint_files([ROOT / "src/heldout/walkforward.py", ROOT / "tests/test_walkforward_ext.py",
                                              ROOT / "experiments/phase8_state_continuity_supplement/PREREGISTRATION_S8.md"]).items() if v}
    assert bad == {}, bad


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
