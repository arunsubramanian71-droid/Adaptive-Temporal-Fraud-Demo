"""Arm "Tr" of the walk-forward function (drift-triggered refresh) on the small synthetic dataset of tests/test_walkforward_ext.py. Needs PyTorch (skipped and named in a core-only environment)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from src.heldout import walkforward as W  # noqa: E402
from tests.test_walkforward_ext import WINDOWS, build, model_pair, same  # noqa: E402

KW = dict(pos_weight=torch.tensor(3.0), fine_tune_epochs=2, base_seed=7, train_day_min=1, train_day_max=8, burn_in_through=8)


def raises(fn, exc):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def run(arm, trigger=None):
    ds = build()
    m, _ = model_pair()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    torch.manual_seed(11)
    out = W.run_walk_forward_ext(arm, m, ds, WINDOWS, opt=opt, trigger=trigger, **KW)
    return out, m


def test_a_trigger_that_never_fires_equals_the_frozen_arm_and_changes_no_parameter():
    ds = build()
    m1, m2 = model_pair()
    before = [p.detach().clone() for p in m2.parameters()]
    torch.manual_seed(11)
    a = W.run_walk_forward_ext("F", m1, ds, WINDOWS, burn_in_through=8)
    torch.manual_seed(11)
    b = W.run_walk_forward_ext("Tr", m2, ds, WINDOWS, opt=torch.optim.Adam(m2.parameters(), lr=1e-2), trigger=lambda k, s: False, **KW)
    assert same(a, b) and all(torch.equal(x, y) for x, y in zip(before, m2.parameters()))


def test_a_trigger_that_always_fires_equals_the_adaptive_arm():
    ds = build()
    m1, m2 = model_pair()
    torch.manual_seed(11)
    a = W.run_walk_forward_ext("Ad", m1, ds, WINDOWS, opt=torch.optim.Adam(m1.parameters(), lr=1e-2), **KW)
    torch.manual_seed(11)
    b = W.run_walk_forward_ext("Tr", m2, ds, WINDOWS, opt=torch.optim.Adam(m2.parameters(), lr=1e-2), trigger=lambda k, s: True, **KW)
    assert same(a, b)


def test_the_trigger_sees_only_scores_of_the_window_just_scored_and_never_after_the_last_window():
    calls = []

    def trig(k, scores):
        calls.append((k, type(scores).__name__, len(scores)))
        return False
    out, _ = run("Tr", trig)
    assert [c[0] for c in calls] == [1]                                   # two windows: the monitor runs after window 1 only (an update after the last window would be unused)
    assert calls[0][1] == "ndarray" and calls[0][2] == len(out[0][1])


def test_tr_requires_a_trigger_and_no_other_arm_accepts_one():
    ds = build()
    m, _ = model_pair()
    opt = torch.optim.Adam(m.parameters(), lr=1e-2)
    raises(lambda: W.run_walk_forward_ext("Tr", m, ds, WINDOWS, opt=opt, **KW), ValueError)
    raises(lambda: W.run_walk_forward_ext("Ad", m, ds, WINDOWS, opt=opt, trigger=lambda k, s: True, **KW), ValueError)


def test_a_label_trigger_receives_only_the_labels_of_the_window_just_scored_and_equal_to_the_dataset_labels():
    ds = build()
    seen = []

    class Probe:
        needs_labels = True

        def __call__(self, k, scores, labels):
            seen.append((k, np.asarray(scores).copy(), np.asarray(labels).copy()))
            return False
    m, _ = model_pair()
    torch.manual_seed(11)
    out = W.run_walk_forward_ext("Tr", m, ds, WINDOWS, opt=torch.optim.Adam(m.parameters(), lr=1e-2), trigger=Probe(), **KW)
    assert [s[0] for s in seen] == [1]                                       # after window 1 only, never after the last window
    first = WINDOWS[0]
    expected = torch.cat([ds.labels(d) for d in range(first[0], first[1] + 1)]).numpy()
    assert np.array_equal(seen[0][2], expected) and len(seen[0][1]) == len(seen[0][2]) == len(out[0][1])


def test_a_label_free_trigger_is_never_given_labels():
    got = []

    def trig(k, scores):                                                     # a two-argument callable: passing labels would raise TypeError
        got.append(k)
        return False
    run("Tr", trig)
    assert got == [1]


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print("[PASS]", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print("[FAIL]", fn.__name__, repr(e))
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
