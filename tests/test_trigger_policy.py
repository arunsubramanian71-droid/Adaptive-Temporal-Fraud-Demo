"""Drift-triggered refresh policy (src/trigger/policy.py): PSI properties and trigger behaviour. Pure NumPy; no data, no model, no labels."""
from __future__ import annotations

import ast
import inspect
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.trigger import policy as P  # noqa: E402


def raises(fn, exc):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_psi_is_zero_for_identical_samples_and_grows_with_a_shift():
    rng = np.random.default_rng(0)
    a = rng.beta(2, 8, 5000)
    assert P.psi(a, a) == 0.0
    vals = [P.psi(a, np.clip(a + s, 0, 1)) for s in (0.0, 0.02, 0.05, 0.1, 0.2)]
    assert all(x <= y + 1e-12 for x, y in zip(vals, vals[1:])) and vals[-1] > 0.25 and vals[1] < vals[-1]
    raises(lambda: P.psi([], [1.0]), ValueError)


def test_psi_is_small_for_two_samples_of_the_same_distribution():
    rng = np.random.default_rng(1)
    assert P.psi(rng.beta(2, 8, 4000), rng.beta(2, 8, 4000)) < 0.02


def test_trigger_baselines_on_the_first_window_fires_on_a_shift_and_rebaselines_after_a_refresh():
    rng = np.random.default_rng(2)
    base = rng.beta(2, 8, 3000)
    t = P.ScorePSITrigger(threshold=0.2)
    assert t(1, base) is False and t.reference is not None
    assert t(2, rng.beta(2, 8, 3000)) is False                         # same distribution: no trigger
    assert t(3, np.clip(base + 0.2, 0, 1)) is True                    # shifted: fires
    assert t.reference is None                                          # cleared: the refreshed model re-baselines
    assert t(4, np.clip(base + 0.2, 0, 1)) is False                    # first window after the refresh is the new reference, never a trigger
    assert [r["fired"] for r in t.log] == [False, False, True, False] and t.log[0]["psi"] is None and t.log[2]["psi"] > 0.2
    raises(lambda: P.ScorePSITrigger(0.0), ValueError)


def test_trigger_is_label_free_by_construction():
    assert list(inspect.signature(P.ScorePSITrigger.__call__).parameters) == ["self", "k", "scores"]
    tree = ast.parse((ROOT / "src" / "trigger" / "policy.py").read_text(encoding="utf-8"))
    imported = {n.names[0].name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)} | {n.module.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
    assert imported <= {"__future__", "numpy"}, imported                # no data loader, no model, no label source can be reached from the policy module


def _window(rng, quality, n=4000, rate=0.05):
    """Scores whose separation of the positives is ``quality`` (0 = random, larger = better); returns (scores, labels)."""
    y = (rng.random(n) < rate).astype(int)
    return np.clip(rng.normal(0.2, 0.1, n) + quality * y, 0, 1), y


def test_performance_trigger_baselines_fires_on_a_relative_drop_and_rebaselines():
    from src.trigger.performance import PerformanceTrigger
    rng = np.random.default_rng(3)
    t = PerformanceTrigger(delta=0.2)
    assert t.needs_labels is True
    assert t(1, *_window(rng, 0.5)) is False and t.reference is not None            # reference window
    assert t(2, *_window(rng, 0.5)) is False                                         # same quality: no trigger
    assert t(3, *_window(rng, 0.0)) is True and t.reference is None                  # quality collapses: fires, reference cleared
    assert t(4, *_window(rng, 0.0)) is False                                         # first window after the refresh re-baselines
    assert [r["fired"] for r in t.log] == [False, False, True, False] and t.log[2]["relative_drop"] > 0.2 and t.log[0]["relative_drop"] is None


def test_performance_trigger_validates_its_inputs():
    from src.trigger.performance import PerformanceTrigger
    rng = np.random.default_rng(4)
    for bad in (0.0, 1.0, -0.1, 1.5):
        raises(lambda b=bad: PerformanceTrigger(b), ValueError)
    s, y = _window(rng, 0.5)
    raises(lambda: PerformanceTrigger(0.2)(1, s, y[:-1]), ValueError)
    raises(lambda: PerformanceTrigger(0.2)(1, s, np.zeros_like(y)), ValueError)


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
