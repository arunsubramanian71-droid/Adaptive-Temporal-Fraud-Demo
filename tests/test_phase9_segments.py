"""Phase 9 unit tests: array-level segmentation (toy/random cases only; the real-data equivalence is hard gate G0)."""
from __future__ import annotations

import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.coordination.g0 import toy_cases
from src.coordination.segments_array import COLD, NOEDGE, WARM, segments_from_arrays


def test_toy_cases_all_pass():
    bad = [(n, d) for n, ok, d in toy_cases() if not ok]
    assert not bad, bad


def test_matches_pandas_reference_on_random_arrays():
    """Independent re-implementation with pandas cumcount (the existing function's logic) on random data."""
    import pandas as pd
    rng = np.random.default_rng(0)
    day = np.sort(rng.integers(1, 30, 2000))
    key = rng.integers(-1, 60, 2000)
    order = np.argsort(day, kind="stable")
    s = pd.Series(np.where(key[order] >= 0, key[order], np.nan))
    prior = s.groupby(s).cumcount()
    warm_sorted = (key[order] >= 0) & (prior.fillna(0).to_numpy() > 0)
    warm = np.zeros(len(day), bool)
    warm[order] = warm_sorted
    expect = np.where(key < 0, NOEDGE, np.where(warm, WARM, COLD))
    got = segments_from_arrays(day, key)
    assert (got == expect).all()


def test_no_label_argument():
    import inspect
    assert list(inspect.signature(segments_from_arrays).parameters) == ["day", "pa_key"]


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
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
