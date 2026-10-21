"""Mutation-sensitivity check for the comparison-B tests: inject known defects into an in-memory copy of
src/features/graph_features.py and verify that tests/test_graph_features.py fails for each of them (BA9).

No file is modified; no data is read. Writes experiments/comparison_B_graph_features/test_results/test_sensitivity.json.
Usage:  .venv/Scripts/python.exe -B scripts/check_comparison_b_test_sensitivity.py
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys
import types

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import tests.test_graph_features as TG  # noqa: E402

SRC = (ROOT / "src" / "features" / "graph_features.py").read_text(encoding="utf-8")
OUT = ROOT / "experiments" / "comparison_B_graph_features" / "test_results"

MUTANTS = [
    ("M01 ties count as prior (side=right)", 'pos = np.searchsorted(skey, key, side="left")', 'pos = np.searchsorted(skey, key, side="right")'),
    ("M02 one-day window excludes the boundary", "np.maximum(tv - DAY, 0), side=\"left\")", "np.maximum(tv - DAY + 1, 0), side=\"left\")"),
    ("M03 seven-day window uses one day", "np.maximum(tv - WEEK, 0), side=\"left\")", "np.maximum(tv - DAY, 0), side=\"left\")"),
    ("M04 last-seen uses the first prior row", "st[np.maximum(pos - 1, 0)]", "st[start]"),
    ("M05 distinct count includes the current row", "cf[pos] - cf[start]", "cf[np.minimum(pos + 1, len(cf) - 1)] - cf[start]"),
    ("M06 amount mean includes the current row", "cs[np.maximum(pos - 1, 0)]", "cs[np.minimum(pos, len(cs) - 1)]"),
    ("M07 counts leak across entities", "start = np.searchsorted(skey, ev * TIME_LIMIT, side=\"left\")", "start = np.zeros(len(pos), dtype=np.int64)"),
    ("M08 edge flag inverted", "has[valid] = 1.0", "has[:] = 1.0\n        has[valid] = 0.0"),
    ("M09 tie order depends on input order", "order = np.lexsort((logamt[valid], tv, ev))", "order = np.argsort(key, kind=\"stable\")"),
    ("M10 missing-edge rows collide with entity 0", "np.where(ok, se * width + o, -1 - np.arange(len(se)))", "se * width + np.where(ok, o, 0)"),
    ("M11 label argument added", "def compute_graph_features(time, amount, entities: dict) -> pd.DataFrame:", "def compute_graph_features(time, amount, entities: dict, y=None) -> pd.DataFrame:"),
    ("M12 prior count uses the whole entity (future rows)", "prior = pos - start", "prior = np.searchsorted(skey, (ev + 1) * TIME_LIMIT, side=\"left\") - start"),
    ("M13 amount deviation sign flipped", "cols[\"amt_dev\"][valid] = logamt[valid] - mean", "cols[\"amt_dev\"][valid] = mean - logamt[valid]"),
    ("M14 time-limit check removed", "if n and (time.min() < 0 or time.max() >= TIME_LIMIT):", "if False:"),
]


def make_module(source: str, name: str):
    mod = types.ModuleType(name)
    mod.__file__ = str(ROOT / "src" / "features" / "graph_features.py")
    exec(compile(source, mod.__file__, "exec"), mod.__dict__)  # noqa: S102 - in-memory copy of the project's own source
    return mod


def run_tests(mod):
    original = TG.G
    TG.G = mod
    failed = []
    try:
        for name in [k for k, v in vars(TG).items() if k.startswith("test_") and callable(v)]:
            try:
                getattr(TG, name)()
            except Exception:  # noqa: BLE001
                failed.append(name)
    finally:
        TG.G = original
    return failed


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    unmutated_failed = run_tests(make_module(SRC, "graph_features_unmutated"))
    rows, undetected = [], []
    for label, old, new in MUTANTS:
        if SRC.count(old) != 1:
            rows.append({"mutant": label, "status": "NOT APPLIED", "detail": f"pattern found {SRC.count(old)} times"})
            undetected.append(label)
            continue
        try:
            failed = run_tests(make_module(SRC.replace(old, new), "graph_features_mutant"))
        except Exception as e:  # noqa: BLE001
            failed = [f"module error: {type(e).__name__}: {e}"]
        rows.append({"mutant": label, "status": "detected" if failed else "UNDETECTED", "failing_tests": failed})
        if not failed:
            undetected.append(label)
        print(f"[{'detected' if failed else 'UNDETECTED'}] {label} -> {len(failed)} failing test(s)")
    rec = {"purpose": "mutation sensitivity of tests/test_graph_features.py", "run_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
           "unmutated_failures": unmutated_failed, "n_mutants": len(MUTANTS), "n_detected": sum(r["status"] == "detected" for r in rows), "mutants": rows,
           "scope_statement": "In-memory mutants only; no file modified; no data read. Detection shows these tests fail on these defects, not that no other defect exists."}
    (OUT / "test_sensitivity.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    print(f"unmutated failures: {unmutated_failed or 'none'}; detected {rec['n_detected']}/{rec['n_mutants']}")
    return 0 if not unmutated_failed and not undetected else 1


if __name__ == "__main__":
    sys.exit(main())
