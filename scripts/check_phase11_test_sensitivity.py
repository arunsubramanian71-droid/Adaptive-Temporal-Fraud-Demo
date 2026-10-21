"""Test-sensitivity (mutation) check for the Phase 11 first-pass tests.

Each mutant is an in-memory text edit of ``src/explain/xgb_contrib.py`` or ``src/explain/cards.py`` (files on disk are never modified).
A mutant is DETECTED if at least one Phase 11 test fails against it; the unmutated modules must pass everything. This checks the tests,
not the research.  Usage:  .venv/Scripts/python.exe -B scripts/check_phase11_test_sensitivity.py
"""
from __future__ import annotations

import contextlib
import datetime
import hashlib
import importlib
import io
import json
import pathlib
import sys
import types

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments" / "phase11_explainable_evidence" / "test_results"
SUITES = ("tests.test_phase11_explain", "tests.test_phase11_outputs")
XC, CD = "xgb_contrib", "cards"

MUTANTS = {
    "M1 early-stopping iteration range ignored": (XC, "return (0, int(model.best_iteration) + 1)      # what predict_proba uses after early stopping", "return (0, 0)"),
    "M2 uniform draws not excluding the top group": (XC, "rest = np.sort(idd[~np.isin(idd, top)])", "rest = np.sort(idd)"),
    "M3 score ties broken by higher id": (XC, "order = np.lexsort((idd, -ss))", "order = np.lexsort((-idd, -ss))"),
    "M4 draw seed ignores the day": (XC, "np.random.SeedSequence([int(seed), d])", "np.random.SeedSequence([int(seed)])"),
    "M5 day-window guard removed": (XC, "if days.size and (days.min() < lo or days.max() > hi):", "if False:"),
    "M6 label column not excluded from reads": (XC, '_NEVER_LOADED = ("isFraud",)', "_NEVER_LOADED = ()"),
    "M7 TEST-file guard removed": (XC, 'if "test" in p.name.lower():', "if False:"),
    "M8 top-feature tie-break reversed": (XC, "np.lexsort((np.arange(len(contrib_row)), -np.abs(contrib_row)))", "np.lexsort((-np.arange(len(contrib_row)), -np.abs(contrib_row)))"),
    "M9 alert rule strict (ties at tau not alerted)": (CD, 'df["score_st"] >= tau, "track"', 'df["score_st"] > tau, "track"'),
    "M10 group (d) uses u_s rank only": (CD, '_rank_position(t["peak_u_s"], tr) + _rank_position(t["peak_hub_share"], tr)', '_rank_position(t["peak_u_s"], tr)'),
    "M11 group (b) ranks by observed days": (CD, 't.sort_values(["peak_n_cards", "track"], ascending=[False, True])', 't.sort_values(["n_days", "track"], ascending=[False, True])'),
    "M12 group (c) seed offset wrong": (CD, "SAMPLE_SEED_BASE + int(seed)", "SAMPLE_SEED_BASE + 1"),
    "M13 right-censoring flag wrong": (CD, '"right_censored": last == DAYS[1]', '"right_censored": last == DAYS[1] - 1'),
    "M14 Tier 0 day window not enforced": (CD, "if df[\"day\"].min() < DAYS[0] or df[\"day\"].max() > DAYS[1]:", "if False:"),
}


def load(overrides: dict) -> None:
    for m in [k for k in sys.modules if k.startswith("src.explain") or k.startswith("tests.")]:
        del sys.modules[m]
    pkg = importlib.import_module("src.explain")
    for name in (XC, CD):
        code = overrides.get(name) or (ROOT / "src" / "explain" / f"{name}.py").read_text(encoding="utf-8")
        mod = types.ModuleType(f"src.explain.{name}")
        mod.__file__ = f"mutant_{name}"
        sys.modules[f"src.explain.{name}"] = mod
        setattr(pkg, name, mod)
        exec(compile(code, mod.__file__, "exec"), mod.__dict__)


def failing() -> list[str]:
    bad = []
    for s in SUITES:
        m = importlib.import_module(s)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            m._run_all()
        bad += [l.split(":")[0].replace("[FAIL] ", "") for l in buf.getvalue().splitlines() if l.startswith("[FAIL]")]
    return bad


def main() -> int:
    src = {n: (ROOT / "src" / "explain" / f"{n}.py").read_text(encoding="utf-8") for n in (XC, CD)}
    load({})
    base = failing()
    rows = []
    for label, (mod, old, new) in MUTANTS.items():
        if src[mod].count(old) != 1:
            rows.append({"mutant": label, "status": "pattern_not_unique", "failing_tests": []})
            continue
        load({mod: src[mod].replace(old, new)})
        f = failing()
        rows.append({"mutant": label, "status": "detected" if f else "NOT_DETECTED", "failing_tests": f})
    detected = sum(r["status"] == "detected" for r in rows)
    record = {
        "purpose": "Test-sensitivity (mutation) check of the Phase 11 first-pass tests; in-memory mutants only, no project file modified.",
        "run_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "module_sha256": {n: hashlib.sha256((ROOT / "src" / "explain" / f"{n}.py").read_bytes()).hexdigest() for n in (XC, CD)},
        "unmutated_failures": base, "summary": {"mutants": len(rows), "detected": detected}, "mutants": rows,
        "interpretation": "A detected mutant shows that at least one test fails when that specific defect is injected. "
                          "It does not show that no other defect exists.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_sensitivity.json").write_bytes((json.dumps(record, indent=1, ensure_ascii=True) + "\n").encode("ascii"))
    print(f"unmutated failures: {base}\nmutants detected: {detected}/{len(rows)}")
    for r in rows:
        print(f"  {r['status']:>16}  {r['mutant']}")
    return 0 if (not base and detected == len(rows)) else 1


if __name__ == "__main__":
    sys.exit(main())
