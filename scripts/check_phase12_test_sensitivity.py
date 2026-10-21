"""Test-sensitivity (mutation) check for the Phase 12 tests.

Each mutant is an in-memory text edit of a module in ``src/evidence/`` (files on disk are never modified). A mutant is DETECTED if at least
one Phase 12 test fails (or errors) against it; the unmutated modules must pass everything. This checks the tests, not the research.
Usage:  .venv/Scripts/python.exe -B scripts/check_phase12_test_sensitivity.py
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
import warnings

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
OUT = ROOT / "experiments" / "phase12_validation_evidence_table" / "test_results"
SUITES = ("tests.test_phase12_evidence", "tests.test_phase12_outputs")
ORDER = ("outputs", "sources", "table", "build")
SRC = ROOT / "src" / "evidence"

MUTANTS = {
    "M1 guard no longer refuses TEST-named paths": ("sources", 'if rel not in self.allowed and rel not in self.extra:', None,
                                                     '"test" in low.replace("latest", "") or ', "False or "),
    "M2 guard stops logging opens": ("sources", "self.log.append((rel, kind))", None, None, "pass"),
    "M3 fingerprint separator changed": ("sources", 'f"{p}\\t{hashes[p]}\\n"', None, None, 'f"{p} {hashes[p]}\\n"'),
    "M4 excluded folders no longer refused": ("sources", "any(f in low for f in EXCLUDED_FRAGMENTS)", None, None, "False"),
    "M5 registered tolerance loosened": ("table", "TOL = 1e-9", None, None, "TOL = 1e-3"),
    "M6 label identity check dropped": ("table", 'n = int((b["isFraud"].astype(int) != b["isFraud_ref"].astype(int)).sum())', None, None, "n = 0"),
    "M7 day/segment identity checks dropped": ("table", 'n = int((b[c] != b[c + "_ref"]).sum())', None, None, "n = 0"),
    "M8 F window consistency check removed": ("table", 'if not (frame["window"] == w).all() or not (frame["day"].map(window_of_day) == w).all():', None, None, "if False:"),
    "M9 F mean-of-windows uses the median": ("table", "return float(np.mean(vals)) if vals else None", None, None, "return float(np.median(vals)) if vals else None"),
    "M10 SD uses ddof = 0": ("table", 'float(a.std(ddof=1)) if len(a) > 1 else None', None, None, 'float(a.std(ddof=0)) if len(a) > 1 else None'),
    "M11 strict-segment mask inverted": ("table", "m = seg == name", None, None, "m = seg != name"),
    "M12 segment join re-sorts the rows": ("table", 'frame = frame.merge(ref[["TransactionID"] + missing], on="TransactionID", how="left", validate="many_to_one")', None, None,
                                            'frame = frame.merge(ref[["TransactionID"] + missing], on="TransactionID", how="left", validate="many_to_one").sort_values("TransactionID").reset_index(drop=True)'),
    "M13 scope sidecar gate removed": ("build", "if scope_hash != sidecar:", None, None, "if False:"),
    "M14 fingerprint gate removed": ("build", "if fp_start != expected_fp or len(hashes) != 165:", None, None, "if False:"),
    "M15 recorded-tolerance gate removed": ("build", "if failing:", None, None, "if False:"),
    "M16 B row no longer marked MISSING": ("build", '"status": "MISSING", "note"', None, None, '"status": "n/a", "note"'),
    "M17 comparability labels overwritten": ("build", '"label": lab,', None, None, '"label": "Comparable",'),
}


def load(overrides: dict) -> None:
    for m in [k for k in sys.modules if k.startswith("src.evidence") or k.startswith("tests.")]:
        del sys.modules[m]
    pkg = importlib.import_module("src.evidence")
    for name in ORDER:
        code = overrides.get(name) or (SRC / f"{name}.py").read_text(encoding="utf-8")
        mod = types.ModuleType(f"src.evidence.{name}")
        mod.__file__ = f"mutant_{name}"
        sys.modules[f"src.evidence.{name}"] = mod
        setattr(pkg, name, mod)
        exec(compile(code, mod.__file__, "exec"), mod.__dict__)


def failing() -> list:
    bad = []
    for s in SUITES:
        m = importlib.import_module(s)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            m._run_all()
        bad += [l.split(":")[0].replace("[FAIL] ", "") for l in buf.getvalue().splitlines() if l.startswith("[FAIL]")]
    return bad


def main() -> int:
    src = {n: (SRC / f"{n}.py").read_text(encoding="utf-8") for n in ORDER}
    load({})
    base = failing()
    rows = []
    for label, (mod, old, _unused, inner_old, new) in MUTANTS.items():
        target_old = old if inner_old is None else old
        if inner_old is not None:                      # M1: replace an inner fragment of the guarded condition line
            if src[mod].count(inner_old) != 1:
                rows.append({"mutant": label, "status": "pattern_not_unique", "failing_tests": []})
                continue
            mutated = src[mod].replace(inner_old, new)
        else:
            if src[mod].count(target_old) != 1:
                rows.append({"mutant": label, "status": "pattern_not_unique", "failing_tests": []})
                continue
            mutated = src[mod].replace(target_old, new)
        load({mod: mutated})
        f = failing()
        rows.append({"mutant": label, "status": "detected" if f else "NOT_DETECTED", "failing_tests": f})
    detected = sum(r["status"] == "detected" for r in rows)
    record = {
        "purpose": "Test-sensitivity (mutation) check of the Phase 12 tests; in-memory mutants only, no project file modified.",
        "run_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "module_sha256": {n: hashlib.sha256((SRC / f"{n}.py").read_bytes()).hexdigest() for n in ORDER},
        "unmutated_failures": base, "summary": {"mutants": len(rows), "detected": detected}, "mutants": rows,
        "interpretation": "A detected mutant shows that at least one test fails when that specific defect is injected. It does not show that no other defect exists.",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "test_sensitivity.json").write_bytes((json.dumps(record, indent=1, ensure_ascii=True) + "\n").encode("ascii"))
    print(f"unmutated failures: {base}\nmutants detected: {detected}/{len(rows)}")
    for r in rows:
        print(f"  {r['status']:>16}  {r['mutant']}")
    return 0 if (not base and detected == len(rows)) else 1


if __name__ == "__main__":
    sys.exit(main())
