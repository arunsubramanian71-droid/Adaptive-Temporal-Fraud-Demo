"""Run the Phase 12 tests (unit tests on synthetic data and verification of the generated evidence table) and record per-test results.

Scope: no training, scoring or prediction; reads only Phase 12 outputs and the declared stored-VAL source files through the guarded reader.
Usage:  .venv/Scripts/python.exe -B scripts/run_phase12_tests.py
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import io
import json
import pathlib
import sys
import time
import traceback
import warnings

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

MODULES = ("tests.test_phase12_evidence", "tests.test_phase12_outputs")
OUT = ROOT / "experiments" / "phase12_validation_evidence_table" / "test_results"
FILES = ["src/evidence/__init__.py", "src/evidence/sources.py", "src/evidence/table.py", "src/evidence/build.py", "src/evidence/outputs.py",
         "scripts/run_phase12_table.py", "scripts/run_phase12_tests.py", "scripts/check_phase12_test_sensitivity.py", "scripts/finalize_phase12_integrity.py",
         "tests/test_phase12_evidence.py", "tests/test_phase12_outputs.py", "src/models/metrics.py"]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results, log = [], io.StringIO()
    for name in MODULES:
        mod = importlib.import_module(name)
        for fn_name in [k for k, v in vars(mod).items() if k.startswith("test_") and callable(v)]:
            t0 = time.perf_counter()
            try:
                getattr(mod, fn_name)()
                status, err = "pass", None
            except Exception as e:  # noqa: BLE001
                status, err = "fail", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            dt = round(time.perf_counter() - t0, 3)
            results.append({"module": name, "test": fn_name, "status": status, "seconds": dt, "error": err})
            log.write(f"[{status.upper()}] {name}.{fn_name} ({dt}s)\n")
            if err:
                log.write(err + "\n")
    n_pass = sum(r["status"] == "pass" for r in results)
    record = {
        "purpose": "Phase 12 tests: unit tests on synthetic data and verification of the evidence table against the registered acceptance criteria.",
        "run_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "python": sys.version.split()[0],
        "files_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES},
        "summary": {"total": len(results), "passed": n_pass, "failed": len(results) - n_pass},
        "tests": results,
        "scope_statement": "No training, no scoring, no new predictions, no TEST contents, no checkpoint opened, no Phase 10 empirical tier, no ablation. "
                           "Passing tests establish only what the assertions check.",
    }
    log.write(f"\n{n_pass}/{len(results)} passed\n")
    (OUT / "test_log.txt").write_bytes(log.getvalue().encode("utf-8"))
    (OUT / "test_results.json").write_bytes((json.dumps(record, indent=1, ensure_ascii=True) + "\n").encode("ascii"))
    print(log.getvalue())
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
