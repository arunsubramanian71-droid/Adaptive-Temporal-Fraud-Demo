"""Run the Phase 11 first-pass tests (toy tests and output verification) and record per-test results with file hashes.

Scope: toy/synthetic inputs plus read-only verification of the Phase 11 outputs. No training, no installation, no TEST access, no label read.
Usage:  .venv/Scripts/python.exe -B scripts/run_phase11_tests.py
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

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODULES = ("tests.test_phase11_explain", "tests.test_phase11_outputs")
OUT = ROOT / "experiments" / "phase11_explainable_evidence" / "test_results"
FILES = ["src/explain/__init__.py", "src/explain/xgb_contrib.py", "src/explain/cards.py", "src/explain/outputs.py",
         "scripts/run_phase11_explanations.py", "scripts/run_phase11_cards.py", "scripts/run_phase11_tests.py",
         "tests/test_phase11_explain.py", "tests/test_phase11_outputs.py"]


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
        "purpose": "Phase 11 first-pass tests: toy tests for src/explain and verification of the generated outputs against the scope record.",
        "run_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "python": sys.version.split()[0],
        "files_sha256": {f: hashlib.sha256((ROOT / f).read_bytes()).hexdigest() for f in FILES},
        "summary": {"total": len(results), "passed": n_pass, "failed": len(results) - n_pass},
        "tests": results,
        "scope_statement": "No training, no installation, no TEST access, no label read, no discovery re-run, no Phase 10 empirical tier. "
                           "Passing tests establish only what the assertions check.",
    }
    log.write(f"\n{n_pass}/{len(results)} passed\n")
    (OUT / "test_log.txt").write_bytes(log.getvalue().encode("utf-8"))
    (OUT / "test_results.json").write_bytes((json.dumps(record, indent=1, ensure_ascii=True) + "\n").encode("ascii"))
    print(log.getvalue())
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
