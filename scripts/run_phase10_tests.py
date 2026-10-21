"""Run the Phase 10 core tests (toy, property/accounting, compatibility) and record per-test results with file hashes.

Authorized scope only: toy/synthetic inputs; the Phase 9 ``Tracker`` is imported read-only for the compatibility comparison.
No Stage 0, no TEST data, no synthetic benchmark, no null-model tier, no real-data evaluation.
Usage:  .venv/Scripts/python.exe -B scripts/run_phase10_tests.py
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

MODULES = ("tests.test_phase10_toys", "tests.test_phase10_properties", "tests.test_phase10_compat")
OUT = ROOT / "experiments" / "phase10_community_evolution" / "core_implementation"


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    files = ["src/evolution/__init__.py", "src/evolution/tracker.py", "tests/phase10_helpers.py", "tests/test_phase10_toys.py",
             "tests/test_phase10_properties.py", "tests/test_phase10_compat.py", "scripts/run_phase10_tests.py"]
    n_pass = sum(r["status"] == "pass" for r in results)
    record = {
        "purpose": "Phase 10 core implementation tests (toy, property/accounting, compatibility). Toy/synthetic inputs only.",
        "run_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "python": sys.version.split()[0],
        "files_sha256": {f: sha(ROOT / f) for f in files},
        "summary": {"total": len(results), "passed": n_pass, "failed": len(results) - n_pass},
        "tests": results,
        "scope_statement": "No Stage 0, no TEST data, no synthetic benchmark, no null-model tier, no real-data evaluation. "
                           "Phase 9 Tracker imported read-only. Passing tests establish only what the assertions check.",
    }
    log.write(f"\n{n_pass}/{len(results)} passed\n")
    (OUT / "core_test_log.txt").write_bytes(log.getvalue().encode("utf-8"))
    (OUT / "core_test_results.json").write_bytes((json.dumps(record, indent=1, ensure_ascii=True) + "\n").encode("ascii"))
    print(log.getvalue())
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
