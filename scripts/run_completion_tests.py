"""Run every tests/test_*.py module and record per-test results in submission/test_results.json (with a SHA-256 sidecar).

No training, no scoring of research models and no TEST data: the tests use synthetic fixtures, recorded outputs and the project's own
guards. Usage:  .venv/Scripts/python.exe -B scripts/run_completion_tests.py
"""
from __future__ import annotations

import datetime
import hashlib
import importlib
import io
import json
import pathlib
import platform
import sys
import time
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
OUT = ROOT / "submission"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    results, modules = [], sorted(p.stem for p in (ROOT / "tests").glob("test_*.py"))
    for name in modules:
        mod = importlib.import_module(f"tests.{name}")
        for fn_name in [k for k, v in vars(mod).items() if k.startswith("test_") and callable(v)]:
            t0 = time.perf_counter()
            buf = io.StringIO()
            try:
                with redirect_stdout(buf), redirect_stderr(buf):
                    getattr(mod, fn_name)()
                status, err = "pass", None
            except Exception as e:  # noqa: BLE001
                status, err = "fail", f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            results.append({"module": name, "test": fn_name, "status": status, "seconds": round(time.perf_counter() - t0, 3), "error": err})
            print(f"[{status.upper()}] {name}.{fn_name}", flush=True)
    n_pass = sum(r["status"] == "pass" for r in results)
    per_module = {m: {"total": sum(r["module"] == m for r in results), "passed": sum(r["module"] == m and r["status"] == "pass" for r in results)} for m in modules}
    rec = {"run_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "python": platform.python_version(),
           "command": ".venv/Scripts/python.exe -B scripts/run_completion_tests.py", "summary": {"modules": len(modules), "total": len(results), "passed": n_pass, "failed": len(results) - n_pass},
           "per_module": per_module, "tests": results,
           "scope_statement": "Synthetic fixtures, recorded outputs and guards only; no research model trained or scored; no TEST data. Passing tests establish only what their assertions check."}
    data = (json.dumps(rec, indent=1) + "\n").encode("utf-8")
    (OUT / "test_results.json").write_bytes(data)
    (OUT / "test_results.sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print(f"\n{n_pass}/{len(results)} passed in {len(modules)} modules; wrote submission/test_results.json")
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
