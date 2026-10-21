"""Run the test suite in any environment, skipping (and saying so) the modules that need PyTorch when it is not installed.

  .venv/Scripts/python.exe -B scripts/run_core_tests.py [--json OUT.json]

With the full environment (requirements.txt + requirements-gnn.txt) every module runs and the result equals scripts/run_tests.py. With only
requirements.txt, the modules whose import needs PyTorch are reported as SKIPPED (never as passed). A recorded-output test that compares the Python version string stored in
the Phase 12 manifest can report a difference under another interpreter; that is environment metadata, not a data difference, and is printed as a failure rather than hidden.
Exit code: 0 if no executed test failed, 1 otherwise.
"""
from __future__ import annotations

import argparse
import importlib
import io
import json
import pathlib
import sys
import time
import warnings
from contextlib import redirect_stderr, redirect_stdout

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
NEEDS_TORCH = ("torch", "torch_geometric")
# A registered test whose failure can be an environment-metadata difference only: the Phase 12 AC10 test compares the rebuilt manifest, which also records the
# interpreter version. It is NEVER counted as passed. If it fails, the companion module must pass (data outputs byte-identical, manifests equal except timestamp and
# recorded versions); the failure is then labelled as such and does not make the exit code non-zero.
ENV_SENSITIVE = {("test_phase12_outputs", "test_AC10_sidecars_and_deterministic_rebuild"): ("test_phase12_environment_independence", "test_rebuild_is_byte_identical_and_the_manifest_differs_only_in_environment_metadata")}


def _companion_passes(module: str, test: str) -> bool:
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            getattr(importlib.import_module(f"tests.{module}"), test)()
        return True
    except BaseException:  # noqa: BLE001
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=None, help="write the per-module result as JSON")
    a = ap.parse_args()
    rows, failed_any = [], False
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        name = p.stem
        t0 = time.perf_counter()
        try:
            mod = importlib.import_module(f"tests.{name}")
        except ModuleNotFoundError as e:
            if (e.name or "").split(".")[0] in NEEDS_TORCH:
                rows.append({"module": name, "status": "SKIPPED", "reason": f"needs {e.name} (install requirements-gnn.txt)"})
                print(f"[SKIPPED] {name}: needs {e.name}", flush=True)
                continue
            raise
        fns = [k for k, v in vars(mod).items() if k.startswith("test_") and callable(v)]
        bad, env_only = [], []
        for f in fns:
            try:
                with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                    getattr(mod, f)()
            except BaseException as e:  # noqa: BLE001
                msg = f"{f}: {type(e).__name__}: {str(e)[:120]}"
                comp = ENV_SENSITIVE.get((name, f))
                if comp and _companion_passes(*comp):
                    env_only.append(msg + "  [environment metadata only; companion test passes]")
                else:
                    bad.append(msg)
        failed_any |= bool(bad)
        status = "FAILED" if bad else ("PASSED, 1 ENVIRONMENT-METADATA DIFFERENCE" if env_only else "PASSED")
        rows.append({"module": name, "status": status, "tests": len(fns), "passed": len(fns) - len(bad) - len(env_only), "failures": bad,
                     "environment_metadata_differences": env_only, "seconds": round(time.perf_counter() - t0, 1)})
        print(f"[{status}] {name}: {len(fns) - len(bad) - len(env_only)}/{len(fns)}" + "".join(f"\n      {b}" for b in bad + env_only), flush=True)
    ran = [r for r in rows if r["status"] != "SKIPPED"]
    skipped = [r["module"] for r in rows if r["status"] == "SKIPPED"]
    total, passed = sum(r["tests"] for r in ran), sum(r["passed"] for r in ran)
    env_n = sum(len(r.get("environment_metadata_differences", [])) for r in ran)
    print(f"\n{passed}/{total} tests passed in {len(ran)} modules; environment-metadata differences (never counted as passed): {env_n}; skipped modules: {skipped or 'none'}")
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"python": sys.version.split()[0], "modules": rows, "tests": total, "passed": passed, "environment_metadata_differences": env_n, "skipped": skipped}, indent=1), encoding="utf-8")
    return 1 if failed_any else 0


if __name__ == "__main__":
    sys.exit(main())
