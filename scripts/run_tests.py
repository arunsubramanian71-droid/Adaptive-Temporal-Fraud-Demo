"""Run every tests/test_*.py module without requiring pytest to be installed.

Each test module exposes a _run_all() that executes its test_* functions and
returns True iff all passed. This script just orchestrates across modules
and sets the process exit code accordingly.
"""
from __future__ import annotations

import importlib
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    test_dir = ROOT / "tests"
    modules = sorted(p.stem for p in test_dir.glob("test_*.py"))
    all_ok = True
    for name in modules:
        print(f"\n----- tests.{name} -----")
        mod = importlib.import_module(f"tests.{name}")
        ok = mod._run_all()
        all_ok = all_ok and ok
    print(f"\n{'ALL TEST MODULES PASSED' if all_ok else 'SOME TEST MODULES FAILED'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
