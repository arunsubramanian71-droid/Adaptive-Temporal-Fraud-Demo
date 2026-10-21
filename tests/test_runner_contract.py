"""Test-runner contract: every test module exposes `_run_all()` returning a truthy value only when all its tests passed (scripts/run_tests.py relies on it), and a failing test is detected.
Source-level checks plus one injected failure; no research data is read."""
from __future__ import annotations

import importlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_every_test_module_exposes_run_all_that_returns_a_boolean_not_an_exit_code():
    bad = []
    for p in sorted((ROOT / "tests").glob("test_*.py")):
        src = p.read_text(encoding="utf-8")
        if "def _run_all" not in src:
            bad.append(f"{p.name}: no _run_all")
        elif re.search(r"return\s+1\s+if\s+failed\s+else\s+0", src):
            bad.append(f"{p.name}: _run_all returns an exit code (truthy on failure)")
    assert bad == [], bad


def test_run_all_reports_false_when_a_test_fails_and_true_when_all_pass():
    mod = importlib.import_module("tests.test_scope_reconciliation_records")
    assert mod._run_all.__globals__ is vars(mod)
    original = dict(vars(mod))
    try:
        def test_zz_injected_failure():
            raise AssertionError("injected")
        vars(mod)["test_zz_injected_failure"] = test_zz_injected_failure
        import io
        from contextlib import redirect_stdout
        with redirect_stdout(io.StringIO()):
            assert mod._run_all() is False
    finally:
        vars(mod).pop("test_zz_injected_failure", None)
    assert "test_zz_injected_failure" not in vars(mod) and set(original) <= set(vars(mod))
    import io
    from contextlib import redirect_stdout
    with redirect_stdout(io.StringIO()):
        assert mod._run_all() is True


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print("[PASS]", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print("[FAIL]", fn.__name__, repr(e))
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
