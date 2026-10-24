"""scripts/check_setup.py must import every required package, not only read its version metadata.

Found 2026-10-23 on macOS (Apple Silicon): the XGBoost wheel links the OpenMP runtime through one search path (/opt/homebrew/opt/libomp/lib).
Without Homebrew's libomp, ``import xgboost`` fails, yet the check printed "xgboost 3.4.1 ok" and "SETUP OK" with exit code 0, because it read only
the installed version. These tests plant a package that is installed (its version metadata exists) but cannot be imported, in a child process,
so they run on every operating system. No data is read.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_setup.py"
# XGBoost's own message (xgboost 3.4.1, captured 2026-10-23 on a Mac whose libomp was made unreachable): it lists every platform's OpenMP runtime
# before the loader's error, so the fix must be chosen by platform, not by the first runtime name found in the text.
BOILERPLATE = ("XGBoost Library (libxgboost.dylib) could not be loaded.\nLikely causes:\n  * OpenMP runtime is not installed\n"
               "    - vcomp140.dll or libgomp-1.dll for Windows\n    - libomp.dylib for Mac OSX\n    - libgomp.so for Linux and other UNIX-like OSes\n"
               "    Mac OSX users: Run `brew install libomp` to install OpenMP runtime.\n\n  * You are running 32-bit Python on a 64-bit OS\n\n")
LIBOMP_ERROR = BOILERPLATE + ("Error message(s): [\"dlopen(.../xgboost/lib/libxgboost.dylib, 0x0006): Library not loaded: @rpath/libomp.dylib\\n"
                              "  Reason: tried: '/opt/homebrew/opt/libomp/lib/libomp.dylib' (no such file)\"]")
LIBGOMP_ERROR = BOILERPLATE.replace("libxgboost.dylib", "libxgboost.so") + "Error message(s): ['libgomp.so.1: cannot open shared object file: No such file or directory']"
FIX_WORDS = {"darwin": "brew install libomp", "linux": "libgomp1", "win32": "Visual C++ Redistributable"}


def _module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_setup_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)                     # defines constants and functions only; main() runs under __main__
    return mod


def _run(extra_path: pathlib.Path | None = None) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    if extra_path is not None:
        env["PYTHONPATH"] = str(extra_path)
    return subprocess.run([sys.executable, "-B", str(SCRIPT)], cwd=ROOT, env=env, capture_output=True, text=True, timeout=300)


def _broken_package(name: str, message: str) -> pathlib.Path:
    """A directory holding a package ``name`` whose import raises like a wheel whose native library cannot be loaded."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="setup_check_"))
    (d / name).mkdir()
    (d / name / "__init__.py").write_text(f"raise OSError({message!r})\n", encoding="utf-8")
    return d


def test_the_check_passes_in_a_working_environment_and_says_it_imported_each_package():
    r = _run()
    assert r.returncode == 0, r.stdout + r.stderr
    assert "SETUP OK for the demonstration" in r.stdout
    lines = r.stdout.splitlines()
    for name, _ in _module().requirements(ROOT / "requirements.txt"):
        line = next((l for l in lines if l.strip().split(" ")[0] == name), "")
        assert "imports" in line and "CANNOT" not in line, (name, line)


def test_an_installed_package_that_cannot_be_imported_fails_the_check():
    d = _broken_package("xgboost", LIBOMP_ERROR)
    try:
        r = _run(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert r.returncode == 1, r.stdout + r.stderr
    assert "SETUP OK" not in r.stdout and "SETUP INCOMPLETE" in r.stdout, r.stdout
    line = next((l for l in r.stdout.splitlines() if l.strip().startswith("xgboost")), "")
    assert "CANNOT BE IMPORTED" in line, r.stdout


def test_a_missing_openmp_runtime_gets_this_platforms_fix_and_the_loaders_own_reason():
    d = _broken_package("xgboost", LIBOMP_ERROR)
    try:
        r = _run(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    platform = "darwin" if sys.platform == "darwin" else "win32" if sys.platform.startswith("win") else "linux"
    fixes = [l for l in r.stdout.splitlines() if l.strip().startswith("fix:")]
    assert len(fixes) == 1 and FIX_WORDS[platform] in fixes[0], fixes
    line = next(l for l in r.stdout.splitlines() if l.strip().startswith("xgboost"))
    assert "Library not loaded: @rpath/libomp.dylib" in line and "Likely causes" not in line, line


def test_the_fix_is_chosen_by_platform_not_by_the_first_runtime_named_in_the_message():
    """Review 2026-10-23: matching words picked 'libomp' (listed first in XGBoost's boilerplate) on every platform, telling Linux users to use Homebrew."""
    m = _module()
    for platform, words in FIX_WORDS.items():
        for err in (LIBOMP_ERROR, LIBGOMP_ERROR):
            fix = m.openmp_fix(err, platform)
            assert fix and words in fix and all(w not in fix for p, w in FIX_WORDS.items() if p != platform), (platform, fix)
    assert m.openmp_fix("No module named 'pandas'", "linux") is None
    assert m.loader_reason(OSError(LIBGOMP_ERROR)) == "OSError: libgomp.so.1: cannot open shared object file: No such file or directory"


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print("[PASS]", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print("[FAIL]", fn.__name__, repr(e)[:300])
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
