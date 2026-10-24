"""The end-to-end verifiers must not overwrite the project's tracked, dated evidence records unless asked to (--record).

Found 2026-10-23: running scripts/verify_dashboard_e2e.py on a collaborator's Mac replaced
experiments/phase13_serving_dashboard/test_results/e2e_verification.json (recorded on the development machine on 2026-10-22), so the file no longer
matched its SHA-256 in submission/INTEGRITY_SUBMISSION.json. Each test leaves the tracked files byte-identical, restoring them if a verifier changed them.
The dashboard check starts the documented demo server (synthetic data only); no research data is read.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "experiments" / "phase13_serving_dashboard" / "test_results"
TRACKED = {"dashboard": RESULTS / "e2e_verification.json", "research": RESULTS / "research_e2e_verification.json"}


def _run(script: str, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", f"scripts/{script}", *args], cwd=ROOT, capture_output=True, text=True, timeout=600)


class _Guard:
    """Snapshot the tracked records; afterwards report whether any changed and put the original bytes back."""

    def __enter__(self):
        self.before = {k: (p.read_bytes() if p.exists() else None) for k, p in TRACKED.items()}
        return self

    def __exit__(self, *exc):
        self.changed = []
        for k, p in TRACKED.items():
            now = p.read_bytes() if p.exists() else None
            if now != self.before[k]:
                self.changed.append(p.name)
                if self.before[k] is None:
                    p.unlink()
                else:
                    p.write_bytes(self.before[k])
        return False


def test_the_dashboard_check_writes_where_it_is_told_and_leaves_the_tracked_record_alone():
    d = pathlib.Path(tempfile.mkdtemp(prefix="e2e_out_"))
    out = d / "record.json"
    try:
        with _Guard() as g:
            r = _run("verify_dashboard_e2e.py", "--out", str(out))
        assert g.changed == [], f"tracked record(s) overwritten: {g.changed}"
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        rec = json.loads(out.read_text(encoding="utf-8"))
        assert rec["all_pass"] is True and len(rec["checks"]) >= 30, rec.get("checks")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_record_and_out_together_are_refused():
    with _Guard() as g:
        r = _run("verify_dashboard_e2e.py", "--record", "--out", "x.json")
    assert g.changed == [] and r.returncode == 2 and "choose --record or --out" in r.stderr, (r.returncode, r.stderr)


def test_the_research_check_accepts_out_and_leaves_its_tracked_record_alone():
    """Without the withheld research artifacts the check exits 3 before scoring; with them it runs. Either way the tracked record is untouched."""
    d = pathlib.Path(tempfile.mkdtemp(prefix="research_out_"))
    try:
        with _Guard() as g:
            r = _run("verify_research_serving_e2e.py", "--out", str(d / "record.json"))
        assert g.changed == [], f"tracked record(s) overwritten: {g.changed}"
        assert r.returncode in (0, 3), r.stdout[-2000:] + r.stderr[-2000:]
        if r.returncode == 0:
            assert json.loads((d / "record.json").read_text(encoding="utf-8"))["all_pass"] is True
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _browser_harness():
    """scripts/verify_dashboard_browser.py imports only the standard library at module level (Playwright is imported inside main), so its argument parser
    can be checked here without a browser."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("verify_dashboard_browser", ROOT / "scripts" / "verify_dashboard_browser.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_browser_check_defaults_to_untracked_output_and_this_platforms_interpreter():
    """Before 2026-10-23 the defaults were the tracked record folder and the Windows interpreter path (.venv/Scripts/python.exe)."""
    h = _browser_harness()
    a = h.parse_args([])
    out = pathlib.Path(a.out_dir)
    assert out == ROOT / "demo_output" / "verification" / "browser" and "experiments" not in out.parts, out
    expected = ("Scripts", "python.exe") if sys.platform == "win32" else ("bin", "python")
    assert pathlib.Path(a.python).parts[-2:] == expected, a.python
    assert pathlib.Path(h.parse_args(["--record"]).out_dir) == RESULTS / "browser"
    for bad in (["--record", "--out-dir", "x"], ["--out-dir", str(RESULTS / "browser")], ["--out-dir", str(RESULTS / "browser" / "screenshots")]):
        try:
            h.parse_args(bad)
            raise AssertionError(f"{bad} was accepted")
        except SystemExit as e:
            assert e.code == 2, bad


def test_out_cannot_point_into_the_tracked_records_without_record():
    """Security review 2026-10-23: --out experiments/... would otherwise bypass the --record guard."""
    with _Guard() as g:
        r = _run("verify_dashboard_e2e.py", "--out", str(TRACKED["dashboard"]))
    assert g.changed == [] and r.returncode == 2 and "may not point into experiments/" in r.stderr, (r.returncode, r.stderr[-300:])


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
