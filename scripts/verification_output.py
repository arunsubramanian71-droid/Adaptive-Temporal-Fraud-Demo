"""Where the end-to-end verifiers write their record.

Before 2026-10-23 scripts/verify_dashboard_e2e.py and scripts/verify_research_serving_e2e.py always overwrote their tracked, dated records under
experiments/phase13_serving_dashboard/test_results/. Running the documented check on another machine therefore replaced the recorded evidence and broke
the hash in submission/INTEGRITY_SUBMISSION.json. Now a run writes to demo_output/verification/ (generated output, not tracked) unless the operator asks
for the tracked record with --record, or names a file with --out.
"""
from __future__ import annotations

import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "demo_output" / "verification"


def output_path(argv, tracked: pathlib.Path, description: str) -> pathlib.Path:
    """Parse --out / --record and return the record path. ``tracked`` is the dated record inside experiments/ (written only with --record)."""
    ap = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help=f"write the record to this file (default: demo_output/verification/{tracked.name})")
    ap.add_argument("--record", action="store_true", help=f"overwrite the tracked record {tracked.relative_to(ROOT).as_posix()} (only when updating the project's evidence)")
    a = ap.parse_args(argv)
    if a.record and a.out:
        ap.error("choose --record or --out, not both")
    if a.record:
        return tracked
    out = pathlib.Path(a.out).resolve() if a.out else DEFAULT_DIR / tracked.name
    if out.is_relative_to(ROOT / "experiments"):
        ap.error("--out may not point into experiments/ (the tracked records); use --record to update the tracked record deliberately")
    return out


def write_record(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
