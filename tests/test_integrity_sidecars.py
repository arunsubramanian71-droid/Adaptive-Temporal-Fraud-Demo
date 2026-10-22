"""Every tracked SHA-256 sidecar matches the file it describes (added 2026-10-22 after a repository-wide sweep).

A sidecar ``X.sha256`` holds the bare SHA-256 of its companion ``X.<ext>`` in the same folder. Two documented exceptions: the Phase 9 sealed manifests hash
the manifest *text* (``src/coordination/runner.py``), and the Phase 9 closure-report sidecar describes ``reports/phase9_closure_report.md``. A sidecar whose
companion is a withheld artifact (row-level data, model files) cannot be checked in a checkout without it; that is reported, never counted as a match.
No held-out file is opened.
"""
from __future__ import annotations

import hashlib
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

HELD_OUT = re.compile(r"test_features|test_transaction|test_identity|heldout_results|AUTHORIZATION")
OTHER_FOLDER = {"experiments/phase9_coordinated_fraud/phase9_closure_report.sha256": "reports/phase9_closure_report.md"}


def sidecar_problems(root: pathlib.Path, sidecars: list, tracked: set) -> tuple:
    """(problems, n_matched, n_withheld) for the given sidecar paths (relative, posix)."""
    def sha256_text(txt: str) -> str:                              # the sealing rule of src/coordination/benchmark.py (inlined: that package needs PyTorch)
        return hashlib.sha256(txt.encode("utf-8")).hexdigest()
    problems, matched, withheld = [], 0, 0
    for rel in sidecars:
        if HELD_OUT.search(rel):
            continue
        side = root / rel
        recorded = side.read_text(encoding="utf-8", errors="replace").strip()
        if not re.fullmatch(r"[0-9a-f]{64}", recorded):
            problems.append(f"{rel}: not a bare SHA-256")
            continue
        if rel.endswith("/sealed/manifest.sha256") and "phase9_coordinated_fraud" in rel:
            ok = sha256_text((side.parent / "manifest_sealed.json").read_text(encoding="utf-8")) == recorded
            matched += ok
            problems += [] if ok else [f"{rel}: sealed manifest text does not match"]
            continue
        if rel in OTHER_FOLDER:
            comps = [root / OTHER_FOLDER[rel]]
        else:
            comps = [c for c in side.parent.glob(side.stem + ".*") if c.suffix != ".sha256" and c.is_file()]
        if len(comps) > 1:
            problems.append(f"{rel}: one sidecar for several files {sorted(c.name for c in comps)} (one hash overwrites the other)")
            continue
        if not comps:
            if any(t.startswith(rel[: -len(".sha256")] + ".") for t in tracked):
                problems.append(f"{rel}: its tracked companion is missing")
            else:
                withheld += 1
            continue
        if HELD_OUT.search(comps[0].name):
            continue
        if hashlib.sha256(comps[0].read_bytes()).hexdigest() == recorded:
            matched += 1
        else:
            problems.append(f"{rel}: does not match {comps[0].relative_to(root).as_posix()}")
    return problems, matched, withheld


def _tracked() -> set:
    if not (ROOT / ".git").exists():
        return {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*") if p.is_file() and ".venv" not in p.parts and ".git" not in p.parts}
    return set(subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True).stdout.split("\0")) - {""}


def test_every_tracked_sidecar_matches_its_file():
    tracked = _tracked()
    sidecars = sorted(t for t in tracked if t.endswith(".sha256"))
    problems, matched, withheld = sidecar_problems(ROOT, sidecars, tracked)
    assert problems == [], problems
    assert matched >= 100, (matched, withheld)                    # 132+ in the project directory; a fresh clone lacks the withheld companions


def test_the_checker_catches_a_shared_sidecar_and_a_mismatch():
    d = pathlib.Path(tempfile.mkdtemp(prefix="sidecar_"))
    try:
        (d / "R.md").write_text("review", encoding="utf-8")
        (d / "R.json").write_text("{}", encoding="utf-8")
        (d / "R.sha256").write_text(hashlib.sha256(b"{}").hexdigest(), encoding="ascii")          # the PROTOCOL_REVIEW pattern before 2026-10-22
        (d / "M.txt").write_text("data", encoding="utf-8")
        (d / "M.sha256").write_text("0" * 64, encoding="ascii")
        (d / "W.sha256").write_text(hashlib.sha256(b"x").hexdigest(), encoding="ascii")            # companion withheld and untracked
        problems, matched, withheld = sidecar_problems(d, ["R.sha256", "M.sha256", "W.sha256"], {"R.md", "R.json", "M.txt"})
        assert any("several files" in p for p in problems) and any("does not match" in p for p in problems) and withheld == 1 and matched == 0, problems
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_protocol_review_files_each_have_their_own_sidecar():
    fh = ROOT / "experiments" / "final_heldout_protocol"
    for name in ("PROTOCOL_REVIEW.md", "PROTOCOL_REVIEW_FACTS.json"):
        p = fh / name
        assert hashlib.sha256(p.read_bytes()).hexdigest() == p.with_suffix(".sha256").read_text(encoding="ascii").strip(), name
    assert not (fh / "PROTOCOL_REVIEW.json").exists()


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
