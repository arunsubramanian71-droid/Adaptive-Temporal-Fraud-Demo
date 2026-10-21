"""Check that the environment can run the demonstration, the tests and the dashboard (no data is read, no model is trained).

  .venv/Scripts/python.exe -B scripts/check_setup.py

Reports each requirement from requirements.txt (installed version versus the pinned one), the optional neural-network stack, the presence of the
data files the research experiments need (existence only; nothing is opened, and held-out-named files are never touched), and the folders the
scripts write to. Exit code 0 if everything required for the demonstration is available, 1 otherwise.
"""
from __future__ import annotations

import importlib.metadata as md
import pathlib
import re
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON_TESTED = (3, 12)
IMPORT_NAMES = {"scikit-learn": "sklearn", "pyyaml": "yaml"}
RESEARCH_FILES = ["data/raw/train_transaction.csv", "data/raw/train_identity.csv", "data/features/train_features.parquet", "data/features/val_features.parquet",
                  "data/processed/graph/hetero_graph.pt", "experiments/phase9_coordinated_fraud/cache/real_le151.npz"]


def requirements(path: pathlib.Path):
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.-]+)==([^\s#]+)", line.strip())
        if m:
            yield m.group(1), m.group(2)


def main() -> int:
    ok = True
    v = sys.version_info
    print(f"Python {v.major}.{v.minor}.{v.micro} (tested with {PYTHON_TESTED[0]}.{PYTHON_TESTED[1]}): {'ok' if (v.major, v.minor) >= PYTHON_TESTED else 'OLDER than tested; may not work'}")
    ok &= (v.major, v.minor) >= (3, 10)
    print("\nrequirements.txt (needed for the demonstration, tests, scorer and dashboard):")
    for name, pinned in requirements(ROOT / "requirements.txt"):
        try:
            have = md.version(name)
            status = "ok" if have == pinned else f"version differs (pinned {pinned}); usually fine, rerun the tests if unsure"
        except md.PackageNotFoundError:
            have, status = "-", f"MISSING: run  python -m pip install -r requirements.txt"
            ok = False
        print(f"  {name:<14} {have:<10} {status}")
    print("\nrequirements-gnn.txt (only for the neural-network experiments):")
    for name, pinned in requirements(ROOT / "requirements-gnn.txt"):
        try:
            print(f"  {name:<16} {md.version(name)}  (pinned {pinned})")
        except md.PackageNotFoundError:
            print(f"  {name:<16} not installed (not needed for the demonstration)")
    print("\nresearch data (existence only; needed to re-run experiments, not for the demonstration):")
    for rel in RESEARCH_FILES:
        print(f"  {'present' if (ROOT / rel).exists() else 'absent ':<8} {rel}")
    print("  held-out-named files are never opened by any check or demonstration command.")
    print("\nwritable output locations:")
    for rel in ("demo_output", "submission"):
        d = ROOT / rel
        try:
            d.mkdir(exist_ok=True)
            probe = d / ".write_probe"
            probe.write_text("x", encoding="utf-8")
            probe.unlink()
            print(f"  ok  {rel}/")
        except OSError as e:
            print(f"  NOT WRITABLE {rel}/ ({e})")
            ok = False
    print("\nSETUP", "OK for the demonstration" if ok else "INCOMPLETE: see messages above")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
