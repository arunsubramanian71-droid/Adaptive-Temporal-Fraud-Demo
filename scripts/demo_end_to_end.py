"""End-to-end SOFTWARE demonstration on a small synthetic fixture (no dataset, no GPU, no TEST data; about a minute on a laptop).

  .venv/Scripts/python.exe -B scripts/demo_end_to_end.py [--out-dir demo_output] [--seed 0]

It runs the project's real modules (graph features, XGBoost protocol, metrics, history-based scoring, explanations, candidate-community
discovery) on synthetic data and prints each stage. It demonstrates mechanics and reproduces no research experiment. To see the
recorded research results instead, read reports/PROJECT_SYNTHESIS_REPORT.md and submission/README.md.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.demo.pipeline import run_demo  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="demo_output")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    out = pathlib.Path(a.out_dir)
    if "experiments" in out.resolve().parts:
        print("error: refusing to write demo output inside experiments/ (it would mix synthetic files with research artifacts)", file=sys.stderr)
        return 2
    run_demo(out, seed=a.seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
