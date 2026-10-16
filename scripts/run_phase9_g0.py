"""Phase 9 hard gate G0 (TEST-free segmentation equivalence).
Writes experiments/phase9_coordinated_fraud/segmentation_equivalence_gate.json.

  python -u scripts/run_phase9_g0.py
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.coordination.g0 import run_g0


def main() -> int:
    cfg = load_config()
    out = cfg.experiments_dir / "phase9_coordinated_fraud"
    rec = run_g0(cfg, out)
    for k, v in rec["criteria"].items():
        brief = {a: b for a, b in v.items() if a not in ("passed", "cases", "label_counts")}
        print(f"  [{'PASS' if v['passed'] else 'FAIL'}] {k}: " + json.dumps(brief)[:400])
    print(f"G0 {'PASSED' if rec['passed'] else 'FAILED'} in {rec.get('elapsed_s')}s; "
          f"test_file_opened={rec['test_file_opened']}; shared_graph_loaded={rec['shared_graph_file_loaded']}")
    return 0 if rec["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
