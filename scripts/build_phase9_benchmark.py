"""Phase 9 evaluation benchmark generation (design 11): run-once, only AFTER G0, the timing pilot, and the locked configuration exist.

  python -u scripts/build_phase9_benchmark.py            # eval benchmarks for seeds 42, 123, 2024
Generator seed 20000+S; planted in days 122-151 (formation <= 146); pools from real days 112..(planted day - 1); counts 36+18 scaled to
the <=5% cap. Writes detector-visible arrays and a SEALED manifest/ground truth; computes validity gates G3 and G5 (G1, G2, G4 need
the risk streams: scripts/run_phase9_benchmark_gates.py).
"""
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.coordination import frozen as F
from src.coordination.benchmark import check_manifest, plant
from src.coordination.data import get_real
from src.coordination.runner import bench_dir, exp_dir, load_locked, require_gates, save_benchmark


def main() -> int:
    cfg = load_config()
    require_gates(cfg)
    load_locked(cfg)                                   # raises unless the dev selection was locked first
    real = get_real(cfg)
    rec = {}
    for S in F.SEEDS:
        if (bench_dir(cfg, "eval", S) / "arrays.npz").exists():
            raise RuntimeError(f"eval benchmark for seed {S} already exists: it is generated exactly once")
        aug, x_src, man, is_pl, rg = plant(real, F.EVAL_PERIOD, F.EVAL_COUNTS, F.GENERATOR_SEED_EVAL + S, is_eval=True, pool_min_day=112)
        g5 = check_manifest(aug, man, rg)
        info = save_benchmark(cfg, "eval", S, aug, x_src, man, is_pl, rg)
        rec[S] = {**info, "planted_fraction": man["planted_fraction"], "G3_pass": man["planted_fraction"] <= F.PLANT_CAP, "G5": g5,
                  "scale_factor": man["scale_factor"], "n_planted_before_thinning": man["n_planted_before_thinning"],
                  "n_coordinated": man["n_coordinated"], "n_decoy": man["n_decoy"], "n_syn_net": man["n_syn_net"], "n_syn_dev": man["n_syn_dev"]}
        print(f"seed {S}: eval planted {man['n_planted']} of cap {man['cap_n']} ({man['planted_fraction']:.3%}); G3 {rec[S]['G3_pass']}; G5 {g5['passed']}", flush=True)
        if not (rec[S]["G3_pass"] and g5["passed"]):
            raise RuntimeError("benchmark validity gate (G3/G5) failed: STOP and report (design S6)")
    with open(exp_dir(cfg) / "eval_benchmark_G3_G5.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
