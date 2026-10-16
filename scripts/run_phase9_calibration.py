"""Phase 9 null calibration (design 7.1/7.2/15.1): 10 rewirings x 8 sampled TRAIN windows per (W, gamma); 9 configurations; locked afterwards.
Real clean TRAIN data only; seed-independent (rewiring seed 30000+42, Louvain seed 42).

  python -u scripts/run_phase9_calibration.py
"""
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.coordination import frozen as F
from src.coordination.data import get_real
from src.coordination.nullmodel import calibrate, calibration_days, save_tables
from src.coordination.runner import exp_dir, require_gates
from src.coordination.scoring import ArrivalTable


def main() -> int:
    cfg = load_config()
    gates = require_gates(cfg)
    real = get_real(cfg)
    v = real.view()
    table = ArrivalTable(v)
    days = calibration_days()
    out = exp_dir(cfg)
    tables, summ, t0 = {}, {}, time.time()
    for W in F.WINDOWS:
        for g in F.GAMMAS:
            t1 = time.time()
            nt = calibrate(v, table, W, g, days)
            tables[(W, g)] = nt
            summ[f"W{W}_g{g}"] = {**nt.summary(), "seconds": round(time.time() - t1, 1)}
            print(f"  calibrated W={W} gamma={g} in {summ[f'W{W}_g{g}']['seconds']}s", flush=True)
    save_tables(tables, out / "null_tables.pkl")
    with open(out / "null_calibration_summary.json", "w", encoding="utf-8") as fh:
        json.dump({"gates": gates, "calibration_days": days, "k_null": F.K_NULL, "n_windows": F.N_NULL_WINDOWS,
                   "louvain_seed": 42, "rewiring_seed_base": F.NULL_SEED_BASE + 42, "total_seconds": round(time.time() - t0, 1),
                   "configs": summ}, fh, indent=1)
    print(f"calibration complete in {time.time() - t0:.1f}s; days {days}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
