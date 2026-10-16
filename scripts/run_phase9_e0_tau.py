"""E0 step 4 (NOT AUTHORIZED YET): TRAIN-only alert threshold tau_S per seed, then lock (write + hash).

Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 protocol halted at S6.
Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under the original protocol.

Refuses to run unless the E0 addendum hash, a passing E0 validation record and the sources recorded in it all verify. Uses only real TRAIN days
(<= 121); VAL is not touched. If no admissible tau exists the record says so and the script stops (no replacement rule).
"""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.coordination import frozen as F
from src.coordination.data import get_real
from src.coordination.e0 import (ADDENDUM, GAMMA_E0, TAU_N_DAYS, TAU_TARGET, W_E0, record_scores, require_e0_gates, select_tau, tau_days)
from src.coordination.realrun import BANNER, load_locked_context, out_dir, run_discovery
from src.coordination.scoring import ArrivalTable


def main() -> int:
    cfg = load_config()
    od = out_dir(cfg)
    val = require_e0_gates(od)
    locked, tables, gates = load_locked_context(cfg)
    real = get_real(cfg)
    days = tau_days()
    assert len(days) == TAU_N_DAYS and max(days) <= 121
    table = ArrivalTable(real.view())
    add_sha = hashlib.sha256((pathlib.Path(__file__).resolve().parent.parent / ADDENDUM).read_bytes()).hexdigest()
    bad = False
    for S in F.SEEDS:
        p = od / f"e0_tau_seed{S}.json"
        if p.exists():
            raise RuntimeError(f"{p.name} already exists: tau is calibrated exactly once")
        out = run_discovery(real, tables[(W_E0, GAMMA_E0)], W_E0, GAMMA_E0, S, days, table=table)
        sel = select_tau(record_scores(out), len(days), TAU_TARGET)
        rec = {"banner": BANNER, "seed": S, "train_days": days, "W": W_E0, "gamma": GAMMA_E0, "addendum_sha256": add_sha,
               "validation_record_sha256": hashlib.sha256((od / "validation_E0.json").read_bytes()).hexdigest(), **sel}
        txt = json.dumps(rec, indent=1)
        p.write_text(txt, encoding="utf-8")
        (od / f"e0_tau_seed{S}.sha256").write_text(hashlib.sha256(txt.encode("utf-8")).hexdigest())
        print(f"seed {S}: feasible={sel['feasible']} tau={sel.get('tau')} rate={sel.get('realized_rate_per_day')} tied={sel.get('n_records_tied_at_tau')}", flush=True)
        bad |= not sel["feasible"]
    return 6 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
