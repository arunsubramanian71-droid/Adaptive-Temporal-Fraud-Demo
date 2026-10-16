"""E0 step 5 (NOT AUTHORIZED YET): degree-preserving null only, K = 100 coherent null histories per seed, descriptive per-seed results.

Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 protocol halted at S6.
Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under the original protocol.

Refuses to run unless: the addendum hash, a passing E0 validation record (sources unchanged), locked tau files for every seed (hash-verified,
feasible), and the Tier 0 record all verify. The observed run must reproduce the Tier 0 output hash. No threshold, no verdict.
"""
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import load_config
from src.coordination import frozen as F
from src.coordination.data import get_real
from src.coordination.e0 import (GAMMA_E0, K_E0, W_E0, render_report, require_e0_gates, run_observed, run_seed)
from src.coordination.lint import lint_text
from src.coordination.realrun import hash_output, load_locked_context, out_dir
from src.coordination.scoring import ArrivalTable


def main() -> int:
    cfg = load_config()
    od = out_dir(cfg)
    require_e0_gates(od)
    locked, tables, gates = load_locked_context(cfg)
    tier0 = json.loads((od / "tier0_summary.json").read_text(encoding="utf-8"))
    taus = {}
    for S in F.SEEDS:
        p = od / f"e0_tau_seed{S}.json"
        if not p.exists() or hashlib.sha256(p.read_bytes()).hexdigest() != (od / f"e0_tau_seed{S}.sha256").read_text().strip():
            raise RuntimeError(f"tau lock missing or modified for seed {S}")
        taus[S] = json.loads(p.read_text(encoding="utf-8"))
        if not taus[S]["feasible"]:
            raise RuntimeError(f"no admissible tau for seed {S}: E0 not computable")
    if (od / "e0_results.json").exists():
        raise RuntimeError("e0_results.json already exists: E0 is run exactly once")
    real = get_real(cfg)
    table = ArrivalTable(real.view())
    nt = tables[(W_E0, GAMMA_E0)]
    results = {}
    t0 = time.time()
    for S in F.SEEDS:
        obs_hash = hash_output(run_observed(real, nt, S, table))
        if obs_hash != tier0["configs"][f"W{W_E0}_seed{S}"]["output_hash"]:
            raise RuntimeError(f"observed E0 run does not reproduce the Tier 0 output hash for seed {S}")
        results[S] = run_seed(real, nt, S, taus[S]["tau"], table, K=K_E0)
        results[S]["observed_hash_matches_tier0"] = True
        print(f"seed {S} done ({time.time() - t0:.0f}s)", flush=True)
    with open(od / "e0_results.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results, "taus": taus}, fh, indent=1)
    rep = render_report(results, taus)
    bad = lint_text(rep)
    assert not bad, bad
    (od / "E0_REPORT.md").write_text(rep, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
