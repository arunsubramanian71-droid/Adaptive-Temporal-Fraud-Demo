"""Phase 9 development benchmark + pre-registered configuration selection (design 11.6).

For each seed S: plant the DEV benchmark (generator seed 10000+S, planted in days 62-121, pools from days < planted day), run the
risk-free S+T discovery (A2) for each of the 9 frozen (W, gamma) configurations on days 62-121, and score track-level AUPRC at J>=0.5
over all planted coordinated groups. gamma is selected PER W by the highest mean dev AUPRC across the 3 seeds (ties -> smaller gamma).
Primary analyses use W = 7; W = 3 and 14 are pre-registered sensitivity analyses. The risk component is NOT tuned. The selection is
written to locked_config.json (+ sha256) BEFORE the evaluation benchmark may be generated.

  python -u scripts/run_phase9_dev_selection.py
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
from src.coordination.benchmark import check_manifest, plant
from src.coordination.data import get_real
from src.coordination.metrics import GroupIndex, annotate, method_records, track_ap
from src.coordination.nullmodel import load_tables
from src.coordination.pipeline import discover
from src.coordination.runner import exp_dir, load_sealed, locked_config_path, require_gates, save_benchmark
from src.coordination.scoring import ArrivalTable


def main() -> int:
    cfg = load_config()
    gates = require_gates(cfg)
    out = exp_dir(cfg)
    if locked_config_path(cfg).exists():
        raise RuntimeError("locked_config.json already exists: dev selection is run exactly once")
    real = get_real(cfg)
    tables = load_tables(out / "null_tables.pkl")
    days = list(range(F.DEV_PERIOD[0], F.DEV_PERIOD[1] + 1))
    results: dict = {}
    bench_rec = {}
    t0 = time.time()
    for S in F.SEEDS:
        aug, x_src, man, is_pl, rg = plant(real, F.DEV_PERIOD, F.DEV_COUNTS, F.GENERATOR_SEED_DEV + S, is_eval=False, pool_min_day=1)
        g5 = check_manifest(aug, man, rg)
        info = save_benchmark(cfg, "dev", S, aug, x_src, man, is_pl, rg)
        bench_rec[S] = {**info, "planted_fraction": man["planted_fraction"], "G3_pass": man["planted_fraction"] <= F.PLANT_CAP,
                        "G5": g5, "scale_factor": man["scale_factor"], "n_coordinated": man["n_coordinated"], "n_decoy": man["n_decoy"]}
        print(f"seed {S}: dev planted {man['n_planted']} txns ({man['planted_fraction']:.3%}), G5 {g5['passed']}", flush=True)
        assert bench_rec[S]["G3_pass"] and g5["passed"], "dev benchmark validity gate failed (G3/G5): STOP"
        man, _, _ = load_sealed(cfg, "dev", S)           # evaluation side opens the sealed manifest only now, after generation
        gi = GroupIndex(man)
        view = aug.view()
        table = ArrivalTable(view)
        for W in F.WINDOWS:
            for g in F.GAMMAS:
                t1 = time.time()
                res = discover(view, table, tables[(W, g)], W, g, seed=S, days=days, risk=None, sets=("comm",))
                recs = method_records(res, "A2_ST")
                annotate(recs, gi)
                ap = track_ap(recs, gi, regime=None, thr=F.JACCARD_PRIMARY)
                recs1 = method_records(res, "A1_S")
                ap1 = track_ap(recs1, gi, regime=None, thr=F.JACCARD_PRIMARY)
                results.setdefault(f"W{W}_g{g}", {})[S] = {"A2_ST_ap": ap["ap"], "A2_recovered": ap["n_recovered"], "n_groups": ap["n_groups"],
                                                         "A1_S_ap": ap1["ap"], "n_tracks": ap["n_tracks"], "seconds": round(time.time() - t1, 1)}
                print(f"  seed {S} W={W} gamma={g}: A2 AP {ap['ap']:.3f} ({ap['n_recovered']}/{ap['n_groups']} groups), A1 AP {ap1['ap']:.3f}", flush=True)

    selected = {}
    table_rows = {}
    for W in F.WINDOWS:
        means = {g: float(np.mean([results[f"W{W}_g{g}"][S]["A2_ST_ap"] for S in F.SEEDS])) for g in F.GAMMAS}
        best = max(F.GAMMAS, key=lambda g: (round(means[g], 12), -g))      # ties -> smaller gamma
        selected[str(W)] = best
        table_rows[str(W)] = means
    locked = {"locked_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "primary_W": F.PRIMARY_W, "gamma_per_W": selected,
              "selection_metric": "mean over seeds of dev track-level AUPRC@J>=0.5 of A2 (S+T), all planted coordinated groups",
              "dev_mean_ap_by_W_gamma": table_rows, "gates": gates, "dev_benchmarks": bench_rec, "frozen_registry": "DESIGN section 15.1",
              "design_sha256": hashlib.sha256(open(out / "DESIGN_BEFORE_EXECUTION.md", "rb").read()).hexdigest(),
              "null_tables_sha256": hashlib.sha256(open(out / "null_tables.pkl", "rb").read()).hexdigest()}
    with open(out / "dev_selection.json", "w", encoding="utf-8") as fh:
        json.dump({"results": results, "locked": locked}, fh, indent=1)
    p = locked_config_path(cfg)
    p.write_text(json.dumps(locked, indent=1), encoding="utf-8")
    (out / "locked_config.sha256").write_text(hashlib.sha256(open(p, "rb").read()).hexdigest())
    print(f"LOCKED: gamma per W = {selected} (primary W={F.PRIMARY_W}); elapsed {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
