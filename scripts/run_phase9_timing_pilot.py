"""Phase 9 timing-pilot gate (design section 18, approved R-E / U4).

Protocol (fixed BEFORE any timing was observed): one TRAIN window ending day 121 (no outcome information involved). For each of the
9 frozen (W, gamma) configurations: 3 repeats of [window rows -> aggregate -> Louvain -> connectivity split -> eligibility -> Q]
(seed 42), plus 3 repeats of the same on a degree-preserving rewired window (null-calibration/E0 run), plus a cumulative-graph run
(ablation A10). Gate:
  * STOP if the MEDIAN per-window community-detection time over the real-window runs exceeds 60 s;
  * STOP if the projected total CPU runtime exceeds 24 h, where
        projected = 1.5 x sum(planned Louvain runs x measured time of the matching configuration)
    (the 1.5 factor is a declared allowance for non-Louvain CPU work: graph construction, scoring, tracking, I/O).
No fallback is pre-authorized: if a gate trips, this script reports it and the run STOPS for a rescoping decision.
"""
import json
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from src.config import load_config
from src.coordination import frozen as F
from src.coordination.data import get_real
from src.coordination.graphs import aggregate, louvain_candidates, rewire, window_rows

PILOT_DAY = 121
REPEATS = 3
OVERHEAD = 1.5
GATE_MEDIAN_S = 60.0
GATE_TOTAL_S = 24 * 3600.0


def timed(fn):
    t0 = time.perf_counter()
    r = fn()
    return time.perf_counter() - t0, r


def main() -> int:
    cfg = load_config()
    out_dir = cfg.experiments_dir / "phase9_coordinated_fraud"
    real = get_real(cfg)
    v = real.view()
    runs = {}
    all_real = []
    for W in F.WINDOWS:
        for g in F.GAMMAS:
            ts, tn, info = [], [], {}
            for r in range(REPEATS):
                def run_real():
                    e = window_rows(v, PILOT_DAY, W)
                    gr = aggregate(e)
                    return gr, louvain_candidates(gr, g, seed=42)
                dt, (gr, cands) = timed(run_real)
                ts.append(dt)
                info = {"n_edges": int(len(gr.card)), "n_cards": len(gr.kc), "n_infra": len(gr.ki), "n_candidates": len(cands)}
                rng = np.random.default_rng(F.NULL_SEED_BASE + 42 + r)

                def run_null():
                    e = rewire(window_rows(v, PILOT_DAY, W), rng)
                    gr2 = aggregate(e)
                    return louvain_candidates(gr2, g, seed=42)
                dtn, _ = timed(run_null)
                tn.append(dtn)
            runs[f"W{W}_g{g}"] = {"real_s": ts, "null_s": tn, "real_median_s": statistics.median(ts), "null_median_s": statistics.median(tn), **info}
            all_real += ts
            print(f"  W={W:>2} gamma={g}: real {statistics.median(ts):6.2f}s  null {statistics.median(tn):6.2f}s  {info}", flush=True)
    # cumulative graph (ablation A10), gamma = 1.0
    tc = []
    for r in range(REPEATS):
        dt, _ = timed(lambda: louvain_candidates(aggregate(window_rows(v, PILOT_DAY, None)), 1.0, seed=42))
        tc.append(dt)
    cum_med = statistics.median(tc)
    print(f"  cumulative graph (A10): {cum_med:.2f}s", flush=True)

    # planned Louvain runs (fixed before timing; see design sections 7, 11.6, 14, 15, 10)
    def real_t(W, g):
        return runs[f"W{W}_g{g}"]["real_median_s"]

    def null_t(W, g):
        return runs[f"W{W}_g{g}"]["null_median_s"]

    def mean_over_gamma(fn, W):
        return statistics.mean(fn(W, g) for g in F.GAMMAS)

    plan = []
    for W in F.WINDOWS:
        for g in F.GAMMAS:
            plan.append(("null_calibration", f"W{W}_g{g}", F.K_NULL * F.N_NULL_WINDOWS, null_t(W, g)))
            plan.append(("dev_discovery", f"W{W}_g{g}", len(range(F.DEV_PERIOD[0], F.DEV_PERIOD[1] + 1)) * len(F.SEEDS), real_t(W, g)))
    for W in F.WINDOWS:
        n_eval = 30 * len(F.SEEDS)
        plan.append(("eval_discovery", f"W{W}", n_eval, mean_over_gamma(real_t, W)))
        plan.append(("real_clean_discovery", f"W{W}", n_eval, mean_over_gamma(real_t, W)))
    plan.append(("E0_degree_preserving_null", "W7", 4 * F.K_NULL_E0 * len(F.SEEDS), mean_over_gamma(null_t, 7)))
    plan.append(("A10_cumulative", "cumulative", 30 * len(F.SEEDS), cum_med))
    plan.append(("tau_calibration_TRAIN_background", "W7", 30, mean_over_gamma(real_t, 7)))
    rows = [{"category": c, "config": k, "n_runs": n, "seconds_per_run": s, "total_s": n * s} for c, k, n, s in plan]
    louvain_total = sum(r["total_s"] for r in rows)
    projected = OVERHEAD * louvain_total
    median_real = statistics.median(all_real)
    n_runs_total = sum(r["n_runs"] for r in rows)
    rec = {"design": "section 18 timing-pilot gate", "pilot_day": PILOT_DAY, "repeats": REPEATS, "louvain_seed": 42,
           "per_config": runs, "cumulative_graph_median_s": cum_med, "median_real_window_s": median_real,
           "max_real_window_s": max(all_real), "planned_runs": rows, "n_planned_runs": n_runs_total,
           "louvain_total_s": louvain_total, "overhead_factor": OVERHEAD, "projected_total_cpu_s": projected,
           "projected_total_cpu_h": projected / 3600.0, "gate_median_s": GATE_MEDIAN_S, "gate_total_h": GATE_TOTAL_S / 3600.0,
           "gate_median_tripped": bool(median_real > GATE_MEDIAN_S), "gate_total_tripped": bool(projected > GATE_TOTAL_S)}
    rec["passed"] = not (rec["gate_median_tripped"] or rec["gate_total_tripped"])
    with open(out_dir / "timing_pilot.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=1)
    print(f"median real-window time {median_real:.2f}s (gate {GATE_MEDIAN_S}s) | planned runs {n_runs_total} | "
          f"projected CPU {projected/3600:.2f}h (gate 24h) -> {'PASSED' if rec['passed'] else 'TRIPPED: STOP and request rescoping decision'}")
    return 0 if rec["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
