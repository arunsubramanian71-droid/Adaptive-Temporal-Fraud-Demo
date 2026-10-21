"""Phase 9b: synthetic recovery benchmark with a corrected interference gate (experiments/phase9b_recovery_benchmark/PREREGISTRATION_9B.md).

  .venv/Scripts/python.exe -B scripts/run_phase9b_benchmark.py

Run-once. Refuses to run unless the pre-registration matches its hashed sidecar and the Phase 9 gates (G0, timing pilot), locked configuration and
null tables verify. Step 1, for every seed: generate the benchmark in memory (generator seed 70000+S), compute the risk stream on it and evaluate
G1, G2, G3, G5 and G4-9b (plus the original G4 quantity, descriptively). Any failure stops the run before any detector output is computed.
Step 2: discovery and track-level AUPRC for A2, B1, B2, B3a, B3b, B0 (and A5 descriptively). Step 3: G-cal on the clean real validation days with the
TRAIN-locked tau. Writes experiments/phase9b_recovery_benchmark/outputs/{results.json, RESULTS_9B.md} (+ sidecars). Days <= 151 only; no
row-level array is written to disk.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import sys
import time

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

from src.config import load_config  # noqa: E402
from src.coordination import frozen as F  # noqa: E402
from src.coordination.benchmark import check_manifest, manifest_json, plant, sha256_text  # noqa: E402
from src.coordination.data import get_real  # noqa: E402
from src.coordination.e0 import record_scores  # noqa: E402
from src.coordination.metrics import GroupIndex, annotate, method_records, track_ap  # noqa: E402
from src.coordination.pipeline import discover  # noqa: E402
from src.coordination.realrun import load_locked_context, run_discovery  # noqa: E402
from src.coordination.risk import ckpt_sha256, risk_stream  # noqa: E402
from src.coordination.runner import exp_dir  # noqa: E402
from src.coordination.scoring import ArrivalTable  # noqa: E402

EXP = ROOT / "experiments" / "phase9b_recovery_benchmark"
OUT = EXP / "outputs"
PRE = EXP / "PREREGISTRATION_9B.md"
GEN_SEED = 70000
W = F.PRIMARY_W
DAYS = list(range(F.EVAL_PERIOD[0], F.EVAL_PERIOD[1] + 1))
COMPARED = {"B1": "B1_naive", "B2": "A1_S", "B3a": "B3a_riskfirst", "B3b": "A4_Rp"}
DESCRIPTIVE = {"B0": "B0_random", "A5": "A5_STRp"}
REGIMES = {"alpha0": 0.0, "alpha0.5": 0.5, "alpha1": 1.0, "all": None}


def write(path: pathlib.Path, data: bytes):
    path.write_bytes(data)
    path.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))


def within_day_pct(logit, day):
    out = np.zeros(len(logit))
    for d in np.unique(day):
        m = day == d
        out[m] = rankdata(logit[m], method="average") / m.sum()
    return out


def gates_for(S, real, cfg, d9):
    aug, x_src, man, is_pl, rg = plant(real, F.EVAL_PERIOD, F.EVAL_COUNTS, GEN_SEED + S, is_eval=True, pool_min_day=112)
    g5 = check_manifest(aug, man, rg)
    aug.X = real.X[x_src]
    t0 = time.time()
    risk = risk_stream(aug, cfg, S, torch.device("cpu"), snapshot_days=set(range(119, 152)))
    rc = np.load(d9 / "risk" / f"risk_clean_seed{S}.npz")
    gmeta = {g["group_id"]: g for g in man["groups"]}

    def frac(pred):
        sel = np.array([(r >= 0) and pred(gmeta[int(r)]) for r in rg])
        return float((risk.pct[sel] > 0.99).mean()), int(sel.sum())
    f0, n0 = frac(lambda g: g["kind"] == "coordinated" and g["regime_alpha"] == 0.0)
    f1, n1 = frac(lambda g: g["kind"] == "coordinated" and g["regime_alpha"] == 1.0)
    real_rows = np.nonzero((~is_pl) & (aug.day >= F.EVAL_PERIOD[0]))[0]
    pos = {int(t): i for i, t in enumerate(real.txn_id.tolist())}
    ci = np.array([pos[int(t)] for t in aug.txn_id[real_rows].tolist()])
    clean_pct, clean_logit = rc["pct"][ci], rc["logit"][ci]
    g4_as_defined = float(np.abs(risk.pct[real_rows] - clean_pct).mean() * 100)
    day_r = aug.day[real_rows]
    g4_9b = float(np.abs(within_day_pct(risk.logit[real_rows], day_r) - within_day_pct(clean_logit, day_r)).mean() * 100)
    g = {"seed": S, "generator_seed": GEN_SEED + S, "manifest_sha256": sha256_text(manifest_json(man)), "checkpoint_sha256": ckpt_sha256(cfg, S),
         "n_rows": int(len(aug.day)), "n_planted": int(is_pl.sum()), "planted_fraction": float(man["planted_fraction"]),
         "n_coordinated": int(man["n_coordinated"]), "n_decoy": int(man["n_decoy"]), "risk_seconds": round(time.time() - t0, 1),
         "G1_alpha0_top1_fraction": f0, "G1_n": n0, "G1_pass": bool(f0 <= F.G1_MAX_TOP1),
         "G2_alpha1_top1_fraction": f1, "G2_n": n1, "G2_pass": bool(f1 > 0 and f1 >= F.G2_MIN_RATIO * f0),
         "G3_pass": bool(man["planted_fraction"] <= F.PLANT_CAP), "G5_pass": bool(g5["passed"]),
         "G4_9b_mean_abs_delta_pct_points": g4_9b, "G4_9b_pass": bool(g4_9b <= F.G4_MAX_DELTA_PCT_POINTS),
         "G4_original_formula_mean_abs_delta_pct_points_descriptive": g4_as_defined,
         "G4_original_formula_would_pass_descriptive": bool(g4_as_defined <= F.G4_MAX_DELTA_PCT_POINTS)}
    g["all_registered_gates_pass"] = bool(g["G1_pass"] and g["G2_pass"] and g["G3_pass"] and g["G5_pass"] and g["G4_9b_pass"])
    return g, (aug, man, risk)


def main() -> int:
    sidecar = PRE.with_suffix(".sha256")
    if not sidecar.exists() or hashlib.sha256(PRE.read_bytes()).hexdigest() != sidecar.read_text().strip():
        print("error: the pre-registration does not match its hashed sidecar", file=sys.stderr)
        return 2
    if (OUT / "results.json").exists():
        print("error: Phase 9b has already been run (run-once)", file=sys.stderr)
        return 2
    cfg = load_config()
    locked, tables, gates0 = load_locked_context(cfg)
    gamma = float(locked["gamma_per_W"][str(W)])
    d9 = exp_dir(cfg)
    real = get_real(cfg, with_X=True)
    assert int(real.day.max()) <= 151, "the real-data cache must be truncated at day 151"
    t0 = time.time()
    rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "preregistration_sha256": sidecar.read_text().strip(),
           "locked_config": {"W": W, "gamma": gamma, "null_tables_sha256": locked["null_tables_sha256"]}, "phase9_gates": gates0,
           "test_data_read": False, "max_day": int(real.day.max())}
    # ---- step 1: validity gates for every seed, before any detector output
    gate_rows, kept = {}, {}
    for S in F.SEEDS:
        g, k = gates_for(S, real, cfg, d9)
        gate_rows[str(S)] = g
        kept[S] = k
        print(f"seed {S}: G1 {g['G1_pass']} G2 {g['G2_pass']} G3 {g['G3_pass']} G5 {g['G5_pass']} G4-9b {g['G4_9b_pass']} ({g['G4_9b_mean_abs_delta_pct_points']:.4f} pts; "
              f"original formula {g['G4_original_formula_mean_abs_delta_pct_points_descriptive']:.4f})", flush=True)
    rec["gates"] = gate_rows
    rec["all_gates_pass"] = all(g["all_registered_gates_pass"] for g in gate_rows.values())
    OUT.mkdir(parents=True, exist_ok=True)
    if not rec["all_gates_pass"]:
        rec["outcome"] = "STOPPED: a registered validity gate failed; no detector output was computed (hard stop, no regeneration)"
        write(OUT / "results.json", (json.dumps(rec, indent=1) + "\n").encode("utf-8"))
        print(rec["outcome"])
        return 3
    # ---- step 2: discovery and track-level AUPRC
    ap = {}
    for S in F.SEEDS:
        aug, man, risk = kept[S]
        gi = GroupIndex(man)
        v = aug.view()
        t1 = time.time()
        out = discover(v, ArrivalTable(v), tables[(W, gamma)], W, gamma, seed=S, days=DAYS, risk=risk, sets=("comm", "star", "rand", "riskfirst"))
        for name, method in {"A2": "A2_ST", **COMPARED, **DESCRIPTIVE}.items():
            recs = method_records(out, method)
            annotate(recs, gi)
            for rn, regime in REGIMES.items():
                r = track_ap(recs, gi, regime, F.JACCARD_PRIMARY)
                ap.setdefault(name, {}).setdefault(rn, {})[str(S)] = {"ap": r["ap"], "n_recovered": r["n_recovered"], "n_groups": r["n_groups"], "n_tracks": r["n_tracks"]}
        print(f"seed {S}: discovery {time.time() - t1:.0f}s; alpha0 AP " + ", ".join(f"{n} {ap[n]['alpha0'][str(S)]['ap']:.3f}" for n in ap), flush=True)
    rec["track_auprc"] = ap
    # ---- decision rule (design 15) on alpha = 0
    vals = {n: [ap[n]["alpha0"][str(S)]["ap"] for S in F.SEEDS] for n in ap}
    comp = {}
    for b in COMPARED:
        wins = sum(a > x for a, x in zip(vals["A2"], vals[b]))
        diff = float(np.mean(vals["A2"]) - np.mean(vals[b]))
        bound = max(float(np.std(vals["A2"], ddof=1)), float(np.std(vals[b], ddof=1)))
        comp[b] = {"A2_wins_of_3": wins, "mean_diff": diff, "larger_sd": bound, "met": bool(wins >= 2 and diff > 0 and diff > bound)}
    # ---- step 3: G-cal on the clean real validation days with the TRAIN-locked tau
    gcal = {}
    for S in F.SEEDS:
        tau = json.loads((d9 / "exploratory_realdata" / f"e0_tau_seed{S}.json").read_text(encoding="utf-8"))
        assert tau["W"] == W and float(tau["gamma"]) == gamma and tau["feasible"]
        sc = record_scores(run_discovery(real, tables[(W, gamma)], W, gamma, S, DAYS))
        per_day = float((sc >= tau["tau"]).sum() / len(DAYS))
        gcal[str(S)] = {"tau": tau["tau"], "false_alerts_per_day": per_day, "limit": F.GCAL_FACTOR * F.ALERT_TARGET_PER_DAY,
                        "pass": bool(per_day <= F.GCAL_FACTOR * F.ALERT_TARGET_PER_DAY)}
    rec["g_cal"] = gcal
    # ---- S8 non-diagnostic check
    b1_easy = all(np.mean([ap["B1"][rn][str(S)]["ap"] for S in F.SEEDS]) >= 0.95 for rn in ("alpha0", "alpha0.5", "alpha1"))
    def near_b0(n, rn):
        a = [ap[n][rn][str(S)]["ap"] for S in F.SEEDS]
        b = [ap["B0"][rn][str(S)]["ap"] for S in F.SEEDS]
        return np.mean(a) - np.mean(b) <= max(np.std(a, ddof=1), np.std(b, ddof=1))
    too_hard = all(near_b0(n, rn) for n in ("A2", *COMPARED, "A5") for rn in ("alpha0", "alpha0.5", "alpha1"))
    rec["S8_non_diagnostic"] = {"B1_too_easy_everywhere": bool(b1_easy), "every_method_within_noise_of_B0_everywhere": bool(too_hard)}
    rec["comparisons_alpha0"] = comp
    gcal_ok = all(g["pass"] for g in gcal.values())
    if b1_easy or too_hard:
        rec["outcome"] = "NON-DIAGNOSTIC (S8): no conclusion is drawn"
    else:
        rec["outcome"] = "H5a-9b SUPPORTED" if all(c["met"] for c in comp.values()) and gcal_ok else "H5a-9b NOT SUPPORTED"
    rec["seconds"] = round(time.time() - t0, 1)
    write(OUT / "results.json", (json.dumps(rec, indent=1) + "\n").encode("utf-8"))
    print(rec["outcome"], json.dumps(comp), json.dumps({k: v["false_alerts_per_day"] for k, v in gcal.items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
