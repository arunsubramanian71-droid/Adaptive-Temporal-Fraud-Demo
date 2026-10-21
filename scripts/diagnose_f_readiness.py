"""Post-result diagnosis of the Amendment 1 module F readiness gate (E, seed 123: window-mean PR-AUC differed from the recorded supplement run by 0.0104 against the registered 0.01).

Question: is that difference a reproducible code difference or run-to-run numerical variation of a trained arm? Validation days 122-151 only, no held-out data, label use exactly as in the
readiness run. The identical arm and seed (burn-in variant, same checkpoint, same seeding) is run N times in one process; the spread among identical runs is compared with the 0.0104 and with
the 0.01 tolerance. The registered tolerance and the registered result are NOT changed; this is a labelled post-hoc diagnostic.

Usage:  .venv/Scripts/python.exe -B scripts/diagnose_f_readiness.py [--arm E] [--seed 123] [--runs 3]
Writes experiments/phase8_state_continuity_supplement/diagnostics/F_readiness_diagnosis_<arm>_seed<seed>.json (+ sidecar).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

from src.adaptive.windows import build_windows  # noqa: E402
from src.config import load_config  # noqa: E402
from src.graph.temporal import build_temporal_dataset, load_raw_arrays  # noqa: E402
from src.heldout.walkforward import run_walk_forward_ext  # noqa: E402
from src.models.temporal_graphsage import TemporalGraphSAGE  # noqa: E402

P8 = ROOT / "experiments" / "phase8_adaptive_drift"
SUP = ROOT / "experiments" / "phase8_state_continuity_supplement" / "outputs"
ARM_DIR = {"F": "frozen", "Ad": "adaptive", "E": "replay_control"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="E", choices=["Ad", "E"])
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu"], help="cpu: identical-run determinism check without GPU kernels (compares only the identical runs with each other)")
    ap.add_argument("--max-windows", type=int, default=5, help="score only the first N windows (CPU check)")
    a = ap.parse_args(argv)
    cfg = load_config()
    dev = torch.device("cpu") if a.device == "cpu" else torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    raw, _graph, _ = load_raw_arrays(cfg)
    y_all = pd.Series(raw["y"], index=raw["txn_id"])
    ds = build_temporal_dataset(raw, 151, dev, cfg)
    tr = ds.txn_day <= cfg.split.train_day_max
    y_tr = raw["y"][:len(tr)][tr]
    pos_weight = torch.tensor(float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), device=dev)
    windows = build_windows(122, 151)[:a.max_windows]
    hp = json.loads((P8 / "shared_init" / f"hparams_seed{a.seed}.json").read_text(encoding="utf-8"))
    runs = []
    for r in range(a.runs):
        torch.manual_seed(a.seed)
        model = TemporalGraphSAGE(raw["X"].shape[1]).to(dev)
        model.load_state_dict(torch.load(P8 / "shared_init" / f"model_checkpoint_initial_seed{a.seed}.pt", map_location=dev))
        opt = torch.optim.Adam(model.parameters(), lr=hp["fine_tune_lr"], weight_decay=1e-5)
        res = run_walk_forward_ext(a.arm, model, ds, windows, opt=opt, pos_weight=pos_weight, fine_tune_epochs=hp["fine_tune_epochs"], base_seed=a.seed,
                                   train_day_min=cfg.split.train_day_min, train_day_max=cfg.split.train_day_max, burn_in_through=windows[0][0] - 1)
        scores = [pd.Series(s, index=ids) for ids, s, _d in res]
        prs = [float(average_precision_score(y_all.loc[ids].to_numpy(), s)) for ids, s, _d in res]
        runs.append({"scores": scores, "window_pr_auc": prs, "mean": float(np.mean(prs))})
        print(f"[run {r + 1}/{a.runs}] {a.arm} seed {a.seed}: window PR-AUC {[round(p, 4) for p in prs]} mean {np.mean(prs):.4f}", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    cpu = a.device == "cpu"
    if cpu:
        pair = [{"runs": [i + 1, j + 1], "window_mean_abs_diff": abs(runs[i]["mean"] - runs[j]["mean"]),
                 "max_abs_score_diff_per_window": [float(np.abs(runs[i]["scores"][k].to_numpy() - runs[j]["scores"][k].loc[runs[i]["scores"][k].index].to_numpy()).max()) for k in range(len(windows))]}
                for i in range(len(runs)) for j in range(i + 1, len(runs))]
        rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "arm": a.arm, "seed": a.seed, "n_identical_runs": a.runs, "device": str(dev),
               "windows_scored": len(windows), "identical_run_window_means": [r["mean"] for r in runs], "identical_run_pairwise": pair,
               "all_identical_runs_bitwise_equal": all(max(p_["max_abs_score_diff_per_window"]) == 0.0 for p_ in pair),
               "reading": "post-hoc, labelled diagnostic on the CPU: if identical-seed runs are bitwise equal here, the run-to-run differences seen on the GPU come from nondeterministic GPU kernels, not from the code or the seeding.",
               "test_data_read": False, "seconds": round(time.time() - t0, 1)}
        out = ROOT / "experiments" / "phase8_state_continuity_supplement" / "diagnostics"
        out.mkdir(parents=True, exist_ok=True)
        data = (json.dumps(rec, indent=1) + chr(10)).encode("utf-8")
        p = out / f"F_readiness_diagnosis_{a.arm}_seed{a.seed}_cpu.json"
        p.write_bytes(data)
        p.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
        print(json.dumps({k: rec[k] for k in ("identical_run_window_means", "all_identical_runs_bitwise_equal")}))
        return 0
    ref = [pd.read_parquet(SUP / "burn_in" / ARM_DIR[a.arm] / f"seed_{a.seed}" / f"predictions_W{k}.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"] for k in range(1, 6)]
    rdy = [pd.read_parquet(SUP / "readiness" / ARM_DIR[a.arm] / f"seed_{a.seed}" / f"predictions_W{k}.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"] for k in range(1, 6)]
    ref_mean = float(np.mean([json.loads((SUP / "burn_in" / ARM_DIR[a.arm] / f"seed_{a.seed}" / f"metrics_W{k}.json").read_text(encoding="utf-8"))["pooled"]["pr_auc"] for k in range(1, 6)]))
    rdy_mean = float(np.mean([json.loads((SUP / "readiness" / ARM_DIR[a.arm] / f"seed_{a.seed}" / f"metrics_W{k}.json").read_text(encoding="utf-8"))["pooled"]["pr_auc"] for k in range(1, 6)]))
    pair = []
    for i in range(len(runs)):
        for j in range(i + 1, len(runs)):
            pair.append({"runs": [i + 1, j + 1], "window_mean_abs_diff": abs(runs[i]["mean"] - runs[j]["mean"]),
                         "max_abs_score_diff_per_window": [float(np.abs(runs[i]["scores"][k].to_numpy() - runs[j]["scores"][k].loc[runs[i]["scores"][k].index].to_numpy()).max()) for k in range(5)]})
    means = [r["mean"] for r in runs]
    all_means = means + [ref_mean, rdy_mean]
    rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "arm": a.arm, "seed": a.seed, "n_identical_runs": a.runs,
           "device": str(dev), "identical_run_window_means": means, "identical_run_pairwise": pair,
           "recorded_supplement_window_mean": ref_mean, "recorded_readiness_window_mean": rdy_mean,
           "readiness_vs_supplement_abs_diff": abs(ref_mean - rdy_mean),
           "max_spread_of_window_means_over_all_runs": float(max(all_means) - min(all_means)),
           "window1_max_score_diff_supplement_vs_readiness": float(np.abs(ref[0].to_numpy() - rdy[0].loc[ref[0].index].to_numpy()).max()),
           "registered_tolerance": 0.01,
           "reading": "post-hoc, labelled diagnostic. If identical-seed runs themselves differ in window mean by a size comparable to the readiness difference, the readiness difference is run-to-run numerical variation, not a deterministic code difference.",
           "test_data_read": False, "seconds": round(time.time() - t0, 1)}
    out = ROOT / "experiments" / "phase8_state_continuity_supplement" / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(rec, indent=1) + "\n").encode("utf-8")
    p = out / f"F_readiness_diagnosis_{a.arm}_seed{a.seed}.json"
    p.write_bytes(data)
    p.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print(json.dumps({k: rec[k] for k in ("identical_run_window_means", "recorded_supplement_window_mean", "recorded_readiness_window_mean", "max_spread_of_window_means_over_all_runs")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
