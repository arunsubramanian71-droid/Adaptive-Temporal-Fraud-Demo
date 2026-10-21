"""Performance-triggered refresh (arm TrP), validation period only: experiments/phase8_triggered_refresh/PREREGISTRATION_TRP.md.

  .venv/Scripts/python.exe -B scripts/run_triggered_refresh_perf.py [--seeds 42,123,2024]

For each seed: (1) delta from TRAIN days only: relative difference r of the initial checkpoint's PR-AUC on TRAIN days 112-116 and 117-121, delta = min(0.5, max(0.10, 2 x r));
(2) arm TrP over the validation windows 122-151 with the entity memory built over days 1-121 first; after each scored window the monitor sees that window's labels and decides the refresh.
Outputs: experiments/phase8_triggered_refresh/outputs_performance/seed_S/{predictions_W{k}.parquet, metrics_W{k}.json, trigger_log.json}, summary.json (+ sidecars). The dataset is truncated at day 151.
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

from src.adaptive.update_loop import score_window  # noqa: E402
from src.adaptive.windows import build_windows  # noqa: E402
from src.config import load_config  # noqa: E402
from src.graph.temporal import build_temporal_dataset, load_raw_arrays  # noqa: E402
from src.heldout.walkforward import burn_in, run_walk_forward_ext  # noqa: E402
from src.models.metrics import compute_metrics  # noqa: E402
from src.models.temporal_graphsage import TemporalGraphSAGE  # noqa: E402
from src.trigger.performance import PerformanceTrigger  # noqa: E402

P8 = ROOT / "experiments" / "phase8_adaptive_drift"
OUT = ROOT / "experiments" / "phase8_triggered_refresh" / "outputs_performance"
FLOOR, CAP = 0.10, 0.5


def write(path: pathlib.Path, data: bytes):
    path.write_bytes(data)
    path.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))


def load_model(raw, dev, seed):
    m = TemporalGraphSAGE(raw["X"].shape[1]).to(dev)
    m.load_state_dict(torch.load(P8 / "shared_init" / f"model_checkpoint_initial_seed{seed}.pt", map_location=dev))
    return m


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", default="42,123,2024")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    cfg = load_config()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    raw, _graph, _ = load_raw_arrays(cfg)
    y_all = pd.Series(raw["y"], index=raw["txn_id"])
    ds = build_temporal_dataset(raw, 151, dev, cfg)
    assert ds.max_day == 151
    tr_mask = ds.txn_day <= cfg.split.train_day_max
    y_tr = raw["y"][:len(tr_mask)][tr_mask]
    pos_weight = torch.tensor(float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), device=dev)
    windows = build_windows(122, 151)
    summary = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "seeds": seeds, "windows": windows, "per_seed": {}, "delta_floor": FLOOR,
               "delta_cap": CAP, "test_data_read": False}
    for seed in seeds:
        hp = json.loads((P8 / "shared_init" / f"hparams_seed{seed}.json").read_text(encoding="utf-8"))
        # (1) delta from TRAIN days only (the checkpoint was fitted on days 1-111; days 112-121 are its internal hold-out)
        torch.manual_seed(seed)
        m0 = load_model(raw, dev, seed)
        st = m0.init_state(ds)
        burn_in(m0, ds, st, 111)
        ids_a, s_a, _ = score_window(m0, st, ds, (112, 116))
        ids_b, s_b, _ = score_window(m0, st, ds, (117, 121))
        ap_a = float(average_precision_score(y_all.loc[ids_a].to_numpy(), s_a))
        ap_b = float(average_precision_score(y_all.loc[ids_b].to_numpy(), s_b))
        r = abs(ap_a - ap_b) / ((ap_a + ap_b) / 2)
        delta = min(CAP, max(FLOOR, 2.0 * r))
        del m0, st
        # (2) arm TrP
        torch.manual_seed(seed)
        model = load_model(raw, dev, seed)
        opt = torch.optim.Adam(model.parameters(), lr=hp["fine_tune_lr"], weight_decay=1e-5)
        trig = PerformanceTrigger(delta)
        ta = time.time()
        res = run_walk_forward_ext("Tr", model, ds, windows, opt=opt, pos_weight=pos_weight, fine_tune_epochs=hp["fine_tune_epochs"], base_seed=seed,
                                   train_day_min=cfg.split.train_day_min, train_day_max=cfg.split.train_day_max, burn_in_through=windows[0][0] - 1, trigger=trig)
        d = OUT / f"seed_{seed}"
        d.mkdir(parents=True, exist_ok=True)
        prs = []
        for k, (ids, scores, days) in enumerate(res, start=1):
            y = y_all.loc[ids].to_numpy()
            met = {"pooled": compute_metrics(y, scores, threshold=hp["threshold"])}
            (d / f"metrics_W{k}.json").write_text(json.dumps(met, indent=1), encoding="utf-8")
            pd.DataFrame({"TransactionID": ids, "isFraud": y, "y_score": scores, "day": days, "window": k}).to_parquet(d / f"predictions_W{k}.parquet", index=False)
            prs.append(met["pooled"]["pr_auc"])
        n_ref = sum(1 for row in trig.log if row["fired"])
        rec = {"seed": seed, "pr_auc_train_days_112_116": ap_a, "pr_auc_train_days_117_121": ap_b, "relative_noise_r": r, "delta": delta, "fine_tune_epochs": hp["fine_tune_epochs"],
               "monitor_log": trig.log, "n_refreshes": n_ref, "window_pr_auc": prs, "window_mean_pr_auc": float(np.mean(prs))}
        write(d / "trigger_log.json", (json.dumps(rec, indent=1) + "\n").encode("utf-8"))
        summary["per_seed"][str(seed)] = {k: rec[k] for k in ("relative_noise_r", "delta", "n_refreshes", "window_pr_auc", "window_mean_pr_auc")}
        print(f"[done] TrP seed {seed}: delta {delta:.3f} (r {r:.3f}) refreshes {n_ref}/4; window PR-AUC {[round(p, 4) for p in prs]} mean {np.mean(prs):.4f} ({time.time() - ta:.0f}s)", flush=True)
        del model, opt
        torch.cuda.empty_cache()
    summary["seconds"] = round(time.time() - t0, 1)
    OUT.mkdir(parents=True, exist_ok=True)
    write(OUT / "summary.json", (json.dumps(summary, indent=1) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
