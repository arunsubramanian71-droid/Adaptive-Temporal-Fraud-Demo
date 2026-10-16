"""Shared reporting helpers for the Phase 7A arms (identical metric/aggregation code for B and C,
mirroring what Phases 6A and 7 did)."""
from __future__ import annotations

import json
import pathlib
import platform
import time

import numpy as np
import pandas as pd
import torch

from src.models.metrics import compute_metrics, select_best_f1_threshold

SEGMENTS = ["has_edge_warm", "has_edge_cold", "no_pseudo_account_edge"]
AGG_KEYS = {"pooled_pr_auc": ("pooled", "pr_auc"), "pooled_roc_auc": ("pooled", "roc_auc"),
            "cold_pr_auc": ("has_edge_cold", "pr_auc"), "cold_roc_auc": ("has_edge_cold", "roc_auc"),
            "warm_pr_auc": ("has_edge_warm", "pr_auc"), "warm_roc_auc": ("has_edge_warm", "roc_auc"),
            "no_edge_pr_auc": ("no_pseudo_account_edge", "pr_auc"), "no_edge_roc_auc": ("no_pseudo_account_edge", "roc_auc")}


def final_metrics(ids, y_true, scores, seg):
    th = select_best_f1_threshold(y_true, scores)          # threshold chosen on VALIDATION only
    pooled = compute_metrics(y_true, scores, threshold=th["threshold"])
    pooled["threshold_selection"] = th
    metrics = {"pooled": pooled}
    for name in SEGMENTS:
        m = seg == name
        yt, sc = y_true[m], scores[m]
        metrics[name] = (compute_metrics(yt, sc, threshold=th["threshold"]) if 0 < yt.sum() < len(yt)
                         else {"n_rows": int(len(yt)), "n_fraud": int(yt.sum()), "note": "insufficient class balance"})
    return metrics, th


def save_seed(out_dir: pathlib.Path, ids, y_true, scores, days_arr, seg, th, metrics, history, resource, best_state, config):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model_checkpoint.pt")
    for name, obj in [("metrics_val.json", metrics), ("training_history.json", history),
                      ("resource_measurements.json", resource), ("config.json", config)]:
        with open(out_dir / name, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=1)
    pd.DataFrame({"TransactionID": ids, "isFraud": y_true, "y_score": scores, "day": days_arr,
                  "y_pred": (scores >= th["threshold"]).astype(int), "segment": seg}).to_parquet(out_dir / "predictions_val.parquet", index=False)


def aggregate_and_plot(results, base_dir: pathlib.Path, title: str, seeds, extra: dict, t_start: float, meta_extra: dict):
    print(f"\n{'='*70}\n=== Cross-seed aggregation ===\n{'='*70}", flush=True)
    agg = {}
    for key, (seg, m) in AGG_KEYS.items():
        v = [r["metrics"][seg][m] for r in results]
        agg[key] = {"values": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                    "min": float(np.min(v)), "max": float(np.max(v))}
        print(f"  {key:<16}: {[round(x, 4) for x in v]}  mean={agg[key]['mean']:.4f} std={agg[key]['std']:.4f} "
              f"min={agg[key]['min']:.4f} max={agg[key]['max']:.4f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    col = {42: "tab:blue", 123: "tab:orange", 2024: "tab:green"}
    for r in results:
        h = pd.DataFrame(r["history"])
        ax[0].plot(h.epoch, h.train_loss, color=col[r["seed"]], label=f"seed {r['seed']}")
        ax[1].plot(h.epoch, h.val_pr_auc, color=col[r["seed"]], label=f"seed {r['seed']}")
        ax[1].axvline(r["resource"]["best_epoch"], color=col[r["seed"]], ls=":", alpha=.5)
        ax[2].plot(h.epoch, h.val_roc_auc, color=col[r["seed"]], label=f"seed {r['seed']}")
    ax[0].set_title(f"{title}: training loss"); ax[1].set_title("Validation PR-AUC (rolling, pooled)")
    ax[2].set_title("Validation ROC-AUC (rolling, pooled)")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(base_dir / "convergence_curves.png", dpi=120); plt.close()

    summary = {"seeds": seeds, **extra,
               "per_seed": [{"seed": r["seed"], **r["resource"],
                             "pooled_pr_auc": r["metrics"]["pooled"]["pr_auc"], "pooled_roc_auc": r["metrics"]["pooled"]["roc_auc"],
                             "cold_pr_auc": r["metrics"]["has_edge_cold"]["pr_auc"], "warm_pr_auc": r["metrics"]["has_edge_warm"]["pr_auc"],
                             "no_edge_pr_auc": r["metrics"]["no_pseudo_account_edge"]["pr_auc"]} for r in results],
               "aggregate": agg, "total_script_seconds": round(time.time() - t_start, 2)}
    with open(base_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    import torch_geometric as pyg
    with open(base_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"timestamp_utc": pd.Timestamp.now("UTC").isoformat(), "python": platform.python_version(),
                   "torch": torch.__version__, "torch_geometric": pyg.__version__,
                   "device": "cuda:0" if torch.cuda.is_available() else "cpu", **meta_extra,
                   "total_script_seconds": round(time.time() - t_start, 2)}, fh, indent=1)
    return agg
