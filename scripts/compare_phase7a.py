"""Phase 7A analysis: A (static 6A) vs B (static, schedule-matched) vs C (temporal, memory-ablated) vs D (temporal GRU, Phase 7).
Read-only over saved artifacts; trains nothing. Writes experiments/temporal_ablation/comparison.json + plots.
Pairwise p-values are DESCRIPTIVE (n=3 seeds, uncorrected); reading rules were fixed in DESIGN_BEFORE_EXECUTION.md."""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

from src.config import load_config

SEEDS = [42, 123, 2024]
ARMS = {"A": ("graphsage_robust", "Static GraphSAGE (Phase 6A: 1 step/epoch)"),
        "B": ("temporal_ablation/static_schedule_matched", "Static GraphSAGE, schedule-matched (18 steps/epoch, clip 1.0)"),
        "C": ("temporal_ablation/memory_ablated_temporal", "Temporal pipeline, cumulative-mean memory (memory ablated)"),
        "D": ("temporal_graphsage", "Temporal GraphSAGE, GRU memory (Phase 7)")}
SCALAR = {"pooled_pr_auc": ("pooled", "pr_auc"), "pooled_roc_auc": ("pooled", "roc_auc"),
          "cold_pr_auc": ("has_edge_cold", "pr_auc"), "warm_pr_auc": ("has_edge_warm", "pr_auc"),
          "no_edge_pr_auc": ("no_pseudo_account_edge", "pr_auc"),
          "precision": ("pooled", "precision"), "recall": ("pooled", "recall"), "f1": ("pooled", "f1")}
TOPK = ["0.1%", "0.5%", "1%", "5%"]
STEPS_PER_EPOCH = {"A": 1, "B": 18, "C": 18, "D": 18}


def load_arm(exp: pathlib.Path, rel: str, key: str) -> dict:
    per = []
    for s in SEEDS:
        d = exp / rel / f"seed_{s}"
        m = json.load(open(d / "metrics_val.json"))
        r = json.load(open(d / "resource_measurements.json"))
        h = json.load(open(d / "training_history.json"))
        row = {"seed": s, **{k: m[a][b] for k, (a, b) in SCALAR.items()}}
        for k in TOPK:
            row[f"P@{k}"] = m["pooled"]["top_k"][k]["precision_at_k"]
            row[f"R@{k}"] = m["pooled"]["top_k"][k]["recall_at_k"]
        row.update({"params": r["parameter_count"], "epochs_run": r["epochs_run"], "best_epoch": r["best_epoch"],
                    "stopped_early": r["stopped_early"], "train_seconds": r["training_wall_seconds"],
                    "peak_gpu_reserved_gib": r.get("peak_gpu_memory_reserved_gib"),
                    "peak_gpu_allocated_gib": r.get("peak_gpu_memory_allocated_gib"),
                    "steps_per_epoch": STEPS_PER_EPOCH[key],
                    "total_optimizer_steps": r["epochs_run"] * STEPS_PER_EPOCH[key],
                    "steps_to_best": r["best_epoch"] * STEPS_PER_EPOCH[key],
                    "final_train_loss": h[-1]["train_loss"], "best_epoch_train_loss": h[r["best_epoch"] - 1]["train_loss"]})
        per.append(row)
    df = pd.DataFrame(per)
    num = [c for c in df.columns if c not in ("seed", "stopped_early")]
    return {"per_seed": per, "mean": df[num].mean().to_dict(), "std": df[num].std(ddof=1).to_dict(),
            "min": df[num].min().to_dict(), "max": df[num].max().to_dict()}


def pair(a: dict, b: dict, metric: str) -> dict:
    """x - y seed-matched (same seed number)."""
    x = np.array([r[metric] for r in a["per_seed"]]); y = np.array([r[metric] for r in b["per_seed"]])
    d = x - y
    t = stats.ttest_rel(x, y); w = stats.ttest_ind(x, y, equal_var=False)
    return {"diff_per_seed": d.tolist(), "mean_diff": float(d.mean()), "wins": int((d > 0).sum()),
            "paired_t_p_descriptive": float(t.pvalue), "welch_p_descriptive": float(w.pvalue),
            "sd_x": float(x.std(ddof=1)), "sd_y": float(y.std(ddof=1)),
            "exceeds_larger_sd": bool(abs(d.mean()) > max(x.std(ddof=1), y.std(ddof=1)))}


def main() -> None:
    cfg = load_config()
    exp = cfg.experiments_dir
    arms = {k: load_arm(exp, rel, k) for k, (rel, _) in ARMS.items()}
    out = {"arms": {k: {"description": ARMS[k][1], **v} for k, v in arms.items()}, "pairs": {}, "decision_rules": {}}
    for name, (x, y) in {"D_minus_B": ("D", "B"), "D_minus_C": ("D", "C"), "C_minus_B": ("C", "B"), "B_minus_A": ("B", "A"),
                         "D_minus_A": ("D", "A"), "C_minus_A": ("C", "A")}.items():
        out["pairs"][name] = {m: pair(arms[x], arms[y], m) for m in ["pooled_pr_auc", "pooled_roc_auc", "cold_pr_auc", "warm_pr_auc", "no_edge_pr_auc"]}
    for name in ["D_minus_B", "D_minus_C"]:
        p = out["pairs"][name]["pooled_pr_auc"]
        out["decision_rules"][name] = {"wins_3_of_3": p["wins"] == 3, "mean_diff_exceeds_larger_seed_sd": p["exceeds_larger_sd"],
                                       "criterion_met": bool(p["wins"] == 3 and p["mean_diff"] > 0 and p["exceeds_larger_sd"])}
    with open(exp / "temporal_ablation" / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)

    # console tables
    def line(k, m):
        a = arms[k]; return f"{a['mean'][m]:.4f} +/- {a['std'][m]:.4f} [{a['min'][m]:.4f}, {a['max'][m]:.4f}]"
    print("arm | " + " | ".join(SCALAR))
    for k in ARMS:
        print(k, "|", " | ".join(f"{arms[k]['mean'][m]:.4f}+/-{arms[k]['std'][m]:.4f}" for m in SCALAR))
    for k in ARMS:
        print(k, "per-seed pooled PR-AUC:", [round(r["pooled_pr_auc"], 4) for r in arms[k]["per_seed"]])
    for name, blk in out["pairs"].items():
        p = blk["pooled_pr_auc"]
        print(f"{name}: mean diff {p['mean_diff']:+.4f}, per-seed {[round(v, 4) for v in p['diff_per_seed']]}, wins {p['wins']}/3, "
              f"paired p={p['paired_t_p_descriptive']:.3f}, Welch p={p['welch_p_descriptive']:.3f}, > larger SD: {p['exceeds_larger_sd']}")
    print("decision rules:", out["decision_rules"])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    cols = {"A": "tab:gray", "B": "tab:blue", "C": "tab:orange", "D": "tab:green"}
    for k in ARMS:
        for s, r in zip(SEEDS, arms[k]["per_seed"]):
            ax[0].scatter([k], [r["pooled_pr_auc"]], color=cols[k], s=40, zorder=3)
        ax[0].errorbar([k], [arms[k]["mean"]["pooled_pr_auc"]], yerr=[arms[k]["std"]["pooled_pr_auc"]], color="k", capsize=6, zorder=2)
        h = None
    ax[0].set_title("Pooled VAL PR-AUC per seed (bar = mean +/- SD)"); ax[0].set_ylabel("PR-AUC")
    for k in ARMS:
        for s in SEEDS:
            hist = json.load(open(exp / ARMS[k][0] / f"seed_{s}" / "training_history.json"))
            ax[1].plot([e["epoch"] for e in hist], [e["val_pr_auc"] for e in hist], color=cols[k], alpha=.55, lw=1,
                       label=k if s == SEEDS[0] else None)
    ax[1].set_title("Validation PR-AUC vs epoch (all seeds)"); ax[1].set_xlabel("epoch"); ax[1].legend()
    plt.tight_layout(); plt.savefig(exp / "temporal_ablation" / "comparison_plot.png", dpi=120); plt.close()
    print("wrote comparison.json and comparison_plot.png")


if __name__ == "__main__":
    main()
