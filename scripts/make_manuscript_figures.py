"""Manuscript figures drawn ONLY from recorded validation-period results (nothing is trained, scored or re-estimated; no held-out data).

  fig1_validation_pr_auc.png  : pooled validation PR-AUC per model/cell (experiments/comparison_B_graph_features/outputs/table_with_B.csv); error bars are the sample SD over seeds where there are several runs
  fig2_drift_adaptation.png   : window PR-AUC of the frozen, adaptive and replay arms over the five validation windows (Phase 8 recorded per-window metrics; mean of three seeds),
                                with the state-continuity supplement (memory built over days 1-121) as dashed lines
  fig3_class_weighting.png    : XGBoost PR-AUC under three class-weighting recipes (experiments/imbalance_check_xgboost/outputs/arms_table.csv)

Also writes submission/figures/figure_data.json (the plotted numbers) and a SHA-256 sidecar per file. Usage:  .venv/Scripts/python.exe -B scripts/make_manuscript_figures.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "submission" / "figures"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Okabe-Ito colour-blind-safe palette; every mark is also distinguished by position, marker or line style
BLUE, ORANGE, GREEN, VERMILION, PURPLE, GREY = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#666666"
CELL_COLOR = {"A": BLUE, "B": GREEN, "C": ORANGE, "C-control": GREY, "D": VERMILION, "E": PURPLE, "E-control": GREY}
ARM = {"F": ("frozen", "Frozen (F)", BLUE, "o"), "Ad": ("adaptive", "Adaptive (Ad)", VERMILION, "s"), "E": ("replay_control", "Replay control (E)", GREY, "^")}
SEEDS = (42, 123, 2024)
META = {"Software": None}


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / name
    fig.savefig(p, dpi=150, metadata=META, bbox_inches="tight")
    plt.close(fig)
    p.with_suffix(".sha256").write_bytes(hashlib.sha256(p.read_bytes()).hexdigest().encode("ascii"))


def fig1(data: dict) -> None:
    d = pd.read_csv(ROOT / "experiments/comparison_B_graph_features/outputs/table_with_B.csv")
    d = d[~d["model"].str.contains("pipeline control")].reset_index(drop=True)
    short = {"logistic_regression": "Logistic regression", "random_forest": "Random forest", "xgboost": "XGBoost (A)"}
    labels = []
    for m in d["model"]:
        labels.append(short.get(m, m.replace("XGBoost, A features, subsample 0.8 / colsample 0.8 (3 seeds)", "XGBoost A, subsample 0.8")
                                .replace("XGBoost, A features + 40 graph features (B, A recipe)", "XGBoost + graph features (B)")
                                .replace("XGBoost, B features, subsample 0.8 / colsample 0.8 (3 seeds)", "XGBoost + graph features, subsample 0.8")
                                .replace("GraphSAGE (Phase 6A)", "GraphSAGE").replace("MLP control, no relations (Phase 6A extended)", "MLP, no relations")
                                .replace("GAT (Phase 6B)", "GAT (not converged)").replace("Temporal GraphSAGE (GRU memory, Phase 7)", "Temporal GraphSAGE")
                                .replace("Static GraphSAGE, schedule-matched (Phase 7A)", "Static, schedule-matched").replace("Temporal, memory ablated (Phase 7A)", "Temporal, memory ablated")))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    y = np.arange(len(d))[::-1]
    for yi, (_, r) in zip(y, d.iterrows()):
        ax.errorbar(r["pr_auc"], yi, xerr=None if pd.isna(r["pr_auc_sd"]) else r["pr_auc_sd"], fmt="o" if r["runs"] == 1 else "D", color=CELL_COLOR[r["cell"]], capsize=3, ms=6)
        ax.text(r["pr_auc"] + (0 if pd.isna(r["pr_auc_sd"]) else r["pr_auc_sd"]) + 0.012, yi, f"{r['pr_auc']:.3f}", va="center", fontsize=8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}  {l}" for c, l in zip(d["cell"], labels)], fontsize=8)
    ax.set_xlabel("Validation PR-AUC (days 122-151); bars = sample SD over seeds")
    ax.set_xlim(0.25, 0.68)
    ax.grid(axis="x", alpha=0.3)
    ax.set_title("Validation-period PR-AUC by model (circle = 1 run, diamond = mean of 3 seeds)", fontsize=9)
    save(fig, "fig1_validation_pr_auc.png")
    data["fig1"] = [{"cell": r["cell"], "model": r["model"], "runs": int(r["runs"]), "pr_auc": float(r["pr_auc"]), "pr_auc_sd": None if pd.isna(r["pr_auc_sd"]) else float(r["pr_auc_sd"])} for _, r in d.iterrows()]


def fig2(data: dict) -> None:
    P8 = ROOT / "experiments/phase8_adaptive_drift"
    SUP = ROOT / "experiments/phase8_state_continuity_supplement/outputs/burn_in"

    def curves(base, fmt):
        out = {}
        for a, (dname, *_rest) in ARM.items():
            out[a] = np.mean([[json.loads((base / dname / f"seed_{s}" / fmt.format(k=k)).read_text(encoding="utf-8"))["pooled"]["pr_auc"] for k in range(1, 6)] for s in SEEDS], axis=0)
        return out
    rec, sup = curves(P8, "metrics_val_W{k}.json"), curves(SUP, "metrics_W{k}.json")
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(1, 6)
    for a, (_d, label, color, marker) in ARM.items():
        ax.plot(x, rec[a], marker=marker, color=color, label=f"{label}, as recorded (Phase 8)", lw=1.8)
        ax.plot(x, sup[a], marker=marker, color=color, ls="--", alpha=0.8, label=f"{label}, memory built first", lw=1.4, mfc="white")
    ax.set_xticks(x)
    ax.set_xticklabels(["W1\n122-127", "W2\n128-133", "W3\n134-139", "W4\n140-145", "W5\n146-151"])
    ax.set_ylabel("Window PR-AUC (mean of 3 seeds)")
    ax.set_title("Drift and adaptation on validation windows (three seeds are not independent)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=1, loc="upper right")
    save(fig, "fig2_drift_adaptation.png")
    data["fig2"] = {"windows": ["W1", "W2", "W3", "W4", "W5"], "recorded_phase8": {a: [float(v) for v in rec[a]] for a in ARM}, "memory_built_first": {a: [float(v) for v in sup[a]] for a in ARM}}


def fig3(data: dict) -> None:
    d = pd.read_csv(ROOT / "experiments/imbalance_check_xgboost/outputs/arms_table.csv")
    lab = {"none": "No weighting\n(weight 1)", "sqrt": "Square-root\nweighting (5.2)", "full": "Full weighting\n(27.4)"}
    fig, ax = plt.subplots(figsize=(5.6, 3.8))
    x = np.arange(len(d))
    w = 0.38
    b1 = ax.bar(x - w / 2, d["pr_auc"], w, color=BLUE, label="Pooled PR-AUC")
    b2 = ax.bar(x + w / 2, d["pr_auc_cold"], w, color=ORANGE, label="Cold-start PR-AUC", hatch="//", edgecolor="white")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008, f"{b.get_height():.3f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([lab[a] for a in d["arm"]], fontsize=8)
    ax.set_ylabel("Validation PR-AUC")
    ax.set_ylim(0, 0.7)
    ax.set_title("XGBoost (comparison-A recipe): effect of class weighting", fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig3_class_weighting.png")
    data["fig3"] = [{"arm": r["arm"], "scale_pos_weight": float(r["scale_pos_weight"]), "pr_auc": float(r["pr_auc"]), "pr_auc_cold": float(r["pr_auc_cold"])} for _, r in d.iterrows()]


def main() -> int:
    data: dict = {"sources": ["experiments/comparison_B_graph_features/outputs/table_with_B.csv", "experiments/phase8_adaptive_drift/*/seed_*/metrics_val_W*.json",
                              "experiments/phase8_state_continuity_supplement/outputs/burn_in/*/seed_*/metrics_W*.json", "experiments/imbalance_check_xgboost/outputs/arms_table.csv"],
                  "statement": "Drawn from recorded validation-period results only; nothing was trained or re-estimated; no held-out data."}
    fig1(data)
    fig2(data)
    fig3(data)
    OUT.mkdir(parents=True, exist_ok=True)
    b = (json.dumps(data, indent=1) + "\n").encode("utf-8")
    (OUT / "figure_data.json").write_bytes(b)
    (OUT / "figure_data.sha256").write_bytes(hashlib.sha256(b).hexdigest().encode("ascii"))
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
