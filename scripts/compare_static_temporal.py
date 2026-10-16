"""Phase 7 analysis (read-only over saved artifacts; trains nothing):
  - Static GraphSAGE (Phase 6A) vs Temporal GraphSAGE: seed-matched ablation
  - Re-stratification of every baseline's saved VAL predictions onto the strict
    3-way segment definition, so all models are compared on identical populations
  - Time-bucketed PR-AUC across the validation period (future-time robustness)
Established reference numbers are NOT replaced; the re-stratified columns are supplementary.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config
from src.graph.schema import compute_pseudo_account_segments

SEEDS = [42, 123, 2024]
SEGMENTS = ["has_edge_warm", "has_edge_cold", "no_pseudo_account_edge"]


def ap(y, s):
    return float(average_precision_score(y, s)) if 0 < y.sum() < len(y) else float("nan")


def seg_metrics(df: pd.DataFrame) -> dict:
    out = {"pooled": ap(df.isFraud.to_numpy(), df.y_score.to_numpy())}
    for s in SEGMENTS:
        d = df[df.segment == s]
        out[s] = ap(d.isFraud.to_numpy(), d.y_score.to_numpy())
    return out


def main() -> None:
    cfg = load_config()
    exp = cfg.experiments_dir
    graph = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    seg_by_id = pd.Series(compute_pseudo_account_segments(graph), index=graph["transaction"].transaction_id.numpy())
    day_by_id = pd.Series(graph["transaction"].day.numpy(), index=graph["transaction"].transaction_id.numpy())

    def load_preds(path: pathlib.Path) -> pd.DataFrame:
        p = pd.read_parquet(path)
        p["segment"] = seg_by_id.loc[p.TransactionID.to_numpy()].to_numpy()
        p["day"] = day_by_id.loc[p.TransactionID.to_numpy()].to_numpy()
        return p

    out: dict = {}

    # ---- 1. seed-matched static vs temporal ----
    static = {s: json.load(open(exp / "graphsage_robust" / f"seed_{s}" / "metrics_val.json")) for s in SEEDS}
    temporal = {s: json.load(open(exp / "temporal_graphsage" / f"seed_{s}" / "metrics_val.json")) for s in SEEDS}
    rows = []
    for key, seg in [("pooled", "pooled"), ("cold", "has_edge_cold"), ("warm", "has_edge_warm"), ("no_edge", "no_pseudo_account_edge")]:
        for metric in ["pr_auc", "roc_auc"]:
            st = np.array([static[s][seg][metric] for s in SEEDS])
            tp = np.array([temporal[s][seg][metric] for s in SEEDS])
            d = tp - st
            rows.append({"segment": key, "metric": metric, "static_values": st.tolist(), "temporal_values": tp.tolist(),
                         "static_mean": float(st.mean()), "static_std": float(st.std(ddof=1)),
                         "temporal_mean": float(tp.mean()), "temporal_std": float(tp.std(ddof=1)),
                         "seed_matched_diff": d.tolist(), "diff_mean": float(d.mean()), "diff_std": float(d.std(ddof=1)),
                         "temporal_wins_in_n_of_3_seeds": int((d > 0).sum())})
    out["static_vs_temporal"] = rows
    print("=== Static GraphSAGE (Phase 6A) vs Temporal GraphSAGE, seed-matched ===")
    for r in rows:
        if r["metric"] == "pr_auc":
            print(f"  {r['segment']:<8} PR-AUC  static {r['static_mean']:.4f}±{r['static_std']:.4f}  temporal {r['temporal_mean']:.4f}±{r['temporal_std']:.4f}  "
                  f"diff {r['diff_mean']:+.4f}±{r['diff_std']:.4f}  temporal wins {r['temporal_wins_in_n_of_3_seeds']}/3 seeds  "
                  f"per-seed diff {[round(x,4) for x in r['seed_matched_diff']]}")

    # ---- 2. strict re-stratification of all models ----
    models = {
        "Logistic Regression": [exp / "logistic_regression" / "predictions_val.parquet"],
        "Random Forest": [exp / "random_forest" / "predictions_val.parquet"],
        "XGBoost": [exp / "xgboost" / "predictions_val.parquet"],
        "XGBoost historical control": [exp / "xgboost_historical_control" / "predictions_val.parquet"],
        "MLP control (Phase 6A extended)": [exp / "graphsage_robust" / "mlp_control_extended" / "predictions_val.parquet"],
        "GraphSAGE Phase 6A (3 seeds)": [exp / "graphsage_robust" / f"seed_{s}" / "predictions_val.parquet" for s in SEEDS],
        "GAT Phase 6B (3 seeds)": [exp / "gat" / f"seed_{s}" / "predictions_val.parquet" for s in SEEDS],
        "Temporal GraphSAGE (3 seeds)": [exp / "temporal_graphsage" / f"seed_{s}" / "predictions_val.parquet" for s in SEEDS],
    }
    strict = {}
    print("\n=== PR-AUC on the STRICT 3-way segments (same populations for every model; mean over seeds where applicable) ===")
    print(f"  {'model':<34}{'pooled':>9}{'cold':>9}{'warm':>9}{'no-edge':>9}")
    for name, paths in models.items():
        per = [seg_metrics(load_preds(p)) for p in paths]
        m = {k: float(np.nanmean([x[k] for x in per])) for k in per[0]}
        strict[name] = m
        print(f"  {name:<34}{m['pooled']:>9.4f}{m['has_edge_cold']:>9.4f}{m['has_edge_warm']:>9.4f}{m['no_pseudo_account_edge']:>9.4f}")
    out["strict_segment_pr_auc"] = strict

    # ---- 3. time-bucketed PR-AUC over the validation period ----
    buckets = [(122, 127), (128, 133), (134, 139), (140, 145), (146, 151)]
    def bucket_ap(paths):
        res = []
        for lo, hi in buckets:
            vals = []
            for p in paths:
                d = load_preds(p)
                d = d[(d.day >= lo) & (d.day <= hi)]
                vals.append(ap(d.isFraud.to_numpy(), d.y_score.to_numpy()))
            res.append(float(np.mean(vals)))
        return res
    tb = {name: bucket_ap(models[name]) for name in ["XGBoost", "GraphSAGE Phase 6A (3 seeds)", "Temporal GraphSAGE (3 seeds)"]}
    out["time_bucketed_pr_auc"] = {"buckets": buckets, **tb}
    print("\n=== PR-AUC by validation time bucket (days) ===")
    print("  " + f"{'model':<34}" + "".join(f"{f'{a}-{b}':>10}" for a, b in buckets))
    for name, v in tb.items():
        print("  " + f"{name:<34}" + "".join(f"{x:>10.4f}" for x in v))

    # ---- 4. temporal top-K / operating points (mean over seeds) ----
    tk = {}
    for k in ["0.1%", "0.5%", "1%", "5%"]:
        tk[k] = {f: float(np.mean([temporal[s]["pooled"]["top_k"][k][f] for s in SEEDS]))
                 for f in ["k_alerts", "fraud_captured", "precision_at_k", "recall_at_k"]}
    out["temporal_top_k_mean"] = tk
    op = {f: [temporal[s]["pooled"][f] for s in SEEDS] for f in ["threshold", "precision", "recall", "f1", "n_alerts", "false_positives", "false_negatives"]}
    out["temporal_operating_points_per_seed"] = op

    with open(exp / "temporal_graphsage" / "comparison.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nWrote {exp / 'temporal_graphsage' / 'comparison.json'}")


if __name__ == "__main__":
    main()
