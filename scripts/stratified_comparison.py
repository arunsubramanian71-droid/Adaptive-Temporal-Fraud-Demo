"""Stratify each baseline's VAL predictions by cold-start vs. has-history
(account_has_prior_history), per Phase 1's explicit recommendation that
pooled metrics on this dataset are misleading. Read-only: loads existing
predictions_val.parquet files, computes no new model.
"""
from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config

EXPERIMENTS = ["logistic_regression", "random_forest", "xgboost", "xgboost_historical_control"]


def stratified_metrics(y_true, y_score) -> dict:
    n, n_fraud = len(y_true), int(y_true.sum())
    if n_fraud == 0 or n_fraud == n:
        return {"n": n, "n_fraud": n_fraud, "pr_auc": float("nan"), "roc_auc": float("nan")}
    return {"n": n, "n_fraud": n_fraud, "fraud_rate": n_fraud / n,
            "pr_auc": float(average_precision_score(y_true, y_score)),
            "roc_auc": float(roc_auc_score(y_true, y_score))}


def main() -> None:
    cfg = load_config()
    hist = pd.read_parquet(cfg.features_dir / "historical_features.parquet")[
        ["TransactionID", "account_has_prior_history"]]

    results = {}
    for name in EXPERIMENTS:
        pred_path = cfg.experiments_dir / name / "predictions_val.parquet"
        preds = pd.read_parquet(pred_path).merge(hist, on="TransactionID", how="left")
        cold = preds["account_has_prior_history"] == 0
        warm = ~cold
        results[name] = {
            "pooled": stratified_metrics(preds["isFraud"].to_numpy(), preds["y_score"].to_numpy()),
            "cold_start": stratified_metrics(preds.loc[cold, "isFraud"].to_numpy(), preds.loc[cold, "y_score"].to_numpy()),
            "has_history": stratified_metrics(preds.loc[warm, "isFraud"].to_numpy(), preds.loc[warm, "y_score"].to_numpy()),
        }
        r = results[name]
        print(f"=== {name} ===")
        print(f"  pooled     : n={r['pooled']['n']:,} fraud={r['pooled']['n_fraud']:,} "
              f"PR-AUC={r['pooled']['pr_auc']:.4f} ROC-AUC={r['pooled']['roc_auc']:.4f}")
        print(f"  cold-start : n={r['cold_start']['n']:,} fraud={r['cold_start']['n_fraud']:,} "
              f"rate={r['cold_start']['fraud_rate']*100:.3f}% PR-AUC={r['cold_start']['pr_auc']:.4f} "
              f"ROC-AUC={r['cold_start']['roc_auc']:.4f}")
        print(f"  has-history: n={r['has_history']['n']:,} fraud={r['has_history']['n_fraud']:,} "
              f"rate={r['has_history']['fraud_rate']*100:.3f}% PR-AUC={r['has_history']['pr_auc']:.4f} "
              f"ROC-AUC={r['has_history']['roc_auc']:.4f}")
        print()

    out_path = cfg.reports_dir / "phase3_stratified_metrics.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
