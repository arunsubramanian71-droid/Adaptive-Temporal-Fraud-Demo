"""Baseline 4 (control): XGBoost trained on ONLY the strictly historical /
account-level "blocklist-style" features from Phase 1 reconnaissance —
no TransactionAmt, no C/D/V columns, no device/network features at all.

Purpose (per instruction): isolate how much predictive power comes from
account-history alone, so a future graph model's apparent gains can be
judged against this rather than mistaken for new signal. If this minimal,
12-feature baseline captures most of full-feature XGBoost's performance,
that is strong evidence that a graph model's gains would need to be
compared against — not just above zero, but above THIS.

Same model hyperparameters as the full-feature XGBoost baseline
(scripts/train_xgboost.py), deliberately, so any performance difference is
attributable to the feature set, not to a hyperparameter change.
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb

from src.config import load_config
from src.data.splits import split_frame
from src.features.historical import historical_feature_columns
from src.features.preprocessing import assert_features_exclude_blocklist
from src.models.metrics import compute_metrics, select_best_f1_threshold

RANDOM_STATE = 42
EXPERIMENT_NAME = "xgboost_historical_control"


def main() -> None:
    t_start = time.time()
    cfg = load_config()
    exp_dir = cfg.experiments_dir / EXPERIMENT_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading merged frame + historical features, then splitting ===")
    merged = pd.read_parquet(cfg.processed_dir / "train_merged.parquet")
    hist = pd.read_parquet(cfg.features_dir / "historical_features.parquet")
    feat_cols = historical_feature_columns()
    df = merged[["TransactionID", "TransactionDT", "isFraud"]].merge(hist, on="TransactionID", how="left")
    splits = split_frame(df, cfg)
    train_df, val_df = splits["train"], splits["val"]
    print(f"  train: {len(train_df):,} rows, val: {len(val_df):,} rows (TEST split not touched)")
    print(f"  feature count: {len(feat_cols)} — strictly historical/account-level only: {feat_cols}")
    assert_features_exclude_blocklist(feat_cols, cfg)

    X_train = train_df[feat_cols].to_numpy(dtype=np.float32)
    X_val = val_df[feat_cols].to_numpy(dtype=np.float32)
    y_train = train_df["isFraud"].to_numpy()
    y_val = val_df["isFraud"].to_numpy()

    est_train_mb = X_train.nbytes / 1e6
    est_val_mb = X_val.nbytes / 1e6
    print(f"  estimated/actual matrix memory: TRAIN {est_train_mb:.1f} MB, VAL {est_val_mb:.1f} MB — "
          "trivial at only 12 features; no separate approval pause needed.")
    print(f"  expected runtime: seconds — same algorithm as the full-feature XGBoost run but ~40x fewer "
          "features, so materially faster.")

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"\n=== 2. Training XGBoost (historical features only) ===")
    print(f"  scale_pos_weight={scale_pos_weight:.2f}; NOT optimizing accuracy, NOT using SMOTE")
    clf = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.05,
        tree_method="hist",
        eval_metric="aucpr",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=50,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    t0 = time.time()
    clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    t_fit = time.time() - t0
    print(f"  fit time: {t_fit:.1f}s, best_iteration={clf.best_iteration} (of max 1000, early_stopping_rounds=50)")

    print("\n=== 3. Scoring VALIDATION ===")
    val_scores = clf.predict_proba(X_val)[:, 1]

    print("\n=== 4. Threshold selection (VALIDATION only) ===")
    thresh_info = select_best_f1_threshold(y_val, val_scores)
    print(f"  best-F1 threshold: {thresh_info['threshold']:.4f} "
          f"(F1={thresh_info['f1']:.4f}, P={thresh_info['precision']:.4f}, R={thresh_info['recall']:.4f})")

    print("\n=== 5. Computing full metric set on VALIDATION ===")
    metrics = compute_metrics(y_val, val_scores, threshold=thresh_info["threshold"])
    metrics["threshold_selection"] = thresh_info
    print(f"  PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
          f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    for k, v in metrics["top_k"].items():
        print(f"    top {k:>6}: alerts={v['k_alerts']:>6,}  fraud_captured={v['fraud_captured']:>5,}  "
              f"P@K={v['precision_at_k']:.4f}  R@K={v['recall_at_k']:.4f}")

    print("\n=== 6. Saving artifacts ===")
    joblib.dump(clf, exp_dir / "model.joblib")
    importances = sorted(zip(feat_cols, clf.feature_importances_.tolist()), key=lambda t: -t[1])
    with open(exp_dir / "feature_importances.json", "w", encoding="utf-8") as fh:
        json.dump([{"feature": f, "importance": v} for f, v in importances], fh, indent=1)

    config_record = {
        "experiment": EXPERIMENT_NAME,
        "purpose": "Control baseline: isolates the predictive power of strictly historical, "
                   "account-level features alone (Phase 1 'Level 3.5 blocklist baseline'), to test "
                   "whether apparent future graph-model gains are actually recovering this same "
                   "account-level labeling structure rather than discovering new signal.",
        "random_state": RANDOM_STATE,
        "split": {"train_days": [cfg.split.train_day_min, cfg.split.train_day_max],
                  "val_days": [cfg.split.val_day_min, cfg.split.val_day_max],
                  "test_days": [cfg.split.test_day_min, cfg.split.test_day_max]},
        "model": {"type": "XGBClassifier", "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.05,
                  "tree_method": "hist", "scale_pos_weight": scale_pos_weight,
                  "early_stopping_rounds": 50, "best_iteration": int(clf.best_iteration),
                  "note": "identical hyperparameters to scripts/train_xgboost.py, deliberately, so any "
                          "performance gap is attributable to the feature set alone"},
        "feature_count": len(feat_cols),
        "features": feat_cols,
        "feature_source": "src/features/historical.py — strictly information_time < transaction_time; "
                           "computed over the full chronological labeled dataset, verified via an "
                           "independent leakage self-check (scripts/build_historical_features.py)",
    }
    with open(exp_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config_record, fh, indent=1)

    with open(exp_dir / "metrics_val.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)

    preds = pd.DataFrame({
        "TransactionID": val_df["TransactionID"].to_numpy(),
        "isFraud": y_val,
        "y_score": val_scores,
        "y_pred": (val_scores >= thresh_info["threshold"]).astype(int),
    })
    preds.to_parquet(exp_dir / "predictions_val.parquet", index=False)

    metadata = {
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "python_version": platform.python_version(),
        "xgboost_version": xgb.__version__,
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "config_path": "configs/data.yaml",
        "model_fit_seconds": round(t_fit, 2),
        "total_script_seconds": round(time.time() - t_start, 2),
    }
    with open(exp_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=1)

    for f in ["model.joblib", "feature_importances.json", "config.json", "metrics_val.json",
              "predictions_val.parquet", "metadata.json"]:
        print(f"  wrote {exp_dir / f}")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s. TEST split was not loaded or touched.")


if __name__ == "__main__":
    main()
