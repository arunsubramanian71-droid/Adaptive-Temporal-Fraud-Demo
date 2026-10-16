"""Baseline 1: Logistic Regression.

TRAIN (days 1-121) -> fit. VALIDATION (days 122-151) -> evaluate + pick
threshold. TEST (days 152-182) is never touched by this script.

Run: .venv/Scripts/python.exe scripts/train_logistic_regression.py
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
from sklearn.linear_model import LogisticRegression

from src.config import load_config
from src.features.preprocessing import assert_features_exclude_blocklist
from src.models.lr_features import LogisticRegressionFeatureBuilder
from src.models.metrics import compute_metrics, select_best_f1_threshold

RANDOM_STATE = 42
EXPERIMENT_NAME = "logistic_regression"


def main() -> None:
    t_start = time.time()
    cfg = load_config()
    exp_dir = cfg.experiments_dir / EXPERIMENT_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading TRAIN and VALIDATION feature frames (TEST not touched) ===")
    train_df = pd.read_parquet(cfg.features_dir / "train_features.parquet")
    val_df = pd.read_parquet(cfg.features_dir / "val_features.parquet")
    print(f"  train: {train_df.shape[0]:,} rows x {train_df.shape[1]} cols")
    print(f"  val  : {val_df.shape[0]:,} rows x {val_df.shape[1]} cols")

    print("\n=== 2. Building Logistic Regression feature matrix (fit on TRAIN only) ===")
    t0 = time.time()
    builder = LogisticRegressionFeatureBuilder(cfg=cfg).fit(train_df)
    n_features = len(builder.feature_names_)
    assert_features_exclude_blocklist(builder.feature_names_, cfg)

    # Resource estimate, reported before the (cheap) transform + fit steps below.
    est_train_matrix_mb = train_df.shape[0] * n_features * 4 / 1e6
    est_val_matrix_mb = val_df.shape[0] * n_features * 4 / 1e6
    print(f"  feature count: {n_features} (post one-hot/frequency-encoding; see report for the breakdown)")
    print(f"  estimated dense matrix memory: TRAIN {est_train_matrix_mb:.0f} MB, VAL {est_val_matrix_mb:.0f} MB (float32)")
    print("  expected runtime: seconds to low tens of seconds (lbfgs on this size is comparable to the Phase 2 "
          "build step, ~37s for a much larger CSV-parsing workload) — proceeding without a separate approval "
          "pause, per this phase's own rule to pause only for POTENTIALLY EXPENSIVE operations.")

    X_train = builder.transform(train_df)
    X_val = builder.transform(val_df)
    y_train = train_df["isFraud"].to_numpy()
    y_val = val_df["isFraud"].to_numpy()
    t_feat = time.time() - t0
    print(f"  actual feature-matrix build time: {t_feat:.1f}s")
    print(f"  actual X_train memory: {X_train.nbytes/1e6:.0f} MB, X_val memory: {X_val.nbytes/1e6:.0f} MB")

    print("\n=== 3. Training LogisticRegression ===")
    print("  class_weight='balanced' (inverse-frequency weighting for the 27.58:1 imbalance; "
          "NOT optimizing accuracy, NOT using SMOTE — see report section B)")
    clf = LogisticRegression(
        class_weight="balanced",
        solver="lbfgs",
        max_iter=2000,
        random_state=RANDOM_STATE,
        n_jobs=None,
    )
    t0 = time.time()
    clf.fit(X_train, y_train)
    t_fit = time.time() - t0
    print(f"  fit time: {t_fit:.1f}s, converged: {clf.n_iter_[0]} iterations (max_iter=2000)")

    print("\n=== 4. Scoring VALIDATION ===")
    val_scores = clf.predict_proba(X_val)[:, 1]

    print("\n=== 5. Threshold selection (VALIDATION only) ===")
    thresh_info = select_best_f1_threshold(y_val, val_scores)
    print(f"  best-F1 threshold on VAL: {thresh_info['threshold']:.4f} "
          f"(F1={thresh_info['f1']:.4f}, P={thresh_info['precision']:.4f}, R={thresh_info['recall']:.4f})")

    print("\n=== 6. Computing full metric set on VALIDATION ===")
    metrics = compute_metrics(y_val, val_scores, threshold=thresh_info["threshold"])
    metrics["threshold_selection"] = thresh_info
    print(f"  PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
          f"Precision={metrics['precision']:.4f}  Recall={metrics['recall']:.4f}  F1={metrics['f1']:.4f}")
    for k, v in metrics["top_k"].items():
        print(f"    top {k:>6}: alerts={v['k_alerts']:>6,}  fraud_captured={v['fraud_captured']:>5,}  "
              f"P@K={v['precision_at_k']:.4f}  R@K={v['recall_at_k']:.4f}")

    print("\n=== 7. Saving artifacts ===")
    joblib.dump(clf, exp_dir / "model.joblib")
    builder.save(exp_dir / "feature_builder")

    coef = sorted(zip(builder.feature_names_, clf.coef_[0].tolist()), key=lambda t: -abs(t[1]))
    with open(exp_dir / "coefficients.json", "w", encoding="utf-8") as fh:
        json.dump([{"feature": f, "coefficient": c} for f, c in coef], fh, indent=1)

    config_record = {
        "experiment": EXPERIMENT_NAME,
        "random_state": RANDOM_STATE,
        "split": {"train_days": [cfg.split.train_day_min, cfg.split.train_day_max],
                  "val_days": [cfg.split.val_day_min, cfg.split.val_day_max],
                  "test_days": [cfg.split.test_day_min, cfg.split.test_day_max]},
        "model": {"type": "LogisticRegression", "class_weight": "balanced", "solver": "lbfgs",
                  "max_iter": 2000, "n_iter_actual": int(clf.n_iter_[0])},
        "feature_count": n_features,
        "encoding_strategy": {
            "continuous_median_imputed_then_scaled": len(builder.continuous_cols_),
            "boolean_flags": len(builder.bool_cols_),
            "frequency_encoded_extra": list(builder.freq_encoder_.columns),
            "frequency_encoded_phase2": ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain", "R_emaildomain"],
            "one_hot_encoded": list(builder.onehot_categories_.keys()),
            "amount_transform": "log1p",
            "time_transform": "sin/cos on hour (period 24) and day_of_week (period 7)",
        },
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
        "sklearn_version": sklearn.__version__,
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "config_path": "configs/data.yaml",
        "feature_matrix_build_seconds": round(t_feat, 2),
        "model_fit_seconds": round(t_fit, 2),
        "total_script_seconds": round(time.time() - t_start, 2),
    }
    with open(exp_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=1)

    for f in ["model.joblib", "coefficients.json", "config.json", "metrics_val.json", "predictions_val.parquet", "metadata.json"]:
        print(f"  wrote {exp_dir / f}")
    print(f"  wrote {exp_dir / 'feature_builder'}/ (fitted encoders, imputer, scaler)")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s. TEST split was not loaded or touched.")


if __name__ == "__main__":
    main()
