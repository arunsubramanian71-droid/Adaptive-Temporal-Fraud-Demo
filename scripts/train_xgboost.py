"""Baseline 3: XGBoost, full feature set.

TRAIN (days 1-121) -> fit. VALIDATION (days 122-151) -> early-stopping eval
set, threshold selection, and final metrics. TEST (days 152-182) is never
touched by this script. Early stopping on VAL is a legitimate use of VAL for
tuning (this project's leakage rule prohibits using TEST for tuning; VAL
exists exactly for this).
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
from src.features.preprocessing import assert_features_exclude_blocklist
from src.models.metrics import compute_metrics, select_best_f1_threshold
from src.models.tree_features import TreeFeatureBuilder

RANDOM_STATE = 42
EXPERIMENT_NAME = "xgboost"


def main() -> None:
    t_start = time.time()
    cfg = load_config()
    exp_dir = cfg.experiments_dir / EXPERIMENT_NAME
    exp_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading TRAIN and VALIDATION feature frames (TEST not touched) ===")
    train_df = pd.read_parquet(cfg.features_dir / "train_features.parquet")
    val_df = pd.read_parquet(cfg.features_dir / "val_features.parquet")
    print(f"  train: {train_df.shape[0]:,} rows, val: {val_df.shape[0]:,} rows")

    print("\n=== 2. Building tree feature matrix (fit on TRAIN only; XGB variant keeps NaN native) ===")
    t0 = time.time()
    builder = TreeFeatureBuilder(cfg=cfg).fit(train_df)
    n_features = len(builder.feature_names_)
    assert_features_exclude_blocklist(builder.feature_names_, cfg)

    est_train_mb = train_df.shape[0] * n_features * 4 / 1e6
    est_val_mb = val_df.shape[0] * n_features * 4 / 1e6
    print(f"  feature count: {n_features}")
    print(f"  estimated dense matrix memory: TRAIN {est_train_mb:.0f} MB, VAL {est_val_mb:.0f} MB (float32)")
    print("  expected runtime: XGBoost tree_method='hist', up to 1000 rounds with early stopping on "
          f"~417K rows x ~{n_features} features — expect well under a minute (XGBoost's histogram "
          "algorithm is fast at this scale). Proceeding automatically per the current execution policy.")

    X_train = builder.to_matrix(train_df, for_model="xgb")
    X_val = builder.to_matrix(val_df, for_model="xgb")
    y_train = train_df["isFraud"].to_numpy()
    y_val = val_df["isFraud"].to_numpy()
    t_feat = time.time() - t0
    n_nan = np.isnan(X_train).sum()
    print(f"  actual feature-matrix build time: {t_feat:.1f}s")
    print(f"  actual X_train memory: {X_train.nbytes/1e6:.0f} MB, X_val memory: {X_val.nbytes/1e6:.0f} MB")
    print(f"  X_train contains {n_nan:,} NaN values ({100*n_nan/X_train.size:.2f}%) — left native for XGBoost's "
          "learned missing-value split direction, not imputed")

    scale_pos_weight = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))
    print(f"\n=== 3. Training XGBoost ===")
    print(f"  scale_pos_weight={scale_pos_weight:.2f} (neg:pos ratio on TRAIN, addressing 27.58:1 imbalance); "
          "NOT optimizing accuracy, NOT using SMOTE")
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

    print("\n=== 4. Scoring VALIDATION ===")
    val_scores = clf.predict_proba(X_val)[:, 1]

    print("\n=== 5. Threshold selection (VALIDATION only) ===")
    thresh_info = select_best_f1_threshold(y_val, val_scores)
    print(f"  best-F1 threshold: {thresh_info['threshold']:.4f} "
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

    importances = sorted(zip(builder.feature_names_, clf.feature_importances_.tolist()), key=lambda t: -t[1])
    with open(exp_dir / "feature_importances.json", "w", encoding="utf-8") as fh:
        json.dump([{"feature": f, "importance": v} for f, v in importances], fh, indent=1)

    config_record = {
        "experiment": EXPERIMENT_NAME,
        "random_state": RANDOM_STATE,
        "split": {"train_days": [cfg.split.train_day_min, cfg.split.train_day_max],
                  "val_days": [cfg.split.val_day_min, cfg.split.val_day_max],
                  "test_days": [cfg.split.test_day_min, cfg.split.test_day_max]},
        "model": {"type": "XGBClassifier", "n_estimators": 1000, "max_depth": 6, "learning_rate": 0.05,
                  "tree_method": "hist", "scale_pos_weight": scale_pos_weight,
                  "early_stopping_rounds": 50, "best_iteration": int(clf.best_iteration)},
        "feature_count": n_features,
        "encoding_strategy": {
            "continuous_raw_nan_native": len(builder.continuous_cols_),
            "boolean_flags": len(builder.bool_cols_),
            "frequency_encoded": list(builder.freq_encoder_.columns) +
                                 ["ProductCD", "card4", "card6", "DeviceType", "P_emaildomain", "R_emaildomain"],
            "ordinal_encoded_low_cardinality": list(builder.ordinal_maps_.keys()),
            "raw_time_columns": ["hour", "day_of_week"],
            "missing_value_handling": "native XGBoost learned-direction split on real NaN (no imputation)",
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
        "xgboost_version": xgb.__version__,
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

    for f in ["model.joblib", "feature_importances.json", "config.json", "metrics_val.json",
              "predictions_val.parquet", "metadata.json"]:
        print(f"  wrote {exp_dir / f}")
    print(f"  wrote {exp_dir / 'feature_builder'}/ (fitted encoders, ordinal maps, imputer)")

    print(f"\nTotal runtime: {time.time()-t_start:.1f}s. TEST split was not loaded or touched.")


if __name__ == "__main__":
    main()
