"""Freeze the two unweighted XGBoost models of held-out amendment 1, module U (experiments/final_heldout_protocol/AMENDMENT_1.md): A_none (A features) and B_none (A + 40 graph features),
comparison-A recipe, `scale_pos_weight` = 1, seed 42, early stopping on validation as in every other XGBoost arm.

The unweighted follow-up kept metrics and predictions but not the model files, so the models are re-fitted here with the identical recipe and inputs (training and validation data only) and accepted
ONLY if they reproduce the recorded validation results: B_none within 1e-9 on every validation score and the same best iteration; A_none (which has recorded metrics but no stored predictions) the same
pooled PR-AUC to 1e-9 and the same best iteration. If a model does not reproduce, nothing is saved as frozen. No held-out file is read.

Usage:  .venv/Scripts/python.exe -B scripts/freeze_unweighted_xgboost.py
Writes experiments/final_heldout_protocol/frozen_amendment/{A_none,B_none}/ (model.joblib, feature_columns.json, reproduction.json) with sidecars.
"""
from __future__ import annotations

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
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

import run_comparison_b as RB  # noqa: E402
from src.features import graph_features as G  # noqa: E402
from src.features.preprocessing import assert_features_exclude_blocklist  # noqa: E402
from src.models.tree_features import TreeFeatureBuilder  # noqa: E402

FH = ROOT / "experiments" / "final_heldout_protocol"
OUT = FH / "frozen_amendment"
REC_A = ROOT / "experiments/imbalance_check_xgboost/outputs/arms/none/metrics_val.json"
REC_B_DIR = ROOT / "experiments/imbalance_check_xgboost/outputs/arms_unweighted/B_none"
ALLOWED = [RB.TRAIN_F, RB.VAL_F, RB.CACHE]


def wr(path: pathlib.Path, data: bytes):
    path.write_bytes(data)
    path.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))


def main() -> int:
    import xgboost as xgb
    t0 = time.time()
    reader = RB.Reader(ROOT, allowed=ALLOWED, extra=())
    cfg, train_df, val_df, z, ents, checks = RB.load_inputs(reader)
    assert all(checks.values()), checks
    n_train = len(train_df)
    full = pd.concat([train_df, val_df], ignore_index=True)
    gf = G.compute_graph_features(full["TransactionDT"].to_numpy(dtype=np.int64), full["TransactionAmt"].to_numpy(dtype=np.float64), ents)
    builder = TreeFeatureBuilder(cfg=cfg).fit(train_df)
    assert_features_exclude_blocklist(builder.feature_names_, cfg)
    Xa = np.vstack([builder.to_matrix(train_df, for_model="xgb"), builder.to_matrix(val_df, for_model="xgb")]).astype(np.float32)
    a_cols = list(builder.feature_names_)
    Xb = np.hstack([Xa, gf.to_numpy(dtype=np.float32)])
    b_cols = a_cols + G.feature_names()
    y = np.concatenate([train_df["isFraud"].to_numpy(), val_df["isFraud"].to_numpy()])
    yv = y[n_train:]
    ids = val_df["TransactionID"].to_numpy()
    report = {}
    for name, X, cols in (("A_none", Xa, a_cols), ("B_none", Xb, b_cols)):
        clf = xgb.XGBClassifier(**RB.BASE, scale_pos_weight=1.0, random_state=42)
        clf.fit(X[:n_train], y[:n_train], eval_set=[(X[n_train:], yv)], verbose=False)
        score = clf.predict_proba(X[n_train:])[:, 1]
        pr = float(average_precision_score(yv, score))
        if name == "B_none":
            rec = pd.read_parquet(REC_B_DIR / "predictions_val.parquet").set_index("TransactionID")["y_score"]
            diff = float(np.abs(pd.Series(score, index=ids).loc[rec.index].to_numpy() - rec.to_numpy()).max())
            rec_it = json.loads((REC_B_DIR / "metrics_val.json").read_text(encoding="utf-8"))["best_iteration"]
            ok = diff <= 1e-9 and int(clf.best_iteration) == rec_it
            check = {"max_abs_score_difference_vs_recorded_predictions": diff, "recorded_best_iteration": rec_it}
        else:
            rec = json.loads(REC_A.read_text(encoding="utf-8"))
            diff = abs(pr - rec["pr_auc"])
            ok = diff <= 1e-9 and int(clf.best_iteration) == rec["best_iteration"]
            check = {"abs_pr_auc_difference_vs_recorded": diff, "recorded_best_iteration": rec["best_iteration"]}
        report[name] = {"reproduces_recorded_validation_result": bool(ok), "best_iteration": int(clf.best_iteration), "pr_auc": pr, **check}
        print(f"{name}: PR-AUC {pr:.6f} best_iter {clf.best_iteration} reproduces recorded: {ok}", flush=True)
        if ok:
            d = OUT / name
            d.mkdir(parents=True, exist_ok=True)
            joblib.dump(clf, d / "model.joblib")
            (d / "model.sha256").write_bytes(hashlib.sha256((d / "model.joblib").read_bytes()).hexdigest().encode("ascii"))
            wr(d / "feature_columns.json", json.dumps(cols).encode("utf-8"))
    rec_all = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "models": report, "inputs": {p: reader.sha256(p) for p in ALLOWED},
               "recipe": {**RB.BASE, "scale_pos_weight": 1.0, "random_state": 42}, "test_data_read": False, "io_audit": reader.audit(), "seconds": round(time.time() - t0, 1)}
    OUT.mkdir(parents=True, exist_ok=True)
    wr(OUT / "reproduction.json", (json.dumps(rec_all, indent=1) + "\n").encode("utf-8"))
    return 0 if all(r["reproduces_recorded_validation_result"] for r in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
