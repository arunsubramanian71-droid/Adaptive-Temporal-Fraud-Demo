"""Unweighted XGBoost arms of the follow-up (experiments/imbalance_check_xgboost/PREREGISTRATION_UNWEIGHTED_FOLLOWUP.md): B_none and the
subsampled A_none_sub / B_none_sub arms, `scale_pos_weight` = 1. A_none (seed 42, A features) already exists as arm `none` of the imbalance check.

Same inputs, guards and metric code as scripts/run_comparison_b.py; writes only into experiments/imbalance_check_xgboost/outputs/arms_unweighted/.
Usage:  .venv/Scripts/python.exe -B scripts/run_unweighted_xgboost.py
"""
from __future__ import annotations

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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import run_comparison_b as RB  # noqa: E402
from src.evidence import table as T  # noqa: E402
from src.features import graph_features as G  # noqa: E402
from src.features.preprocessing import assert_features_exclude_blocklist  # noqa: E402
from src.models.metrics import select_best_f1_threshold  # noqa: E402
from src.models.tree_features import TreeFeatureBuilder  # noqa: E402

EXP = "experiments/imbalance_check_xgboost"
OUT = ROOT / EXP / "outputs" / "arms_unweighted"
PREREG = f"{EXP}/PREREGISTRATION_UNWEIGHTED_FOLLOWUP.md"
PREREG_SIDECAR = f"{EXP}/PREREGISTRATION_UNWEIGHTED_FOLLOWUP.sha256"
PREREG_B = RB.PREREG
ARMS = [("B_none", "B", 42, {}), *[(f"A_none_sub_s{s}", "A", s, {"subsample": 0.8, "colsample_bytree": 0.8}) for s in RB.SEEDS],
        *[(f"B_none_sub_s{s}", "B", s, {"subsample": 0.8, "colsample_bytree": 0.8}) for s in RB.SEEDS]]


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    reader = RB.Reader(ROOT, allowed=[*RB.ALLOWED, PREREG, PREREG_SIDECAR], extra=())
    pre = reader.sha256(PREREG)
    assert pre == reader.read_text(PREREG_SIDECAR).strip(), "follow-up pre-registration does not match its sidecar"
    cfg, train_df, val_df, z, ents, checks = RB.load_inputs(reader)
    if not all(checks.values()):
        raise SystemExit(f"BLOCKED: {checks}")
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
    ref = T.make_reference(reader)
    ids = val_df["TransactionID"].to_numpy()
    for name, fset, seed, extra in ARMS:
        d = OUT / name
        cfg_rec = {"arm": name, "features": fset, "seed": seed, "extra_params": extra, "scale_pos_weight": 1.0, "base_params": RB.BASE}
        if (d / "metrics_val.json").exists() and json.loads((d / "metrics_val.json").read_text(encoding="utf-8")).get("config") == cfg_rec:
            print(f"[skip] {name}")
            continue
        d.mkdir(exist_ok=True)
        X, cols = (Xa, a_cols) if fset == "A" else (Xb, b_cols)
        clf, score, used, fit_s = RB.fit_arm(name, fset, seed, extra, None, X, y, 1.0, cols, n_train, {})
        yv = y[n_train:]
        thr = select_best_f1_threshold(yv, score)
        frame = T.attach_missing(pd.DataFrame({"TransactionID": ids, "isFraud": yv, "y_score": score}), ref)
        probs = T.population_problems(ref, frame, name)
        if probs:
            raise SystemExit(f"BLOCKED: {probs}")
        ev = T.evaluate_frame(frame, thr["threshold"], windows=False)
        rec = {"config": cfg_rec, "n_features": len(used), "best_iteration": int(clf.best_iteration), "fit_seconds": round(fit_s, 1), "threshold_selection": thr,
               "pooled": ev["pooled"], "segments": {k: ev[k] for k in T.SEGMENTS}, "pr_auc": ev["pooled"]["pr_auc"], "pr_auc_cold": ev["has_edge_cold"]["pr_auc"]}
        (d / "metrics_val.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        pd.DataFrame({"TransactionID": ids, "isFraud": yv, "y_score": score}).to_parquet(d / "predictions_val.parquet", index=False)
        print(f"[done] {name}: PR-AUC {rec['pr_auc']:.4f} cold {rec['pr_auc_cold']:.4f} iter {rec['best_iteration']} fit {fit_s:.0f}s", flush=True)
    (OUT / "run_record.json").write_text(json.dumps({"preregistration_sha256": pre, "io_audit": reader.audit(), "test_data_read": False,
                                                     "seconds": round(time.time() - t0, 1)}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
