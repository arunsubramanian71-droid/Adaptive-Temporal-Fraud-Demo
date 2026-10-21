"""Phase 11 first pass: XGBoost per-transaction explanations for the pre-registered, label-free VAL selection.

Scope record: experiments/phase11_explainable_evidence/SCOPE_AND_ACCEPTANCE.md (P1-P4, A1-A7). VAL days 122-151 only; TEST is never opened;
the label column is never loaded; no model is trained; nothing is installed. Acceptance checks A1-A3 run on the real rows BEFORE any
output is written; if one fails the script exits with status 2 and writes nothing.
Usage:  .venv/Scripts/python.exe -B scripts/run_phase11_explanations.py
"""
from __future__ import annotations

import datetime
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import load_config
from src.explain import outputs as O
from src.explain import xgb_contrib as X
from src.models.tree_features import TreeFeatureBuilder

OUT = ROOT / "experiments" / "phase11_explainable_evidence" / "xgb_explanations_val"
TOL_ADD, TOL_PROB, TOL_BATCH = 1e-4, 1e-6, 1e-5            # fixed in the scope record (A1, A2, A3)
STATEMENT = ("Contributions explain the XGBoost model's output in log-odds margin space. They are not evidence of fraud, causality or "
             "coordination. VAL labels shaped the model's iteration count and threshold, so these VAL explanations are descriptive and "
             "not held-out. Features are anonymised: columns are named, mechanisms are not.")


def main() -> int:
    cfg = load_config()
    exp = cfg.experiments_dir / "xgboost"
    val_path = cfg.features_dir / "val_features.parquet"
    pred_path = exp / "predictions_val.parquet"

    ids_days = X.load_ids_days(val_path, cfg.day_seconds)
    scores = X.load_scores(pred_path)
    merged = ids_days.merge(scores, on="TransactionID", how="inner", validate="one_to_one")
    assert len(merged) == len(ids_days) == len(scores), "VAL ids and stored scores do not align"
    sel = X.select_transactions(merged["TransactionID"], merged["day"], merged["y_score"])

    rows = X.load_feature_rows(val_path, sel["TransactionID"])
    builder = TreeFeatureBuilder.load(cfg, exp / "feature_builder")
    Xs = builder.to_matrix(rows, for_model="xgb")
    names = list(builder.feature_names_)
    model = joblib.load(exp / "model.joblib")
    contrib, bias, margin = X.contributions(model, Xs)

    stored = sel["y_score"].to_numpy(dtype=np.float64)
    add_err = np.abs(contrib.sum(axis=1) + bias - margin)
    prob_err = np.abs(X.sigmoid(margin) - stored)
    proba_err = np.abs(model.predict_proba(Xs)[:, 1].astype(np.float64) - stored)
    batch_err = 0.0
    for i in range(0, len(Xs), 15):                                   # 20 rows evaluated one at a time
        c1, b1, m1 = X.contributions(model, Xs[i:i + 1])
        batch_err = max(batch_err, float(np.abs(c1[0] - contrib[i]).max()), abs(float(b1[0] - bias[i])), abs(float(m1[0] - margin[i])))
    checks = {"A1_max_abs_additivity_error": float(add_err.max()), "A2_max_abs_prob_difference_vs_stored_y_score": float(prob_err.max()),
              "A2_max_abs_predict_proba_difference_vs_stored_y_score": float(proba_err.max()), "A3_max_abs_batch_difference": float(batch_err)}
    print(checks)
    if not (add_err.max() <= TOL_ADD and prob_err.max() <= TOL_PROB and batch_err <= TOL_BATCH):
        print("ACCEPTANCE CHECK FAILED - no output written (scope record: report, never relax)")
        return 2

    sel_out = sel.copy()
    sel_out["margin"], sel_out["bias"] = margin.astype(np.float64), bias.astype(np.float64)
    contrib_df = pd.DataFrame(contrib.astype(np.float64), columns=[f"c::{n}" for n in names])
    contrib_df.insert(0, "margin", margin.astype(np.float64))
    contrib_df.insert(0, "bias", bias.astype(np.float64))
    contrib_df.insert(0, "TransactionID", sel["TransactionID"].to_numpy())
    top = [{"TransactionID": int(t), "day": int(d), "group": g, "y_score": float(s), "margin": float(m), "bias": float(b),
            "top_features": X.top_features(contrib[i], names, Xs[i], 10)}
           for i, (t, d, g, s, m, b) in enumerate(zip(sel["TransactionID"], sel["day"], sel["group"], stored, margin, bias))]

    hashes = {}
    hashes["selection_val.csv"] = O.write_csv(OUT / "selection_val.csv", sel_out)
    hashes["contributions_val.csv"] = O.write_csv(OUT / "contributions_val.csv", contrib_df)
    hashes["top_features_val.json"] = O.write_json(OUT / "top_features_val.json", {"statement": STATEMENT, "transactions": top})
    summary = {
        "scope_record": "experiments/phase11_explainable_evidence/SCOPE_AND_ACCEPTANCE.md",
        "statement": STATEMENT, "n_selected": int(len(sel)), "n_per_group": sel["group"].value_counts().to_dict(),
        "n_days": int(sel["day"].nunique()), "day_range": [int(sel["day"].min()), int(sel["day"].max())],
        "selection": {"top_per_day": X.N_TOP, "uniform_draws_per_day": X.N_DRAW, "seed": X.SELECTION_SEED,
                      "generator": "numpy.random.default_rng(SeedSequence([seed, day])) over the day's non-top transactions sorted by TransactionID",
                      "label_used": False},
        "model": {"path": "experiments/xgboost/model.joblib", "best_iteration": int(model.best_iteration),
                  "iteration_range_used": [0, int(model.best_iteration) + 1], "n_features": len(names), "xgboost_version": xgb.__version__},
        "acceptance_checks": checks, "tolerances": {"A1": TOL_ADD, "A2": TOL_PROB, "A3": TOL_BATCH},
        "mean_abs_margin": float(np.abs(margin).mean()),
    }
    hashes["summary_val.json"] = O.write_json(OUT / "summary_val.json", summary)
    inputs = {str(p.relative_to(ROOT)).replace("\\", "/"): O.sha256_file(p) for p in
              [exp / "model.joblib", val_path, pred_path, ROOT / "experiments/phase11_explainable_evidence/SCOPE_AND_ACCEPTANCE.md"]
              + sorted((exp / "feature_builder").glob("*.json"))}
    manifest = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                "inputs_sha256": inputs,
                "outputs_sha256": {f"experiments/phase11_explainable_evidence/xgb_explanations_val/{k}": v for k, v in hashes.items()},
                "test_features_opened": False, "label_column_loaded": False, "dependencies_installed": False}
    O.write_json(OUT / "manifest_val.json", manifest)
    print("written:", sorted(hashes), "| n_selected", len(sel))
    return 0


if __name__ == "__main__":
    sys.exit(main())
