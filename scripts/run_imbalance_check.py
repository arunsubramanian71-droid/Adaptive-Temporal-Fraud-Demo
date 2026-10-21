"""Class-imbalance check for the comparison-A XGBoost (experiments/imbalance_check_xgboost/PREREGISTRATION_IMBALANCE.md).

Fits two new arms (no class weighting; square-root weighting) with the comparison-A recipe and compares them with the recorded
full-weight arm (the B pipeline's A_refit). Reads only the files named in the pre-registration. Writes only into
experiments/imbalance_check_xgboost/outputs/.

Usage:  .venv/Scripts/python.exe -B scripts/run_imbalance_check.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import platform
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
import xgboost as xgb  # noqa: E402

import run_comparison_b as RB  # noqa: E402  (shared constants; main() is not run on import)
from src.config import load_config  # noqa: E402
from src.evidence import table as T  # noqa: E402
from src.features.preprocessing import assert_features_exclude_blocklist  # noqa: E402
from src.models.metrics import select_best_f1_threshold  # noqa: E402
from src.models.tree_features import TreeFeatureBuilder  # noqa: E402

EXP = "experiments/imbalance_check_xgboost"
OUT = ROOT / EXP / "outputs"
PREREG, PREREG_SIDECAR = f"{EXP}/PREREGISTRATION_IMBALANCE.md", f"{EXP}/PREREGISTRATION_IMBALANCE.sha256"
FULL = "experiments/comparison_B_graph_features/outputs/arms/A_refit/metrics_val.json"
ALLOWED = [PREREG, PREREG_SIDECAR, RB.TRAIN_F, RB.VAL_F, T.REF_REL, FULL]
TOL = 0.005


def verdict(prs: dict) -> dict:
    vals = list(prs.values())
    spread = max(vals) - min(vals)
    others = max(v for k, v in prs.items() if k != "full")
    if spread <= TOL:
        v = "WEIGHTING MAKES NO MATERIAL DIFFERENCE (threshold-free metric)"
    elif prs["full"] - others > TOL:
        v = "WEIGHTING HELPS"
    elif others - prs["full"] > TOL:
        v = "WEIGHTING HURTS"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "spread": float(spread), "tolerance": TOL, "pr_auc": prs}


def write(name, data: bytes):
    (OUT / name).write_bytes(data)
    h = hashlib.sha256(data).hexdigest()
    (OUT / (name.rsplit(".", 1)[0] + ".sha256")).write_bytes(h.encode("ascii"))
    return h


def main() -> int:
    t0 = time.time()
    (OUT / "arms").mkdir(parents=True, exist_ok=True)
    reader = RB.Reader(ROOT, allowed=ALLOWED, extra=())
    prereg_hash = reader.sha256(PREREG)
    assert prereg_hash == reader.read_text(PREREG_SIDECAR).strip(), "pre-registration does not match its sidecar"
    cfg = load_config()
    train_df, val_df = reader.read_parquet(RB.TRAIN_F), reader.read_parquet(RB.VAL_F)
    ref = T.make_reference(reader)
    builder = TreeFeatureBuilder(cfg=cfg).fit(train_df)
    assert_features_exclude_blocklist(builder.feature_names_, cfg)
    Xt, Xv = builder.to_matrix(train_df, for_model="xgb"), builder.to_matrix(val_df, for_model="xgb")
    yt, yv = train_df["isFraud"].to_numpy(), val_df["isFraud"].to_numpy()
    ratio = float((yt == 0).sum() / max((yt == 1).sum(), 1))
    spw = {"none": 1.0, "sqrt": float(np.sqrt(ratio))}
    ids = val_df["TransactionID"].to_numpy()
    results = {}
    for name, w in spw.items():
        done = OUT / "arms" / name / "metrics_val.json"
        if done.exists():
            results[name] = json.loads(done.read_text(encoding="utf-8"))
            print(f"[skip] {name}")
            continue
        (OUT / "arms" / name).mkdir(exist_ok=True)
        clf = xgb.XGBClassifier(**RB.BASE, scale_pos_weight=w, random_state=42)
        t1 = time.time()
        clf.fit(Xt, yt, eval_set=[(Xv, yv)], verbose=False)
        fit_s = time.time() - t1
        sc = clf.predict_proba(Xv)[:, 1]
        thr = select_best_f1_threshold(yv, sc)
        frame = T.attach_missing(pd.DataFrame({"TransactionID": ids, "isFraud": yv, "y_score": sc}), ref)
        probs = T.population_problems(ref, frame, name)
        if probs:
            raise SystemExit(f"BLOCKED: {probs}")
        ev = T.evaluate_frame(frame, thr["threshold"], windows=False)
        rec = {"arm": name, "scale_pos_weight": w, "best_iteration": int(clf.best_iteration), "fit_seconds": round(fit_s, 1), "threshold_selection": thr,
               "pooled": ev["pooled"], "segments": {k: ev[k] for k in T.SEGMENTS}, "pr_auc": ev["pooled"]["pr_auc"]}
        done.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        results[name] = rec
        print(f"[done] {name}: scale_pos_weight {w:.2f} PR-AUC {rec['pr_auc']:.4f} ROC {ev['pooled']['roc_auc']:.4f} alerts {ev['pooled']['n_alerts']} iter {rec['best_iteration']} fit {fit_s:.0f}s", flush=True)
    full = json.loads(reader.read_text(FULL))
    assert abs(full["config"]["base_params"]["scale_pos_weight"] - ratio) < 1e-9 and full["config"]["arm"] == "A_refit"
    results["full"] = {"arm": "full", "scale_pos_weight": ratio, "best_iteration": full["best_iteration"], "pooled": full["pooled"], "segments": full["segments"], "pr_auc": full["pr_auc"]}
    dec = verdict({k: results[k]["pr_auc"] for k in ("none", "sqrt", "full")})
    rows = []
    for k in ("none", "sqrt", "full"):
        r = results[k]
        p = r["pooled"]
        rows.append((k, r["scale_pos_weight"], r["best_iteration"], p["pr_auc"], p["roc_auc"], r["segments"]["has_edge_cold"]["pr_auc"], p["top_k"]["1%"]["precision_at_k"],
                     p["top_k"]["1%"]["recall_at_k"], p["threshold"], p["n_alerts"], p["precision"], p["recall"], p["f1"]))
    cols = ["arm", "scale_pos_weight", "best_iteration", "pr_auc", "roc_auc", "pr_auc_cold", "precision_at_1pct", "recall_at_1pct", "threshold", "n_alerts", "precision", "recall", "f1"]
    table = pd.DataFrame(rows, columns=cols)
    write("arms_table.csv", table.to_csv(index=False, float_format="%.10g", lineterminator="\n").encode("utf-8"))
    md = ["# Class-imbalance check (comparison-A XGBoost): results\n",
          "**VAL-period (days 122-151), one seed per arm, VAL used for early stopping and the best-F1 threshold. No TEST data was read. Descriptive.**\n",
          f"Pre-registered verdict: **{dec['verdict']}** (PR-AUC spread {dec['spread']:.4f}; tolerance {TOL}).\n",
          "| Arm | scale_pos_weight | Best iteration | PR-AUC | ROC-AUC | PR-AUC cold-start | P@1% | R@1% | Alerts at best-F1 threshold | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        md.append(f"| {r.arm} | {r.scale_pos_weight:.2f} | {r.best_iteration} | {r.pr_auc:.4f} | {r.roc_auc:.4f} | {r.pr_auc_cold:.4f} | {r.precision_at_1pct:.4f} | {r.recall_at_1pct:.4f} | {r.n_alerts} | {r.precision:.4f} | {r.recall:.4f} | {r.f1:.4f} |")
    md += ["", "Scope: the XGBoost comparison-A recipe only. The weighted loss of the neural graph models was not re-evaluated and focal loss was not tried. "
           "A threshold-free metric can be indifferent to weighting while alert counts and calibration change; the alert columns show that effect at the best-F1 threshold of each arm.", ""]
    h_md = write("IMBALANCE_RESULTS.md", ("\n".join(md)).encode("utf-8"))
    h_dec = write("decision.json", (json.dumps(dec, indent=1) + "\n").encode("utf-8"))
    manifest = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "preregistration_sha256": prereg_hash,
                "inputs_sha256": {p: reader.sha256(p) for p in (RB.TRAIN_F, RB.VAL_F, T.REF_REL, FULL)}, "outputs_sha256": {"IMBALANCE_RESULTS.md": h_md, "decision.json": h_dec},
                "io_audit": reader.audit(), "test_data_read": False, "versions": {"python": platform.python_version(), "xgboost": xgb.__version__}, "seconds": round(time.time() - t0, 1)}
    write("manifest_imbalance.json", (json.dumps(manifest, indent=1) + "\n").encode("utf-8"))
    print(json.dumps(dec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
