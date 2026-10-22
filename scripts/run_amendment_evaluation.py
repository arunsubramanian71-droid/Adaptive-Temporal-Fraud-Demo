"""Amendment 1 modules H and U of the held-out protocol (experiments/final_heldout_protocol/AMENDMENT_1.md). Module F (adaptive arms) is scripts/run_f_walkforward.py.

  readiness gate (validation period, allowed now):
      .venv/Scripts/python.exe -B scripts/run_amendment_evaluation.py --split val --modules H,U
  held-out period (days 152-182): REFUSED unless AUTHORIZATION.txt quotes the protocol hash and sentence, the amendment hash, and a sentence
      'I authorize the following Amendment 1 modules: <letters>' naming every requested module. H and U share one output folder and one one-shot flag, so
      --modules must name exactly the H/U modules that AUTHORIZATION.txt authorizes. One-shot: refused once the output folder holds any file.

Nothing is fitted, tuned or re-thresholded; thresholds are the recorded validation best-F1 values. The existing evaluator, its protocol and its 19-model registry are untouched.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import platform
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.heldout import amendment as A  # noqa: E402
from src.heldout import codehash as CH  # noqa: E402
from src.heldout import evaluator as E  # noqa: E402
from src.heldout import oneshot as OS  # noqa: E402

FOLDER = ROOT / "experiments" / "final_heldout_protocol"
HIST_REC = ROOT / A.HIST_DIR / "predictions_val.parquet"
B_NONE_REC = ROOT / "experiments/imbalance_check_xgboost/outputs/arms_unweighted/B_none/predictions_val.parquet"
A_NONE_METRICS = ROOT / "experiments/imbalance_check_xgboost/outputs/arms/none/metrics_val.json"
TOL = {"xgb": (1e-9, 1e-9), "neural": (1e-4, 1e-6)}           # (score, PR-AUC), as in the base readiness gate


def write(path: pathlib.Path, text: str):
    data = text.encode("utf-8")
    path.write_bytes(data)
    path.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))


def recorded_scores(model_id: str):
    if model_id == "xgboost_historical":
        return pd.read_parquet(HIST_REC, columns=["TransactionID", "y_score"]), "xgb"
    if model_id == "B_none":
        return pd.read_parquet(B_NONE_REC, columns=["TransactionID", "y_score"]), "xgb"
    if model_id == "A_none":
        return None, "xgb"
    pat = (A.UNW_STATIC.get(model_id.rsplit("_s", 1)[0]) or A.UNW_TEMPORAL[model_id.rsplit("_s", 1)[0]])[0]
    return pd.read_parquet(ROOT / pat.format(s=int(model_id.rsplit("_s", 1)[1])) / "predictions_val.parquet", columns=["TransactionID", "y_score"]), "neural"


def readiness(res: dict, out: pathlib.Path) -> bool:
    from sklearn.metrics import average_precision_score
    lab = res["labels"].set_index("TransactionID")
    rows, ok_all = [], True
    for mid, df in sorted(res["scores"].items()):
        rec, fam = recorded_scores(mid)
        got = df.set_index("TransactionID")["y_score"]
        tol_s, tol_p = TOL[fam]
        if rec is None:                                   # A_none: only metrics were recorded
            y = lab.loc[got.index, "isFraud"].to_numpy()
            pr_new, pr_old = float(average_precision_score(y, got.to_numpy())), float(json.loads(A_NONE_METRICS.read_text(encoding="utf-8"))["pr_auc"])
            ok = bool(len(got) == len(lab) and abs(pr_new - pr_old) <= tol_p)
            row = {"model": mid, "compared_against": "recorded PR-AUC (no stored predictions)", "ids_identical": True, "max_abs_score_diff": None, "pr_auc_recomputed": pr_new, "pr_auc_recorded": pr_old}
        else:
            r = rec.set_index("TransactionID")["y_score"]
            same = len(got) == len(r) and bool(got.index.isin(r.index).all())
            diff = float(np.max(np.abs(got.loc[r.index].to_numpy() - r.to_numpy()))) if same else float("nan")
            y = lab.loc[r.index, "isFraud"].to_numpy()
            pr_new, pr_old = float(average_precision_score(y, got.loc[r.index].to_numpy())), float(average_precision_score(y, r.to_numpy()))
            ok = bool(same and diff <= tol_s and abs(pr_new - pr_old) <= tol_p)
            row = {"model": mid, "compared_against": "recorded validation predictions", "ids_identical": bool(same), "max_abs_score_diff": diff, "pr_auc_recomputed": pr_new, "pr_auc_recorded": pr_old}
        row.update({"tolerance_score": tol_s, "tolerance_pr_auc": tol_p, "pass": ok})
        ok_all &= ok
        rows.append(row)
        print(f"[{'ok ' if ok else 'FAIL'}] {mid:<34} PR-AUC {pr_new:.6f} vs {pr_old:.6f}  max|diff| {row['max_abs_score_diff']}", flush=True)
    rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "split": "val", "all_pass": ok_all, "python": platform.python_version(),
           "models": rows, "code_sha256": CH.loaded_project_code(), "statement": "Validation-period re-scoring of the Amendment 1 frozen models (modules H, U) for the code-readiness gate. No held-out data was read."}
    write(out / "readiness_report.json", json.dumps(rec, indent=1) + "\n")
    return ok_all


def heldout(res: dict, out: pathlib.Path) -> None:
    lab = res["labels"]
    rows = []
    for mid, df in sorted(res["scores"].items()):
        m = lab.merge(df, on="TransactionID", how="inner", validate="one_to_one")
        if len(m) != len(lab):
            raise RuntimeError(f"{mid}: scored {len(m)} of {len(lab)} held-out rows")
        thr = A.recorded_threshold_hist() if mid == "xgboost_historical" else A.recorded_threshold_u(mid)
        ev = E.evaluate_frame_metrics(m["isFraud"], m["y_score"], m["segment"], m["day"], thr)
        m[["TransactionID", "isFraud", "y_score"]].to_parquet(out / f"predictions_heldout_{mid}.parquet", index=False)
        p = ev["pooled"]
        rows.append({"model": mid, "recorded_val_threshold": thr, "pr_auc": p["pr_auc"], "roc_auc": p["roc_auc"],
                     **{f"pr_auc_{s}": ev[s]["pr_auc"] for s in ("has_edge_warm", "has_edge_cold", "no_pseudo_account_edge")},
                     "precision_at_1pct": p["top_k"]["1%"]["precision_at_k"], "recall_at_1pct": p["top_k"]["1%"]["recall_at_k"], "precision": p["precision"], "recall": p["recall"], "f1": p["f1"],
                     "n_alerts": p["n_alerts"], "false_positives": p["false_positives"], "n_rows": p["n_rows"], "n_fraud": p["n_fraud"]})
    pd.DataFrame(rows).to_csv(out / "heldout_metrics_amendment.csv", index=False, float_format="%.10g", lineterminator="\n")
    manifest = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "authorization": res["auth"], "models": sorted(res["scores"]),
                "statement": "One-shot held-out evaluation of the Amendment 1 frozen models; no model, threshold or setting was changed in response to these results."}
    write(out / "manifest_heldout_amendment.json", json.dumps(manifest, indent=1) + "\n")
    (out / ".completed").write_text(manifest["created_local"], encoding="utf-8")
    print(f"held-out metrics for {len(rows)} models written to {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "heldout"], required=True)
    ap.add_argument("--modules", default="H,U")
    a = ap.parse_args(argv)
    mods = tuple(m.strip() for m in a.modules.split(",") if m.strip())
    if not mods or len(set(mods)) != len(mods) or not set(mods) <= {"H", "U"}:
        print(f"error: --modules must name H and/or U once each (module F is run by scripts/run_f_walkforward.py); got {a.modules!r}", file=sys.stderr)
        return 2
    try:
        if a.split == "heldout":
            out = FOLDER / "heldout_results_amendment"
            OS.require_unused(out, "the amendment held-out evaluation")   # Amendment 1 section 7: any held-out output means the run was executed
            auth = A.require_amendment_authorization("heldout", mods)   # raises before anything is read
            granted = set(auth.get("amendment_modules_authorized", ())) & {"H", "U"}
            if set(mods) != granted:
                print(f"error: AUTHORIZATION.txt authorizes the H/U modules {sorted(granted)}, but --modules requests {sorted(mods)}. H and U share one output folder "
                      f"and one one-shot flag (Amendment 1 section 7), so they must run together in one invocation. Nothing was read.", file=sys.stderr)
                return 2
            A.require_amendment_readiness(mods)                         # registered readiness gates must have passed
            out.mkdir(parents=True, exist_ok=True)
            heldout(A.run_modules("heldout", mods), out)
            return 0
        out = FOLDER / "readiness_val_amendment"
        out.mkdir(parents=True, exist_ok=True)
        ok = readiness(A.run_modules("val", mods), out)
        print("READINESS:", "PASS" if ok else "FAIL (see readiness_report.json)")
        return 0 if ok else 1
    except (E.AuthorizationError, A.ReadinessError, OS.OneShotError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
