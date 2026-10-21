"""Evaluate the frozen models of experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL.md.

  readiness gate (validation period, allowed now):
      .venv/Scripts/python.exe -B scripts/run_heldout_evaluation.py --split val
  held-out period (days 152-182): REFUSED unless the user has created experiments/final_heldout_protocol/AUTHORIZATION.txt quoting the
      protocol's SHA-256 and the authorization sentence (protocol section 9). One-shot: refuses to run again once results exist.

`--split val` re-scores the frozen models on VAL through the same code path and compares with the recorded VAL predictions
(tolerances: tabular and B 1e-9 on scores; neural models 1e-4 on scores and 1e-6 on PR-AUC, because GPU kernels are not bit-reproducible).
Nothing is fitted, tuned or re-thresholded.
"""
from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import platform
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.heldout import evaluator as E  # noqa: E402

FOLDER = ROOT / "experiments" / "final_heldout_protocol"
TOL = {"tabular": 1e-9, "b": 1e-9, "static": 1e-4, "temporal": 1e-4}
TOL_PR = {"tabular": 1e-9, "b": 1e-9, "static": 1e-6, "temporal": 1e-6}


def family_of(model_id: str) -> str:
    return E.frozen_registry()[model_id]["family"]


def write(path: pathlib.Path, text: str):
    import hashlib
    data = text.encode("utf-8")
    path.write_bytes(data)
    path.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))


def readiness(res: dict, out: pathlib.Path) -> bool:
    from sklearn.metrics import average_precision_score
    reg = E.frozen_registry()
    lab = res["labels"].set_index("TransactionID")
    rows, ok_all = [], True
    for mid, df in sorted(res["scores"].items()):
        d = ROOT / reg[mid]["dir"]
        rec = pd.read_parquet(d / "predictions_val.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
        got = df.set_index("TransactionID")["y_score"]
        same_ids = len(got) == len(rec) and got.index.isin(rec.index).all()
        diff = float(np.max(np.abs(got.loc[rec.index].to_numpy() - rec.to_numpy()))) if same_ids else float("nan")
        y = lab.loc[rec.index, "isFraud"].to_numpy()
        pr_new, pr_old = float(average_precision_score(y, got.loc[rec.index].to_numpy())), float(average_precision_score(y, rec.to_numpy()))
        fam = reg[mid]["family"]
        ok = bool(same_ids and diff <= TOL[fam] and abs(pr_new - pr_old) <= TOL_PR[fam])
        ok_all &= ok
        rows.append({"model": mid, "family": fam, "ids_identical": bool(same_ids), "max_abs_score_diff": diff, "pr_auc_recomputed": pr_new, "pr_auc_recorded_scores": pr_old,
                     "tolerance_score": TOL[fam], "tolerance_pr_auc": TOL_PR[fam], "pass": ok})
        print(f"[{'ok ' if ok else 'FAIL'}] {mid:<32} max|diff| {diff:.2e}  PR-AUC {pr_new:.6f} vs {pr_old:.6f}", flush=True)
    rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "split": "val", "all_pass": ok_all,
           "python": platform.python_version(), "models": rows,
           "statement": "Validation-period re-scoring of the frozen models for the code-readiness gate. No held-out data was read."}
    write(out / "readiness_report.json", json.dumps(rec, indent=1) + "\n")
    return ok_all


def heldout(res: dict, out: pathlib.Path) -> None:
    reg = E.frozen_registry()
    lab = res["labels"]
    rows = []
    for mid, df in sorted(res["scores"].items()):
        m = lab.merge(df, on="TransactionID", how="inner", validate="one_to_one")
        if len(m) != len(lab):
            raise RuntimeError(f"{mid}: scored {len(m)} of {len(lab)} held-out rows")
        thr = E.recorded_threshold(ROOT / reg[mid]["dir"])
        ev = E.evaluate_frame_metrics(m["isFraud"], m["y_score"], m["segment"], m["day"], thr)
        m[["TransactionID", "isFraud", "y_score"]].to_parquet(out / f"predictions_heldout_{mid}.parquet", index=False)
        p = ev["pooled"]
        rows.append({"model": mid, "family": reg[mid]["family"], "recorded_val_threshold": thr, "pr_auc": p["pr_auc"], "roc_auc": p["roc_auc"],
                     **{f"pr_auc_{s}": ev[s]["pr_auc"] for s in ("has_edge_warm", "has_edge_cold", "no_pseudo_account_edge")},
                     "precision_at_1pct": p["top_k"]["1%"]["precision_at_k"], "recall_at_1pct": p["top_k"]["1%"]["recall_at_k"],
                     "precision": p["precision"], "recall": p["recall"], "f1": p["f1"], "n_alerts": p["n_alerts"], "false_positives": p["false_positives"],
                     "n_rows": p["n_rows"], "n_fraud": p["n_fraud"]})
    pd.DataFrame(rows).to_csv(out / "heldout_metrics.csv", index=False, float_format="%.10g", lineterminator="\n")
    manifest = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "authorization": res["auth"],
                "models": {mid: {"dir": reg[mid]["dir"], "files_sha256": {f: E.sha256_file(ROOT / reg[mid]["dir"] / f) for f in reg[mid]["files"]}} for mid in sorted(res["scores"])},
                "statement": "One-shot held-out evaluation of frozen models; no model, threshold or setting was changed in response to these results."}
    write(out / "manifest_heldout.json", json.dumps(manifest, indent=1) + "\n")
    (out / ".completed").write_text(manifest["created_local"], encoding="utf-8")
    print(f"held-out metrics for {len(rows)} models written to {out}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "heldout"], required=True)
    ap.add_argument("--families", default=",".join(E.FAMILIES))
    a = ap.parse_args(argv)
    fams = tuple(f for f in a.families.split(",") if f)
    try:
        if a.split == "heldout":
            out = FOLDER / "heldout_results"
            if (out / ".completed").exists():
                print("error: the held-out evaluation has already been executed; it is one-shot (protocol section 7).", file=sys.stderr)
                return 2
            E.require_authorization("heldout")                  # raises before anything is read
            out.mkdir(parents=True, exist_ok=True)
            heldout(E.run("heldout", fams), out)
            return 0
        out = FOLDER / "readiness_val"
        out.mkdir(parents=True, exist_ok=True)
        ok = readiness(E.run("val", fams), out)
        print("READINESS:", "PASS" if ok else "FAIL (see readiness_report.json)")
        return 0 if ok else 1
    except E.AuthorizationError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
