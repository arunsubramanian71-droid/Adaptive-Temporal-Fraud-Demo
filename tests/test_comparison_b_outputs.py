"""Comparison B, imbalance check and graph evidence: verification of the RECORDED outputs against their pre-registered acceptance criteria.

Reads only recorded outputs and the Phase 12 reference predictions (VAL, ids/labels/segments); no model is fitted or scored, no TEST file is
opened. Criterion labels (BA*, IA*, GE*) are those of the pre-registration records.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.lint import lint_files  # noqa: E402
from src.evidence import table as T  # noqa: E402
from src.features import b_decision as D  # noqa: E402
from src.features import graph_features as G  # noqa: E402
from src.models.metrics import compute_metrics  # noqa: E402

B = ROOT / "experiments" / "comparison_B_graph_features"
OUT = B / "outputs"
IM = ROOT / "experiments" / "imbalance_check_xgboost"
GEV = B / "graph_evidence"
SEEDS = (42, 123, 2024)
_C = {}


def sha(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def arm(name):
    return json.loads((OUT / "arms" / name / "metrics_val.json").read_text(encoding="utf-8"))


def ref():
    if "ref" not in _C:
        _C["ref"] = pd.read_parquet(ROOT / T.REF_REL, columns=["TransactionID", "isFraud", "day", "segment"]).sort_values("TransactionID").reset_index(drop=True)
    return _C["ref"]


def arm_names():
    return ["A_refit", "B_full", *[f"A_sub_s{s}" for s in SEEDS], *[f"B_sub_s{s}" for s in SEEDS], *[f"B_{g}" for g in G.ablation_groups()]]


def test_ba1_preregistration_precedes_every_output_and_hashes_match():
    pre = B / "PREREGISTRATION_B.md"
    assert sha(pre) == (B / "PREREGISTRATION_B.sha256").read_text().strip()
    rec = json.loads((OUT / "run_record.json").read_text(encoding="utf-8"))
    assert rec["preregistration_sha256"] == sha(pre)
    t0 = os.stat(pre).st_mtime
    produced = [OUT / "arms" / a / "metrics_val.json" for a in arm_names()] + [OUT / "decision.json", ROOT / "src" / "features" / "graph_features.py"]
    assert all(os.stat(p).st_mtime > t0 for p in produced), "an output or the feature code is older than the pre-registration"


def test_all_thirteen_arms_exist_and_follow_the_registered_configuration():
    names = arm_names()
    assert len(names) == 13
    for a in names:
        c = arm(a)["config"]
        assert c["arm"] == a and c["base_params"]["n_estimators"] == 1000 and c["base_params"]["max_depth"] == 6 and c["base_params"]["learning_rate"] == 0.05
        assert c["base_params"]["early_stopping_rounds"] == 50 and c["base_params"]["tree_method"] == "hist"
        if "_sub_" in a:
            assert c["extra_params"] == {"subsample": 0.8, "colsample_bytree": 0.8} and c["seed"] == int(a.rsplit("s", 1)[1])
        else:
            assert c["extra_params"] == {} and c["seed"] == 42
        assert arm(a)["n_features"] == (470 if a.startswith("A_") else 510 - (len(G.ablation_groups()[a[2:]]) if a[2:] in G.ablation_groups() else 0))


def test_ba4_only_registered_inputs_were_opened_and_no_test_named_path():
    rec = json.loads((OUT / "run_record.json").read_text(encoding="utf-8"))
    opened = rec["io_audit"]["opened"]
    allowed = {"experiments/comparison_B_graph_features/PREREGISTRATION_B.md", "experiments/comparison_B_graph_features/PREREGISTRATION_B.sha256",
               "data/features/train_features.parquet", "data/features/val_features.parquet", "experiments/phase9_coordinated_fraud/cache/real_le151.npz",
               T.REF_REL, "experiments/phase12_validation_evidence_table/outputs/evidence_runs.csv"}
    assert set(opened) <= allowed, set(opened) - allowed
    assert not any("test" in p.lower().replace("latest", "") for p in opened) and rec["test_data_read"] is False
    assert all(rec["input_checks"].values()) and all(v["bit_identical"] for v in rec["truncation_check"].values())


def test_ba5_each_arm_scores_exactly_the_reference_population():
    r = ref()
    for a in arm_names():
        p = pd.read_parquet(OUT / "arms" / a / "predictions_val.parquet").sort_values("TransactionID").reset_index(drop=True)
        assert len(p) == len(r) == 83655 and np.array_equal(p["TransactionID"].to_numpy(), r["TransactionID"].to_numpy()), a
        assert np.array_equal(p["isFraud"].to_numpy().astype(int), r["isFraud"].to_numpy().astype(int)), a


def test_ba7_recorded_metrics_equal_a_recomputation_from_the_stored_predictions():
    r = ref()
    for a in ("A_refit", "B_full", "B_sub_s42", "B_drop_dev"):
        p = pd.read_parquet(OUT / "arms" / a / "predictions_val.parquet")
        frame = T.attach_missing(p[["TransactionID", "isFraud", "y_score"]], r)
        rec = arm(a)
        ev = T.evaluate_frame(frame, rec["threshold_selection"]["threshold"], windows=False)
        assert ev["pooled"]["pr_auc"] == rec["pooled"]["pr_auc"] and ev["pooled"]["roc_auc"] == rec["pooled"]["roc_auc"], a
        for s in T.SEGMENTS:
            assert ev[s]["pr_auc"] == rec["segments"][s]["pr_auc"], (a, s)
        assert ev["pooled"]["top_k"] == rec["pooled"]["top_k"]


def test_ba6_pipeline_control_reproduces_the_recorded_comparison_a():
    assert abs(arm("A_refit")["pr_auc"] - 0.5395) <= 0.005
    dec = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    assert dec["BA6_pipeline_control"]["within_0.005"] is True


def test_ba8_decisions_equal_the_registered_rules_applied_to_the_recorded_values():
    dec = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    a_sub = [arm(f"A_sub_s{s}")["pr_auc"] for s in SEEDS]
    b_sub = [arm(f"B_sub_s{s}")["pr_auc"] for s in SEEDS]
    assert dec["H1_direct_B_vs_A"] == json.loads(json.dumps(D.decide_h1(arm("B_full")["pr_auc"], arm("A_refit")["pr_auc"], b_sub, a_sub)))
    runs = pd.read_csv(ROOT / "experiments/phase12_validation_evidence_table/outputs/evidence_runs.csv", float_precision="round_trip")
    e = runs[(runs["cell"] == "E") & runs["model"].str.startswith("Temporal GraphSAGE (GRU")].sort_values("seed")["pr_auc"].to_numpy()
    assert dec["H2_E_vs_B_sub"] == json.loads(json.dumps(D.decide_h2(e, b_sub)))
    ab = {f"B_{g}": arm(f"B_{g}")["pr_auc"] for g in G.ablation_groups()}
    assert dec["ablations"] == json.loads(json.dumps(D.ablation_table(arm("B_full")["pr_auc"], ab, b_sub)))


def test_ba9_sidecars_and_outputs_wording():
    for f in ("B_RESULTS.md", "table_with_B.csv", "decision.json", "run_record.json"):
        p = OUT / f
        assert p.exists(), f
    for f in ("B_RESULTS", "table_with_B"):
        side = (OUT / f"{f}.sha256").read_text().strip()
        assert side == sha(OUT / ("B_RESULTS.md" if f == "B_RESULTS" else "table_with_B.csv")), f
    bad = {str(k): v for k, v in lint_files([OUT / "B_RESULTS.md", B / "PREREGISTRATION_B.md", B / "GRAPH_EVIDENCE_PREREG.md"]).items() if v}
    assert bad == {}, bad
    txt = (OUT / "B_RESULTS.md").read_text(encoding="utf-8")
    assert "No TEST data was read" in txt and "Validation-period" in txt


# ------------------------------------------------------------------------------------------------ imbalance check
def test_ia_imbalance_outputs_follow_their_preregistration():
    pre = IM / "PREREGISTRATION_IMBALANCE.md"
    assert sha(pre) == (IM / "PREREGISTRATION_IMBALANCE.sha256").read_text().strip()
    man = json.loads((IM / "outputs" / "manifest_imbalance.json").read_text(encoding="utf-8"))
    assert man["preregistration_sha256"] == sha(pre) and man["test_data_read"] is False
    assert not any("test" in p.lower().replace("latest", "") for p in man["io_audit"]["opened"])
    dec = json.loads((IM / "outputs" / "decision.json").read_text(encoding="utf-8"))
    prs = dec["pr_auc"]
    assert set(prs) == {"none", "sqrt", "full"} and prs["full"] == arm("A_refit")["pr_auc"]            # IA4
    spread = max(prs.values()) - min(prs.values())
    others = max(v for k, v in prs.items() if k != "full")
    exp = ("WEIGHTING MAKES NO MATERIAL DIFFERENCE (threshold-free metric)" if spread <= 0.005 else "WEIGHTING HELPS" if prs["full"] - others > 0.005
           else "WEIGHTING HURTS" if others - prs["full"] > 0.005 else "INCONCLUSIVE")
    assert dec["verdict"] == exp
    r = ref()
    for a in ("none", "sqrt"):
        rec = json.loads((IM / "outputs" / "arms" / a / "metrics_val.json").read_text(encoding="utf-8"))
        assert rec["pooled"]["n_rows"] == len(r) == 83655 and rec["pooled"]["n_fraud"] == int(r["isFraud"].sum())
    assert (IM / "outputs" / "IMBALANCE_RESULTS.sha256").read_text().strip() == sha(IM / "outputs" / "IMBALANCE_RESULTS.md")
    assert lint_files([IM / "outputs" / "IMBALANCE_RESULTS.md", pre])[str(IM / "outputs" / "IMBALANCE_RESULTS.md")] == []


# ------------------------------------------------------------------------------------------------ graph evidence
def test_ge_graph_evidence_outputs_meet_their_criteria_and_hashes_link():
    pre = B / "GRAPH_EVIDENCE_PREREG.md"
    assert sha(pre) == (B / "GRAPH_EVIDENCE_PREREG.sha256").read_text().strip()
    man = json.loads((GEV / "manifest_graph_evidence.json").read_text(encoding="utf-8"))
    assert man["preregistration_sha256"] == sha(pre) and man["test_data_read"] is False and man["label_loaded"] is False
    for k, h in man["outputs_sha256"].items():
        assert sha(GEV / k) == h, k
    for k, h in man["inputs_sha256"].items():
        assert sha(ROOT / k) == h, k
    s = json.loads((GEV / "summary.json").read_text(encoding="utf-8"))
    assert all(s["acceptance"].values()), s["acceptance"]
    assert s["GE1_additivity"]["max_abs_additivity_error"] < 1e-4 and s["GE2_selection"]["n_ids"] == 300
    rows = pd.read_csv(GEV / "relation_evidence.csv")
    sel = pd.read_csv(ROOT / "experiments/phase11_explainable_evidence/xgb_explanations_val/selection_val.csv")
    assert rows["TransactionID"].tolist() == sel["TransactionID"].tolist()
    comp = rows[[f"contrib_{r}" for r in (*G.RELATIONS, "tabular")]].sum(axis=1) + rows["bias"]
    assert float(np.max(np.abs(comp - rows["margin"]))) < 1e-4
    assert lint_files([GEV / "summary.json", pre])[str(GEV / "summary.json")] == []


# ------------------------------------------------------------------------------------------------ unweighted follow-up
def test_ua_unweighted_followup_follows_its_preregistration():
    pre = IM / "PREREGISTRATION_UNWEIGHTED_FOLLOWUP.md"
    assert sha(pre) == (IM / "PREREGISTRATION_UNWEIGHTED_FOLLOWUP.sha256").read_text().strip()
    rec = json.loads((IM / "outputs" / "arms_unweighted" / "run_record.json").read_text(encoding="utf-8"))
    assert rec["preregistration_sha256"] == sha(pre) and rec["test_data_read"] is False
    assert not any("test" in q.lower().replace("latest", "") for q in rec["io_audit"]["opened"])
    r = ref()
    for a in ("B_none", *[f"{k}_none_sub_s{s}" for k in "AB" for s in SEEDS]):
        m = json.loads((IM / "outputs" / "arms_unweighted" / a / "metrics_val.json").read_text(encoding="utf-8"))
        assert m["config"]["scale_pos_weight"] == 1.0 and m["pooled"]["n_rows"] == len(r) == 83655, a
        p = pd.read_parquet(IM / "outputs" / "arms_unweighted" / a / "predictions_val.parquet").sort_values("TransactionID").reset_index(drop=True)
        assert np.array_equal(p["TransactionID"].to_numpy(), r["TransactionID"].to_numpy()) and np.array_equal(p["isFraud"].to_numpy().astype(int), r["isFraud"].to_numpy().astype(int)), a
    for d in ("unweighted_graphsage", "unweighted_temporal_graphsage"):                        # UA3 and UA5
        for s in SEEDS:
            c = json.loads((ROOT / "experiments" / d / f"seed_{s}" / "config.json").read_text(encoding="utf-8"))
            assert c["effective_pos_weight"] == 1.0 and c["seed"] == s
            p = pd.read_parquet(ROOT / "experiments" / d / f"seed_{s}" / "predictions_val.parquet").sort_values("TransactionID").reset_index(drop=True)
            assert np.array_equal(p["TransactionID"].to_numpy(), r["TransactionID"].to_numpy()) and int(p["day"].max()) <= 151 and int(p["day"].min()) >= 122
            assert np.array_equal(p["isFraud"].to_numpy().astype(int), r["isFraud"].to_numpy().astype(int))
    dec = json.loads((IM / "outputs" / "unweighted_decision.json").read_text(encoding="utf-8"))
    v = dec["values"]
    assert dec["rule2_H1_direct_unweighted"] == json.loads(json.dumps(D.decide_h1(v["B_none"], v["A_none"], v["B_none_sub"], v["A_none_sub"])))
    assert dec["rule3_H2_unweighted_E"] == json.loads(json.dumps(D.decide_h2(v["E_none"], v["B_none_sub"])))
    assert dec["rule4_ordering"]["tabular_above_every_neural_seed"] == bool(min(v["A_none_sub"] + v["B_none_sub"]) > max(v["C_none"] + v["E_none"]))
    assert (IM / "outputs" / "UNWEIGHTED_RESULTS.sha256").read_text().strip() == sha(IM / "outputs" / "UNWEIGHTED_RESULTS.md")
    assert lint_files([IM / "outputs" / "UNWEIGHTED_RESULTS.md", pre])[str(IM / "outputs" / "UNWEIGHTED_RESULTS.md")] == []


def test_heldout_readiness_gate_record_passes_and_no_heldout_artifact_exists():
    fh = ROOT / "experiments" / "final_heldout_protocol"
    rep = json.loads((fh / "readiness_val" / "readiness_report.json").read_text(encoding="utf-8"))
    assert rep["all_pass"] is True and rep["split"] == "val" and len(rep["models"]) == 19
    assert all(m["pass"] and m["ids_identical"] for m in rep["models"])
    assert not (fh / "AUTHORIZATION.txt").exists() and not (fh / "heldout_results").exists()
    assert (fh / "readiness_val" / "readiness_report.sha256").read_text().strip() == sha(fh / "readiness_val" / "readiness_report.json")


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
