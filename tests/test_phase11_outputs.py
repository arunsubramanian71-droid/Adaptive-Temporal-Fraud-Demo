"""Phase 11 first pass: verification of the generated outputs against the scope record's acceptance tests (A1, A2, A4, A6, A7, A8, A9, A10).

Reads only the Phase 11 outputs, the label-free id/day/score columns of the VAL files, the Tier 0 CSVs and hash records. No model is
trained, nothing is installed, TEST is not opened, no label column is read."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.lint import lint_files, lint_text  # noqa: E402
from src.explain import cards as C  # noqa: E402
from src.explain import xgb_contrib as X  # noqa: E402

P11 = ROOT / "experiments" / "phase11_explainable_evidence"
XGB = P11 / "xgb_explanations_val"
CARDS = P11 / "community_cards"
E = ROOT / "experiments" / "phase9_coordinated_fraud" / "exploratory_realdata"


def _h(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def _outputs(folder, suffixes=(".csv", ".json", ".md")):
    return sorted(p for p in folder.iterdir() if p.suffix in suffixes)


def test_scope_record_sidecar_matches_and_precedes_the_outputs():
    sc = P11 / "SCOPE_AND_ACCEPTANCE.md"
    assert _h(sc) == (P11 / "SCOPE_AND_ACCEPTANCE.sha256").read_text().strip()
    assert lint_text(sc.read_text(encoding="utf-8")) == []
    assert sc.stat().st_mtime <= min(p.stat().st_mtime for p in _outputs(XGB) + _outputs(CARDS))


def test_A7_every_output_has_a_matching_sidecar_and_the_manifests_match_disk():
    for folder in (XGB, CARDS):
        for p in _outputs(folder):
            assert _h(p) == p.with_suffix(".sha256").read_text().strip(), p.name
        assert len({p.stem.lower() for p in _outputs(folder)}) == len(_outputs(folder))      # no case-insensitive sidecar collision
    for mf in (XGB / "manifest_val.json", CARDS / "manifest_cards.json"):
        m = json.loads(mf.read_text(encoding="utf-8"))
        for rel, h in {**m["inputs_sha256"], **m["outputs_sha256"]}.items():
            assert _h(ROOT / rel) == h, rel
        assert m["dependencies_installed"] is False
    mv = json.loads((XGB / "manifest_val.json").read_text(encoding="utf-8"))
    assert mv["test_features_opened"] is False and mv["label_column_loaded"] is False
    for need in ("experiments/xgboost/model.joblib", "data/features/val_features.parquet", "experiments/xgboost/predictions_val.parquet"):
        assert need in mv["inputs_sha256"]
    assert not any("test" in k.lower() for k in mv["inputs_sha256"])


def test_A4_A6_selection_has_300_rows_10_per_day_within_val_and_is_reproduced_from_label_free_inputs():
    sel = pd.read_csv(XGB / "selection_val.csv", float_precision="round_trip")      # exact float64 round trip of the written repr
    assert len(sel) == 300 and sel["TransactionID"].is_unique
    assert (sel.groupby("day").size() == 10).all() and sel["day"].between(122, 151).all() and sel["day"].nunique() == 30
    assert sel["group"].value_counts().to_dict() == {"top_score": 150, "uniform_draw": 150}
    ids_days = X.load_ids_days(ROOT / "data" / "features" / "val_features.parquet")
    scores = X.load_scores(ROOT / "experiments" / "xgboost" / "predictions_val.parquet")
    m = ids_days.merge(scores, on="TransactionID", validate="one_to_one")
    again = X.select_transactions(m["TransactionID"], m["day"], m["y_score"])
    key = ["TransactionID", "day", "group", "rank_in_group"]
    assert again[key].reset_index(drop=True).equals(sel[key].reset_index(drop=True))
    assert np.array_equal(again["y_score"].to_numpy(), sel["y_score"].to_numpy())
    # the selected rows' days really are what TransactionDT says
    chk = sel.merge(m, on="TransactionID", suffixes=("", "_v"))
    assert (chk["day"] == chk["day_v"]).all() and len(chk) == 300


def test_A1_A2_stored_contributions_are_additive_and_reproduce_the_stored_scores():
    cd = pd.read_csv(XGB / "contributions_val.csv")
    sel = pd.read_csv(XGB / "selection_val.csv")
    assert cd["TransactionID"].tolist() == sel["TransactionID"].tolist() and cd.shape[1] == 3 + 470
    feat = cd[[c for c in cd.columns if c.startswith("c::")]].to_numpy(dtype=np.float64)
    assert feat.shape == (300, 470)
    assert np.abs(feat.sum(axis=1) + cd["bias"].to_numpy() - cd["margin"].to_numpy()).max() <= 1e-4
    assert np.abs(X.sigmoid(cd["margin"].to_numpy()) - sel["y_score"].to_numpy()).max() <= 1e-6
    s = json.loads((XGB / "summary_val.json").read_text(encoding="utf-8"))
    assert s["acceptance_checks"]["A1_max_abs_additivity_error"] <= s["tolerances"]["A1"]
    assert s["acceptance_checks"]["A2_max_abs_prob_difference_vs_stored_y_score"] <= s["tolerances"]["A2"]
    assert s["acceptance_checks"]["A3_max_abs_batch_difference"] <= s["tolerances"]["A3"]
    assert s["selection"]["label_used"] is False and s["n_selected"] == 300 and s["model"]["best_iteration"] == 798


def test_top_feature_lists_agree_with_the_contribution_matrix():
    cd = pd.read_csv(XGB / "contributions_val.csv")
    names = [c[3:] for c in cd.columns if c.startswith("c::")]
    top = json.loads((XGB / "top_features_val.json").read_text(encoding="utf-8"))
    assert "not evidence of fraud" in top["statement"] and len(top["transactions"]) == 300
    feat = cd[[c for c in cd.columns if c.startswith("c::")]].to_numpy(dtype=np.float64)
    for i in (0, 1, 149, 150, 299):
        t = top["transactions"][i]
        assert t["TransactionID"] == int(cd["TransactionID"].iloc[i]) and len(t["top_features"]) == 10
        order = np.lexsort((np.arange(len(names)), -np.abs(feat[i])))[:10]
        assert [f["feature"] for f in t["top_features"]] == [names[j] for j in order]
        assert all(abs(f["contribution"] - feat[i, names.index(f["feature"])]) <= 1e-12 for f in t["top_features"])


def test_A9_cards_match_a_fresh_recomputation_from_the_existing_aggregates_only():
    meta = json.loads((CARDS / "cards_W7_seed42.json").read_text(encoding="utf-8"))
    assert meta["standing_statement"] == C.STANDING and meta["not_established"] == C.NOT_ESTABLISHED
    tl = json.loads((E / "e0_tau_seed42.json").read_text(encoding="utf-8"))
    df = C.load_tier0(E / "tier0_community_days_W7_seed42.csv")
    groups = C.select_groups(df, float(tl["tau"]), 42)
    assert {k: v for k, v in meta["meta"]["groups"].items()} == groups
    mem = C.memberships(groups)
    assert [c["track"] for c in meta["cards"]] == list(mem) and len(meta["cards"]) == 24
    for c in meta["cards"]:
        assert c == json.loads(json.dumps(C.build_card(df, c["track"], mem[c["track"]], float(tl["tau"]))))
        assert c["unavailable_fields"] == list(C.UNAVAILABLE_FIELDS) and c["standing_statement"] == C.STANDING
    app = pd.read_csv(CARDS / "appendix_selection_all_seeds_W7.csv")
    assert sorted(app["seed"].unique().tolist()) == [42, 123, 2024]
    for S in (123, 2024):
        d = C.load_tier0(E / f"tier0_community_days_W7_seed{S}.csv")
        tau = float(json.loads((E / f"e0_tau_seed{S}.json").read_text(encoding="utf-8"))["tau"])
        want = list(C.memberships(C.select_groups(d, tau, S)))
        assert app.loc[app["seed"] == S, "track"].tolist() == want
    s = json.loads((CARDS / "cards_summary.json").read_text(encoding="utf-8"))
    assert s["discovery_rerun"] is False and s["labels_or_risk_used"] is False and s["a2_score_equals_mean_of_u_s_and_u_t_max_abs_difference"] < 1e-12


def test_A8_wording_lint_on_all_phase11_text_outputs_and_sources():
    files = [p for p in P11.rglob("*") if p.suffix in (".md", ".json", ".csv") and p.is_file()]
    files += sorted((ROOT / "src" / "explain").glob("*.py")) + sorted(ROOT.glob("scripts/*phase11*.py")) + sorted(ROOT.glob("tests/*phase11*.py"))
    names = {f.name for f in files}
    assert {"SCOPE_AND_ACCEPTANCE.md", "selection_val.csv", "contributions_val.csv", "top_features_val.json", "summary_val.json",
            "cards_W7_seed42.json", "cards_W7_seed42_report.md", "appendix_selection_all_seeds_W7.csv", "xgb_contrib.py", "cards.py",
            "run_phase11_explanations.py", "run_phase11_cards.py", "test_phase11_explain.py", "test_phase11_outputs.py"} <= names, names
    bad = {str(k): v for k, v in lint_files(files).items() if v}
    assert bad == {}, bad
    assert C.STANDING in (CARDS / "cards_W7_seed42_report.md").read_text(encoding="utf-8")


def test_A10_frozen_phase9_and_phase10_records_are_unchanged():
    p9 = json.loads((ROOT / "experiments/phase9_coordinated_fraud/phase9_closure_integrity.json").read_text(encoding="utf-8"))
    assert all(_h(ROOT / k) == v for k, v in p9["artifacts_relied_on_sha256"].items())
    p10 = ROOT / "experiments" / "phase10_community_evolution"
    for f in ("DESIGN_BEFORE_EXECUTION.md", "INSPECTION_REPORT.md", "INTEGRITY.json", "DESIGN_ADDENDUM_D12_approved_decisions.md",
              "DESIGN_ADDENDUM_ERRATA_1_B1_B5.md", "DESIGN_ADDENDUM_ERRATA_2_targets_interface_censoring.md",
              "INTEGRITY_CORE_IMPLEMENTATION.json", "INTEGRITY_CORE_FINALIZATION.json"):
        assert _h(p10 / f) == (p10 / f).with_suffix(".sha256").read_text().strip(), f


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
