"""Records of the scope reconciliation: amendment and scope hashes, base protocol unchanged, S8 supplement, NE5 diagnosis, F-readiness diagnosis, freeze reproduction, readiness records.
Reads only recorded files; no held-out file is opened and no model is run."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FH = ROOT / "experiments" / "final_heldout_protocol"
BASE_HASH = "59bf1565fb16653a02a2d743cc445e114fb2a6a42f1de7b2fa56261e6f68c806"


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def sidecar_ok(p: pathlib.Path) -> bool:
    return sha(p) == p.with_suffix(".sha256").read_text().strip()


def test_base_protocol_is_unchanged_and_amendment_and_scope_are_hashed():
    assert sha(FH / "HELDOUT_EVALUATION_PROTOCOL.md") == BASE_HASH
    for n in ("AMENDMENT_1.md", "SCOPE_RECONCILIATION.md"):
        assert sidecar_ok(FH / n), n
    assert BASE_HASH in (FH / "AMENDMENT_1.md").read_text(encoding="utf-8")


def test_amendment_names_the_modules_the_gate_and_the_failed_as_written_f_readiness():
    t = (FH / "AMENDMENT_1.md").read_text(encoding="utf-8")
    for token in ("I authorize the following Amendment 1 modules:", "Module F", "Module H", "Module U", "FAIL as written", "NOT EXECUTED"):
        assert token in t, token


def test_f_readiness_record_matches_what_the_amendment_reports():
    r = json.loads((ROOT / "experiments/phase8_state_continuity_supplement/outputs/readiness/readiness_F.json").read_text(encoding="utf-8"))
    det = [x for x in r["rows"] if x["window"] != "mean"]
    tr = [x for x in r["rows"] if x["window"] == "mean"]
    assert len(det) == 21 and all(x["pass"] for x in det)
    assert len(tr) == 6 and sum(x["pass"] for x in tr) == 5 and r["all_pass"] is False      # registered result stands; the tolerance was not changed


def test_f_readiness_diagnosis_shows_run_to_run_variation_and_leaves_the_registered_result_unchanged():
    p = ROOT / "experiments/phase8_state_continuity_supplement/diagnostics/F_readiness_diagnosis_E_seed123.json"
    assert sidecar_ok(p)
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["registered_tolerance"] == 0.01 and r["test_data_read"] is False and r["n_identical_runs"] >= 3
    means = r["identical_run_window_means"]
    assert max(means) - min(means) > 0.002, "identical-seed runs should differ (numerical variation); if they do not, the diagnosis must be revisited"
    assert abs(r["readiness_vs_supplement_abs_diff"] - 0.0103920760) < 1e-6        # the failing difference of the registered gate
    assert r["window1_max_score_diff_supplement_vs_readiness"] < 1e-5               # the deterministic part is identical


def test_freeze_reproduces_the_recorded_unweighted_xgboost_validation_results():
    r = json.loads((FH / "frozen_amendment" / "reproduction.json").read_text(encoding="utf-8"))
    assert r["test_data_read"] is False
    for n in ("A_none", "B_none"):
        assert r["models"][n]["reproduces_recorded_validation_result"] is True
        assert sidecar_ok(FH / "frozen_amendment" / n / "feature_columns.json")
        assert (FH / "frozen_amendment" / n / "model.sha256").read_text().strip() == sha(FH / "frozen_amendment" / n / "model.joblib")


def test_amendment_readiness_on_validation_passed_for_all_models():
    r = json.loads((FH / "readiness_val_amendment" / "readiness_report.json").read_text(encoding="utf-8"))
    assert r["all_pass"] is True and len(r["models"]) == 9          # 8 unweighted + the historical reference
    assert "No held-out data was read" in r["statement"]


def test_state_continuity_supplement_keeps_h4_supported_and_records_the_failed_s1():
    d = json.loads((ROOT / "experiments/phase8_state_continuity_supplement/outputs/decision_compare.json").read_text(encoding="utf-8"))
    assert d["recorded_phase8"]["h4"] == "SUPPORTED" and d["state_continuity_supplement"]["h4"] == "SUPPORTED"
    s1 = json.loads((ROOT / "experiments/phase8_state_continuity_supplement/outputs/as_written/readiness_S1.json").read_text(encoding="utf-8"))
    assert s1["all_pass"] is False                                  # the registered criterion failed and was not changed
    assert sidecar_ok(ROOT / "experiments/phase8_state_continuity_supplement/outputs/S8_RESULTS.md")


def test_ne5_diagnosis_is_numerical_precision_and_the_registered_failure_stands():
    p = ROOT / "experiments/neural_graph_evidence/diagnostics/NE5_diagnosis.json"
    assert sidecar_ok(p)
    r = json.loads(p.read_text(encoding="utf-8"))
    assert r["cpu_exact_zero"] is True and r["device_results"]["gpu"]["noop_n_nonzero"] > 0
    assert r["device_results"]["gpu"]["max_abs_reset_effect"] < 1e-5
    assert "FAILED" in r["registered_result"]


def test_nothing_is_authorized_or_executed():
    assert not (FH / "AUTHORIZATION.txt").exists()
    for n in ("heldout_results", "heldout_results_amendment", "heldout_results_f"):
        assert not (FH / n).exists(), n


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    failed = []
    for fn in fns:
        try:
            fn()
            print("[PASS]", fn.__name__)
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            print("[FAIL]", fn.__name__, repr(e))
    print(f"\n{len(fns) - len(failed)}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
