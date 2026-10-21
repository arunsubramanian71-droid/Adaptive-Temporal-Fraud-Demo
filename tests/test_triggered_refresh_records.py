"""Records of the drift-triggered refresh analysis (experiments/phase8_triggered_refresh): pre-registration hashed before the outputs, outputs consistent with their summary,
the registered outcome reported as it is (NOT SUPPORTED), TRAIN-only thresholds. Reads recorded files only."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments" / "phase8_triggered_refresh"
SEEDS = (42, 123, 2024)


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def test_preregistration_is_hashed_and_precedes_every_output():
    pre = EXP / "PREREGISTRATION_TR.md"
    assert sha(pre) == (EXP / "PREREGISTRATION_TR.sha256").read_text().strip()
    assert all(os.stat(pre).st_mtime <= os.stat(EXP / "outputs" / "summary.json").st_mtime for _ in (0,))


def test_outputs_have_matching_sidecars_and_the_decision_is_reported_as_registered():
    for rel in ("outputs/summary.json", "outputs/TR_RESULTS.md") + tuple(f"outputs/seed_{s}/trigger_log.json" for s in SEEDS):
        p = EXP / rel
        assert sha(p) == p.with_suffix(".sha256").read_text().strip(), rel
    d = json.loads((EXP / "outputs" / "decision.json").read_text(encoding="utf-8"))
    assert d["outcome"] == "NOT SUPPORTED" and d["h4_unchanged"] is True and d["test_data_read"] is False
    assert d["criteria"]["TR1_execution_and_window1_equals_F"] and d["criteria"]["TR2_train_only_threshold_recorded"]
    assert d["criteria"]["TR3_benefit_over_static"] is False and d["criteria"]["TR5_parity_with_periodic"] is False


def test_threshold_rule_was_applied_per_seed_from_train_days_and_the_monitor_never_fired():
    for s in SEEDS:
        r = json.loads((EXP / "outputs" / f"seed_{s}" / "trigger_log.json").read_text(encoding="utf-8"))
        assert abs(r["tau"] - max(0.10, 2 * r["psi_noise_train_days_112_116_vs_117_121"])) < 1e-12
        assert r["n_refreshes"] == 0 and all(not row["fired"] for row in r["monitor_log"])
        assert all(row["psi"] is None or row["psi"] < r["tau"] for row in r["monitor_log"])


def test_performance_trigger_preregistration_precedes_outputs_and_outcome_is_reported_as_registered():
    pre = EXP / "PREREGISTRATION_TRP.md"
    out = EXP / "outputs_performance"
    assert sha(pre) == (EXP / "PREREGISTRATION_TRP.sha256").read_text().strip()
    assert os.stat(pre).st_mtime <= os.stat(out / "summary.json").st_mtime
    for rel in ("summary.json", "TRP_RESULTS.md") + tuple(f"seed_{s}/trigger_log.json" for s in SEEDS):
        p = out / rel
        assert sha(p) == p.with_suffix(".sha256").read_text().strip(), rel
    d = json.loads((out / "decision.json").read_text(encoding="utf-8"))
    assert d["outcome"] == "NOT SUPPORTED" and d["h4_unchanged"] is True and d["test_data_read"] is False
    c = d["criteria"]
    assert c["TRP1_execution_and_window1_equals_F"] and c["TRP2_threshold_rule_from_train_days"] and c["TRP3_benefit_over_static"] and c["TRP4_fewer_refreshes_than_periodic"]
    assert c["TRP5_parity_with_periodic"] is False                      # the registered parity criterion was not met and is reported as such


def test_performance_trigger_threshold_rule_and_log_are_consistent_per_seed():
    for s in SEEDS:
        r = json.loads((EXP / "outputs_performance" / f"seed_{s}" / "trigger_log.json").read_text(encoding="utf-8"))
        assert abs(r["delta"] - min(0.5, max(0.10, 2 * r["relative_noise_r"]))) < 1e-12
        assert abs(r["relative_noise_r"] - abs(r["pr_auc_train_days_112_116"] - r["pr_auc_train_days_117_121"]) / ((r["pr_auc_train_days_112_116"] + r["pr_auc_train_days_117_121"]) / 2)) < 1e-12
        fired = [row["window"] for row in r["monitor_log"] if row["fired"]]
        assert r["n_refreshes"] == len(fired) == 1
        assert all(row["relative_drop"] is None or (row["fired"] == (row["relative_drop"] > r["delta"])) for row in r["monitor_log"])


def test_no_authorization_or_heldout_results_exist():
    fh = ROOT / "experiments" / "final_heldout_protocol"
    assert not (fh / "AUTHORIZATION.txt").exists() and not any((fh / n).exists() for n in ("heldout_results", "heldout_results_f", "heldout_results_amendment"))


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
