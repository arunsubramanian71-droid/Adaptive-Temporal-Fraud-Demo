"""Core-versus-artifact-dependent test workflow: tests/artifact_requirements.json lists the tests that need artifacts the repository does not publish.
This module checks the registry itself (it names real tests, points only at withheld artifacts, never at a held-out path) and the runner's behaviour on a missing artifact.
No held-out file is opened and no model is run."""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_core_tests as RC  # noqa: E402

REGISTRY = {k: v for k, v in json.loads((ROOT / "tests" / "artifact_requirements.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}


def test_every_registered_key_names_a_real_test_function():
    bad = []
    for key in REGISTRY:
        module, test = key.split("::")
        p = ROOT / "tests" / f"{module}.py"
        if not p.exists() or not re.search(rf"^def {re.escape(test)}\(", p.read_text(encoding="utf-8"), re.M):
            bad.append(key)
    assert bad == [], bad


def test_registered_patterns_are_relative_never_held_out_and_never_match_a_tracked_file():
    for key, pats in REGISTRY.items():
        assert pats and all(isinstance(p, str) and not pathlib.PurePath(p).is_absolute() and ".." not in p for p in pats), key
        assert not any(re.search(r"test_features|test_transaction|test_identity|AUTHORIZATION|heldout_results", p, re.I) for p in pats), f"{key}: pattern names a held-out path"
    if not (ROOT / ".git").exists():
        return                                                     # a source archive without git history: the tracked-file check does not apply
    tracked = set(subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True).stdout.split("\0"))
    clash = [(k, p) for k, pats in REGISTRY.items() for p in pats if any(ROOT.glob(p)) and any(t in tracked for t in (x.relative_to(ROOT).as_posix() for x in ROOT.glob(p)))]
    assert clash == [], f"registered as withheld but tracked (the test could run): {clash}"


def test_missing_artifact_helper_reports_missing_patterns_and_never_invents_presence():
    assert RC._missing_artifacts(["experiments/__no_such_artifact__/*.parquet"]) == ["experiments/__no_such_artifact__/*.parquet"]
    assert RC._missing_artifacts(["tests/artifact_requirements.json"]) == []
    assert RC._missing_artifacts([]) == []


ORIGINAL_13 = [
    "test_comparison_b_outputs::test_ba5_each_arm_scores_exactly_the_reference_population",
    "test_comparison_b_outputs::test_ba7_recorded_metrics_equal_a_recomputation_from_the_stored_predictions",
    "test_comparison_b_outputs::test_ge_graph_evidence_outputs_meet_their_criteria_and_hashes_link",
    "test_comparison_b_outputs::test_ia_imbalance_outputs_follow_their_preregistration",
    "test_comparison_b_outputs::test_ua_unweighted_followup_follows_its_preregistration",
    "test_heldout_gate::test_frozen_registry_matches_the_protocol_and_every_artifact_exists",
    "test_phase11_outputs::test_A4_A6_selection_has_300_rows_10_per_day_within_val_and_is_reproduced_from_label_free_inputs",
    "test_phase11_outputs::test_A7_every_output_has_a_matching_sidecar_and_the_manifests_match_disk",
    "test_phase12_environment_independence::test_rebuild_is_byte_identical_and_the_manifest_differs_only_in_environment_metadata",
    "test_phase12_outputs::test_AC10_sidecars_and_deterministic_rebuild",
    "test_phase12_outputs::test_AC1_only_the_declared_source_set_was_read_and_the_fingerprint_is_stable",
    "test_phase12_outputs::test_build_gates_block_before_any_output_is_written",
    "test_scope_reconciliation_records::test_freeze_reproduces_the_recorded_unweighted_xgboost_validation_results",
]
RESEARCH_SERVING = [
    "test_research_serving::test_api_scores_equal_direct_model_scores_and_the_recorded_validation_predictions",
    "test_research_serving::test_invalid_research_requests_get_actionable_400_errors",
    "test_research_serving::test_labels_in_the_request_are_ignored_and_each_row_is_scored_independently",
]
UNPUBLISHED_EVIDENCE = [
    "test_phase11_outputs::test_A1_A2_stored_contributions_are_additive_and_reproduce_the_stored_scores",
    "test_phase11_outputs::test_top_feature_lists_agree_with_the_contribution_matrix",
    "test_phase11_outputs::test_A8_wording_lint_on_all_phase11_text_outputs_and_sources",
    "test_neural_graph_evidence::test_recorded_outputs_follow_the_preregistration_and_report_the_failed_criterion",
    "test_ac12_recovery_records::test_derived_evidence_recomputes_from_the_current_files",
]
FORMERLY_SILENT = [
    "test_splits::test_built_split_matches_expected_regression_counts",
]


def test_registry_is_exactly_the_four_documented_groups():
    """13 tests that failed on the first fresh clone; 3 research-model serving tests; 5 tests that read the six transaction-level evidence files kept
    local since 2026-10-22 (publication policy); 1 test that returned early without its feature tables, which counted as a pass on a fresh clone (found
    2026-10-22). A change to the registry must update these lists deliberately."""
    groups = ORIGINAL_13 + RESEARCH_SERVING + UNPUBLISHED_EVIDENCE + FORMERLY_SILENT
    assert len(ORIGINAL_13) == 13 and len(RESEARCH_SERVING) == 3 and len(UNPUBLISHED_EVIDENCE) == 5 and len(FORMERLY_SILENT) == 1
    assert set(REGISTRY) == set(groups) and len(groups) == len(set(groups)), sorted(set(REGISTRY) ^ set(groups))
    assert all(any("model.joblib" in p for p in REGISTRY[k]) for k in RESEARCH_SERVING)


def test_no_test_returns_early_when_its_input_files_are_missing():
    """A test that returns when its data is absent counts as passed on a fresh clone. Only a missing .git (a source archive) may end a test early; a test that
    needs unpublished files is registered in tests/artifact_requirements.json instead."""
    def early_returns(text):
        lines = text.splitlines()
        out = []
        for i, line in enumerate(lines):
            if re.match(r"\s+return\s*(#.*)?$", line):
                cond = [c for c in lines[max(0, i - 3):i] if re.search(r"\bif not\b.*\.(exists|is_file|is_dir)\(\)", c)]
                if cond and ".git" not in cond[-1]:
                    out.append(i + 1)
        return out
    bad = [f"{p.name}:{n}" for p in sorted((ROOT / "tests").glob("test_*.py")) for n in early_returns(p.read_text(encoding="utf-8"))]
    assert bad == [], bad
    old = 'def t():\n    if not path.exists():\n        print("SKIP: not found")\n        return\n'          # the pattern removed from test_splits on 2026-10-22
    assert early_returns(old) == [4] and early_returns('def t():\n    if not (ROOT / ".git").exists():\n        return\n') == []


def test_no_test_loops_over_all_three_splits_to_read_feature_tables():
    """The held-out feature table is opened only by the gated evaluator (src/heldout/evaluator.py, after AUTHORIZATION.txt). Until 2026-10-22 test_splits looped over
    the three split names and read each '<split>_features.parquet', which opened the sealed table on every full run. Tests that name the table literally do so to
    assert that it is refused."""
    three = re.compile(r"\(\s*[\"']train[\"']\s*,\s*[\"']val[\"']\s*,\s*[\"']test[\"']\s*\)")
    bad = [p.name for p in sorted((ROOT / "tests").glob("test_*.py")) if p.name != pathlib.Path(__file__).name
           and "_features.parquet" in p.read_text(encoding="utf-8") and three.search(p.read_text(encoding="utf-8"))]
    assert bad == [], bad
    assert three.search('for name in ("train", "val", "test"):') and not three.search('for name in ("train", "val"):')


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
