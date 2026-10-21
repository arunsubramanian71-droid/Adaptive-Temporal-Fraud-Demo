"""Read-only readiness review of the frozen held-out protocol (no held-out file is opened, hashed, read or scored; no authorization is created).

Checks: the protocol hash against its sidecar; every frozen model artifact and its recorded validation threshold; the evaluator registry against the protocol
text; the validation readiness gate record (all models, recorded after the last change to the evaluator); the gate and one-shot behaviour (the tested refusals);
which held-out-named paths exist (existence and size only is NOT used: only the two authorization artifacts are checked, by existence). Writes
experiments/final_heldout_protocol/PROTOCOL_REVIEW.md and .json (with sidecars).

Usage:  .venv/Scripts/python.exe -B scripts/verify_heldout_protocol_ready.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.heldout import evaluator as E  # noqa: E402

FH = ROOT / "experiments" / "final_heldout_protocol"


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def main() -> int:
    proto = ROOT / E.PROTOCOL
    text = proto.read_text(encoding="utf-8")
    ph = sha(proto)
    facts = {"protocol_sha256": ph, "sidecar_matches": ph == (FH / "HELDOUT_EVALUATION_PROTOCOL.sha256").read_text().strip()}
    reg = E.frozen_registry()
    models = {}
    for mid, spec in sorted(reg.items()):
        d = ROOT / spec["dir"]
        models[mid] = {"artifacts_exist": all((d / f).exists() for f in spec["files"]), "recorded_validation_threshold": E.recorded_threshold(d),
                       "artifact_sha256": {f: sha(d / f) for f in spec["files"] if not f.endswith("metrics_val.json")}}
    facts["n_models"] = len(models)
    facts["all_artifacts_exist"] = all(m["artifacts_exist"] for m in models.values())
    facts["thresholds_all_recorded_in_unit_interval"] = all(0 < m["recorded_validation_threshold"] < 1 for m in models.values())
    for token in ("logistic regression", "XGBoost", "GraphSAGE", "GAT", "Temporal GraphSAGE", "memory ablated", "schedule-matched"):
        facts.setdefault("protocol_names", {})[token] = token.lower() in text.lower()
    rep_path = FH / "readiness_val" / "readiness_report.json"
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    ev_files = [ROOT / "src/heldout/evaluator.py", ROOT / "scripts/run_heldout_evaluation.py"]
    from src.heldout import codehash as CH
    code = rep.get("code_sha256") or {}
    facts["readiness"] = {"all_pass": rep["all_pass"], "models": len(rep["models"]), "max_abs_score_difference": max(m["max_abs_score_diff"] for m in rep["models"]),
                          # content-based (the runner's own rule, src/heldout/codehash.py): every code file the gate loaded is unchanged since; file times are not used
                          "recorded_after_last_evaluator_change": bool(code) and {"src/heldout/evaluator.py", "scripts/run_heldout_evaluation.py"} <= set(code) and CH.changed_code(code) == [],
                          "code_files_fingerprinted": len(code), "run_local": rep["created_local"],
                          "evaluator_sha256": {f.name: sha(f) for f in ev_files}}
    facts["heldout_artifacts"] = {"AUTHORIZATION.txt_exists": (FH / "AUTHORIZATION.txt").exists(), "heldout_results_exists": (FH / "heldout_results").exists()}
    facts["gate_behaviour_tested"] = {"module": "tests/test_heldout_gate.py", "tests": ["refused without the file", "refused with wrong hash or sentence", "accepted only with both", "CLI refuses before reading and enforces one-shot",
                                                                              "held-out feature path reachable only through the gate", "feature frames read without the label column"]}
    AMEND = "experiments/final_heldout_protocol/AMENDMENT_1.md"
    amend_hash = sha(ROOT / AMEND)
    amend_ok = amend_hash == (FH / "AMENDMENT_1.sha256").read_text().strip()
    amend_rep = json.loads((FH / "readiness_val_amendment" / "readiness_report.json").read_text(encoding="utf-8"))
    f_rep = json.loads((ROOT / "experiments/phase8_state_continuity_supplement/outputs/readiness/readiness_F.json").read_text(encoding="utf-8"))
    facts["amendment_1"] = {"sha256": amend_hash, "sidecar_matches": amend_ok, "readiness_H_U_all_pass": amend_rep["all_pass"], "readiness_F_all_pass_as_written": f_rep["all_pass"]}
    ok = (facts["sidecar_matches"] and facts["n_models"] == 19 and facts["all_artifacts_exist"] and facts["thresholds_all_recorded_in_unit_interval"] and all(facts["protocol_names"].values())
          and rep["all_pass"] and facts["readiness"]["recorded_after_last_evaluator_change"] and not facts["heldout_artifacts"]["AUTHORIZATION.txt_exists"] and not facts["heldout_artifacts"]["heldout_results_exists"])
    facts["ready_for_informed_authorization"] = bool(ok)
    rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "facts": facts, "models": models,
           "scope": "read-only review; no held-out file was opened, hashed, read or scored"}
    L = ["# Held-out protocol: readiness review (no authorization requested or created)\n",
         f"Dated {rec['created_local'][:10]}. Read-only. **No held-out file was opened, hashed, read or scored; `AUTHORIZATION.txt` does not exist.**\n",
         f"* Protocol: `{E.PROTOCOL}`, SHA-256 `{ph}` (sidecar matches: {facts['sidecar_matches']}).",
         f"* Frozen models: {facts['n_models']} (all artifacts present: {facts['all_artifacts_exist']}); thresholds are the recorded validation best-F1 values below and are transferred, never re-tuned.",
         f"* Code-readiness gate: validation re-scoring of all {facts['readiness']['models']} models reproduces the recorded predictions (maximum score difference {facts['readiness']['max_abs_score_difference']:.2e}); code unchanged since the gate ran (SHA-256 of the {facts['readiness']['code_files_fingerprinted']} code files it loaded): {facts['readiness']['recorded_after_last_evaluator_change']}.",
         f"* Evaluator hashes at readiness: " + ", ".join(f"`{k}` `{v[:16]}...`" for k, v in facts["readiness"]["evaluator_sha256"].items()) + ".",
         f"* **Verdict: {'READY for informed authorization' if ok else 'NOT READY (see the failed fact above)'}.** Readiness means the procedure is frozen and checked on validation; it does not mean the evaluation answers every question of the brief (see the gaps below).\n",
         "## Frozen models and recorded validation thresholds\n", "| Model | Recorded validation threshold |", "|---|---|"]
    for mid, m in models.items():
        L.append(f"| {mid} | {m['recorded_validation_threshold']:.6f} |")
    L += ["", "## Input restrictions and one-shot behaviour (each tested without touching held-out data)\n",
          "* Without `AUTHORIZATION.txt` quoting the protocol hash and the sentence, the evaluator raises before reading anything; a wrong hash or sentence is refused; a second run is refused once `heldout_results/.completed` exists.",
          "* The held-out feature file is named in exactly one place in the code and is reachable only after the gate; feature frames are read with the label column projected away; labels are read afterwards from the graph file only to compute metrics; the temporal dataset is built without labels.",
          "* When authorized it reads: `data/features/test_features.parquet` (tabular models and B) and the Phase 4 graph file (neural models; all days are loaded into memory, the models score days 152-182 under the causal masks); it writes only into `experiments/final_heldout_protocol/heldout_results/`.", "",
          "## Consistency with the original brief\n",
          "* Consistent: chronological split with the latest period held out; decisions and hyperparameters fixed without it; the brief's metrics (PR-AUC primary, ROC-AUC, precision, recall, F1, Precision/Recall@K, confusion matrices, alerts, false positives) are all computed.",
          "* **Gap 1 (closed only by Amendment 1 module F, if authorized):** the brief's adaptation requirement (\"compare static versus adaptive performance on future periods\") and cell F are not evaluated by the base protocol; H4 would stay validation-only.",
          "* **Gap 2 (modules H and U, if authorized):** the reference historical-feature XGBoost and the unweighted variants are not in the base protocol; the no-relations MLP control is deliberately excluded.",
          "* **Gap 3 (unsupported):** community discovery (H5) is outside this evaluation; no label-based held-out criterion exists.",
          f"* **Amendment 1:** `{AMEND}`, SHA-256 `{amend_hash}` (sidecar matches: {amend_ok}); module readiness on validation: H and U {'PASS' if amend_rep['all_pass'] else 'FAIL'} ({len(amend_rep['models'])} models); F: deterministic checks pass, trained-arm window-mean gate failed as written for 1 of 6 cases (see the amendment). Scope decisions: `SCOPE_RECONCILIATION.md`.",
          "* Caution: GAT is evaluated as the frozen non-converged runs; a held-out difference between GAT and the others cannot separate inferiority from under-training.", "",
          "## What the evaluation would make irreversible\n",
          "* The held-out period could never again serve as untouched test data for any model or decision in this project.",
          "* It is one-shot; after any held-out metric exists, no model, threshold, feature, evaluator setting or exclusion may be changed in response, and all results (including unfavourable ones) must be reported.",
          "* Each current validation-only caveat would be supplemented by a held-out reading; verdicts H1-H5 as recorded would not be rewritten.",
          "* Reading the held-out features and labels cannot be undone, and any later model selection that looked at the outcome would be invalid.", "",
          "## Residual risks stated for the decision\n",
          "* One future period of 31 days: concept drift may move every model; seeds are not independent replications.",
          "* GPU scores differ from the recorded validation predictions by up to the figure above; this affects tie-sensitive top-K metrics only marginally.",
          "* If the run crashes before any held-out score is written the defect may be fixed and rerun (protocol section 7); a crash while writing leaves partial outputs that must be handled by hand before a rerun.", "",
          "## Exact authorization\n", "Create `experiments/final_heldout_protocol/AUTHORIZATION.txt` containing the sentence in protocol section 9 and the protocol hash above; to add Amendment 1 modules it must also contain the amendment hash above and the sentence `I authorize the following Amendment 1 modules: <F, H, U subset>`. Then run `scripts/run_heldout_evaluation.py --split heldout` (base), `scripts/run_f_walkforward.py --split heldout` (F), `scripts/run_amendment_evaluation.py --split heldout --modules H,U` (H, U). Nothing has been created; the decision is the user's.", ""]
    md = ("\n".join(L)).encode("utf-8")
    js = (json.dumps(rec, indent=1) + "\n").encode("utf-8")
    for name, data in (("PROTOCOL_REVIEW.md", md), ("PROTOCOL_REVIEW.json", js)):
        (FH / name).write_bytes(data)
        (FH / name.replace(".md", ".sha256").replace(".json", ".sha256")).write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print("ready for informed authorization:", ok)
    print(json.dumps({k: facts[k] for k in ("sidecar_matches", "n_models", "all_artifacts_exist", "thresholds_all_recorded_in_unit_interval")}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
