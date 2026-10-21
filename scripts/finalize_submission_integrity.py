"""Final integrity record for the completion work.

Compares the whole project against the last persistent baseline taken before this work (the Phase 12 pre-change manifest, plus the documented
changes made after it), lists every new file with its SHA-256, and writes submission/INTEGRITY_SUBMISSION.json with a sidecar.
TEST-named files are compared by size and modification time only and are never opened.

Usage:  .venv/Scripts/python.exe -B scripts/finalize_submission_integrity.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)

BASELINE = "experiments/phase12_validation_evidence_table/PRECHANGE_MANIFEST_closure_update.tsv"
# Changes made between that baseline and the start of the completion work, each documented in its own record (Phase 12 revision record).
DOCUMENTED_CHANGES = {
    "experiments/phase12_validation_evidence_table/PHASE12_CLOSURE_REPORT.md": "Phase 12 closure report revised (AC12 wording); recorded in its section 8 and in INTEGRITY_PHASE12.json 'revision'",
    "experiments/phase12_validation_evidence_table/INTEGRITY_PHASE12.json": "Phase 12 integrity record updated to cover the revised report",
    "experiments/phase12_validation_evidence_table/INTEGRITY_PHASE12.sha256": "sidecar of the updated integrity record",
}
SYNTHESIS_REPORT_SHA = "4fdb4133b33984e51d81faff16778aa5ccb1e6c7b055972c7b929687d8424353"
OUT = "submission/INTEGRITY_SUBMISSION.json"


def is_test_named(rel: str) -> bool:
    b = os.path.basename(rel).lower()
    return "test_features" in b or (rel.startswith("data/") and "test" in b)


def sha(rel: str) -> str:
    return hashlib.sha256(pathlib.Path(rel).read_bytes()).hexdigest()


def main() -> int:
    base = dict(l.split("\t") for l in pathlib.Path(BASELINE).read_text(encoding="utf-8").splitlines() if l)
    cur = {}
    for root, dirs, files in os.walk("."):
        r = root.replace("\\", "/") + "/"
        if r.startswith("./.venv") or r.startswith("./.git") or "/__pycache__" in r or r.startswith("./demo_output"):
            continue
        for f in files:
            cur[(r + f)[2:]] = None
    changed, missing, unchanged = [], [], 0
    test_named_ok = True
    for rel, h in base.items():
        if rel not in cur:
            missing.append(rel)
            continue
        if is_test_named(rel):
            st = os.stat(rel)
            now = f"stat:{st.st_size}:{st.st_mtime_ns}"
            test_named_ok &= now == h
            unchanged += now == h
            if now != h:
                changed.append(rel)
            continue
        if sha(rel) == h:
            unchanged += 1
        else:
            changed.append(rel)
    undocumented = [c for c in changed if c not in DOCUMENTED_CHANGES]
    new_files = sorted(p for p in cur if p not in base and p not in (OUT, OUT.replace(".json", ".sha256")))
    new_sha = {p: sha(p) for p in new_files if not is_test_named(p)}
    areas = {}
    for p in new_files:
        k = "/".join(p.split("/")[:2]) if p.startswith(("experiments/", "src/")) else p.split("/")[0]
        areas[k] = areas.get(k, 0) + 1
    synth_ok = pathlib.Path("reports/PROJECT_SYNTHESIS_REPORT.md").exists() and sha("reports/PROJECT_SYNTHESIS_REPORT.md") == SYNTHESIS_REPORT_SHA
    prereg = {}
    for rel in ("experiments/comparison_B_graph_features/PREREGISTRATION_B", "experiments/comparison_B_graph_features/GRAPH_EVIDENCE_PREREG",
                "experiments/imbalance_check_xgboost/PREREGISTRATION_IMBALANCE", "experiments/final_heldout_protocol/HELDOUT_EVALUATION_PROTOCOL"):
        prereg[rel + ".md"] = {"sha256": sha(rel + ".md"), "sidecar_matches": sha(rel + ".md") == pathlib.Path(rel + ".sha256").read_text().strip()}
    rec = {
        "created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "purpose": "integrity record of the completion work (comparison B, imbalance check, graph evidence, held-out protocol and evaluator, software demonstration, submission package)",
        "baseline": {"file": BASELINE, "baseline_sha256": sha(BASELINE), "entries": len(base),
                     "note": "the last persistent pre-change manifest; it predates the synthesis report, which is verified separately"},
        "frozen_check": {"baseline_files_unchanged": unchanged, "baseline_files_missing": missing, "changed_since_baseline": changed,
                         "documented_changes": {c: DOCUMENTED_CHANGES[c] for c in changed if c in DOCUMENTED_CHANGES}, "undocumented_changes": undocumented,
                         "test_named_files_unchanged_by_size_and_mtime": bool(test_named_ok),
                         "test_named_files": [p for p in base if is_test_named(p)], "test_named_contents_opened": False},
        "synthesis_report_sha256_matches_reported": synth_ok,
        "preregistrations_hashed_before_use": prereg,
        "new_files": {"count": len(new_files), "by_area": areas, "sha256": new_sha},
        "limitations": ["Phase 12 acceptance criterion AC12 remains BLOCKED (partially verified: 300 of 651 pre-existing files; 322 not retrospectively verifiable). "
                        "This record is a post-hoc comparison against a baseline taken after that history and does not resolve it.",
                        "Files from Phases 1-8 have no per-file integrity record of the Phase 9-12 type; they are compared here only against the Phase 12 pre-change manifest."],
        "result": "PASS" if not missing and not undocumented and test_named_ok and synth_ok else "FAIL",
    }
    data = (json.dumps(rec, indent=1) + "\n").encode("ascii")
    pathlib.Path(OUT).write_bytes(data)
    pathlib.Path(OUT.replace(".json", ".sha256")).write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print(json.dumps({k: rec[k] for k in ("result", "synthesis_report_sha256_matches_reported")}, indent=1))
    print("baseline files unchanged:", unchanged, "| missing:", missing, "| changed:", changed, "| undocumented:", undocumented)
    print("new files:", len(new_files), areas)
    return 0 if rec["result"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
