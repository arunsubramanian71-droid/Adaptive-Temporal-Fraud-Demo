"""Phase 12 finalization: re-verify earlier integrity records, check that no pre-existing file changed, hash the Phase 12 files.

Writes (new files only, inside experiments/phase12_validation_evidence_table/): frozen_files_after.tsv (+ sidecar) and
INTEGRITY_PHASE12.json (+ sidecar). Reads every project file outside .venv/.git/__pycache__ for hashing; files whose names contain
"test_features" or that sit under data/ with "test" in the name are recorded by size and modification time only and are never opened.
Usage:  .venv/Scripts/python.exe -B scripts/finalize_phase12_integrity.py --since 2026-10-19T00:00:00 --prebuild-count 653 --prebuild-phase12-files 2
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
os.chdir(ROOT)
P12 = "experiments/phase12_validation_evidence_table/"
HEX = re.compile(r"^[0-9a-f]{64}$")


def h(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def is_phase12(rel: str) -> bool:
    return rel.startswith(P12) or rel.startswith("src/evidence/") or "phase12" in rel


def walk_hashes(obj, found):
    """Collect (path, sha256) pairs from any JSON record: dict keys that are existing project paths with a 64-hex value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str) and HEX.match(v) and isinstance(k, str) and "/" in k and pathlib.Path(k).is_file():
                found[k] = v
            else:
                walk_hashes(v, found)
    elif isinstance(obj, list):
        for v in obj:
            walk_hashes(v, found)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", required=True)
    ap.add_argument("--prebuild-count", type=int, required=True)
    ap.add_argument("--prebuild-phase12-files", type=int, required=True, help="Phase 12 files that already existed when the pre-build snapshot was taken")
    a = ap.parse_args()
    since = datetime.datetime.fromisoformat(a.since).timestamp()

    # ---- 1. earlier integrity records re-verified live
    records = ["experiments/phase9_coordinated_fraud/phase9_closure_integrity.json", "experiments/phase11_explainable_evidence/INTEGRITY_PHASE11.json"] + \
              sorted(str(p).replace("\\", "/") for p in pathlib.Path("experiments/phase10_community_evolution").glob("INTEGRITY*.json"))
    earlier = {}
    for rec in records:
        found = {}
        walk_hashes(json.loads(pathlib.Path(rec).read_text(encoding="utf-8")), found)
        bad = [p for p, v in found.items() if h(p) != v]
        sidecar = pathlib.Path(rec).with_suffix(".sha256")
        earlier[rec] = {"hashed_paths": len(found), "mismatches": bad, "sidecar_matches": (sidecar.exists() and sidecar.read_text().strip() == h(rec)) or "no sidecar"}
    bad_any = [r for r, v in earlier.items() if v["mismatches"] or v["sidecar_matches"] is False]

    # ---- 2. files and modification times
    files = []
    for root, dirs, fs in os.walk("."):
        r = root.replace("\\", "/") + "/"
        if r.startswith("./.venv") or r.startswith("./.git") or "/__pycache__" in r:
            continue
        files += [(r + f)[2:] for f in fs]
    pre = sorted(f for f in files if not is_phase12(f))
    new = sorted(f for f in files if is_phase12(f))
    modified_pre = [f for f in pre if os.path.getmtime(f) >= since]

    # ---- 3. 'after' manifest of every pre-existing file (TEST-named files by size and mtime only; never opened)
    lines = []
    for f in pre:
        low = pathlib.Path(f).name.lower()
        if "test_features" in low or (f.startswith("data/") and "test" in low):
            st = os.stat(f)
            lines.append(f"{f}\tstat:{st.st_size}:{st.st_mtime_ns}\n")
        else:
            lines.append(f"{f}\t{h(f)}\n")
    manifest_text = "".join(lines)
    (ROOT / P12 / "frozen_files_after.tsv").write_bytes(manifest_text.encode("utf-8"))
    (ROOT / P12 / "frozen_files_after.sha256").write_bytes(hashlib.sha256(manifest_text.encode("utf-8")).hexdigest().encode("ascii"))

    # ---- 4. Phase 12 files
    exclude = {P12 + "INTEGRITY_PHASE12.json", P12 + "INTEGRITY_PHASE12.sha256"}
    p12_files = sorted(f for f in new if f not in exclude and not f.endswith(".sha256"))
    res = json.loads((ROOT / P12 / "test_results/test_results.json").read_text(encoding="utf-8"))
    sens = json.loads((ROOT / P12 / "test_results/test_sensitivity.json").read_text(encoding="utf-8"))
    out = json.loads((ROOT / P12 / "outputs/manifest_phase12.json").read_text(encoding="utf-8"))
    rec = {
        "purpose": "Integrity record for the Phase 12 consolidated validation-evidence table (stored VAL predictions only).",
        "created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
        "scope_record": {P12 + "SCOPE_AND_ACCEPTANCE.md": h(P12 + "SCOPE_AND_ACCEPTANCE.md")},
        "source_set": {"fingerprint": out["source_set_fingerprint"], "n_files": out["n_source_files"]},
        "files_sha256": {f: h(f) for f in p12_files},
        "tests": {"total": res["summary"]["total"], "passed": res["summary"]["passed"], "failed": res["summary"]["failed"],
                  "recorded_file_hashes_match_disk": all(h(f) == v for f, v in res["files_sha256"].items()),
                  "sensitivity": sens["summary"], "sensitivity_unmutated_failures": sens["unmutated_failures"],
                  "sensitivity_module_hashes_match_disk": all(h(f"src/evidence/{n}.py") == v for n, v in sens["module_sha256"].items())},
        "frozen_artifact_check": {
            "registered_method": "SHA-256 before/after comparison of every existing project file",
            "registered_method_status": "BLOCKED: the pre-build snapshot file (taken at the start of the work; 653 files) was lost before the comparison because the harness cleared the scratchpad. "
                                        "The count it reported (653, which included the 2 Phase 12 scope-record files that already existed) is recorded here; its hashes are not available.",
            "pre_build_file_count_reported": a.prebuild_count, "pre_existing_files_now": len(pre), "new_phase12_files_now": len(new),
            "phase12_files_present_at_snapshot": a.prebuild_phase12_files,
            "count_reconciles": len(pre) + a.prebuild_phase12_files == a.prebuild_count,
            "substitute_evidence": {
                "modification_time_scan": {"since": a.since, "pre_existing_files_modified_since": modified_pre,
                                           "note": "weaker than a hash comparison: it detects ordinary edits but not a deliberate restoration of timestamps"},
                "earlier_integrity_records_reverified_now": earlier, "earlier_records_all_ok": not bad_any,
                "after_manifest": {"path": P12 + "frozen_files_after.tsv", "sha256": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
                                   "files": len(lines), "test_named_files_recorded_by_size_and_mtime_only": [f.split("\t")[0] for f in lines if "\tstat:" in f]},
                "writes": "all writes in src/evidence go through outputs.py with paths under the build's output folder (tested: a blocked build leaves the folder empty)"},
        },
        "boundaries": out["boundaries"],
        "hash_convention": "SHA-256 over file bytes; sidecars hold the 64-character lowercase digest only; this record cannot contain its own hash (see INTEGRITY_PHASE12.sha256).",
    }
    path = ROOT / P12 / "INTEGRITY_PHASE12.json"
    path.write_bytes((json.dumps(rec, indent=1, ensure_ascii=True) + "\n").encode("ascii"))
    (ROOT / P12 / "INTEGRITY_PHASE12.sha256").write_bytes(h(path).encode("ascii"))
    print(json.dumps({"earlier_records_ok": not bad_any, "pre_existing_modified_since": modified_pre, "pre_existing_files": len(pre), "count_reconciles": len(pre) + a.prebuild_phase12_files == a.prebuild_count,
                      "phase12_files_hashed": len(p12_files), "after_manifest_files": len(lines)}, indent=1))
    return 0 if not bad_any and not modified_pre else 2


if __name__ == "__main__":
    sys.exit(main())
