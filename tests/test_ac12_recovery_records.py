"""Phase 12 AC12 recovery record (experiments/phase12_validation_evidence_table/AC12_GIT_SNAPSHOT_VERIFICATION.json): internally consistent, no
contradiction, the snapshot commit is in this repository's history, and the record keeps AC12's registered status unchanged. Reads recorded files and
Git metadata only."""
from __future__ import annotations

import hashlib
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
P12 = ROOT / "experiments" / "phase12_validation_evidence_table"
REC = P12 / "AC12_GIT_SNAPSHOT_VERIFICATION.json"


def load():
    return json.loads(REC.read_text(encoding="utf-8"))


def test_record_matches_its_sidecar_and_covers_every_pre_existing_file():
    assert hashlib.sha256(REC.read_bytes()).hexdigest() == REC.with_suffix(".sha256").read_text().strip()
    r = load()
    after = [ln.split("\t")[0] for ln in (P12 / "frozen_files_after.tsv").read_text(encoding="utf-8").splitlines() if "\t" in ln]
    assert r["pre_existing_files"] == len(after) == 651 and sorted(f["path"] for f in r["files"]) == sorted(after)
    assert sum(r["strongest_evidence_counts"].values()) == 651


def test_no_contradiction_and_counts_agree_with_the_per_file_rows():
    r = load()
    assert r["contradictions"] == []
    counts = {}
    for f in r["files"]:
        counts[f["strongest"]] = counts.get(f["strongest"], 0) + 1
        assert (f["strongest"] == "no_pre_build_evidence") == (f["evidence"] == [])
    assert counts == {k: v for k, v in r["strongest_evidence_counts"].items() if v}


def test_it_is_labelled_as_an_alternative_method_and_never_opened_held_out_files():
    r = load()
    assert "not the registered" in r["method"] and "does not change AC12" in r["method"]
    assert r["test_named_or_held_out_files_opened"] is False
    held = [f for f in r["files"] if any(t in f["path"] for t in ("test_features", "test_transaction", "test_identity"))]
    assert len(held) == 3 and all(f["strongest"] == "no_pre_build_evidence" for f in held)
    assert r["snapshot_timestamp_evidence"]["timestamp_utc"] < r["phase12_build_start_local"]


def test_snapshot_commit_is_the_root_of_this_repository_history():
    if not (ROOT / ".git").exists():
        return                                                       # a source archive without history: nothing to check
    r = load()
    roots = subprocess.run(["git", "rev-list", "--max-parents=0", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.split()
    assert r["snapshot_commit"] in roots
    n = len(subprocess.run(["git", "ls-tree", "-r", "--name-only", r["snapshot_commit"]], cwd=ROOT, capture_output=True, text=True).stdout.split("\n")) - 1
    assert n == r["strongest_evidence_counts"]["git_snapshot"] == 434


def test_phase12_closure_status_is_not_rewritten():
    integ = json.loads((P12 / "INTEGRITY_PHASE12.json").read_text(encoding="utf-8"))
    assert integ["phase12_status"].startswith("not formally complete") and "BLOCKED" in integ["frozen_artifact_check"]["registered_method_status"]


DER = P12 / "AC12_DERIVED_EVIDENCE.json"


def test_derived_evidence_record_is_labelled_not_provenance_and_leaves_ac12_unchanged():
    assert hashlib.sha256(DER.read_bytes()).hexdigest() == DER.with_suffix(".sha256").read_text().strip()
    s, r = json.loads(DER.read_text(encoding="utf-8")), load()
    assert "NOT provenance" in s["method"] and "NOT counted" in s["method"] and s["ac12_status"].startswith("unchanged") and s["held_out_files_read"] is False
    assert s["source_record_sha256"] == hashlib.sha256(REC.read_bytes()).hexdigest()
    assert all(hashlib.sha256((ROOT / k).read_bytes()).hexdigest() == v for k, v in s["readiness_records_sha256"].items())
    none = {f["path"] for f in r["files"] if f["strongest"] == "no_pre_build_evidence"}
    assert s["files_without_pre_build_evidence_total"] == len(none) == r["strongest_evidence_counts"]["no_pre_build_evidence"] == 101
    assert {row["file"] for row in s["sidecar_rows"]} == {p for p in none if p.endswith(".sha256")}
    assert {row["file"] for row in s["functional_rows"]} <= none
    for row in s["sidecar_rows"]:
        assert row["derived"] == (row["sidecar_equals_companion_hash"] and row["companion_strongest_evidence"] not in ("no_pre_build_evidence", "not a pre-existing file"))
    for row in s["functional_rows"]:
        assert row["derived"] == (row["readiness_pass"] and row["recorded_predictions_strongest_evidence"] not in ("no_pre_build_evidence", "not a pre-existing file"))
    derived = {row["file"] for row in s["sidecar_rows"] + s["functional_rows"] if row["derived"]}
    assert (s["sidecar_consistency"]["derived"], s["functional_reproduction"]["derived"], len(derived), s["files_with_neither"]) == (20, 18, 38, 63)


def test_derived_evidence_recomputes_from_the_current_files():
    sys.path.insert(0, str(ROOT / "scripts"))
    import check_ac12_derived_evidence as C
    new, old = C.check(), json.loads(DER.read_text(encoding="utf-8"))
    assert new["sidecar_rows"] == old["sidecar_rows"] and new["functional_rows"] == old["functional_rows"]


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
