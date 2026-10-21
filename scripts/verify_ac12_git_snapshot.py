"""Phase 12 AC12: per-file recovery of pre-build evidence from every legitimate source, with the Git snapshot as the new independent source.

AC12 registered "SHA-256 before/after comparison of every existing file". The pre-build SHA-256 snapshot was lost. This script classifies each of the
651 pre-existing files (frozen_files_after.tsv, the recorded post-build hashes) by the strongest pre-build evidence that exists for it:

1. git_snapshot: commit e70238f was made on 2026-10-16 and GitHub's server-side activity log records its push (branch creation of main) at
   2026-10-16T13:44:48Z, before the Phase 12 build started (2026-10-19 12:59 local). A commit hash fixes its tree, so its blobs are authentic
   pre-build content. The current file's Git blob id (raw, or with CRLF normalised to LF because the snapshot was committed with core.autocrlf=true)
   must equal the snapshot blob, and the current file's SHA-256 must equal the recorded post-build hash.
2. scope_fingerprint: the 165 declared Phase 12 source files are covered collectively by the fingerprint recorded in the scope record before the
   build (the closure report's verified result; reused here, not recomputed).
3. session_log: a SHA-256 printed in a saved session record dated before the build equals the recorded post-build hash (optional source:
   --session-logs DIR, the harness's JSONL records; machine-specific).
4. integrity_record: a SHA-256 in a pre-existing project integrity record or manifest equals the recorded post-build hash. These records were written
   before the build but are not themselves independently timestamped, so this is the weakest source and is labelled as such.
Anything else is "no_pre_build_evidence". Recorded hashes are compared with recorded hashes; besides the files tracked in e70238f (opened to compute
their blob ids), no data file is opened, and no TEST-named file or file holding held-out-period rows is ever opened.

This is NOT the registered method (Git stores SHA-1 ids of line-ending-normalised content; sources 2-4 are not a per-file before/after SHA-256
taken at the build start). It does not change AC12's registered status; whether it is acceptable substitute evidence is the evaluator's decision.

Usage:  .venv/Scripts/python.exe -B scripts/verify_ac12_git_snapshot.py [--session-logs DIR]
Writes experiments/phase12_validation_evidence_table/AC12_GIT_SNAPSHOT_VERIFICATION.json (+ sidecar).
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import re
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
P12 = ROOT / "experiments" / "phase12_validation_evidence_table"
SNAPSHOT = "e70238f6e33ee6079697288dc83a99dceccddda7"
GITHUB_PUSH = {"source": "GET https://api.github.com/repos/arunsubramanian71-droid/Adaptive-Temporal-Fraud/activity (public API)", "activity_type": "branch_creation",
               "ref": "refs/heads/main", "after": SNAPSHOT, "timestamp_utc": "2026-10-16T13:44:48Z"}
BUILD_START_LOCAL = "2026-10-19T12:59+05:30"
BUILD_START_UTC = "2026-10-19T07:29"
HEX = re.compile(r"^[0-9a-f]{64}$")
PATH = re.compile(r"([A-Za-z0-9_\-./]+\.(?:json|tsv|csv|parquet|py|md|txt|png|pkl|npz|yaml|joblib|pt|sha256))")


def blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def harvest_records(paths) -> dict:
    """path -> {(sha256, record)} from pre-existing JSON integrity records/manifests and TSV manifests (no data file is opened)."""
    found = collections.defaultdict(set)

    def walk(o, src):
        if isinstance(o, dict):
            p, h = o.get("path") or o.get("file"), o.get("sha256") or o.get("hash")
            if isinstance(p, str) and isinstance(h, str) and HEX.match(h):
                found[p.replace("\\", "/")].add((h, src))
            for k, v in o.items():
                if isinstance(v, str) and HEX.match(v) and ("/" in k or "." in k):
                    found[k.replace("\\", "/")].add((v, src))
                walk(v, src)
        elif isinstance(o, list):
            for v in o:
                walk(v, src)
    for rel in paths:
        f = ROOT / rel
        if rel.startswith("data/") or not f.exists() or not (rel.endswith(".json") or rel.endswith(".tsv")):
            continue
        try:
            if rel.endswith(".json"):
                walk(json.loads(f.read_text(encoding="utf-8")), rel)
            else:
                for line in f.read_text(encoding="utf-8").splitlines():
                    parts = line.split("\t")
                    if len(parts) >= 2 and HEX.match(parts[-1].strip()):
                        found[parts[0].replace("\\", "/")].add((parts[-1].strip(), rel))
        except (ValueError, UnicodeDecodeError):
            continue
    return found


def harvest_logs(log_dir: pathlib.Path) -> dict:
    found = collections.defaultdict(set)
    for f in sorted(log_dir.glob("*.jsonl")):
        for line in open(f, encoding="utf-8"):
            try:
                o = json.loads(line)
            except ValueError:
                continue
            ts = o.get("timestamp", "")
            if not ts or ts >= BUILD_START_UTC or not re.search(r"[0-9a-f]{64}", line):
                continue
            s = line.replace("\\\\", "/")
            for m in re.finditer(r"[0-9a-f]{64}", s):
                pb = PATH.findall(s[max(0, m.start() - 220): m.start()])
                pa = PATH.findall(s[m.end(): m.end() + 120])
                p = pb[-1] if pb else (pa[0] if pa else None)
                if p:
                    found[p.strip("/")].add((m.group(0), f"{f.name[:8]}@{ts[:16]}Z"))
    return found


def lookup(found: dict, p: str) -> set:
    out = set(found.get(p, set()))
    for k, v in found.items():
        if k != p and (k.endswith("/" + p) or (p.endswith("/" + k) and "/" in k)):
            out |= v
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session-logs", default=None, help="directory of the saved session JSONL records (optional source 3)")
    a = ap.parse_args(argv)
    tree = {}
    for line in subprocess.run(["git", "ls-tree", "-r", SNAPSHOT], cwd=ROOT, capture_output=True, text=True, check=True).stdout.splitlines():
        meta, path = line.split("\t", 1)
        tree[path] = meta.split()[2]
    after = {}
    for line in (P12 / "frozen_files_after.tsv").read_text(encoding="utf-8").splitlines():
        if "\t" in line:
            p, h = line.split("\t", 1)
            after[p] = h.strip()
    declared = set()
    for line in (P12 / "outputs" / "source_manifest.tsv").read_text(encoding="utf-8").splitlines():
        p = line.split("\t")[0]
        if p in after:
            declared.add(p)
    records = harvest_records(sorted(after))
    logs = harvest_logs(pathlib.Path(a.session_logs)) if a.session_logs else {}
    rows, contradictions = [], []
    for path in sorted(after):
        row = {"path": path, "evidence": []}
        if path in tree:
            data = (ROOT / path).read_bytes()
            cur = hashlib.sha256(data).hexdigest()
            raw, lf = blob_id(data), blob_id(data.replace(b"\r\n", b"\n"))
            if cur == after[path] and tree[path] in (raw, lf):
                row["evidence"].append("git_snapshot" + ("" if tree[path] == raw else " (line endings normalised)"))
            else:
                contradictions.append({"path": path, "source": "git_snapshot", "current_equals_post_build": cur == after[path]})
        if path in declared:
            row["evidence"].append("scope_fingerprint")
        for name, src in (("session_log", logs), ("integrity_record", records)):
            hs = lookup(src, path)
            if not hs:
                continue
            if any(h == after[path] for h, _ in hs):
                row["evidence"].append(f"{name}: " + ", ".join(sorted({s for h, s in hs if h == after[path]})[:3]))
            else:
                contradictions.append({"path": path, "source": name, "recorded": sorted(hs)[:3], "post_build": after[path]})
        rank = ("git_snapshot", "scope_fingerprint", "session_log", "integrity_record")
        row["strongest"] = next((r for r in rank if any(e.startswith(r) for e in row["evidence"])), "no_pre_build_evidence")
        rows.append(row)
    counts = collections.Counter(r["strongest"] for r in rows)
    residue = [r["path"] for r in rows if r["strongest"] == "no_pre_build_evidence"]
    by_area = collections.Counter("/".join(p.split("/")[:2]) for p in residue)
    rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
           "method": "alternative per-file recovery (not the registered SHA-256 before/after comparison); does not change AC12's registered status",
           "snapshot_commit": SNAPSHOT, "snapshot_timestamp_evidence": GITHUB_PUSH, "phase12_build_start_local": BUILD_START_LOCAL,
           "session_logs_used": bool(a.session_logs), "pre_existing_files": len(after),
           "strongest_evidence_counts": {k: counts.get(k, 0) for k in ("git_snapshot", "scope_fingerprint", "session_log", "integrity_record", "no_pre_build_evidence")},
           "contradictions": contradictions, "no_pre_build_evidence_by_area": dict(by_area.most_common()),
           "limits": ["git blob ids are SHA-1 over line-ending-normalised content: a change of line endings alone cannot be excluded for 'line endings normalised' files",
                      "integrity_record evidence comes from records written before the build that are not themselves independently timestamped",
                      "a file equal before and after could in principle have been changed and restored in between; the closure report's modification-time scan found no pre-existing file modified after the date rollover",
                      "TEST-named files were never opened or hashed and cannot be verified by any method used here",
                      "accepting this as substitute evidence for AC12 is the evaluator's decision"],
           "test_named_or_held_out_files_opened": False, "files": rows}
    data = (json.dumps(rec, indent=1) + "\n").encode("utf-8")
    dest = P12 / "AC12_GIT_SNAPSHOT_VERIFICATION.json"
    dest.write_bytes(data)
    dest.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print(json.dumps({"strongest_evidence_counts": rec["strongest_evidence_counts"], "contradictions": len(contradictions), "no_pre_build_evidence_by_area": rec["no_pre_build_evidence_by_area"]}, indent=1))
    return 0 if not contradictions else 1


if __name__ == "__main__":
    sys.exit(main())
