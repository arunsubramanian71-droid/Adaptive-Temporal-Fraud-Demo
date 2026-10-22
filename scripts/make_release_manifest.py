"""File-level release manifest: every file a clone of the release will contain, compared with a published Git reference, plus the withheld
artifacts and how a clone obtains them. Read-only apart from the output file; it opens no data file (withheld files are only counted).

  .venv/Scripts/python.exe -B scripts/make_release_manifest.py [--base origin/main] [--out submission/RELEASE_MANIFEST.json]

"Release" = the working tree's tracked files plus untracked, non-ignored files (what `git add -A` would publish), minus deleted files.
For each file: SHA-256, category, and its state against --base (unchanged, modified, added). Files of --base that the release drops are listed.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent

REGENERATED_AFTER = {"submission/INTEGRITY_SUBMISSION.json", "submission/INTEGRITY_SUBMISSION.sha256"}
WITHHELD = [  # (group, git pathspec patterns, how a clone obtains it, needed for the supported demonstration?)
    ("IEEE-CIS raw files", ["data/raw/*"], "Kaggle (competition rules; not redistributable): README section 7 step 1", "no; only for research-model mode"),
    ("processed and feature tables, graph file, split record, entity maps", ["data/processed/*", "data/features/*"],
     "scripts/build_processed_data.py, build_graph.py, build_historical_features.py from the raw files", "no; research-model mode needs data/features/val_features.parquet"),
    ("trained models and checkpoints", ["*.joblib", "*.pt"], "scripts/train_*.py (train_xgboost.py regenerates experiments/xgboost byte for byte from the feature tables)",
     "no; research-model mode needs experiments/xgboost/model.joblib"),
    ("raw-vocabulary encoder files", ["*frequency_encoder.json", "*median_imputer.json", "*ordinal_maps.json"], "written by the same train/build scripts", "no; research-model mode needs the xgboost ones"),
    ("row-level prediction files", ["*predictions*.parquet"], "written by the training and evaluation scripts", "no"),
    ("Phase 9 arrays and caches", ["*.npz", "*.npy"], "Phase 9 build and risk scripts", "no"),
    ("per-transaction evidence files (never to be published)", ["*relation_evidence.csv", "*top_features_B.json", "*relation_effects.csv", "*contributions_val.csv",
                                                                 "*selection_val.csv", "*top_features_val.json"], "the Phase 11 and evidence scripts, locally only", "no"),
    ("demonstration output", ["demo_output/*"], "scripts/demo_end_to_end.py", "generated during the demonstration"),
]


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout


def category(f: str) -> str:
    if f.startswith("src/"):
        return "source code"
    if f.startswith("configs/") or f in (".gitignore", ".gitattributes") or f.startswith("requirements"):
        return "configuration and dependency manifests"
    if f.endswith((".joblib", ".pkl", ".pt")):
        return "small preprocessing artifacts"
    if "feature_builder/" in f or f.endswith(("feature_names.json", "column_roles.json", "feature_columns.json")):
        return "feature definitions"
    if f.startswith("scripts/"):
        return "scripts and entry points"
    if f.startswith("tests/"):
        return "tests"
    if f.startswith(("reports/", "submission/")) or f == "README.md":
        return "reports and documentation"
    if f.startswith("experiments/"):
        return "recorded results, designs and integrity records"
    if f.startswith("data/"):
        return "data records without rows"
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="origin/main")
    ap.add_argument("--out", default="submission/RELEASE_MANIFEST.json")
    a = ap.parse_args(argv)
    out_rel = pathlib.Path(a.out).as_posix()
    tracked = set(git("ls-files", "-z").split("\0")) - {""}
    deleted = {l[3:] for l in git("status", "--porcelain").splitlines() if "D" in l[:2]}
    untracked = set(git("ls-files", "-z", "--others", "--exclude-standard").split("\0")) - {""}
    release = sorted((tracked - deleted) | untracked - {out_rel})
    base_files = {}
    for line in git("ls-tree", "-r", "-z", a.base).split("\0"):
        if line:
            meta, path = line.split("\t", 1)
            base_files[path] = meta.split()[2]
    rows, counts = [], {}
    for f in release:
        data = (ROOT / f).read_bytes()
        blob = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()           # git's blob id, to compare with the base tree
        state = "added" if f not in base_files else ("unchanged" if base_files[f] == blob else "modified")
        cat = category(f)
        counts.setdefault(cat, 0)
        counts[cat] += 1
        later = f in REGENERATED_AFTER                                     # these records fingerprint this manifest, so they are written after it
        rows.append({"path": f, "sha256": None if later else hashlib.sha256(data).hexdigest(), "bytes": None if later else len(data), "category": cat,
                     "state_vs_base": state, **({"note": "written after this manifest (it fingerprints it); see its own sidecar"} if later else {})})
    dropped = sorted(set(base_files) - set(release) - {out_rel})
    ignored = [l[3:] for l in git("status", "--porcelain", "--ignored", "--untracked-files=all").splitlines() if l.startswith("!!")]
    ignored = [p for p in ignored if not p.startswith((".venv", ".claude")) and "__pycache__" not in p]
    import fnmatch
    withheld = []
    for group, pats, how, needed in WITHHELD:
        n = sum(1 for p in ignored if any(fnmatch.fnmatch(p, pat) for pat in pats))
        withheld.append({"group": group, "patterns": pats, "files_on_this_machine": n, "how_a_clone_obtains_it": how, "needed_for_the_supported_demonstration": needed})
    rec = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "base": a.base, "base_commit": git("rev-parse", a.base).strip(),
           "summary": {"release_files": len(rows), "by_category": dict(sorted(counts.items())),
                       "vs_base": {s: sum(r["state_vs_base"] == s for r in rows) for s in ("unchanged", "modified", "added")}, "dropped_from_base": len(dropped)},
           "dropped_from_base": dropped, "withheld_artifacts": withheld, "files": rows,
           "statement": "The release contains no raw data, no row-level file and no model checkpoint; the withheld groups are obtained as stated. Withheld groups can overlap (a file may match two groups)."}
    (ROOT / out_rel).write_text(json.dumps(rec, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(rec["summary"], indent=1))
    for w in withheld:
        print(f"withheld: {w['files_on_this_machine']:4d}  {w['group']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
