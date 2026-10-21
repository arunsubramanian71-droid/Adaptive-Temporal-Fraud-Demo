"""AC12 follow-up: derived evidence for files among the 101 that have no pre-build evidence. Reads records and validation files only; no held-out file is read.

Two classes are recorded. Both are **derived consistency, not provenance**. Neither shows that a file existed unchanged before the Phase 12 build. Neither is counted in the
AC12 recovery, and neither changes AC12's registered status.

1. **Sidecar consistency.** A sidecar `X.sha256` holds only the SHA-256 of its companion `X.<ext>`. Suppose the sidecar equals its companion's hash now and the companion has
   pre-build evidence. Then the sidecar holds the only correct content it could have had before the build, so it differs from its pre-build state only if it was wrong then.
2. **Functional reproduction.** A frozen model file is re-scored by a recorded validation readiness gate. The gate reproduced the recorded validation predictions within its
   registered tolerance, and those predictions have pre-build evidence. Then the current file computes the same validation scores as the model that produced the pre-build
   predictions. A change that leaves every validation score unchanged would not be detected.

  .venv/Scripts/python.exe -B scripts/check_ac12_derived_evidence.py
Writes experiments/phase12_validation_evidence_table/AC12_DERIVED_EVIDENCE.json (+ .sha256).
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
P12 = ROOT / "experiments" / "phase12_validation_evidence_table"
REC = "experiments/phase12_validation_evidence_table/AC12_GIT_SNAPSHOT_VERIFICATION.json"
OUT = P12 / "AC12_DERIVED_EVIDENCE.json"
BASE_READY = "experiments/final_heldout_protocol/readiness_val/readiness_report.json"
AMEND_READY = "experiments/final_heldout_protocol/readiness_val_amendment/readiness_report.json"
NONE = "no_pre_build_evidence"
ABSENT = "not a pre-existing file"


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def sidecar_rows(root: pathlib.Path, strongest: dict) -> list:
    rows = []
    for path, s in sorted(strongest.items()):
        if s != NONE or not path.endswith(".sha256"):
            continue
        side = root / path
        comps = sorted(c for c in side.parent.glob(side.stem + ".*") if c.suffix != ".sha256")
        if len(comps) != 1:
            rows.append({"file": path, "companion": None, "derived": False, "note": "no unique companion"})
            continue
        comp = comps[0].relative_to(root).as_posix()
        cs = strongest.get(comp, ABSENT)
        match = side.read_text(encoding="utf-8").strip().split()[0] == sha(comps[0])
        rows.append({"file": path, "companion": comp, "companion_strongest_evidence": cs, "sidecar_equals_companion_hash": match,
                     "derived": bool(match and cs not in (NONE, ABSENT))})
    return rows


def functional_rows(root: pathlib.Path, strongest: dict) -> list:
    from src.heldout import amendment as A
    from src.heldout import evaluator as E
    base = {m["model"]: m for m in json.loads((root / BASE_READY).read_text(encoding="utf-8"))["models"]}
    amend = {m["model"]: m for m in json.loads((root / AMEND_READY).read_text(encoding="utf-8"))["models"]}
    cand = []
    for mid, info in sorted(E.frozen_registry().items()):
        model_file = next(f for f in info["files"] if f.startswith("model"))
        cand.append((mid, f"{info['dir']}/{model_file}", f"{info['dir']}/predictions_val.parquet", base.get(mid), BASE_READY))
    cand.append(("xgboost_historical", f"{A.HIST_DIR}/model.joblib", f"{A.HIST_DIR}/predictions_val.parquet", amend.get("xgboost_historical"), AMEND_READY))
    rows = []
    for mid, mf, pv, row, src in cand:
        if strongest.get(mf) != NONE:
            continue
        ps = strongest.get(pv, ABSENT)
        ok = bool(row and row.get("pass") is True and row.get("ids_identical") is True)
        rows.append({"file": mf, "model": mid, "readiness_record": src, "readiness_pass": ok,
                     "max_abs_score_diff": None if row is None else row.get("max_abs_score_diff"), "recorded_predictions": pv,
                     "recorded_predictions_strongest_evidence": ps, "derived": bool(ok and ps not in (NONE, ABSENT))})
    return rows


def check(root: pathlib.Path = ROOT) -> dict:
    rec = json.loads((root / REC).read_text(encoding="utf-8"))
    strongest = {f["path"]: f["strongest"] for f in rec["files"]}
    side, func = sidecar_rows(root, strongest), functional_rows(root, strongest)
    none = sum(1 for s in strongest.values() if s == NONE)
    derived = {r["file"] for r in side + func if r["derived"]}
    return {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
            "method": "derived consistency (sidecar consistency; functional reproduction by the recorded readiness gates); NOT provenance and NOT counted in the AC12 recovery",
            "source_record": REC, "source_record_sha256": sha(root / REC),
            "readiness_records_sha256": {p: sha(root / p) for p in (BASE_READY, AMEND_READY)},
            "files_without_pre_build_evidence_total": none,
            "sidecar_consistency": {"candidates": len(side), "derived": sum(r["derived"] for r in side)},
            "functional_reproduction": {"candidates": len(func), "derived": sum(r["derived"] for r in func)},
            "files_with_derived_evidence": len(derived), "files_with_neither": none - len(derived),
            "limits": ["derived consistency cannot show that a file existed before the build",
                       "a sidecar that was wrong before the build and corrected during it would look the same",
                       "a model change that leaves every validation score unchanged would not be detected; tolerances are the registered readiness tolerances"],
            "ac12_status": "unchanged: BLOCKED as registered", "held_out_files_read": False,
            "sidecar_rows": side, "functional_rows": func}


def main() -> int:
    r = check()
    data = (json.dumps(r, indent=1) + "\n").encode("utf-8")
    OUT.write_bytes(data)
    OUT.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print(f"of {r['files_without_pre_build_evidence_total']} files without pre-build evidence: sidecar consistency {r['sidecar_consistency']['derived']} of "
          f"{r['sidecar_consistency']['candidates']}, functional reproduction {r['functional_reproduction']['derived']} of {r['functional_reproduction']['candidates']}; "
          f"{r['files_with_neither']} have neither (derived consistency, not provenance; not counted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
