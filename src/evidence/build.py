"""Phase 12 build: verify the scope record and the source-set fingerprint, recompute, check, assemble, then write the outputs.

Nothing is written unless every gate passes (a failed gate raises ``BuildBlocked`` / ``SourceError`` with the exact discrepancy).
Scope record: experiments/phase12_validation_evidence_table/SCOPE_AND_ACCEPTANCE.md.
"""
from __future__ import annotations

import datetime
import hashlib
import pathlib
import platform

import numpy as np
import pandas as pd
import sklearn

from src.evidence import outputs as O
from src.evidence import sources as S
from src.evidence import table as T

APPLIES_TO = [
    "A:logistic_regression; A:random_forest; A:xgboost; C; D; E", "C; Ccontrol:mlp_control_extended", "C; D", "C; E",
    "C; E7A:static_schedule_matched; E7A:memory_ablated_temporal; E", "F:frozen; F:adaptive; F:replay_control", "F (any arm); A; C; D; E",
    "Aref:xgboost_historical_control; any row", "B (missing)", "all cells, window table",
]
CODE_FILES = ("src/evidence/__init__.py", "src/evidence/sources.py", "src/evidence/table.py", "src/evidence/build.py", "src/evidence/outputs.py",
              "src/models/metrics.py", "scripts/run_phase12_table.py")


def _fmt(a, digits=4):
    if a["n"] == 0 or a["mean"] is None:
        return "n/a"
    if a["n"] == 1 or a["sd"] is None:
        return f"{a['mean']:.{digits}f}"
    return f"{a['mean']:.{digits}f} ± {a['sd']:.{digits}f}"


def render_markdown(families, comparability, cross_summary, recompute_summary):
    L = ["# Phase 12 - consolidated validation-evidence table", "", f"**{T.STANDING}**", "",
         f"**{T.F_BANNER}**", "",
         "All values are recomputed from stored validation predictions with one metric implementation (`src/models/metrics.py`, unmodified). "
         "Cells with several seeds show mean ± sample SD (ddof = 1) over seeds; one-run cells show the value. Operating-point metrics "
         "(precision, recall, F1, alerts) are in the CSV with their origin and are not compared across origins.", "",
         f"Recomputation against the recorded `metrics_val.json` files: {recompute_summary}. Cross-check files: {cross_summary}.", "",
         "| Cell | Model / arm | Basis | Runs | PR-AUC | ROC-AUC | PR-AUC warm | PR-AUC cold | PR-AUC no-edge | P@1% | R@1% |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    for f in families:
        a = f["agg"]
        L.append(f"| {f['cell']} | {f['model']} | {f['basis_short']} | {f['n_runs']} | {_fmt(a['pr_auc'])} | {_fmt(a['roc_auc'])} | {_fmt(a['pr_auc_has_edge_warm'])} | "
                 f"{_fmt(a['pr_auc_has_edge_cold'])} | {_fmt(a['pr_auc_no_pseudo_account_edge'])} | {_fmt(a['precision_at_1pct'])} | {_fmt(a['recall_at_1pct'])} |")
    L += [f"| B | tabular + graph features | - | 0 | **MISSING** | | | | | | |", "",
          f"**B is MISSING.** {T.B_ROW['note']}.", "", "## Comparability (labels fixed in the scope record, section 5)", "", "| Comparison | Label | Reason |", "|---|---|---|"]
    for c, lab, why in comparability:
        L.append(f"| {c} | {lab} | {why} |")
    L += ["", "## Notes", "",
          "* \"Evaluation-comparable\" means the same 83,655 VAL rows and labels scored by the same metric code. It never implies that a difference is attributable to the factor named.",
          "* Excluded and never read: `experiments/gat_300epoch/`, `experiments/graphsage/`, `experiments/graphsage_control_mlp/`, the temporal pilot, every checkpoint and feature file.",
          "* The Phase 8 replay-control arm is named F-replay to avoid a clash with cell E.",
          "* No held-out comparison, significance test, interval or causal statement is made here.", ""]
    return "\n".join(L)


def build(root, out_dir):
    root = pathlib.Path(root)
    out_dir = pathlib.Path(out_dir)
    reader = S.GuardedReader(root)
    scope_text = reader.read_text(S.SCOPE_MD)
    sidecar = reader.read_text(S.SCOPE_SIDECAR).strip()
    scope_hash = reader.sha256(S.SCOPE_MD)
    if scope_hash != sidecar:
        raise T.BuildBlocked(f"scope record hash {scope_hash} differs from its sidecar {sidecar}")
    expected_fp = S.parse_scope_fingerprint(scope_text)
    hashes = reader.hash_all()
    fp_start = S.fingerprint(hashes)
    if fp_start != expected_fp or len(hashes) != 165:
        raise T.BuildBlocked(f"source-set fingerprint {fp_start} ({len(hashes)} files) differs from the scope record {expected_fp}")
    groups = S.declared_groups()
    if {k: len(v) for k, v in groups.items()} != S.EXPECTED_COUNTS:
        raise T.BuildBlocked("declared group counts differ from the scope record")
    comp = S.parse_comparability(scope_text)
    if len(comp) != len(APPLIES_TO):
        raise T.BuildBlocked(f"scope record section 5 has {len(comp)} rows, expected {len(APPLIES_TO)}")

    ref = T.make_reference(reader)
    results = [T.process_non_f(reader, ref, sp, hashes) for sp in T.non_f_specs()]
    f_results = [T.process_f(reader, ref, arm, seed, hashes) for arm in S.ARMS_F for seed in S.SEEDS]

    checks = [r["check"] for r in results] + [c for fr in f_results for c in fr["checks"]]
    failing = [c for c in checks if c["mismatches"]]
    if failing:
        raise T.BuildBlocked("recorded metrics outside the registered tolerance: " + "; ".join(
            f"{c['run_id']}: {m}" for c in failing[:3] for m in c["mismatches"][:3]))
    xc = T.cross_checks(reader, results, f_results)
    xc_bad = [c for c in xc if c["abs_diff"] > T.TOL]

    # ---- assemble
    rows = [r["row"] for r in results] + [row for fr in f_results for row in fr["rows"]]
    runs_df = pd.DataFrame(rows)
    fam_defs = []
    seen = {}
    for r in results:
        sp = r["spec"]
        seen.setdefault(sp["family"], {"cell": sp["cell"], "model": sp["model"], "role": sp["role"], "rows": [], "basis_short": "pooled VAL"})["rows"].append(r["row"])
    for k, v in seen.items():
        fam_defs.append({"family_id": k, **v})
    for arm in S.ARMS_F:
        for basis, short in (("basis_mean_of_windows", "F: mean of 5 window PR-AUCs"), ("basis_concatenated", "F: 5 windows concatenated")):
            rr = [row for fr in f_results if fr["arm"] == arm for row in fr["rows"] if row["run_id"].endswith(basis)]
            fam_defs.append({"family_id": f"F:{arm}:{basis}", "cell": "F", "model": f"F-{arm}", "role": "F (distinct basis)", "rows": rr, "basis_short": short})
    families, cell_rows = [], []
    for f in fam_defs:
        agg = T.aggregate_family(f["rows"])
        f["agg"], f["n_runs"] = agg, len(f["rows"])
        families.append(f)
        cr = {"family_id": f["family_id"], "cell": f["cell"], "role": f["role"], "model": f["model"], "basis": f["rows"][0]["metric_basis"], "n_runs": f["n_runs"], "status": "recomputed"}
        for k in T.METRIC_KEYS:
            for stat in ("mean", "sd", "min", "max"):
                cr[f"{k}_{stat}"] = agg[k][stat]
        cell_rows.append(cr)
    cell_rows.append({"family_id": "B", "cell": "B", "role": "A-F cell", "model": "tabular + graph features", "basis": "", "n_runs": 0, "status": "MISSING", "note": T.B_ROW["note"]})
    win_rows = []
    for r in results:
        sp = r["spec"]
        for w, lo, hi in T.WINDOWS:
            win_rows.append({"run_id": sp["id"], "cell": sp["cell"], "model": sp["model"], "seed": r["row"]["seed"], "window": w, "days": f"{lo}-{hi}", **r["eval"]["windows"][w]})
    for fr in f_results:
        for w, lo, hi in T.WINDOWS:
            win_rows.append({"run_id": f"F:{fr['arm']}:seed{fr['seed']}", "cell": "F", "model": f"F-{fr['arm']}", "seed": fr["seed"], "window": w, "days": f"{lo}-{hi}", **fr["windows"][w]})
    matrix = [{"comparison": c, "label": lab, "reason": why, "applies_to": APPLIES_TO[i]} for i, (c, lab, why) in enumerate(comp)]

    max_diff = max(c["max_abs_diff"] for c in checks)
    recompute = {"tolerance": T.TOL, "n_runs_checked": len(checks), "n_values_compared": int(sum(c["n_compared"] for c in checks)), "max_abs_diff": max_diff,
                 "all_within_tolerance": True, "checks": checks,
                 "note": "recorded keys not produced by the metric function (e.g. threshold_selection, which records how a threshold was chosen) are listed under skipped_keys and are not recomputed"}
    cross = {"tolerance": T.TOL, "n_items": len(xc), "n_discrepancies": len(xc_bad), "max_abs_diff": max(c["abs_diff"] for c in xc), "items": xc,
             "note": "recorded comparison files are compared against recomputed values and are never inputs to the table"}
    pop = {"reference": T.REF_REL, "n_reference_rows": int(len(ref)), "all_identical": True,
           "non_f_runs": [{"run_id": r["spec"]["id"], "n_rows": r["check"]["n_rows"], "problems": []} for r in results],
           "f_arm_seeds": [{"arm": fr["arm"], "seed": fr["seed"], "n_rows_union_of_windows": fr["eval_concat"]["pooled"]["n_rows"],
                            "window_rows": [fr["windows"][w]["n_rows"] for w in range(1, 6)], "problems": []} for fr in f_results]}
    recompute_summary = f"{len(checks)} runs/windows checked, max absolute difference {max_diff:.3g} (tolerance {T.TOL:g})"
    cross_summary = f"{len(xc)} items compared, {len(xc_bad)} beyond tolerance"
    md = render_markdown(families, [(c["comparison"], c["label"], c["reason"]) for c in matrix], cross_summary, recompute_summary)
    table_json = {"standing_statement": T.STANDING, "f_banner": T.F_BANNER, "b_row": T.B_ROW, "runs": [{k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()} for r in rows],
                  "cells": cell_rows, "windows": win_rows, "comparability": matrix}

    # ---- end-of-build fingerprint and I/O audit
    fp_end = S.fingerprint(reader.hash_all())
    aud = reader.audit()
    allowed = set(reader.allowed) | set(reader.extra)
    audit = {**aud, "declared_files": len(reader.allowed), "extra_allowed": sorted(reader.extra), "fingerprint_start": fp_start, "fingerprint_end": fp_end,
             "fingerprint_expected": expected_fp, "all_opened_are_declared": set(aud["opened"]) <= allowed, "all_declared_were_opened": set(reader.allowed) <= set(aud["opened"]),
             "excluded_fragments_opened": [p for p in aud["opened"] if any(f in p.lower() for f in S.EXCLUDED_FRAGMENTS)],
             "test_or_checkpoint_paths_opened": [p for p in aud["opened"] if "test" in p.lower() or p.endswith((".pt", ".joblib"))],
             "code_files_hashed_not_part_of_source_set": list(CODE_FILES)}
    if fp_end != expected_fp or not audit["all_opened_are_declared"] or not audit["all_declared_were_opened"] or audit["excluded_fragments_opened"] or audit["test_or_checkpoint_paths_opened"]:
        raise T.BuildBlocked("I/O audit failed (fingerprint changed or an unlisted/refused path was opened)")

    # ---- write
    hs = {}
    hs["evidence_runs.csv"] = O.write_csv(out_dir / "evidence_runs.csv", runs_df)
    hs["evidence_cells.csv"] = O.write_csv(out_dir / "evidence_cells.csv", pd.DataFrame(cell_rows))
    hs["evidence_windows.csv"] = O.write_csv(out_dir / "evidence_windows.csv", pd.DataFrame(win_rows))
    hs["comparability_matrix.csv"] = O.write_csv(out_dir / "comparability_matrix.csv", pd.DataFrame(matrix))
    hs["evidence_table.json"] = O.write_json(out_dir / "evidence_table.json", table_json)
    hs["table_report.md"] = O.write_text(out_dir / "table_report.md", md)
    hs["recomputation_check.json"] = O.write_json(out_dir / "recomputation_check.json", recompute)
    hs["crosscheck_recorded.json"] = O.write_json(out_dir / "crosscheck_recorded.json", cross)
    hs["population_check.json"] = O.write_json(out_dir / "population_check.json", pop)
    hs["io_audit.json"] = O.write_json(out_dir / "io_audit.json", audit)
    hs["source_manifest.tsv"] = O.write_text(out_dir / "source_manifest.tsv", S.fingerprint_text(hashes))
    code = {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in CODE_FILES if (root / f).exists()}
    manifest = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                "scope_record_sha256": scope_hash, "source_set_fingerprint": fp_start, "n_source_files": len(hashes),
                "versions": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit_learn": sklearn.__version__},
                "code_sha256": code, "outputs_sha256": hs, "metric_implementation": "src/models/metrics.py (unmodified, imported)",
                "boundaries": {"training": False, "scoring_new_models": False, "new_predictions": False, "test_contents_accessed": False,
                               "checkpoints_opened": False, "phase10_empirical_tiers": False, "ablation_started": False, "b_status": "MISSING"}}
    hs["manifest_phase12.json"] = O.write_json(out_dir / "manifest_phase12.json", manifest)
    return {"fingerprint": fp_start, "n_runs": len(rows), "max_abs_diff": max_diff, "cross_discrepancies": len(xc_bad), "n_cross_items": len(xc), "outputs": hs,
            "cross_bad": xc_bad}
