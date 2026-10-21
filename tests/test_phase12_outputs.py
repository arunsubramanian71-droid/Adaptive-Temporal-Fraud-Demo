"""Phase 12: verification of the generated evidence table against the registered acceptance criteria (AC1-AC10) and of the build's gates.

Reads the Phase 12 outputs and (through the guarded reader) the declared source files only. Nothing is trained, scored or predicted."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import tempfile

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.lint import lint_files, lint_text  # noqa: E402
from src.evidence import build as B  # noqa: E402
from src.evidence import sources as S  # noqa: E402
from src.evidence import table as T  # noqa: E402

P12 = ROOT / "experiments" / "phase12_validation_evidence_table"
OUT = P12 / "outputs"
SCOPE = P12 / "SCOPE_AND_ACCEPTANCE.md"


def _j(name):
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def _h(p):
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def test_scope_record_sidecar_matches_and_precedes_every_output():
    assert _h(SCOPE) == (P12 / "SCOPE_AND_ACCEPTANCE.sha256").read_text().strip()
    assert SCOPE.stat().st_mtime <= min(p.stat().st_mtime for p in OUT.iterdir())


def test_AC1_only_the_declared_source_set_was_read_and_the_fingerprint_is_stable():
    a = _j("io_audit.json")
    scope_fp = S.parse_scope_fingerprint(SCOPE.read_text(encoding="utf-8"))
    assert a["fingerprint_start"] == a["fingerprint_end"] == a["fingerprint_expected"] == scope_fp
    assert a["all_opened_are_declared"] and a["all_declared_were_opened"] and a["declared_files"] == 165
    assert a["excluded_fragments_opened"] == [] and a["test_or_checkpoint_paths_opened"] == []
    assert set(a["opened"]) == set(S.declared_paths()) | set(a["extra_allowed"])
    lines = (OUT / "source_manifest.tsv").read_text(encoding="utf-8")
    assert hashlib.sha256(lines.encode("utf-8")).hexdigest() == scope_fp and lines.count("\n") == 165
    reader = S.GuardedReader(ROOT)
    for ln in lines.splitlines():
        path, h = ln.split("\t")
        assert reader.sha256(path) == h, path
    m = _j("manifest_phase12.json")
    assert m["source_set_fingerprint"] == scope_fp and m["n_source_files"] == 165 and m["scope_record_sha256"] == _h(SCOPE)
    assert not any(m["boundaries"][k] for k in m["boundaries"] if k != "b_status") and m["boundaries"]["b_status"] == "MISSING"


def test_AC2_single_metric_implementation_and_no_threshold_tuning():
    m = _j("manifest_phase12.json")
    assert m["metric_implementation"].startswith("src/models/metrics.py") and m["code_sha256"]["src/models/metrics.py"] == _h(ROOT / "src" / "models" / "metrics.py")
    runs = pd.read_csv(OUT / "evidence_runs.csv", float_precision="round_trip")      # exact float64 round trip of the written values
    reader = S.GuardedReader(ROOT)
    for sp in T.non_f_specs():
        rec = reader.read_json(sp["metrics"])
        rec = rec["pooled"] if "pooled" in rec else rec
        assert float(runs.loc[runs["run_id"] == sp["id"], "threshold"].iloc[0]) == float(rec["threshold"]), sp["id"]
    for arm in S.ARMS_F:
        for s in S.SEEDS:
            thr = reader.read_json(f"{S.E}phase8_adaptive_drift/shared_init/hparams_seed{s}.json")["threshold"]
            sub = runs[runs["run_id"] == f"F:{arm}:seed{s}:basis_concatenated"]
            assert float(sub["threshold"].iloc[0]) == float(thr)


def test_AC3_AC8_recomputed_values_match_recorded_values_and_cross_check_files():
    r = _j("recomputation_check.json")
    assert r["tolerance"] == 1e-9 and r["all_within_tolerance"] and r["max_abs_diff"] <= 1e-9 and r["n_values_compared"] >= 8000 and r["n_runs_checked"] == 65
    assert all(c["mismatches"] == [] for c in r["checks"])
    assert {k for c in r["checks"] for k in c["skipped_keys"]} == {"/pooled/threshold_selection"}
    x = _j("crosscheck_recorded.json")
    assert x["n_items"] == 173 and x["n_discrepancies"] == 0 and x["max_abs_diff"] <= 1e-9
    assert all(i["abs_diff"] <= 1e-9 for i in x["items"])
    assert {i["source"].split(" ")[0] for i in x["items"]} == {"temporal_graphsage/comparison.json", "temporal_ablation/comparison.json", "phase8_adaptive_drift/comparison.json"}


def test_AC4_population_identity():
    p = _j("population_check.json")
    assert p["all_identical"] and p["n_reference_rows"] == 83655 and p["reference"] == T.REF_REL
    assert len(p["non_f_runs"]) == 20 and all(r["n_rows"] == 83655 and r["problems"] == [] for r in p["non_f_runs"])
    assert len(p["f_arm_seeds"]) == 9
    for f in p["f_arm_seeds"]:
        assert f["n_rows_union_of_windows"] == 83655 and sum(f["window_rows"]) == 83655 and f["problems"] == [] and len(f["window_rows"]) == 5


def test_AC6_rows_carry_the_required_fields_both_F_bases_and_the_B_missing_row():
    runs = pd.read_csv(OUT / "evidence_runs.csv")
    required = {"run_id", "cell", "role", "model", "seed", "metric_basis", "fit_window", "data_window", "operating_point_origin", "includes_val_tuned_selection",
                "val_label_use", "source_paths", "source_sha256", "pr_auc", "roc_auc", "recomputation_max_abs_diff"}
    assert required <= set(runs.columns) and len(runs) == 38
    assert set(runs["cell"]) == {"A", "A-reference", "C", "C-control", "D", "E", "E-control", "F"}
    f = runs[runs["cell"] == "F"]
    assert len(f) == 18 and f["metric_basis"].str.contains("mean of the five window").sum() == 9 and f["metric_basis"].str.contains("concatenated").sum() == 9
    assert set(f["seed"]) == set(S.SEEDS) and (f["fit_window"] == "TRAIN 1-111 (shared checkpoint)").all()
    manifest = {ln.split("\t")[0]: ln.split("\t")[1] for ln in (OUT / "source_manifest.tsv").read_text(encoding="utf-8").splitlines()}
    for _, r in runs.iterrows():
        for pair in r["source_sha256"].split(";"):
            path, h = pair.split("=")
            assert manifest[path] == h
    assert runs[runs["cell"].isin(["A", "C", "D", "E"])]["seed"].notna().all()
    cells = pd.read_csv(OUT / "evidence_cells.csv")
    b = cells[cells["cell"] == "B"]
    assert len(b) == 1 and b["status"].iloc[0] == "MISSING" and b["n_runs"].iloc[0] == 0 and b["pr_auc_mean"].isna().all()
    assert b["model"].iloc[0] == "tabular + graph features"
    hist = cells[cells["model"].str.contains("historical", case=False)]
    assert set(hist["cell"]) == {"A-reference"}                       # the historical control is never presented as B
    md = (OUT / "table_report.md").read_text(encoding="utf-8")
    assert T.STANDING in md and T.F_BANNER in md and "**MISSING**" in md and "not substituted" in md
    texts = []
    for p in OUT.iterdir():
        if p.suffix in (".csv", ".json", ".md", ".tsv"):
            t = p.read_text(encoding="utf-8")
            if p.name == "table_report.md":      # the report names the excluded folders once, in its 'never read' disclosure line
                disclosure = [ln for ln in t.splitlines() if ln.startswith("* Excluded and never read")]
                assert len(disclosure) == 1
                t = t.replace(disclosure[0], "")
            texts.append(t)
    allp = "".join(texts)
    for frag in ("gat_300epoch", "experiments/graphsage/", "graphsage_control_mlp", "/pilot/", ".joblib", "model_checkpoint"):
        assert frag not in allp, frag


def test_AC6_cell_aggregates_use_sample_sd_and_report_the_recorded_headline_values():
    cells = pd.read_csv(OUT / "evidence_cells.csv").set_index("family_id")
    assert round(cells.loc["C:graphsage_robust", "pr_auc_mean"], 4) == 0.4637 and round(cells.loc["C:graphsage_robust", "pr_auc_sd"], 4) == 0.0167
    assert round(cells.loc["A:xgboost", "pr_auc_mean"], 4) == 0.5395 and pd.isna(cells.loc["A:xgboost", "pr_auc_sd"])
    assert round(cells.loc["E:temporal_graphsage", "pr_auc_mean"], 4) == 0.4927 and round(cells.loc["D:gat", "pr_auc_mean"], 4) == 0.3320
    for arm, v in (("frozen", 0.4243), ("adaptive", 0.4580), ("replay_control", 0.4179)):
        assert round(cells.loc[f"F:{arm}:basis_mean_of_windows", "pr_auc_mean"], 4) == v           # reproduces the Phase 8 report basis
        assert cells.loc[f"F:{arm}:basis_concatenated", "pr_auc_mean"] != cells.loc[f"F:{arm}:basis_mean_of_windows", "pr_auc_mean"]


def test_AC7_comparability_matrix_equals_the_scope_record():
    rows = S.parse_comparability(SCOPE.read_text(encoding="utf-8"))
    mx = pd.read_csv(OUT / "comparability_matrix.csv")
    assert len(mx) == len(rows) == 10
    for (c, lab, why), (_, r) in zip(rows, mx.iterrows()):
        assert (r["comparison"], r["label"], r["reason"]) == (c, lab, why)
    labels = set(mx["label"])
    assert {"Not comparable", "Reference line only", "Not available"} <= labels
    tj = _j("evidence_table.json")["comparability"]
    assert len(tj) == 10 and [(r["comparison"], r["label"], r["reason"]) for r in tj] == rows


def test_AC9_wording_lint_on_all_phase12_text_outputs_and_sources():
    files = [p for p in P12.rglob("*") if p.is_file() and p.suffix in (".md", ".json", ".csv", ".tsv")]
    files += sorted((ROOT / "src" / "evidence").glob("*.py")) + sorted(ROOT.glob("scripts/*phase12*.py")) + sorted(ROOT.glob("tests/*phase12*.py"))
    bad = {str(k): v for k, v in lint_files(files).items() if v}
    assert bad == {}, bad
    assert lint_text((OUT / "table_report.md").read_text(encoding="utf-8")) == []


def test_AC10_sidecars_and_deterministic_rebuild():
    for p in OUT.iterdir():
        if p.suffix != ".sha256":
            assert _h(p) == p.with_suffix(".sha256").read_text().strip(), p.name
    assert len({p.stem.lower() for p in OUT.iterdir() if p.suffix != ".sha256"}) == len([p for p in OUT.iterdir() if p.suffix != ".sha256"])
    with tempfile.TemporaryDirectory() as td:
        B.build(ROOT, td)
        for p in OUT.iterdir():
            if p.suffix == ".sha256" or p.name == "manifest_phase12.json":
                continue
            assert _h(p) == _h(pathlib.Path(td) / p.name), p.name
        a, b = _j("manifest_phase12.json"), json.loads((pathlib.Path(td) / "manifest_phase12.json").read_text(encoding="utf-8"))
        a.pop("created_local"), b.pop("created_local")
        assert a == b


def _patched(attr_owner, name, replacement, fn):
    orig = getattr(attr_owner, name)
    setattr(attr_owner, name, replacement(orig))
    try:
        return fn()
    finally:
        setattr(attr_owner, name, orig)


def _expect_blocked(replacement_owner, name, replacement, needle):
    with tempfile.TemporaryDirectory() as td:
        def run():
            try:
                B.build(ROOT, td)
            except (T.BuildBlocked, S.SourceError) as e:
                return str(e)
            return None
        msg = _patched(replacement_owner, name, replacement, run)
        assert msg is not None and needle in msg, msg
        assert list(pathlib.Path(td).iterdir()) == [], "outputs were written despite a failed gate"


def test_build_gates_block_before_any_output_is_written():
    # the EARLY fingerprint gate must fire (its own message) and the build must stop before any run is processed
    reached = []
    orig_ref = T.make_reference
    T.make_reference = lambda reader: (reached.append(1), orig_ref(reader))[1]
    try:
        _expect_blocked(S, "parse_scope_fingerprint", lambda orig: (lambda text: "0" * 64), "differs from the scope record")
    finally:
        T.make_reference = orig_ref
    assert reached == [], "the build went on to process runs after a fingerprint mismatch"
    _expect_blocked(S.GuardedReader, "read_text", lambda orig: (lambda self, rel: "0" * 64 if rel == S.SCOPE_SIDECAR else orig(self, rel)), "scope record hash")

    def inject(orig):
        state = {"n": 0}

        def f(a, b, tol=T.TOL):
            r = orig(a, b, tol)
            if state["n"] == 0:
                r["mismatches"].append({"key": "/injected", "recorded": 1, "recomputed": 2})
            state["n"] += 1
            return r
        return f
    _expect_blocked(T, "compare_recorded", inject, "registered tolerance")
    _expect_blocked(T, "population_problems", lambda orig: (lambda ref, frame, name: ["injected mismatch"] if name == "A:xgboost" else orig(ref, frame, name)), "population mismatch")


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"[PASS] {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed.append(fn.__name__)
            import traceback
            traceback.print_exc()
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return not failed


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
