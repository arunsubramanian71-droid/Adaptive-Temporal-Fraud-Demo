"""Phase 12: unit tests for src/evidence on synthetic data (no real predictions, no checkpoints, no TEST). Covers acceptance criteria
AC1-AC9 and AC11 of experiments/phase12_validation_evidence_table/SCOPE_AND_ACCEPTANCE.md."""
from __future__ import annotations

import copy
import hashlib
import math
import pathlib
import re
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.lint import lint_files, lint_text  # noqa: E402  (read-only use of the Phase 9 wording lint)
from src.evidence import build as B  # noqa: E402
from src.evidence import sources as S  # noqa: E402
from src.evidence import table as T  # noqa: E402
from src.models.metrics import compute_metrics  # noqa: E402

SCOPE = ROOT / "experiments" / "phase12_validation_evidence_table" / "SCOPE_AND_ACCEPTANCE.md"


# ------------------------------------------------------------------------------------------------ synthetic fixtures
def make_ref(n_per_day=8, seed=0):
    rng = np.random.default_rng(seed)
    days = np.repeat(np.arange(122, 152), n_per_day)
    n = len(days)
    seg = np.array([T.SEGMENTS[i % 3] for i in range(n)])
    y = (rng.random(n) < 0.2).astype(np.float32)
    y[:3] = 1
    return pd.DataFrame({"TransactionID": np.arange(5_000_000, 5_000_000 + n), "isFraud": y, "day": days.astype(np.int64), "segment": seg})


def make_scores(ref, seed=1, signal=1.0):
    rng = np.random.default_rng(seed)
    return (rng.random(len(ref)) * 0.6 + signal * 0.4 * ref["isFraud"].to_numpy()).astype(np.float32)


class FakeReader:
    def __init__(self, parquet=None, json=None):
        self.parquet, self.json, self.opened = parquet or {}, json or {}, []

    def read_parquet(self, rel, columns=None):
        self.opened.append(rel)
        df = self.parquet[rel].copy()
        return df[columns] if columns else df

    def read_json(self, rel):
        self.opened.append(rel)
        return copy.deepcopy(self.json[rel])


def recorded_for(frame, thr, strata=True):
    ev = T.evaluate_frame(frame, thr, windows=False)
    if not strata:
        return copy.deepcopy(ev["pooled"])
    return {k: copy.deepcopy(ev[k]) for k in ("pooled",) + T.SEGMENTS}


# ------------------------------------------------------------------------------------------------ source set, guard, fingerprint (AC1)
def test_declared_set_matches_the_scope_record_counts_and_exclusions():
    g = S.declared_groups()
    assert {k: len(v) for k, v in g.items()} == S.EXPECTED_COUNTS and len(S.declared_paths()) == 165
    for p in S.declared_paths():
        low = p.lower()
        assert "test" not in low and not low.endswith((".pt", ".joblib")) and not any(f in low for f in S.EXCLUDED_FRAGMENTS), p
    text = SCOPE.read_text(encoding="utf-8")
    assert "165 files" in text and S.parse_scope_fingerprint(text) == re.search(r"`([0-9a-f]{64})`", text).group(1)


def test_fingerprint_follows_the_registered_recipe():
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        files = {"b/file2.json": b"two", "a/file1.json": b"one"}
        for rel, data in files.items():
            (td / rel).parent.mkdir(parents=True, exist_ok=True)
            (td / rel).write_bytes(data)
        r = S.GuardedReader(td, allowed=set(files), extra=())
        hashes = r.hash_all()
        lines = "".join(f"{p}\t{hashlib.sha256(files[p]).hexdigest()}\n" for p in sorted(files))
        assert S.fingerprint_text(hashes) == lines and S.fingerprint(hashes) == hashlib.sha256(lines.encode("utf-8")).hexdigest()
        h2 = dict(hashes)
        h2["a/file1.json"] = "0" * 64
        assert S.fingerprint(h2) != S.fingerprint(hashes)


def test_guard_refuses_undeclared_test_checkpoint_and_excluded_paths_and_logs_opens():
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        ok = "experiments/xgboost/metrics_val.json"
        bad = ["experiments/other/metrics_val.json", "experiments/xgboost/test_features.parquet", "experiments/gat_300epoch/seed_42/metrics_val.json",
               "experiments/graphsage/metrics_val.json", "experiments/temporal_graphsage/pilot/x.json", "experiments/xgboost/model.joblib",
               "experiments/graphsage_control_mlp/metrics_val.json", "experiments/xgboost/model_checkpoint.pt"]
        for rel in [ok] + bad:
            (td / rel).parent.mkdir(parents=True, exist_ok=True)
            (td / rel).write_text("{}")
        refused = bad[1:]                                            # everything except the merely-undeclared first entry
        for allowed, to_try in ((set([ok]), bad), (set([ok] + refused), refused)):     # undeclared, and declared-but-refused
            r = S.GuardedReader(td, allowed=allowed, extra=())
            r.read_json(ok)
            for rel in to_try:
                try:
                    r.read_json(rel) if rel.endswith(".json") else r.sha256(rel)
                except S.SourceError:
                    pass
                else:
                    raise AssertionError(f"read allowed: {rel}")
            assert r.audit()["opened"] == [ok]


def test_scope_parsing_reads_labels_from_the_record():
    text = SCOPE.read_text(encoding="utf-8")
    rows = S.parse_comparability(text)
    assert len(rows) == len(B.APPLIES_TO) == 10
    labels = [r[1] for r in rows]
    assert labels[2] == "Not comparable" and labels[4] == "Comparable" and labels[8] == "Not available" and labels[7] == "Reference line only"
    assert all(r[0] and r[2] for r in rows)


# ------------------------------------------------------------------------------------------------ comparison with recorded values (AC3)
def test_compare_recorded_tolerance_ints_nan_and_skipped_keys():
    a = {"pr_auc": 0.5, "n": 10, "x": float("nan"), "s": "a", "nested": {"v": 1.0}, "threshold_selection": {"f1": 1}}
    b = {"pr_auc": 0.5 + 5e-10, "n": 10, "x": float("nan"), "s": "a", "nested": {"v": 1.0}}
    r = T.compare_recorded(a, b)
    assert r["mismatches"] == [] and r["skipped_keys"] == ["/threshold_selection"] and 0 < r["max_abs_diff"] < 1e-9
    assert T.compare_recorded(a, {**b, "pr_auc": 0.5 + 2e-9})["mismatches"][0]["key"] == "/pr_auc"
    assert T.compare_recorded(a, {**b, "n": 11})["mismatches"][0]["key"] == "/n"
    assert T.compare_recorded(a, {**b, "x": 0.1})["mismatches"][0]["key"] == "/x"
    assert T.compare_recorded(a, {**b, "s": "b"})["mismatches"][0]["key"] == "/s"
    assert T.compare_recorded(a, {**b, "nested": {"v": 1.001}})["mismatches"][0]["key"] == "/nested/v"


def test_evaluate_frame_equals_the_metric_module_and_respects_stored_row_order():
    ref = make_ref()
    frame = ref.copy()
    frame["y_score"] = make_scores(ref)
    ev = T.evaluate_frame(frame, 0.5)
    m = compute_metrics(frame["isFraud"].to_numpy(), frame["y_score"].to_numpy(), 0.5)
    assert ev["pooled"] == m and abs(m["pr_auc"] - average_precision_score(frame["isFraud"], frame["y_score"])) < 1e-15
    w = frame["day"].between(128, 133)
    assert ev["windows"][2]["pr_auc"] == compute_metrics(frame["isFraud"][w].to_numpy(), frame["y_score"][w].to_numpy(), 0.5)["pr_auc"]
    # ties: top-K uses a stable sort, so the stored row order matters and must not be altered by the evaluation
    tied = frame.copy()
    tied["y_score"] = np.float32(0.5)
    rev = tied.iloc[::-1].reset_index(drop=True)
    assert T.evaluate_frame(tied, 0.5, windows=False)["pooled"]["top_k"] == compute_metrics(tied["isFraud"].to_numpy(), tied["y_score"].to_numpy(), 0.5)["top_k"]
    assert T.evaluate_frame(rev, 0.5, windows=False)["pooled"]["top_k"] == compute_metrics(rev["isFraud"].to_numpy(), rev["y_score"].to_numpy(), 0.5)["top_k"]


# ------------------------------------------------------------------------------------------------ population identity (AC4)
def test_population_problems_detect_each_defect():
    ref = make_ref()
    base = ref[["TransactionID", "isFraud"]].copy()
    base["y_score"] = 0.5
    assert T.population_problems(ref, base, "ok") == []
    full = ref.copy()
    full["y_score"] = 0.5
    assert T.population_problems(ref, full, "ok") == []
    assert any("rows" in p for p in T.population_problems(ref, base.iloc[:-1], "short"))
    dup = pd.concat([base, base.iloc[:1]], ignore_index=True)
    assert any("duplicated" in p for p in T.population_problems(ref, dup, "dup"))
    other = base.copy()
    other.loc[0, "TransactionID"] = 9
    assert any("id sets differ" in p for p in T.population_problems(ref, other, "ids"))
    lab = base.copy()
    lab.loc[5, "isFraud"] = 1 - lab.loc[5, "isFraud"]
    assert any("isFraud differs" in p for p in T.population_problems(ref, lab, "label"))
    dd = full.copy()
    dd.loc[7, "day"] += 1
    assert any("day differs" in p for p in T.population_problems(ref, dd, "day"))
    sg = full.copy()
    sg.loc[9, "segment"] = "has_edge_cold" if sg.loc[9, "segment"] != "has_edge_cold" else "has_edge_warm"
    assert any("segment differs" in p for p in T.population_problems(ref, sg, "seg"))
    shuffled = full.sample(frac=1, random_state=3).reset_index(drop=True)
    assert T.population_problems(ref, shuffled, "order") == []        # row order is not a population difference


def test_attach_missing_adds_day_and_segment_by_id_and_keeps_row_order():
    ref = make_ref()
    frame = ref[["TransactionID", "isFraud"]].sample(frac=1, random_state=4).reset_index(drop=True)
    frame["y_score"] = 0.3
    out = T.attach_missing(frame, ref)
    assert out["TransactionID"].tolist() == frame["TransactionID"].tolist()
    chk = out.merge(ref, on="TransactionID", suffixes=("", "_r"))
    assert (chk["day"] == chk["day_r"]).all() and (chk["segment"] == chk["segment_r"]).all()
    only_day = frame.assign(segment=out["segment"])
    assert "segment" in T.attach_missing(only_day, ref).columns


def test_window_definition_matches_phase_8():
    assert [T.window_of_day(d) for d in (122, 127, 128, 133, 134, 139, 140, 145, 146, 151)] == [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]
    assert T.window_of_day(121) is None and T.window_of_day(152) is None


# ------------------------------------------------------------------------------------------------ run processing with a fake reader
def _non_f_setup(tamper=None, drop_label=False):
    ref = make_ref()
    frame = ref[["TransactionID", "isFraud", "day", "segment"]].copy()
    frame["y_score"] = make_scores(ref)
    frame["y_pred"] = (frame["y_score"] >= 0.5).astype(int)
    rec = recorded_for(frame, 0.5)
    if tamper:
        tamper(rec)
    d = "experiments/fake_run"
    spec = dict(id="C:seed42", cell="C", role="A-F cell", family="C:graphsage_robust", model="m", seed=42, pred=f"{d}/predictions_val.parquet",
                metrics=f"{d}/metrics_val.json", config=f"{d}/config.json", files=[f"{d}/predictions_val.parquet", f"{d}/metrics_val.json", f"{d}/config.json"])
    reader = FakeReader({spec["pred"]: frame}, {spec["metrics"]: rec, spec["config"]: {"seed": 42, "parameter_count": 7}})
    return reader, ref, spec, {p: "0" * 64 for p in spec["files"]}


def test_process_non_f_reproduces_recorded_metrics_and_flags_a_tampered_record():
    reader, ref, spec, hashes = _non_f_setup()
    res = T.process_non_f(reader, ref, spec, hashes)
    assert res["check"]["mismatches"] == [] and res["check"]["max_abs_diff"] == 0.0 and res["check"]["n_compared"] > 30
    assert res["row"]["seed"] == 42 and res["row"]["parameter_count"] == 7 and res["row"]["pr_auc"] == res["eval"]["pooled"]["pr_auc"]
    for tamper in (lambda r: r["pooled"].update(pr_auc=r["pooled"]["pr_auc"] + 1e-6), lambda r: r["has_edge_cold"]["confusion_matrix"].update(tp=999),
                   lambda r: r["has_edge_warm"]["top_k"]["1%"].update(recall_at_k=0.0)):
        reader2, ref2, spec2, hashes2 = _non_f_setup(tamper)
        res = T.process_non_f(reader2, ref2, spec2, hashes2)
        assert res["check"]["mismatches"], "tampered record not detected"


def test_process_non_f_blocks_on_population_mismatch():
    reader, ref, spec, hashes = _non_f_setup()
    bad = reader.parquet[spec["pred"]].copy()
    bad.loc[3, "isFraud"] = 1 - bad.loc[3, "isFraud"]
    reader.parquet[spec["pred"]] = bad
    try:
        T.process_non_f(reader, ref, spec, hashes)
    except T.BuildBlocked as e:
        assert "population mismatch" in str(e)
    else:
        raise AssertionError("population mismatch not blocked")


def test_process_f_reports_both_bases_and_checks_the_window_partition():
    ref = make_ref()
    frame_all = ref.copy()
    frame_all["y_score"] = make_scores(ref, seed=7)
    thr = 0.55
    pq, js, hashes = {}, {"experiments/phase8_adaptive_drift/shared_init/hparams_seed42.json": {"threshold": thr}}, {}
    base = "experiments/phase8_adaptive_drift/frozen/seed_42"
    for w, lo, hi in T.WINDOWS:
        fw = frame_all[frame_all["day"].between(lo, hi)].copy()
        fw["window"] = w
        fw = fw[["TransactionID", "isFraud", "y_score", "day", "window", "segment"]]
        pq[f"{base}/predictions_val_W{w}.parquet"] = fw
        js[f"{base}/metrics_val_W{w}.json"] = recorded_for(fw, thr)
    for p in list(pq) + list(js):
        hashes[p] = "0" * 64
    res = T.process_f(FakeReader(pq, js), ref, "frozen", 42, hashes)
    mean_row, cat_row = res["rows"]
    wins = [T.evaluate_frame(pq[f"{base}/predictions_val_W{w}.parquet"], thr, windows=False)["pooled"]["pr_auc"] for w, _, _ in T.WINDOWS]
    assert abs(mean_row["pr_auc"] - float(np.mean(wins))) < 1e-15 and "mean of the five window" in mean_row["metric_basis"]
    assert cat_row["n_rows"] == len(ref) and abs(cat_row["pr_auc"] - compute_metrics(
        pd.concat([pq[f"{base}/predictions_val_W{w}.parquet"] for w, _, _ in T.WINDOWS])["isFraud"].to_numpy(),
        pd.concat([pq[f"{base}/predictions_val_W{w}.parquet"] for w, _, _ in T.WINDOWS])["y_score"].to_numpy(), thr)["pr_auc"]) < 1e-15
    assert mean_row["pr_auc"] != cat_row["pr_auc"] and all(c["mismatches"] == [] for c in res["checks"])
    assert mean_row["operating_point_origin"].startswith("TRAIN days 112-121") and "TRAIN 1-111" in mean_row["fit_window"]
    # a window file with a wrong window label, and a dropped window, are both blocked
    pq2 = dict(pq)
    wrong = pq2[f"{base}/predictions_val_W2.parquet"].copy()
    wrong["window"] = 3
    pq2[f"{base}/predictions_val_W2.parquet"] = wrong
    try:
        T.process_f(FakeReader(pq2, js), ref, "frozen", 42, hashes)
    except T.BuildBlocked as e:
        assert "window column inconsistent" in str(e)
    else:
        raise AssertionError("window inconsistency not blocked")
    pq3 = dict(pq)
    pq3[f"{base}/predictions_val_W5.parquet"] = pq3[f"{base}/predictions_val_W5.parquet"].iloc[:-3]
    js3 = dict(js)
    js3[f"{base}/metrics_val_W5.json"] = recorded_for(pq3[f"{base}/predictions_val_W5.parquet"], thr)
    try:
        T.process_f(FakeReader(pq3, js3), ref, "frozen", 42, hashes)
    except T.BuildBlocked as e:
        assert "population mismatch" in str(e)
    else:
        raise AssertionError("missing rows not blocked")


# ------------------------------------------------------------------------------------------------ aggregation, rendering, static checks
def test_aggregate_uses_sample_sd_and_single_runs_have_no_sd():
    a = T.aggregate([0.45147076224113775, 0.45686847659519864, 0.48272217635730197])
    assert abs(a["mean"] - 0.4636871383978794) < 1e-12 and abs(a["sd"] - 0.016704290845453478) < 1e-12 and a["n"] == 3
    one = T.aggregate([0.5])
    assert one["mean"] == 0.5 and one["sd"] is None and T.aggregate([None, None])["n"] == 0
    assert T.aggregate([1.0, None, 3.0])["n"] == 2


def test_markdown_rendering_carries_banners_missing_b_row_and_passes_the_wording_lint():
    agg = {k: T.aggregate([0.5]) for k in T.METRIC_KEYS}
    fam = [{"cell": "A", "model": "m", "basis_short": "pooled VAL", "n_runs": 1, "agg": agg}]
    rows = [(c, l, r) for c, l, r in S.parse_comparability(SCOPE.read_text(encoding="utf-8"))]
    md = B.render_markdown(fam, rows, "x", "y")
    assert T.STANDING in md and T.F_BANNER in md and "**MISSING**" in md and "not substituted" in md
    assert lint_text(md) == [] and "attribution" in md


def test_sources_never_call_threshold_selection_or_read_outside_the_guard_or_touch_models():
    for f in sorted((ROOT / "src" / "evidence").glob("*.py")) + [ROOT / "scripts" / "run_phase12_table.py"]:
        src = f.read_text(encoding="utf-8")
        assert "select_best_f1_threshold" not in src, f
        scan = re.sub(r"endswith\(\([^)]*\)\)", "", src)          # the refusal expressions that block checkpoints are allowed to name them
        assert not re.search(r"joblib|torch|state_dict|\.pt\b|sklearn\.(linear|ensemble)|predict_proba|\.fit\(", scan), f
        if f.name != "sources.py":
            assert not re.search(r"read_table\(|read_parquet\(|open\(|read_text\(|read_bytes\(", src.replace("reader.read_parquet(", "").replace("reader.read_text(", "")
                                 .replace("reader.read_json(", "").replace("(root / f).read_bytes()", "")), f
    assert "compute_metrics" in (ROOT / "src" / "evidence" / "table.py").read_text(encoding="utf-8")


def test_wording_lint_on_all_phase12_sources_and_tests():
    files = sorted((ROOT / "src" / "evidence").glob("*.py")) + sorted(ROOT.glob("tests/*phase12*.py")) + sorted(ROOT.glob("scripts/*phase12*.py")) + [SCOPE]
    names = {f.name for f in files}
    assert {"sources.py", "table.py", "build.py", "outputs.py", "run_phase12_table.py", "test_phase12_evidence.py", "SCOPE_AND_ACCEPTANCE.md"} <= names, names
    bad = {str(k): v for k, v in lint_files(files).items() if v}
    assert bad == {}, bad


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
