"""Scoring bundle, command line and end-to-end demonstration: tests on the SYNTHETIC fixture (no real data, no TEST period)."""
from __future__ import annotations

import hashlib
import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.coordination.lint import lint_files  # noqa: E402
from src.demo.pipeline import run_demo  # noqa: E402
from src.demo.synthetic import FEATURE_COLUMNS, make_transactions  # noqa: E402
from src.inference import scorer as S  # noqa: E402

SCHEMA = S.Schema(features=tuple(FEATURE_COLUMNS))
_CACHE = {}


def data():
    if "df" not in _CACHE:
        df = make_transactions(seed=3, n_days=22, per_day=90, n_cards=200, n_devices=40, n_nets=15, n_planted_cards=8, burst_days=(14, 18))
        _CACHE["df"] = df
        _CACHE["tr"] = df[df["day"] <= 12].reset_index(drop=True)
        _CACHE["va"] = df[(df["day"] > 12) & (df["day"] <= 16)].reset_index(drop=True)
        _CACHE["ho"] = df[df["day"] > 16].reset_index(drop=True)
    return _CACHE["df"], _CACHE["tr"], _CACHE["va"], _CACHE["ho"]


def bundle(use_graph=True):
    key = ("b", use_graph)
    if key not in _CACHE:
        _, tr, va, _ = data()
        _CACHE[key] = S.train_bundle(tr, va, SCHEMA, use_graph=use_graph, params={"n_estimators": 60})
    return _CACHE[key]


def tmpdir():
    return pathlib.Path(tempfile.mkdtemp(prefix="inf_"))


# ------------------------------------------------------------------------------------------------ fixture
def test_synthetic_fixture_is_deterministic_and_chronological():
    kw = dict(n_days=8, per_day=40, burst_days=(3, 5))
    a, b = make_transactions(seed=5, **kw), make_transactions(seed=5, **kw)
    assert a.equals(b)
    assert not a.equals(make_transactions(seed=6, **kw))
    assert a["TransactionDT"].is_monotonic_increasing and a["TransactionID"].is_unique
    assert set(a["isFraud"].unique()) <= {0, 1} and a["is_planted"].any()


# ------------------------------------------------------------------------------------------------ scoring
def test_scores_are_probabilities_one_per_row_in_caller_order():
    df, tr, va, ho = data()
    hist = pd.concat([tr, va], ignore_index=True)
    shuffled = ho.sample(frac=1.0, random_state=1).reset_index(drop=True)
    out = S.score(bundle(), hist, shuffled)
    assert out["TransactionID"].tolist() == shuffled["TransactionID"].tolist()
    assert out["score"].between(0, 1).all() and len(out) == len(ho) and out["alert"].dtype == bool
    ref = S.score(bundle(), hist, ho).set_index("TransactionID")["score"]
    assert np.array_equal(out.set_index("TransactionID")["score"].loc[ref.index].to_numpy(), ref.to_numpy())


def test_scoring_ignores_labels_and_is_causal():
    df, tr, va, ho = data()
    hist = pd.concat([tr, va], ignore_index=True)
    base = S.score(bundle(), hist, ho)["score"].to_numpy()
    poisoned_h, poisoned_n = hist.copy(), ho.copy()
    poisoned_h["isFraud"] = 1 - poisoned_h["isFraud"]
    poisoned_n["isFraud"] = 1 - poisoned_n["isFraud"]
    assert np.array_equal(S.score(bundle(), poisoned_h, poisoned_n)["score"].to_numpy(), base)
    nohist = S.score(bundle(), hist.drop(columns=["isFraud"]), ho.drop(columns=["isFraud"]))["score"].to_numpy()
    assert np.array_equal(nohist, base)
    k = len(ho) // 2
    prefix = S.score(bundle(), hist, ho.iloc[:k].reset_index(drop=True))["score"].to_numpy()
    assert np.array_equal(prefix, base[:k]), "scores of early rows changed when later rows were removed"


def test_graph_features_change_scores_but_tabular_bundle_ignores_entities():
    df, tr, va, ho = data()
    hist = pd.concat([tr, va], ignore_index=True)
    a = S.score(bundle(False), hist, ho)["score"].to_numpy()
    scrambled = ho.copy()
    for c in SCHEMA.entities.values():
        scrambled[c] = scrambled[c].sample(frac=1.0, random_state=2).to_numpy()
    assert np.array_equal(S.score(bundle(False), hist, scrambled)["score"].to_numpy(), a), "tabular-only bundle depends on entity keys"
    assert not np.array_equal(S.score(bundle(True), hist, scrambled)["score"].to_numpy(), S.score(bundle(True), hist, ho)["score"].to_numpy())


def test_explanations_only_for_alerted_rows_and_names_are_model_columns():
    df, tr, va, ho = data()
    hist = pd.concat([tr, va], ignore_index=True)
    out = S.score(bundle(), hist, ho, explain=True, top_k=4)
    assert out.loc[~out["alert"], "top_features"].isna().all()
    if out["alert"].any():
        tf = json.loads(out.loc[out["alert"], "top_features"].iloc[0])
        assert len(tf) == 4 and {t["feature"] for t in tf} <= set(bundle().columns)


# ------------------------------------------------------------------------------------------------ errors
def _raises(fn, exc, text):
    try:
        fn()
    except exc as e:
        assert text in str(e), (text, str(e))
        return
    raise AssertionError(f"expected {exc.__name__} containing {text!r}")


def test_invalid_inputs_give_actionable_errors():
    df, tr, va, ho = data()
    hist = pd.concat([tr, va], ignore_index=True)
    b = bundle()
    _raises(lambda: S.score(b, hist, ho.drop(columns=["f3"])), S.ScoringError, "missing required column")
    dup = pd.concat([ho, ho.iloc[:1]], ignore_index=True)
    _raises(lambda: S.score(b, hist, dup), S.ScoringError, "duplicated")
    bad_t = ho.copy()
    bad_t.loc[0, "TransactionDT"] = -5
    _raises(lambda: S.score(b, hist, bad_t), S.ScoringError, "must lie in")
    _raises(lambda: S.score(b, ho, hist), S.ScoringError, "later than the earliest new")
    _raises(lambda: S.score(b, hist.iloc[:5], hist.iloc[3:10]), S.ScoringError, "share TransactionID")
    nonnum = ho.copy()
    nonnum["f1"] = "x"
    _raises(lambda: S.score(b, hist, nonnum), S.ScoringError, "must be numeric")
    _raises(lambda: S.train_bundle(va, tr, SCHEMA), S.ScoringError, "strictly after")
    one_class = tr.copy()
    one_class["isFraud"] = 0
    _raises(lambda: S.train_bundle(one_class, va, SCHEMA), S.ScoringError, "both classes")
    _raises(lambda: S.score(b, None, "not a frame"), S.ScoringError, "DataFrame")


def test_transaction_ids_must_be_integers_and_amounts_finite():
    """Found 2026-10-22 through the HTTP API: a text or boolean id caused an internal error, a fractional id (1.5) was reported back as 1, and a missing or
    infinite amount turned the backward-looking amount features of every later row of the same entity into NaN or inf."""
    b = bundle()
    _, _, _, ho = data()
    new = ho.head(5).reset_index(drop=True)
    assert S.integer_valued(pd.Series([1, 2, 3])) and S.integer_valued(pd.Series([1.0, 2.0])) and S.integer_valued([10 ** 20]) and S.integer_valued([10 ** 30])
    assert S.integer_valued(pd.Series([1, None], dtype="object"), allow_missing=True) and S.integer_valued(pd.Series([1.0, np.nan]), allow_missing=True)
    for bad in ([1.5], ["7"], [True], [None], [np.nan], [np.inf], [[1]]):
        assert not S.integer_valued(pd.Series(bad, dtype="object")), bad
    assert not S.integer_valued(pd.Series([True, False]))
    for bad_id in ("abc", 1.5, None, True):
        frame = new.copy()
        frame["TransactionID"] = frame["TransactionID"].astype(object)
        frame.loc[0, "TransactionID"] = bad_id
        try:
            S.score(b, None, frame)
        except S.ScoringError as e:
            assert "TransactionID must be integers" in str(e), e
        else:
            raise AssertionError(f"id {bad_id!r} accepted")
    for bad_amt in (np.nan, np.inf, -np.inf):
        frame = new.copy()
        frame.loc[2, "TransactionAmt"] = bad_amt
        try:
            S.score(b, None, frame)
        except S.ScoringError as e:
            assert "TransactionAmt must be finite" in str(e), e
        else:
            raise AssertionError(f"amount {bad_amt!r} accepted")
    as_float = new.copy()
    as_float["TransactionID"] = as_float["TransactionID"].astype(float)
    assert S.score(b, None, as_float)["TransactionID"].tolist() == new["TransactionID"].astype(float).tolist()      # 5.0 is the integer 5


def test_research_sample_refuses_a_non_positive_count_before_reading_anything():
    import subprocess
    r = subprocess.run([sys.executable, "-B", "scripts/make_research_sample.py", "--n", "-5", "--out", str(pathlib.Path(tempfile.gettempdir()) / "x.json")],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "--n must be at least 1" in r.stderr, (r.returncode, r.stderr[-300:])        # before 2026-10-22: [:-5] wrote ~83,000 rows
    r = subprocess.run([sys.executable, "-B", "scripts/make_research_sample.py", "--out", str(ROOT / "experiments" / "x.json")], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 2 and "outside the repository" in r.stderr and not (ROOT / "experiments" / "x.json").exists()


def test_sealed_path_guard_and_missing_file():
    for p in ("data/features/test_features.parquet", "data/raw/test_transaction.csv", "C:/x/test/holdout.csv", "scores_test.csv"):
        _raises(lambda p=p: S.refuse_sealed_path(p), S.SealedDataError, "sealed")
    for p in ("data/features/train_features.parquet", "reports/latest_scores.csv", "demo_output/holdout_scores.csv"):
        S.refuse_sealed_path(p)
    _raises(lambda: S.read_table("definitely_missing_file.csv"), S.ScoringError, "not found")
    _raises(lambda: S.read_table("notes.json"), S.ScoringError, "not found")


def test_bundle_round_trip_and_tamper_detection():
    df, tr, va, ho = data()
    hist = pd.concat([tr, va], ignore_index=True)
    d = tmpdir()
    try:
        m = S.save_bundle(bundle(), d / "b")
        loaded = S.load_bundle(d / "b")
        assert m["bundle_sha256"] == hashlib.sha256((d / "b" / "bundle.joblib").read_bytes()).hexdigest()
        assert np.array_equal(S.score(loaded, hist, ho)["score"].to_numpy(), S.score(bundle(), hist, ho)["score"].to_numpy())
        (d / "b" / "bundle.joblib").write_bytes((d / "b" / "bundle.joblib").read_bytes() + b"x")
        _raises(lambda: S.load_bundle(d / "b"), S.ScoringError, "does not match the hash")
        _raises(lambda: S.load_bundle(d / "nothing_here"), S.ScoringError, "bundle_manifest")
        (d / "b" / "bundle_manifest.json").write_text("{not json", encoding="utf-8")                  # before 2026-10-22: a bare JSON parser message
        _raises(lambda: S.load_bundle(d / "b"), S.ScoringError, "is damaged")
        (d / "b" / "bundle_manifest.json").write_text('{"schema": {}}', encoding="utf-8")
        _raises(lambda: S.load_bundle(d / "b"), S.ScoringError, "lacks required fields")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ command line
def test_command_line_train_and_score_round_trip_and_error_codes():
    sys.path.insert(0, str(ROOT / "scripts"))
    import score_transactions as CLI
    df, tr, va, ho = data()
    hist = pd.concat([tr, va], ignore_index=True)
    d = tmpdir()
    try:
        tr.to_csv(d / "train.csv", index=False)
        va.to_csv(d / "valid.csv", index=False)
        hist.to_csv(d / "history.csv", index=False)
        ho.to_csv(d / "new.csv", index=False)
        feats = ",".join(FEATURE_COLUMNS)
        assert CLI.main(["train", "--train", str(d / "train.csv"), "--valid", str(d / "valid.csv"), "--features", feats, "--out-dir", str(d / "bundle")]) == 0
        assert CLI.main(["score", "--bundle", str(d / "bundle"), "--history", str(d / "history.csv"), "--new", str(d / "new.csv"), "--out", str(d / "scores.csv"), "--explain"]) == 0
        out = pd.read_csv(d / "scores.csv")
        assert len(out) == len(ho) and {"TransactionID", "score", "alert", "top_features"} <= set(out.columns)
        sealed = d / "test_new.csv"
        shutil.copy(d / "new.csv", sealed)
        assert CLI.main(["score", "--bundle", str(d / "bundle"), "--new", str(sealed), "--out", str(d / "o.csv")]) == 2
        assert CLI.main(["score", "--bundle", str(d / "nope"), "--new", str(d / "new.csv"), "--out", str(d / "o.csv")]) == 2
        assert not (d / "o.csv").exists()
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ------------------------------------------------------------------------------------------------ demonstration
def test_demo_runs_is_deterministic_and_shows_its_invariants():
    a, b = tmpdir(), tmpdir()
    try:
        s = run_demo(a, seed=0, verbose=False)
        run_demo(b, seed=0, verbose=False)
        assert (a / "demo_summary.json").read_bytes() == (b / "demo_summary.json").read_bytes(), "demo output is not deterministic"
        summ = json.loads((a / "demo_summary.json").read_text(encoding="utf-8"))
        assert "SYNTHETIC" in summ["banner"] and all(summ["invariants"].values()), summ["invariants"]
        assert max(r["best_overlap_with_planted"] for r in summ["candidate_days"]) >= 10
        for f in ("holdout_scores.csv", "candidate_communities_by_day.csv", "demo_summary.json"):
            assert (a / f).exists(), f
        assert s["seconds"] < 120
    finally:
        shutil.rmtree(a, ignore_errors=True)
        shutil.rmtree(b, ignore_errors=True)


def test_demo_script_refuses_to_write_inside_experiments():
    sys.path.insert(0, str(ROOT / "scripts"))
    import demo_end_to_end as DEMO
    assert DEMO.main(["--out-dir", str(ROOT / "experiments" / "demo_should_not_exist")]) == 2
    assert not (ROOT / "experiments" / "demo_should_not_exist").exists()


def test_torch_free_stand_in_carries_the_same_relations_and_fails_clearly_otherwise():
    from src.demo import _compat
    from src.graph import temporal as real            # PyTorch is installed in the full environment; the stand-in is compared with the real constant
    stub = _compat.make_stub()
    assert stub.RELATIONS == real.RELATIONS and stub.ENTITY_TYPES == real.ENTITY_TYPES
    for name in ("load_raw_arrays", "build_temporal_dataset"):
        try:
            getattr(stub, name)
        except ImportError as e:
            assert "requirements-gnn.txt" in str(e)
        else:
            raise AssertionError("stand-in exposed a torch-dependent name")
    assert _compat.install_if_needed() is False       # real module (or torch) present: nothing is replaced


def test_wording_lint_on_new_software_sources():
    files = [ROOT / "src" / "inference" / "scorer.py", ROOT / "src" / "demo" / "synthetic.py", ROOT / "src" / "demo" / "pipeline.py", ROOT / "src" / "demo" / "_compat.py",
             ROOT / "scripts" / "score_transactions.py", ROOT / "scripts" / "demo_end_to_end.py", ROOT / "tests" / "test_inference_demo.py"]
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
