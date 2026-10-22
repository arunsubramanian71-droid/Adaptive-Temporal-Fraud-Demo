"""Research-model serving (src/inference/research_model.py, /api/research/*): the API returns exactly what the recorded model returns.

Tests that need the trained model and validation features are listed in tests/artifact_requirements.json (NOT RUN on a fresh clone, never counted as
passed). Validation rows only; no held-out file is opened; nothing is trained."""
from __future__ import annotations

import json
import pathlib
import sys
import threading
import urllib.error
import urllib.request

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.inference import scorer as S  # noqa: E402
from src.serving import app  # noqa: E402

VAL = ROOT / "data" / "features" / "val_features.parquet"
_MODEL = None


def model():
    global _MODEL
    if _MODEL is None:
        from src.inference.research_model import ResearchModel
        _MODEL = ResearchModel()
    return _MODEL


def val_rows(n=200):
    from src.heldout.evaluator import read_features
    return read_features(VAL).head(n)


def as_json_rows(df):
    return json.loads(df.replace({np.nan: None}).to_json(orient="records"))


class Live:
    def __init__(self, services):
        self.srv = app.make_server(services, "127.0.0.1", 0)
        self.port = self.srv.server_address[1]
        self.t = threading.Thread(target=self.srv.serve_forever, daemon=True)

    def __enter__(self):
        self.t.start()
        return self

    def __exit__(self, *a):
        self.srv.shutdown()
        self.srv.server_close()

    def call(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode("utf-8")
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=data, method=method, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())


def raises(fn, exc, text=None):
    try:
        fn()
    except exc as e:
        assert text is None or text in str(e), str(e)
        return
    raise AssertionError(f"expected {exc.__name__}")


# ---------------------------------------------------------------- no artifact needed
def test_without_the_research_model_the_endpoints_answer_404_and_405_and_status_says_so():
    with Live(app.Services()) as s:
        st, j = s.call("GET", "/api/research/model")
        assert st == 404 and "--research-model" in j["error"]
        st, j = s.call("POST", "/api/research/score", {"rows": [{"a": 1}]})
        assert st == 404
        st, j = s.call("GET", "/api/research/score")
        assert st == 405
        st, j = s.call("GET", "/api/status")
        assert st == 200 and j["research_model_loaded"] is False


def test_research_model_refuses_held_out_paths_and_explains_a_missing_model():
    from src.inference.research_model import ResearchModel
    raises(lambda: ResearchModel("experiments/test_models/xgboost"), S.SealedDataError)
    raises(lambda: ResearchModel("experiments/__no_such_model__"), FileNotFoundError, "scripts/train_xgboost.py")


def test_a_damaged_model_file_gets_a_clear_error_the_launcher_reports():
    """Before 2026-10-22 a truncated model.joblib raised a bare EOFError, which the launcher (it catches OSError) did not handle: a traceback, not a message."""
    import shutil
    import tempfile
    from src.inference.research_model import ResearchModel, ResearchModelError
    d = pathlib.Path(tempfile.mkdtemp(prefix="rm_"))
    try:
        for content in (b"\x80\x04truncated", b"not a pickle at all", b""):
            (d / "model.joblib").write_bytes(content)
            raises(lambda: ResearchModel(d), ResearchModelError, "could not be loaded")
        assert issubclass(ResearchModelError, OSError)                     # scripts/run_dashboard.py prints it and exits 2
        import joblib
        joblib.dump({"not": "a model"}, d / "model.joblib")
        raises(lambda: ResearchModel(d), ResearchModelError, "predict_proba")
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- artifact-dependent (registered)
def test_api_scores_equal_direct_model_scores_and_the_recorded_validation_predictions():
    m = model()
    df = val_rows(100)                                                   # about 6.6 KB per row: 100 rows stay under the 1 MB body limit (P13-2)
    direct = m.model.predict_proba(m.builder.to_matrix(df, for_model="xgb"))[:, 1]
    rec = pd.read_parquet(ROOT / "experiments/xgboost/predictions_val.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
    with Live(app.Services(research=m)) as s:
        st, j = s.call("POST", "/api/research/score", {"rows": as_json_rows(df)})
    assert st == 200 and j["mode"] == "research_model" and j["synthetic_demo"] is False and j["n"] == len(df)
    api = np.array([r["score"] for r in j["results"]])
    assert np.array_equal(api, direct.astype(np.float64)), float(np.abs(api - direct).max())
    assert np.abs(api - rec.loc[df["TransactionID"]].to_numpy()).max() <= 1e-9
    assert [r["TransactionID"] for r in j["results"]] == df["TransactionID"].tolist()
    assert all(r["alert"] == (r["score"] >= m.threshold) for r in j["results"])
    assert all(("top_features" in r) == r["alert"] for r in j["results"])


def test_labels_in_the_request_are_ignored_and_each_row_is_scored_independently():
    m = model()
    df = val_rows(60)
    with Live(app.Services(research=m)) as s:
        _, base = s.call("POST", "/api/research/score", {"rows": as_json_rows(df)})
        rows1 = [dict(r, isFraud=1) for r in as_json_rows(df)]
        rows0 = [dict(r, isFraud=0) for r in as_json_rows(df)]
        _, j1 = s.call("POST", "/api/research/score", {"rows": rows1})
        _, j0 = s.call("POST", "/api/research/score", {"rows": rows0})
        _, single = s.call("POST", "/api/research/score", {"rows": as_json_rows(df.iloc[[37]])})
        _, reversed_ = s.call("POST", "/api/research/score", {"rows": as_json_rows(df.iloc[::-1])})
    sc = [r["score"] for r in base["results"]]
    assert [r["score"] for r in j1["results"]] == sc == [r["score"] for r in j0["results"]]
    assert j1["labels_in_request_ignored"] is True and base["labels_in_request_ignored"] is False
    assert single["results"][0]["score"] == sc[37]                                # no other row (earlier or later) influences a score
    assert [r["score"] for r in reversed_["results"]] == sc[::-1]


def test_invalid_research_requests_get_actionable_400_errors():
    m = model()
    df = val_rows(3)
    rows = as_json_rows(df)
    with Live(app.Services(research=m)) as s:
        st, j = s.call("POST", "/api/research/score", {"rows": [{k: v for k, v in rows[0].items() if k != "TransactionAmt"}]})
        assert st == 400 and "required input column" in j["error"] and "TransactionAmt" in j["error"]
        st, j = s.call("POST", "/api/research/score", {"rows": [dict(rows[0], TransactionAmt="not a number")]})
        assert st == 400 and "wrong type" in j["error"]
        for bad_id in ("abc", 7.9, True, [1]):                                                       # before 2026-10-22: 500 or a silently rounded id
            st, j = s.call("POST", "/api/research/score", {"rows": [dict(rows[0], TransactionID=bad_id)]})
            assert st == 400 and "TransactionID must be an integer" in j["error"], (bad_id, st, j)
        st, j = s.call("POST", "/api/research/score", {"rows": [dict(rows[0], TransactionID=123456789), dict(rows[1], TransactionID=None), dict(rows[2], TransactionID=5.0)]})
        assert st == 200 and [r["TransactionID"] for r in j["results"]] == [123456789, None, 5], j["results"]
        for bad in ({}, {"rows": []}, {"rows": "x"}, {"rows": [1, 2]}):
            st, j = s.call("POST", "/api/research/score", bad)
            assert st == 400 and "error" in j and "Traceback" not in json.dumps(j), bad
        st, j = s.call("POST", "/api/research/score", {"rows": [{}] * (app.MAX_ROWS + 1)})          # row cap (small rows, under the body limit)
        assert st == 400 and "at most" in j["error"]
        st, j = s.call("POST", "/api/research/score", {"rows": [rows[0]] * 200})                     # about 1.3 MB: the body limit answers first
        assert st == 413 and "larger than" in j["error"]
        st, j = s.call("GET", "/api/research/model")
        assert st == 200 and j["n_required_input_columns"] == len(m.required_columns()) and j["uses_history"] is False and j["threshold"] == m.threshold


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
