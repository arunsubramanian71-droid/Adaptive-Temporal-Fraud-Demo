"""Held-out evaluator: tests of the authorization gate, the one-shot rule, the frozen registry and the load-time truncation.
No held-out file is opened and no model is scored by these tests."""
from __future__ import annotations

import hashlib
import pathlib
import re
import shutil
import sys
import tempfile

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.coordination.lint import lint_files  # noqa: E402
from src.heldout import evaluator as E  # noqa: E402


def fake_root(with_auth=None):
    d = pathlib.Path(tempfile.mkdtemp(prefix="gate_"))
    (d / "experiments" / "final_heldout_protocol").mkdir(parents=True)
    (d / E.PROTOCOL).write_text("frozen protocol text", encoding="utf-8")
    if with_auth is not None:
        (d / E.AUTH).write_text(with_auth, encoding="utf-8")
    return d


def raises(fn, exc, text=None):
    try:
        fn()
    except exc as e:
        assert text is None or text in str(e), str(e)
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_validation_split_needs_no_authorization_and_unknown_split_is_rejected():
    assert E.require_authorization("val")["authorization_required"] is False
    raises(lambda: E.require_authorization("everything"), ValueError, "unknown split")


def test_heldout_is_refused_without_the_authorization_file():
    d = fake_root()
    try:
        raises(lambda: E.require_authorization("heldout", d), E.AuthorizationError, "sealed")
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert not (ROOT / E.AUTH).exists(), "an authorization file exists in the project; it must only be created by the user"


def test_heldout_is_refused_when_hash_or_sentence_is_missing_and_accepted_only_when_both_are_present():
    d = fake_root(with_auth="please let me")
    try:
        raises(lambda: E.require_authorization("heldout", d), E.AuthorizationError, "does not quote")
        h = hashlib.sha256((d / E.PROTOCOL).read_bytes()).hexdigest()
        (d / E.AUTH).write_text(f"{E.AUTH_PHRASE} on the held-out period. protocol {h}", encoding="utf-8")
        ok = E.require_authorization("heldout", d)
        assert ok["authorization_required"] and ok["protocol_sha256"] == h
        (d / E.AUTH).write_text(f"{E.AUTH_PHRASE} on the held-out period. protocol {'0' * 64}", encoding="utf-8")
        raises(lambda: E.require_authorization("heldout", d), E.AuthorizationError, "does not quote")
        (d / E.PROTOCOL).write_text("the protocol was edited after the hash was quoted", encoding="utf-8")
        (d / E.AUTH).write_text(f"{E.AUTH_PHRASE} {h}", encoding="utf-8")
        raises(lambda: E.require_authorization("heldout", d), E.AuthorizationError)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_command_line_refuses_heldout_before_reading_anything_and_enforces_one_shot():
    import run_heldout_evaluation as CLI
    called = []
    original_run = E.run
    E.run = lambda *a, **k: called.append(a) or {}
    try:
        assert CLI.main(["--split", "heldout"]) == 2 and not called, "scoring started without authorization"
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="gate_"))
        old = CLI.FOLDER
        CLI.FOLDER = tmp
        (tmp / "heldout_results").mkdir()
        (tmp / "heldout_results" / ".completed").write_text("x", encoding="utf-8")
        try:
            assert CLI.main(["--split", "heldout"]) == 2 and not called
        finally:
            CLI.FOLDER = old
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        E.run = original_run


def test_heldout_features_path_is_only_reachable_through_the_gate():
    src = (ROOT / "src" / "heldout" / "evaluator.py").read_text(encoding="utf-8")
    assert src.count("test_features") == 1, "the held-out feature file must be named in exactly one place (SPLITS)"
    body = src[src.index("def run("):]
    assert body.index("require_authorization(split, root)") < body.index("import torch"), "the gate must run before anything is loaded"
    assert "test_features" not in (ROOT / "scripts" / "run_heldout_evaluation.py").read_text(encoding="utf-8")


def test_frozen_registry_matches_the_protocol_and_every_artifact_exists():
    reg = E.frozen_registry()
    assert len(reg) == 3 + 1 + 5 * 3        # 3 tabular, B_full, (3 static + 2 temporal families) x 3 seeds
    for mid, spec in reg.items():
        for f in spec["files"]:
            assert (ROOT / spec["dir"] / f).exists(), (mid, f)
        t = E.recorded_threshold(ROOT / spec["dir"])
        assert 0.0 < t < 1.0, (mid, t)
    proto = (ROOT / E.PROTOCOL).read_text(encoding="utf-8")
    for token in ("logistic regression", "XGBoost", "GraphSAGE", "GAT", "Temporal GraphSAGE", "memory ablated", "schedule-matched"):
        assert token.lower() in proto.lower(), token
    assert "F | the adaptive system" in proto and "not evaluated" in proto


def test_truncate_raw_drops_every_later_row_and_edge():
    day = np.array([1, 1, 2, 3, 3, 4, 5])
    raw = {"day": day, "X": np.arange(7 * 2).reshape(7, 2), "y": np.arange(7.0), "txn_id": np.arange(100, 107),
           "relations": {"used_card": (np.arange(7), np.arange(7) % 3, day.copy())}, "n_entities": {"card": 3}}
    t = E.truncate_raw(raw, 3)
    assert len(t["day"]) == 5 and t["day"].max() == 3 and len(t["X"]) == len(t["y"]) == len(t["txn_id"]) == 5
    src, dst, et = t["relations"]["used_card"]
    assert src.max() < 5 and et.max() == 3 and len(src) == len(dst) == len(et) == 5


def test_feature_frames_are_read_without_the_label_column():
    src = (ROOT / "src" / "heldout" / "evaluator.py").read_text(encoding="utf-8")
    assert "pd.read_parquet" not in src, "every feature read must go through read_features (label projected away)"
    assert src.count("read_features(") >= 5
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tempfile
    d = pathlib.Path(tempfile.mkdtemp(prefix="gate_"))
    try:
        pq.write_table(pa.table({"TransactionID": [1, 2], "x": [0.1, 0.2], "isFraud": [0, 1]}), d / "f.parquet")
        df = E.read_features(d / "f.parquet")
        assert list(df.columns) == ["TransactionID", "x"] and len(df) == 2
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_wording_lint_on_heldout_sources():
    files = [ROOT / "src" / "heldout" / "evaluator.py", ROOT / "scripts" / "run_heldout_evaluation.py", ROOT / "tests" / "test_heldout_gate.py", ROOT / E.PROTOCOL]
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
