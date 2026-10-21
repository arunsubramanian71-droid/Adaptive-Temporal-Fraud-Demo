"""Held-out runners refuse before reading anything unless the registered validation readiness gate of every requested part has passed
(protocol section 6; Amendment 1 sections 3-5). No held-out file is opened and no model is run."""
from __future__ import annotations

import json
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

from src.heldout import amendment as A  # noqa: E402
from src.heldout import codehash as CH  # noqa: E402
from src.heldout import evaluator as E  # noqa: E402


def raises(fn, exc, text=None):
    try:
        fn()
    except exc as e:
        assert text is None or text in str(e), str(e)
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_module_f_is_refused_while_its_registered_gate_has_not_passed():
    rec = json.loads((ROOT / A.READINESS["F"]).read_text(encoding="utf-8"))
    assert rec["all_pass"] is False                                   # the registered trained-arm gate failed as written
    raises(lambda: A.require_amendment_readiness(("F",)), A.ReadinessError, "did not pass")


def test_modules_h_and_u_pass_their_recorded_readiness_and_name_their_models():
    ok = A.require_amendment_readiness(("H", "U"))
    assert set(ok) == {"H", "U"} and all(v["all_pass"] for v in ok.values())


def test_missing_or_failing_readiness_records_are_refused():
    d = pathlib.Path(tempfile.mkdtemp(prefix="ready_"))
    try:
        raises(lambda: A.require_amendment_readiness(("H",), d), A.ReadinessError, "does not exist")
        p = d / A.READINESS["H"]
        p.parent.mkdir(parents=True)
        p.write_text(json.dumps({"all_pass": True, "models": [{"model": "A_none", "pass": True}]}), encoding="utf-8")
        raises(lambda: A.require_amendment_readiness(("H",), d), A.ReadinessError, "xgboost_historical")
        p.write_text(json.dumps({"all_pass": False, "models": []}), encoding="utf-8")
        raises(lambda: A.require_amendment_readiness(("U",), d), A.ReadinessError, "did not pass")
        rows = [{"model": "xgboost_historical", "pass": True}]
        p.write_text(json.dumps({"all_pass": True, "models": rows}), encoding="utf-8")
        raises(lambda: A.require_amendment_readiness(("H",), d), A.ReadinessError, "does not fingerprint")
        (d / "src").mkdir()
        (d / "src" / "m.py").write_text("x = 1", encoding="utf-8")
        p.write_text(json.dumps({"all_pass": True, "models": rows, "code_sha256": {"src/m.py": CH.file_sha256(d / "src" / "m.py")}}), encoding="utf-8")
        assert A.require_amendment_readiness(("H",), d)["H"]["code_files_verified"] == 1
        (d / "src" / "m.py").write_text("x = 2", encoding="utf-8")                   # the code changed after the gate
        raises(lambda: A.require_amendment_readiness(("H",), d), A.ReadinessError, "code changed")
        (d / "src" / "m.py").unlink()                                              # or disappeared
        raises(lambda: A.require_amendment_readiness(("H",), d), A.ReadinessError, "code changed")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_amendment_cli_checks_readiness_after_authorization_and_before_any_read():
    import run_amendment_evaluation as CLI
    calls = []
    orig_auth, orig_run = A.require_amendment_authorization, A.run_modules
    A.require_amendment_authorization = lambda split, mods, root=ROOT: {"split": split}       # pretend authorized (no file is created)
    A.run_modules = lambda *a, **k: calls.append(a) or {}
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="ready_"))
    old = CLI.FOLDER
    CLI.FOLDER = tmp
    orig_ready = A.require_amendment_readiness
    try:
        def failing(mods, root=ROOT):
            raise A.ReadinessError("gate not passed")
        A.require_amendment_readiness = failing
        assert CLI.main(["--split", "heldout", "--modules", "H,U"]) == 2 and calls == []
    finally:
        A.require_amendment_authorization, A.run_modules, A.require_amendment_readiness = orig_auth, orig_run, orig_ready
        CLI.FOLDER = old
        shutil.rmtree(tmp, ignore_errors=True)


def test_base_cli_requires_a_passing_readiness_report_that_fingerprints_the_unchanged_code():
    import run_heldout_evaluation as CLI
    ok = CLI.require_base_readiness()                                   # the current record: all 19 frozen models, code unchanged since the gate ran
    assert ok["all_pass"] is True and ok["code_files_verified"] >= 2
    d = pathlib.Path(tempfile.mkdtemp(prefix="ready_"))
    try:
        raises(lambda: CLI.require_base_readiness(d), E.AuthorizationError, "has not been run")
        (d / "readiness_val").mkdir()
        rp = d / "readiness_val" / "readiness_report.json"
        rows = [{"model": m} for m in E.frozen_registry()]
        rp.write_text(json.dumps({"all_pass": False, "models": rows}), encoding="utf-8")
        raises(lambda: CLI.require_base_readiness(d), E.AuthorizationError, "did not pass")
        rp.write_text(json.dumps({"all_pass": True, "models": rows[:-1]}), encoding="utf-8")
        raises(lambda: CLI.require_base_readiness(d), E.AuthorizationError, "did not pass")
        rp.write_text(json.dumps({"all_pass": True, "models": rows}), encoding="utf-8")
        raises(lambda: CLI.require_base_readiness(d), E.AuthorizationError, "does not fingerprint")
        code = {f: CH.file_sha256(ROOT / f) for f in ("src/heldout/evaluator.py", "scripts/run_heldout_evaluation.py")}
        rp.write_text(json.dumps({"all_pass": True, "models": rows, "code_sha256": code}), encoding="utf-8")
        assert CLI.require_base_readiness(d)["all_pass"] is True
        code["src/heldout/evaluator.py"] = "0" * 64                        # a record made with different evaluator code is stale
        rp.write_text(json.dumps({"all_pass": True, "models": rows, "code_sha256": code}), encoding="utf-8")
        raises(lambda: CLI.require_base_readiness(d), E.AuthorizationError, "code changed")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_recorded_readiness_fingerprints_cover_the_scoring_code_and_are_content_based():
    base = json.loads((ROOT / "experiments/final_heldout_protocol/readiness_val/readiness_report.json").read_text(encoding="utf-8"))["code_sha256"]
    amend = json.loads((ROOT / A.READINESS["H"]).read_text(encoding="utf-8"))["code_sha256"]
    assert {"src/heldout/evaluator.py", "scripts/run_heldout_evaluation.py", "src/heldout/codehash.py"} <= set(base)
    assert {"src/heldout/evaluator.py", "src/heldout/amendment.py", "scripts/run_amendment_evaluation.py"} <= set(amend)
    assert all(f.split("/")[0] in ("src", "scripts") for f in {*base, *amend})
    assert CH.changed_code(base) == [] and CH.changed_code(amend) == []
    assert "src/heldout/codehash.py" in CH.loaded_project_code()


def test_no_row_level_prediction_or_held_out_output_file_is_tracked():
    import subprocess
    if not (ROOT / ".git").exists():
        return                                                          # a source archive without history: nothing to check
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True).stdout.split("\0")
    bad = [f for f in tracked if f and (pathlib.PurePosixPath(f).name.startswith("predictions") and f.endswith(".parquet")
                                        or "/heldout_results" in f or f.endswith("AUTHORIZATION.txt"))]
    assert bad == [], bad[:5]


def test_new_decision_and_guard_files_pass_the_wording_lint():
    from src.coordination.lint import lint_files
    files = [ROOT / p for p in ("src/heldout/codehash.py", "tests/test_readiness_guards.py", "scripts/check_ac12_derived_evidence.py",
                                "experiments/final_heldout_protocol/AMENDMENT_2_PROPOSAL_F_GATE_REV1.md", "submission/PHASE13_SCOPE_CONFIRMATION.md",
                                "submission/HELDOUT_DECISION_PACKAGE.md", "submission/MASTER_COMPLETION_CHECKLIST.md", "submission/TRACEABILITY_MATRIX.md")]
    bad = {str(k): v for k, v in lint_files(files).items() if v}
    assert bad == {}, bad


def test_amendment_2_proposals_are_frozen_inactive_and_the_revision_states_the_corrected_error_rates():
    import hashlib
    from scipy import stats
    fh = ROOT / "experiments" / "final_heldout_protocol"
    for name in ("AMENDMENT_2_PROPOSAL_F_GATE", "AMENDMENT_2_PROPOSAL_F_GATE_REV1"):
        md = fh / f"{name}.md"
        assert hashlib.sha256(md.read_bytes()).hexdigest() == (fh / f"{name}.sha256").read_text(encoding="ascii").strip()
        assert "NOT ACTIVE" in md.read_text(encoding="utf-8").splitlines()[0]
    assert hashlib.sha256((fh / "AMENDMENT_2_PROPOSAL_F_GATE.md").read_bytes()).hexdigest().startswith("55776392")    # the original is kept unchanged
    rev = (fh / "AMENDMENT_2_PROPOSAL_F_GATE_REV1.md").read_text(encoding="utf-8")
    assert round(2 * stats.t.sf(3, 4), 3) == 0.040 and round(1 - (1 - 2 * stats.t.sf(3, 4)) ** 6, 3) == 0.217          # A2-a as written: t with 4 df
    assert "4.0 %" in rev and "21.7 %" in rev
    assert round(stats.t.ppf(1 - 0.0027 / 2, 9), 3) == 4.094 and "4.094" in rev                                          # A2-a-r1 critical value
    import math
    diag = json.loads((ROOT / "experiments/phase8_state_continuity_supplement/diagnostics/F_readiness_diagnosis_E_seed123.json").read_text(encoding="utf-8"))
    five = diag["identical_run_window_means"] + [diag["recorded_supplement_window_mean"], diag["recorded_readiness_window_mean"]]
    s = float(np.std(five, ddof=1))
    assert round(s, 4) == 0.0044                                                                                          # the measured run-to-run SD
    p_reg = 2 * stats.norm.sf(0.01 / (math.sqrt(2) * 0.0044))                                                             # registered 0.01 gate, correct code
    assert round(p_reg, 3) == 0.108 and round(1 - (1 - p_reg) ** 6, 2) == 0.50                                           # the package's "10.8 % / 50 %"
    assert round(4.094 * 0.0044 * math.sqrt(1 + 1 / 10), 3) == 0.019                                                      # A2-a-r1 tolerance at s = 0.0044
    for code in ("src/heldout/amendment.py", "src/heldout/evaluator.py", "scripts/run_f_walkforward.py"):
        assert "AMENDMENT_2_PROPOSAL_F_GATE_REV1" not in (ROOT / code).read_text(encoding="utf-8")                      # no code reads it before approval


def test_no_authorization_file_exists():
    assert not (ROOT / E.AUTH).exists()


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
