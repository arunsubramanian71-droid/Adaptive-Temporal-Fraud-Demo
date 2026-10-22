"""Amendment 1 module F runner: the held-out branch is refused without the amendment authorization. Needs PyTorch because the runner imports it at module level
(in an environment without PyTorch the core test runner skips this module and names it; it is never counted as passed there).
No held-out file is opened and no model is run."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_f_walkforward as RF  # noqa: E402


def test_f_runner_refuses_heldout_without_amendment_authorization():
    assert RF.main(["--split", "heldout", "--variant", "burn_in"]) == 2
    assert RF.main(["--split", "heldout", "--variant", "as_written"]) == 2
    assert not (ROOT / "experiments" / "final_heldout_protocol" / "heldout_results_f").exists()


def test_f_runner_refuses_on_its_failed_readiness_gate_even_when_authorized_and_reads_nothing():
    from src.heldout import amendment as A
    orig_auth, orig_load = A.require_amendment_authorization, RF.load_raw_arrays
    A.require_amendment_authorization = lambda split, mods, root=None: {"split": split}       # pretend authorized (no file is created)

    def no_read(*a, **k):
        raise AssertionError("data was loaded although the readiness gate has not passed")
    RF.load_raw_arrays = no_read
    try:
        assert RF.main(["--split", "heldout", "--variant", "burn_in"]) == 2
    finally:
        A.require_amendment_authorization, RF.load_raw_arrays = orig_auth, orig_load
    assert not (ROOT / "experiments" / "final_heldout_protocol" / "heldout_results_f").exists()


def test_f_runner_heldout_scope_is_the_registered_one_and_any_output_counts_as_executed():
    """Amendment 1 section 3 locks arms F, Ad, E and seeds 42, 123, 2024; section 7: any held-out output counts as executed (a repeat only after a crash that wrote
    no held-out score). Before this guard a `--seeds`/`--arms` subset could run and then mark the folder completed, and a crash after the first window file allowed a repeat."""
    import shutil
    import tempfile
    from src.heldout import amendment as A

    class Reached(Exception):
        pass

    def reached(*a, **k):
        raise Reached()
    orig = (A.require_amendment_authorization, A.require_amendment_readiness, RF.load_raw_arrays, RF.HELDOUT_OUT)
    A.require_amendment_authorization = lambda split, mods, root=None: {"split": split}       # pretend authorized (no file is created)
    A.require_amendment_readiness = lambda mods, root=None: {m: {"all_pass": True} for m in mods}
    RF.load_raw_arrays = reached
    tmp = pathlib.Path(tempfile.mkdtemp(prefix="f_once_"))
    RF.HELDOUT_OUT = tmp / "heldout_results_f"
    try:
        (RF.HELDOUT_OUT / "adaptive" / "seed_42").mkdir(parents=True)
        (RF.HELDOUT_OUT / "adaptive" / "seed_42" / "metrics_W1.json").write_text("{}", encoding="utf-8")
        assert RF.main(["--split", "heldout"]) == 2                                     # refused before any read (load_raw_arrays not reached)
        (RF.HELDOUT_OUT / "adaptive" / "seed_42" / "metrics_W1.json").unlink()
        for extra in (["--seeds", "42"], ["--seeds", "42,123"], ["--arms", "F,Ad"], ["--arms", "Ad"], ["--seeds", "42,123,2024,7"]):
            assert RF.main(["--split", "heldout", *extra]) == 2, extra
        try:
            RF.main(["--split", "heldout"])                                              # empty folder, registered scope: the guards pass and loading starts
            raise AssertionError("the registered scope was refused")
        except Reached:
            pass
    finally:
        A.require_amendment_authorization, A.require_amendment_readiness, RF.load_raw_arrays, RF.HELDOUT_OUT = orig
        shutil.rmtree(tmp, ignore_errors=True)
    assert not (ROOT / "experiments" / "final_heldout_protocol" / "heldout_results_f").exists()


def test_f_runner_never_overwrites_a_completed_validation_run_and_refuses_before_reading():
    import hashlib
    sup = ROOT / "experiments" / "phase8_state_continuity_supplement" / "outputs"
    recs = [sup / "readiness" / "readiness_F.json", sup / "burn_in" / "summary.json", sup / "as_written" / "readiness_S1.json"]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in recs]
    orig = RF.load_raw_arrays

    def no_read(*a, **k):
        raise AssertionError("data was loaded although the output folder holds a completed run")
    RF.load_raw_arrays = no_read
    try:
        for variant in ("readiness", "burn_in", "as_written"):
            assert RF.main(["--split", "val", "--variant", variant]) == 2
        assert RF.main(["--split", "val", "--variant", "readiness", "--out-name", "burn_in"]) == 2
        assert RF.main(["--split", "val", "--variant", "readiness", "--out-name", "../x"]) == 2
    finally:
        RF.load_raw_arrays = orig
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in recs] == before


def test_f_diagnosis_never_overwrites_a_preserved_record_and_refuses_before_reading():
    import hashlib
    import diagnose_f_readiness as DG
    recs = [DG.DIAG / "F_readiness_diagnosis_E_seed123.json", DG.DIAG / "F_readiness_diagnosis_E_seed123_cpu.json"]
    before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in recs]
    orig = DG.load_raw_arrays

    def no_read(*a, **k):
        raise AssertionError("data was loaded although the record exists")
    DG.load_raw_arrays = no_read
    try:
        assert DG.main(["--arm", "E", "--seed", "123", "--runs", "1"]) == 2
        assert DG.main(["--arm", "E", "--seed", "123", "--runs", "1", "--device", "cpu"]) == 2
    finally:
        DG.load_raw_arrays = orig
    assert [hashlib.sha256(p.read_bytes()).hexdigest() for p in recs] == before


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
