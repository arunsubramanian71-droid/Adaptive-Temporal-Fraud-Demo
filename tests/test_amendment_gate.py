"""Amendment 1 of the held-out protocol: authorization gate, module subset rule, one-shot rule and registries.
No held-out file is opened and no model is scored by these tests."""
from __future__ import annotations

import hashlib
import pathlib
import shutil
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.coordination.lint import lint_files  # noqa: E402
from src.heldout import amendment as A  # noqa: E402
from src.heldout import evaluator as E  # noqa: E402


def fake_root(auth=None):
    d = pathlib.Path(tempfile.mkdtemp(prefix="amend_"))
    (d / "experiments" / "final_heldout_protocol").mkdir(parents=True)
    (d / E.PROTOCOL).write_text("frozen protocol text", encoding="utf-8")
    (d / A.AMENDMENT).write_text("frozen amendment text", encoding="utf-8")
    if auth is not None:
        (d / E.AUTH).write_text(auth, encoding="utf-8")
    return d


def hashes(d):
    return hashlib.sha256((d / E.PROTOCOL).read_bytes()).hexdigest(), hashlib.sha256((d / A.AMENDMENT).read_bytes()).hexdigest()


def raises(fn, exc, text=None):
    try:
        fn()
    except exc as e:
        assert text is None or text in str(e), str(e)
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_validation_needs_no_authorization_and_unknown_module_is_rejected():
    assert A.require_amendment_authorization("val", ("H", "U"))["authorization_required"] is False
    raises(lambda: A.require_amendment_authorization("val", ("Z",)), ValueError, "unknown amendment module")


def test_heldout_refused_without_base_authorization_even_if_amendment_is_quoted():
    d = fake_root()
    try:
        raises(lambda: A.require_amendment_authorization("heldout", ("H",), d), E.AuthorizationError, "sealed")
        ph, ah = hashes(d)
        (d / E.AUTH).write_text(f"{A.AMEND_PHRASE} H. amendment {ah}", encoding="utf-8")      # amendment quoted, base protocol not
        raises(lambda: A.require_amendment_authorization("heldout", ("H",), d), E.AuthorizationError, "does not quote")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_base_authorization_alone_does_not_unlock_any_amendment_module():
    d = fake_root()
    try:
        ph, ah = hashes(d)
        (d / E.AUTH).write_text(f"{E.AUTH_PHRASE} on the held-out period. protocol {ph}", encoding="utf-8")
        assert E.require_authorization("heldout", d)["authorization_required"]            # the base protocol is authorized ...
        for m in A.MODULES:
            raises(lambda m=m: A.require_amendment_authorization("heldout", (m,), d), E.AuthorizationError, "amendment hash")      # ... the amendment is not
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_modules_must_be_named_and_requested_must_be_a_subset_of_authorized():
    d = fake_root()
    try:
        ph, ah = hashes(d)
        base = f"{E.AUTH_PHRASE} on the held-out period. protocol {ph}. amendment {ah}. "
        (d / E.AUTH).write_text(base, encoding="utf-8")
        raises(lambda: A.require_amendment_authorization("heldout", ("H",), d), E.AuthorizationError, "does not contain the sentence")
        (d / E.AUTH).write_text(base + f"{A.AMEND_PHRASE} H, U", encoding="utf-8")
        ok = A.require_amendment_authorization("heldout", ("H", "U"), d)
        assert ok["amendment_modules_authorized"] == ["H", "U"] and ok["amendment_sha256"] == ah
        raises(lambda: A.require_amendment_authorization("heldout", ("F",), d), E.AuthorizationError, "['F']")
        raises(lambda: A.require_amendment_authorization("heldout", ("H", "F"), d), E.AuthorizationError, "['F']")
        (d / A.AMENDMENT).write_text("the amendment was edited after the hash was quoted", encoding="utf-8")
        raises(lambda: A.require_amendment_authorization("heldout", ("H",), d), E.AuthorizationError, "amendment hash")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_no_authorization_or_amendment_results_exist_in_the_project():
    fh = ROOT / "experiments" / "final_heldout_protocol"
    assert not (fh / "AUTHORIZATION.txt").exists()
    for n in ("heldout_results", "heldout_results_amendment", "heldout_results_f"):
        assert not (fh / n).exists(), n


def test_amendment_cli_refuses_before_reading_and_enforces_one_shot():
    import run_amendment_evaluation as CLI
    called = []
    original = A.run_modules
    A.run_modules = lambda *a, **k: called.append(a) or {}
    try:
        assert CLI.main(["--split", "heldout", "--modules", "H,U"]) == 2 and not called
        tmp = pathlib.Path(tempfile.mkdtemp(prefix="amend_"))
        old = CLI.FOLDER
        CLI.FOLDER = tmp
        try:
            (tmp / "heldout_results_amendment").mkdir()
            (tmp / "heldout_results_amendment" / ".completed").write_text("x")
            assert CLI.main(["--split", "heldout", "--modules", "H,U"]) == 2 and not called
        finally:
            CLI.FOLDER = old
            shutil.rmtree(tmp, ignore_errors=True)
    finally:
        A.run_modules = original


def test_registries_do_not_change_the_base_protocol_and_cover_the_amendment_models():
    assert len(E.frozen_registry()) == 19
    reg = A.u_registry()
    assert set(reg) == {"A_none", "B_none"} | {f"{n}_s{s}" for n in ("graphsage_unweighted", "temporal_graphsage_unweighted") for s in E.SEEDS}
    assert A.authorized_modules(f"{A.AMEND_PHRASE} F, H and U.") == {"F", "H", "U"}
    assert A.authorized_modules("nothing") == set()


def test_modules_lint_clean():
    files = [ROOT / "src/heldout/amendment.py", ROOT / "scripts/run_amendment_evaluation.py", ROOT / "src/heldout/h4.py", ROOT / "tests/test_amendment_gate.py"]
    bad = {str(k): v for k, v in lint_files(files).items() if v}
    assert bad == {}, bad


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
