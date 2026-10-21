"""Core-versus-artifact-dependent test workflow: tests/artifact_requirements.json lists the tests that need artifacts the repository does not publish.
This module checks the registry itself (it names real tests, points only at withheld artifacts, never at a held-out path) and the runner's behaviour on a missing artifact.
No held-out file is opened and no model is run."""
from __future__ import annotations

import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_core_tests as RC  # noqa: E402

REGISTRY = {k: v for k, v in json.loads((ROOT / "tests" / "artifact_requirements.json").read_text(encoding="utf-8")).items() if not k.startswith("_")}


def test_every_registered_key_names_a_real_test_function():
    bad = []
    for key in REGISTRY:
        module, test = key.split("::")
        p = ROOT / "tests" / f"{module}.py"
        if not p.exists() or not re.search(rf"^def {re.escape(test)}\(", p.read_text(encoding="utf-8"), re.M):
            bad.append(key)
    assert bad == [], bad


def test_registered_patterns_are_relative_never_held_out_and_never_match_a_tracked_file():
    for key, pats in REGISTRY.items():
        assert pats and all(isinstance(p, str) and not pathlib.PurePath(p).is_absolute() and ".." not in p for p in pats), key
        assert not any(re.search(r"test_features|test_transaction|test_identity|AUTHORIZATION|heldout_results", p, re.I) for p in pats), f"{key}: pattern names a held-out path"
    if not (ROOT / ".git").exists():
        return                                                     # a source archive without git history: the tracked-file check does not apply
    tracked = set(subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, text=True).stdout.split("\0"))
    clash = [(k, p) for k, pats in REGISTRY.items() for p in pats if any(ROOT.glob(p)) and any(t in tracked for t in (x.relative_to(ROOT).as_posix() for x in ROOT.glob(p)))]
    assert clash == [], f"registered as withheld but tracked (the test could run): {clash}"


def test_missing_artifact_helper_reports_missing_patterns_and_never_invents_presence():
    assert RC._missing_artifacts(["experiments/__no_such_artifact__/*.parquet"]) == ["experiments/__no_such_artifact__/*.parquet"]
    assert RC._missing_artifacts(["tests/artifact_requirements.json"]) == []
    assert RC._missing_artifacts([]) == []


def test_registry_covers_the_thirteen_tests_that_failed_on_a_fresh_clone():
    assert len(REGISTRY) == 13, len(REGISTRY)
    modules = {k.split("::")[0] for k in REGISTRY}
    assert modules == {"test_comparison_b_outputs", "test_heldout_gate", "test_phase11_outputs", "test_phase12_environment_independence", "test_phase12_outputs",
                       "test_scope_reconciliation_records"}


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
