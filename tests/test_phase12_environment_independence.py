"""Phase 12 reproducibility, environment-independent reading (companion to tests/test_phase12_outputs.py::test_AC10_sidecars_and_deterministic_rebuild).

The registered AC10 test compares the whole rebuilt manifest with the recorded one apart from its timestamp. The manifest also records the interpreter and
library versions of the build environment, so under another Python (for example a clean 3.14 environment) that comparison reports a difference in
``versions.python`` only. This module checks what AC10 is about in a way that does not depend on the interpreter: every data output of a rebuild is byte-identical to
the recorded one, the manifests are equal in every field except the timestamp and the recorded environment versions, and the rebuilt manifest records the running
interpreter's own version. It does not replace or weaken the registered test; scripts/run_core_tests.py uses it to classify a failure of that test as an environment-metadata
difference only when this module confirms it.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.evidence import build as B  # noqa: E402

OUT = ROOT / "experiments" / "phase12_validation_evidence_table" / "outputs"
ENVIRONMENT_FIELDS = ("created_local", "versions")


def _h(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def test_rebuild_is_byte_identical_and_the_manifest_differs_only_in_environment_metadata():
    with tempfile.TemporaryDirectory() as td:
        B.build(ROOT, td)
        data_outputs = [p for p in OUT.iterdir() if p.suffix != ".sha256" and p.name != "manifest_phase12.json"]
        assert len(data_outputs) == 11, [p.name for p in data_outputs]
        for p in data_outputs:
            assert _h(p) == _h(pathlib.Path(td) / p.name), f"data output differs on rebuild: {p.name}"
        recorded = json.loads((OUT / "manifest_phase12.json").read_text(encoding="utf-8"))
        rebuilt = json.loads((pathlib.Path(td) / "manifest_phase12.json").read_text(encoding="utf-8"))
    for field in ENVIRONMENT_FIELDS:
        assert field in recorded and field in rebuilt, field
    a = {k: v for k, v in recorded.items() if k not in ENVIRONMENT_FIELDS}
    b = {k: v for k, v in rebuilt.items() if k not in ENVIRONMENT_FIELDS}
    assert a == b, sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
    assert set(recorded["versions"]) == set(rebuilt["versions"]) == {"python", "pandas", "numpy", "scikit_learn"}
    assert rebuilt["versions"]["python"] == platform.python_version(), "the rebuilt manifest must record the interpreter that built it"


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
