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
