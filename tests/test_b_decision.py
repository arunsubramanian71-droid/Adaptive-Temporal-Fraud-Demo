"""Comparison B: tests of the registered decision rules (BA8) on synthetic numbers."""
from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features import b_decision as D  # noqa: E402


def test_h1_supported_needs_all_three_conditions():
    r = D.decide_h1(0.56, 0.54, [0.56, 0.57, 0.555], [0.54, 0.54, 0.54])
    assert r["verdict"] == "SUPPORTED" and all(r["conditions"].values()), r
    # primary comparison fails -> not supported even if seeds agree
    assert D.decide_h1(0.53, 0.54, [0.56, 0.57, 0.555], [0.54, 0.54, 0.54])["verdict"] == "NOT ESTABLISHED"
    # mean difference smaller than the larger SD -> not supported
    r = D.decide_h1(0.545, 0.54, [0.50, 0.55, 0.59], [0.50, 0.55, 0.585])
    assert r["verdict"] == "NOT ESTABLISHED", r
    # only one win -> not supported
    assert D.decide_h1(0.55, 0.54, [0.60, 0.50, 0.50], [0.50, 0.55, 0.55])["verdict"] == "NOT ESTABLISHED"


def test_h1_contradicted_when_b_is_clearly_worse_in_most_seeds():
    r = D.decide_h1(0.50, 0.54, [0.50, 0.50, 0.51], [0.54, 0.54, 0.545])
    assert r["verdict"] == "CONTRADICTED" and r["b_losses"] == 3, r


def test_h1_equal_values_are_not_wins():
    r = D.decide_h1(0.55, 0.54, [0.54, 0.54, 0.54], [0.54, 0.54, 0.54])
    assert r["b_wins"] == 0 and r["verdict"] == "NOT ESTABLISHED"


def test_h2_complete_separation_rule():
    assert D.decide_h2([0.60, 0.61, 0.62], [0.50, 0.55, 0.59])["verdict"] == "LEARNED SUPPORTED"
    assert D.decide_h2([0.40, 0.45, 0.49], [0.50, 0.55, 0.59])["verdict"].startswith("HANDCRAFTED AT LEAST AS GOOD")
    assert D.decide_h2([0.49, 0.55, 0.60], [0.50, 0.55, 0.59])["verdict"] == "INCONCLUSIVE"
    # touching ranges are inconclusive (strict separation required)
    assert D.decide_h2([0.50, 0.55, 0.60], [0.45, 0.48, 0.50])["verdict"] == "INCONCLUSIVE"


def test_ablation_noise_flag_uses_the_b_sub_sd():
    t = D.ablation_table(0.56, {"drop_x": 0.555, "drop_y": 0.50}, [0.55, 0.56, 0.57])
    assert abs(t["seed_noise_sd_B_sub"] - 0.01) < 1e-12
    assert t["rows"]["drop_x"]["larger_than_seed_noise"] is False and t["rows"]["drop_y"]["larger_than_seed_noise"] is True


def test_input_validation():
    for fn in (lambda: D.decide_h1(0.5, 0.5, [0.5, 0.5], [0.5, 0.5]), lambda: D.decide_h2([0.5], [0.5, 0.5, 0.5]), lambda: D.sample_sd([0.5])):
        try:
            fn()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


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
