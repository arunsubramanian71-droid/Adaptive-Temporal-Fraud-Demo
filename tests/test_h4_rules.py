"""The H4 decision function: equality with the recorded Phase 8 verdict, and behaviour on synthetic inputs."""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.heldout import h4 as H  # noqa: E402

P8 = ROOT / "experiments" / "phase8_adaptive_drift"


def recorded():
    arms = {"F": "frozen", "Ad": "adaptive", "E": "replay_control"}
    return {a: {s: [json.loads((P8 / d / f"seed_{s}" / f"metrics_val_W{k}.json").read_text(encoding="utf-8"))["pooled"]["pr_auc"] for k in range(1, 6)] for s in H.SEEDS}
            for a, d in arms.items()}


def test_function_reproduces_the_recorded_phase_8_h4_decision_and_numbers():
    r = H.h4_decision(recorded())
    assert r["h4"] == "SUPPORTED" and r["rule1_ad_degrades_less_in_2_of_3_seeds"]["per_seed_less_degraded"] == [True, True, True]
    assert abs(r["rule2_mean_ad_minus_f"]["mean_diff"] - 0.0338) < 5e-5 and abs(r["rule2_mean_ad_minus_f"]["larger_sd"] - 0.0128) < 5e-5
    assert abs(r["rule3_mean_ad_minus_e"]["mean_diff"] - 0.0402) < 5e-5 and abs(r["rule3_mean_ad_minus_e"]["larger_sd"] - 0.0082) < 5e-5
    rec = json.loads((P8 / "comparison.json").read_text(encoding="utf-8"))
    assert rec["h4_decision"] == "supported"


def test_each_rule_can_fail_independently():
    base = recorded()
    flat = {a: {s: [0.4] * 5 for s in H.SEEDS} for a in base}
    assert H.h4_decision(flat)["h4"].startswith("NOT ESTABLISHED")
    no_rule2 = {a: {s: list(v) for s, v in base[a].items()} for a in base}
    for s in H.SEEDS:
        no_rule2["Ad"][s] = [x - 0.2 for x in base["Ad"][s]]
    r = H.h4_decision(no_rule2)
    assert r["rule2_mean_ad_minus_f"]["result"] is False and r["h4"].startswith("NOT")
    worse_than_replay = {a: {s: list(v) for s, v in base[a].items()} for a in base}
    for s in H.SEEDS:
        worse_than_replay["E"][s] = [x + 0.5 for x in base["E"][s]]
    assert H.h4_decision(worse_than_replay)["rule3_mean_ad_minus_e"]["result"] is False


def test_input_validation():
    base = recorded()
    bad = {a: dict(v) for a, v in base.items()}
    bad["F"].pop(42)
    for broken in (bad, {**base, "Ad": {s: [0.1] * 4 for s in H.SEEDS}}):
        try:
            H.h4_decision(broken)
        except ValueError:
            continue
        raise AssertionError("invalid input accepted")


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
