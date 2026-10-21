"""Relationship-level ablation records (experiments/relation_ablation_temporal): pre-registration hashed before the results, every arm recorded, the
verdicts recomputed from the per-seed metrics, the reference equal to the recorded Phase 7 model. Reads recorded files only."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXP = ROOT / "experiments" / "relation_ablation_temporal"
FULL = ROOT / "experiments" / "temporal_graphsage"
RELATIONS = ("charged_to", "used_card", "via_network", "from_device")
SEEDS = (42, 123, 2024)


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def pooled(p) -> float:
    return float(json.loads(pathlib.Path(p).read_text(encoding="utf-8"))["pooled"]["pr_auc"])


def test_preregistration_hashed_before_any_arm_and_recorded():
    pre = EXP / "PREREGISTRATION_RELATION_ABLATION.md"
    assert sha(pre) == pre.with_suffix(".sha256").read_text().strip()
    r = json.loads((EXP / "results.json").read_text(encoding="utf-8"))
    assert r["preregistration_sha256"] == sha(pre) and r["test_data_read"] is False
    first_arm = min(os.stat(EXP / f"drop_{rel}" / f"seed_{s}" / "metrics_val.json").st_mtime for rel in RELATIONS for s in SEEDS)
    assert os.stat(pre).st_mtime <= first_arm
    for n in ("results.json", "RESULTS_RELATION_ABLATION.md"):
        assert sha(EXP / n) == (EXP / n).with_suffix(".sha256").read_text().strip(), n


def test_verdicts_follow_from_the_recorded_per_seed_metrics():
    r = json.loads((EXP / "results.json").read_text(encoding="utf-8"))
    for rel in RELATIONS:
        f = np.array([pooled(FULL / f"seed_{s}" / "metrics_val.json") for s in SEEDS])
        a = np.array([pooled(EXP / f"drop_{rel}" / f"seed_{s}" / "metrics_val.json") for s in SEEDS])
        d = f - a
        bound = max(float(np.std(f, ddof=1)), float(np.std(a, ddof=1)))
        expected = (f"{rel} contributes" if (d > 0).sum() >= 2 and d.mean() > bound else f"removing {rel} does not hurt" if d.mean() <= 0 else "not established")
        x = r["relations"][rel]
        assert np.allclose(x["pooled_pr_auc_full"], f) and np.allclose(x["pooled_pr_auc_without"], a)
        assert x["verdict"] == expected, (rel, x["verdict"], expected)


def test_noise_reference_rerun_is_recorded():
    r = json.loads((EXP / "results.json").read_text(encoding="utf-8"))
    n = r["full_model_rerun_seed42"]
    assert n is not None and abs(n["recorded"] - pooled(FULL / "seed_42" / "metrics_val.json")) < 1e-12
    assert abs(n["rerun"] - pooled(EXP / "full_rerun" / "seed_42" / "metrics_val.json")) < 1e-12


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
