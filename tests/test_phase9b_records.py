"""Phase 9b records (experiments/phase9b_recovery_benchmark): pre-registration hashed before the outputs, gates evaluated before discovery, the decision
recomputed from the recorded per-seed AUPRC, the original Phase 9 result untouched. Reads recorded files only."""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

EXP = ROOT / "experiments" / "phase9b_recovery_benchmark"
P9 = ROOT / "experiments" / "phase9_coordinated_fraud"
SEEDS = ("42", "123", "2024")


def sha(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def res():
    return json.loads((EXP / "outputs" / "results.json").read_text(encoding="utf-8"))


def test_preregistration_hashed_before_the_results_and_recorded_in_them():
    pre = EXP / "PREREGISTRATION_9B.md"
    assert sha(pre) == pre.with_suffix(".sha256").read_text().strip() == res()["preregistration_sha256"]
    assert os.stat(pre).st_mtime <= os.stat(EXP / "outputs" / "results.json").st_mtime
    for n in ("results.json", "RESULTS_9B.md"):
        p = EXP / "outputs" / n
        assert sha(p) == p.with_suffix(".sha256").read_text().strip(), n


def test_all_registered_gates_passed_and_the_original_formula_is_reported_not_used():
    r = res()
    assert r["all_gates_pass"] is True and r["test_data_read"] is False and r["max_day"] <= 151
    for s in SEEDS:
        g = r["gates"][s]
        assert g["G1_pass"] and g["G2_pass"] and g["G3_pass"] and g["G5_pass"] and g["G4_9b_pass"]
        assert g["G4_9b_mean_abs_delta_pct_points"] <= 0.5 and g["generator_seed"] == 70000 + int(s)
        assert "G4_original_formula_mean_abs_delta_pct_points_descriptive" in g


def test_decision_recomputed_from_the_recorded_auprc_matches_the_recorded_outcome():
    r = res()
    ap = r["track_auprc"]
    a2 = np.array([ap["A2"]["alpha0"][s]["ap"] for s in SEEDS])
    met_all = True
    for b in ("B1", "B2", "B3a", "B3b"):
        x = np.array([ap[b]["alpha0"][s]["ap"] for s in SEEDS])
        wins, diff = int((a2 > x).sum()), float(a2.mean() - x.mean())
        bound = max(float(np.std(a2, ddof=1)), float(np.std(x, ddof=1)))
        met = wins >= 2 and diff > 0 and diff > bound
        c = r["comparisons_alpha0"][b]
        assert c["A2_wins_of_3"] == wins and abs(c["mean_diff"] - diff) < 1e-12 and c["met"] == met
        met_all &= met
    gcal = all(r["g_cal"][s]["false_alerts_per_day"] <= 3.0 for s in SEEDS)
    expected = "H5a-9b SUPPORTED" if met_all and gcal else "H5a-9b NOT SUPPORTED"
    assert r["outcome"] == expected


def test_original_phase9_result_and_lock_are_untouched():
    g4 = json.loads((P9 / "eval_benchmark_G1_G2_G4.json").read_text(encoding="utf-8"))
    assert g4["all_passed"] is False and g4["42"]["G4_pass"] is False and g4["2024"]["G4_pass"] is False
    assert sha(P9 / "locked_config.json") == (P9 / "locked_config.sha256").read_text().strip()
    closure = (ROOT / "reports" / "phase9_closure_report.md").read_text(encoding="utf-8")
    assert "H5a is NOT ESTABLISHED under the original protocol" in closure
    text = (EXP / "outputs" / "RESULTS_9B.md").read_text(encoding="utf-8")
    assert "remains **NOT ESTABLISHED**" in text and "H5a-9b" in text


def test_wording_lint_clean():
    from src.coordination.lint import lint_files
    files = [EXP / "PREREGISTRATION_9B.md", EXP / "outputs" / "RESULTS_9B.md", ROOT / "scripts" / "run_phase9b_benchmark.py", ROOT / "scripts" / "build_phase9b_report.py"]
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
