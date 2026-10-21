"""Manuscript figures: present, hashed, drawn from the recorded validation results (figure_data.json equals the source tables), and referenced by the manuscript.
Reads recorded files only; nothing is trained or scored, no held-out file is opened."""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

FIG = ROOT / "submission" / "figures"
NAMES = ("fig1_validation_pr_auc.png", "fig2_drift_adaptation.png", "fig3_class_weighting.png")


def test_figures_exist_are_png_files_and_match_their_sidecars():
    for n in NAMES + ("figure_data.json",):
        p = FIG / n
        assert p.exists() and p.stat().st_size > 1000, n
        assert hashlib.sha256(p.read_bytes()).hexdigest() == p.with_suffix(".sha256").read_text().strip(), n
    for n in NAMES:
        assert (FIG / n).read_bytes()[:4] == b"\x89PNG", n


def test_figure_data_equals_the_recorded_source_tables():
    d = json.loads((FIG / "figure_data.json").read_text(encoding="utf-8"))
    assert "no held-out data" in d["statement"]
    tb = pd.read_csv(ROOT / "experiments/comparison_B_graph_features/outputs/table_with_B.csv")
    tb = tb[~tb["model"].str.contains("pipeline control")].reset_index(drop=True)
    assert [r["pr_auc"] for r in d["fig1"]] == [float(v) for v in tb["pr_auc"]]
    arms = pd.read_csv(ROOT / "experiments/imbalance_check_xgboost/outputs/arms_table.csv")
    assert [r["pr_auc"] for r in d["fig3"]] == [float(v) for v in arms["pr_auc"]]
    # window means recorded for the three Phase 8 arms (mean of seeds), first window equal across arms within a seed
    assert len(d["fig2"]["recorded_phase8"]["Ad"]) == 5 and len(d["fig2"]["memory_built_first"]["F"]) == 5


def test_manuscript_references_every_figure():
    t = (ROOT / "submission" / "MANUSCRIPT.md").read_text(encoding="utf-8")
    for n in NAMES:
        assert f"figures/{n}" in t, n


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
