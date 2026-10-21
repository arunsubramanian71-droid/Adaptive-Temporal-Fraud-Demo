"""Apply the registered rules of PREREGISTRATION_UNWEIGHTED_FOLLOWUP.md to the recorded unweighted arms and write the result files.

Reads only recorded outputs (the unweighted XGBoost and neural arms, the weighted arms of comparison B and the Phase 12 per-run table).
Usage:  .venv/Scripts/python.exe -B scripts/build_unweighted_results.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.features import b_decision as D  # noqa: E402

SEEDS = (42, 123, 2024)
IM = ROOT / "experiments" / "imbalance_check_xgboost" / "outputs"
BO = ROOT / "experiments" / "comparison_B_graph_features" / "outputs" / "arms"
BANNER = ("Validation-period (VAL, days 122-151) results; VAL used for early stopping and the best-F1 threshold; no TEST data was read. Weighted and unweighted "
          "results are separate readings under different recipes; the recorded weighted results are unchanged.")


def j(p):
    return json.loads(pathlib.Path(p).read_text(encoding="utf-8"))


def effect(unweighted, weighted):
    lo_u, hi_u, lo_w, hi_w = min(unweighted), max(unweighted), min(weighted), max(weighted)
    verdict = "UNWEIGHTED BETTER" if lo_u > hi_w else "WEIGHTED BETTER" if hi_u < lo_w else "NO CLEAR DIFFERENCE"
    return {"verdict": verdict, "unweighted": list(map(float, unweighted)), "weighted": list(map(float, weighted)),
            "unweighted_mean": float(np.mean(unweighted)), "weighted_mean": float(np.mean(weighted))}


def write(path, data: bytes):
    path.write_bytes(data)
    path.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))


def main() -> int:
    runs = pd.read_csv(ROOT / "experiments/phase12_validation_evidence_table/outputs/evidence_runs.csv", float_precision="round_trip")
    rec = lambda prefix, cell: runs[(runs["cell"] == cell) & runs["model"].str.startswith(prefix)].sort_values("seed")["pr_auc"].to_numpy()  # noqa: E731
    a_sub = [j(BO / f"A_sub_s{s}" / "metrics_val.json")["pr_auc"] for s in SEEDS]
    b_sub = [j(BO / f"B_sub_s{s}" / "metrics_val.json")["pr_auc"] for s in SEEDS]
    a_none_sub = [j(IM / "arms_unweighted" / f"A_none_sub_s{s}" / "metrics_val.json")["pr_auc"] for s in SEEDS]
    b_none_sub = [j(IM / "arms_unweighted" / f"B_none_sub_s{s}" / "metrics_val.json")["pr_auc"] for s in SEEDS]
    a_none = j(IM / "arms" / "none" / "metrics_val.json")["pr_auc"]
    b_none = j(IM / "arms_unweighted" / "B_none" / "metrics_val.json")["pr_auc"]
    c_none = [j(ROOT / f"experiments/unweighted_graphsage/seed_{s}/metrics_val.json")["pooled"]["pr_auc"] for s in SEEDS]
    e_none = [j(ROOT / f"experiments/unweighted_temporal_graphsage/seed_{s}/metrics_val.json")["pooled"]["pr_auc"] for s in SEEDS]
    c_w, e_w = rec("GraphSAGE (Phase 6A)", "C"), rec("Temporal GraphSAGE (GRU", "E")
    dec = {
        "rule1_weighting_effect": {"XGBoost A features (sub)": effect(a_none_sub, a_sub), "XGBoost B features (sub)": effect(b_none_sub, b_sub),
                                   "GraphSAGE": effect(c_none, c_w), "Temporal GraphSAGE": effect(e_none, e_w)},
        "rule2_H1_direct_unweighted": D.decide_h1(b_none, a_none, b_none_sub, a_none_sub),
        "rule3_H2_unweighted_E": D.decide_h2(e_none, b_none_sub), "rule3_H2_unweighted_C": D.decide_h2(c_none, b_none_sub),
        "rule4_ordering": {"lowest_tabular_unweighted_seed": float(min(a_none_sub + b_none_sub)), "highest_neural_unweighted_seed": float(max(c_none + e_none)),
                           "tabular_above_every_neural_seed": bool(min(a_none_sub + b_none_sub) > max(c_none + e_none))},
        "values": {"A_none": a_none, "B_none": b_none, "A_none_sub": a_none_sub, "B_none_sub": b_none_sub, "C_none": c_none, "E_none": e_none},
    }
    write(IM / "unweighted_decision.json", (json.dumps(dec, indent=1) + "\n").encode("utf-8"))
    L = ["# Unweighted follow-up: results\n", f"**{BANNER}**\n", "Protocol: `PREREGISTRATION_UNWEIGHTED_FOLLOWUP.md`. Rules are applied mechanically from recorded values.\n",
         "## Rule 1: effect of class weighting (pooled PR-AUC, three seeds each)\n", "| Family | Unweighted seeds | Weighted seeds (recorded) | Verdict |", "|---|---|---|---|"]
    for k, v in dec["rule1_weighting_effect"].items():
        L.append(f"| {k} | {', '.join(f'{x:.4f}' for x in v['unweighted'])} (mean {v['unweighted_mean']:.4f}) | {', '.join(f'{x:.4f}' for x in v['weighted'])} (mean {v['weighted_mean']:.4f}) | **{v['verdict']}** |")
    h1, h2e, h2c, r4 = dec["rule2_H1_direct_unweighted"], dec["rule3_H2_unweighted_E"], dec["rule3_H2_unweighted_C"], dec["rule4_ordering"]
    L += ["", "## Rules 2-4: the registered comparisons under no class weighting\n", "| Question | Verdict under no weighting | Basis |", "|---|---|---|",
          f"| H1-direct (B versus A) | **{h1['verdict']}** | primary B − A = {h1['primary_diff_B_minus_A']:+.4f}; mean seed difference {h1['mean_seed_diff']:+.4f} against the larger SD {h1['larger_sd']:.4f}; B won {h1['b_wins']} of 3 seeds |",
          f"| H2 (learned E versus handcrafted B_none_sub) | **{h2e['verdict']}** | E_none {h2e['learned_min']:.4f}–{h2e['learned_max']:.4f}; B_none_sub {h2e['handcrafted_min']:.4f}–{h2e['handcrafted_max']:.4f} |",
          f"| H2 secondary (C_none) | {h2c['verdict']} | C_none {h2c['learned_min']:.4f}–{h2c['learned_max']:.4f} |",
          f"| Tabular above every neural seed? | **{'yes' if r4['tabular_above_every_neural_seed'] else 'no'}** | lowest tabular unweighted seed {r4['lowest_tabular_unweighted_seed']:.4f}; highest neural unweighted seed {r4['highest_neural_unweighted_seed']:.4f} |",
          "", "Single-fit values: A_none (seed 42, comparison-A recipe) {:.4f}; B_none {:.4f}.".format(a_none, b_none), "",
          "Scope: GAT was not re-run (never converged; about 12 GiB of GPU memory). Three seeds sharing the data are not independent replications. These readings sit beside, and do not replace, the recorded weighted results.", ""]
    write(IM / "UNWEIGHTED_RESULTS.md", ("\n".join(L)).encode("utf-8"))
    print(json.dumps({k: (v["verdict"] if isinstance(v, dict) and "verdict" in v else None) for k, v in dec.items() if k.startswith("rule")}, indent=1))
    print({k: v["verdict"] for k, v in dec["rule1_weighting_effect"].items()}, r4)
    return 0


if __name__ == "__main__":
    sys.exit(main())
