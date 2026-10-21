"""Assemble the comparison-B result files from the recorded arm metrics, the decision record and the Phase 12 evidence cells.

Reads only experiments/comparison_B_graph_features/outputs/{arms,decision.json,run_record.json} and the Phase 12 evidence_cells.csv /
evidence_runs.csv (read-only). Writes B_RESULTS.md and table_with_B.csv (with SHA-256 sidecars) into the B outputs folder. Nothing is
recomputed from data; every number is copied from a recorded file, and the text states the pre-registered verdicts as recorded.

Usage:  .venv/Scripts/python.exe -B scripts/build_b_outputs.py
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

OUT = ROOT / "experiments" / "comparison_B_graph_features" / "outputs"
P12 = ROOT / "experiments" / "phase12_validation_evidence_table" / "outputs"
SEEDS = (42, 123, 2024)
BANNER = ("Validation-period (VAL, days 122-151) results. Comparison A and B use VAL for early stopping and the operating threshold, like every "
          "other model in the table except the F arms. No TEST data was read. Nothing here supports a held-out, significance, causal, fraud or coordination claim.")


def fmt(x, nd=4):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}f}"


def pm(mean, sd, nd=4):
    return f"{fmt(mean, nd)} ± {fmt(sd, nd)}"


def write(path, data: bytes):
    path.write_bytes(data)
    h = hashlib.sha256(data).hexdigest()
    path.with_suffix(".sha256").write_bytes(h.encode("ascii"))
    return h


def arm(name):
    return json.loads((OUT / "arms" / name / "metrics_val.json").read_text(encoding="utf-8"))


def arm_row(name, label, cell):
    m = arm(name)
    seg = m["segments"]
    k = m["pooled"]["top_k"]["1%"]
    return {"cell": cell, "model": label, "basis": "pooled VAL", "runs": 1, "pr_auc": m["pr_auc"], "pr_auc_sd": np.nan, "roc_auc": m["pooled"]["roc_auc"],
            "pr_auc_warm": seg["has_edge_warm"]["pr_auc"], "pr_auc_cold": seg["has_edge_cold"]["pr_auc"], "pr_auc_no_edge": seg["no_pseudo_account_edge"]["pr_auc"],
            "precision_at_1pct": k["precision_at_k"], "recall_at_1pct": k["recall_at_k"], "best_iteration": m["best_iteration"], "source": f"outputs/arms/{name}"}


def sub_row(prefix, label, cell):
    ms = [arm(f"{prefix}_s{s}") for s in SEEDS]
    g = lambda f: np.array([f(m) for m in ms], dtype=float)  # noqa: E731
    sd = lambda a: float(a.std(ddof=1))  # noqa: E731
    pr, roc = g(lambda m: m["pr_auc"]), g(lambda m: m["pooled"]["roc_auc"])
    w, c, n = (g(lambda m, s=s: m["segments"][s]["pr_auc"]) for s in ("has_edge_warm", "has_edge_cold", "no_pseudo_account_edge"))
    p1, r1 = g(lambda m: m["pooled"]["top_k"]["1%"]["precision_at_k"]), g(lambda m: m["pooled"]["top_k"]["1%"]["recall_at_k"])
    return {"cell": cell, "model": label, "basis": "pooled VAL", "runs": 3, "pr_auc": pr.mean(), "pr_auc_sd": sd(pr), "roc_auc": roc.mean(),
            "pr_auc_warm": w.mean(), "pr_auc_cold": c.mean(), "pr_auc_no_edge": n.mean(), "precision_at_1pct": p1.mean(), "recall_at_1pct": r1.mean(),
            "best_iteration": np.nan, "source": f"outputs/arms/{prefix}_s*"}


def phase12_row(family_id, label, cell):
    d = pd.read_csv(P12 / "evidence_cells.csv", float_precision="round_trip")
    r = d[d["family_id"] == family_id]
    assert len(r) == 1, family_id
    r = r.iloc[0]
    return {"cell": cell, "model": label, "basis": "pooled VAL", "runs": int(r["n_runs"]), "pr_auc": r["pr_auc_mean"], "pr_auc_sd": r["pr_auc_sd"],
            "roc_auc": r["roc_auc_mean"], "pr_auc_warm": r["pr_auc_has_edge_warm_mean"], "pr_auc_cold": r["pr_auc_has_edge_cold_mean"],
            "pr_auc_no_edge": r["pr_auc_no_pseudo_account_edge_mean"], "precision_at_1pct": r["precision_at_1pct_mean"], "recall_at_1pct": r["recall_at_1pct_mean"],
            "best_iteration": np.nan, "source": "phase12 evidence_cells.csv"}


def main() -> int:
    dec = json.loads((OUT / "decision.json").read_text(encoding="utf-8"))
    run = json.loads((OUT / "run_record.json").read_text(encoding="utf-8"))
    cells = pd.read_csv(P12 / "evidence_cells.csv", float_precision="round_trip")
    fam = {f: (c, m) for f, c, m in zip(cells["family_id"], cells["cell"], cells["model"])}
    print("phase12 families:", sorted(fam))
    wanted = [("A:logistic_regression", "A"), ("A:random_forest", "A"), ("A:xgboost", "A")]
    rows = [phase12_row(f, fam[f][1], c) for f, c in wanted]
    rows += [arm_row("A_refit", "XGBoost, comparison-A recipe, re-fit in the B pipeline (pipeline control)", "A"),
             sub_row("A_sub", "XGBoost, A features, subsample 0.8 / colsample 0.8 (3 seeds)", "A"),
             arm_row("B_full", "XGBoost, A features + 40 graph features (B, A recipe)", "B"),
             sub_row("B_sub", "XGBoost, B features, subsample 0.8 / colsample 0.8 (3 seeds)", "B")]
    for fid in ("C:graphsage_robust", "Ccontrol:mlp_control_extended", "D:gat", "E:temporal_graphsage", "E7A:static_schedule_matched", "E7A:memory_ablated_temporal"):
        rows.append(phase12_row(fid, fam[fid][1], fam[fid][0]))
    table = pd.DataFrame(rows)
    csv = table.to_csv(index=False, float_format="%.10g", lineterminator="\n").encode("utf-8")
    h_csv = write(OUT / "table_with_B.csv", csv)

    h1, h2, h2c, bad = dec["H1_direct_B_vs_A"], dec["H2_E_vs_B_sub"], dec["H2_secondary_C_vs_B_sub"], dec["BA6_pipeline_control"]
    ps = dec["per_seed"]
    ab = dec["ablations"]
    L = []
    L.append("# Comparison B (tabular + handcrafted graph features): results\n")
    L.append(f"**{BANNER}**\n")
    L.append("Protocol: `PREREGISTRATION_B.md` (written and hashed before any B code or output). Decision rules are applied mechanically by `scripts/run_comparison_b.py` "
             "from recorded values. Verdicts below are those the rules return; they are not adjusted.\n")
    L.append("## Pre-registered verdicts\n")
    L.append("| Question | Verdict | Basis |\n|---|---|---|")
    L.append(f"| H1-direct: do the 40 graph features add to the transaction-level XGBoost? | **{h1['verdict']}** | primary B − A = {h1['primary_diff_B_minus_A']:+.4f}; "
             f"mean seed difference {h1['mean_seed_diff']:+.4f} against the larger SD {h1['larger_sd']:.4f}; B won {h1['b_wins']} of 3 seeds |")
    L.append(f"| H2: do learned graph representations (E, Temporal GraphSAGE) beat B? | **{h2['verdict']}** | E seeds {h2['learned_min']:.4f}–{h2['learned_max']:.4f} "
             f"(mean {h2['learned_mean']:.4f}) versus B_sub {h2['handcrafted_min']:.4f}–{h2['handcrafted_max']:.4f} (mean {h2['handcrafted_mean']:.4f}) |")
    L.append(f"| H2, secondary reading with C (GraphSAGE) | {h2c['verdict']} | C seeds {h2c['learned_min']:.4f}–{h2c['learned_max']:.4f} versus B_sub {h2c['handcrafted_min']:.4f}–{h2c['handcrafted_max']:.4f} |")
    L.append(f"| BA6 pipeline control (A re-fit equals the recorded A) | {'PASS' if bad['within_0.005'] else 'FAIL'} | re-fit {bad['A_refit_pr_auc']:.4f} versus recorded {bad['recorded_A_pr_auc']:.4f} (|diff| {bad['abs_diff']:.4f}, bound 0.005) |")
    L.append("")
    L.append("Per-seed pooled PR-AUC: A_sub " + ", ".join(f"{v:.4f}" for v in ps["A_sub"]) + "; B_sub " + ", ".join(f"{v:.4f}" for v in ps["B_sub"])
             + "; E (recorded) " + ", ".join(f"{v:.4f}" for v in ps["E_recorded"]) + "; C (recorded) " + ", ".join(f"{v:.4f}" for v in ps["C_recorded"]) + " (seeds 42, 123, 2024).\n")
    a_min, e_max = min(ps["A_sub"]), max(ps["E_recorded"])
    arms_all = [q.parent.name for q in (OUT / "arms").glob("*/metrics_val.json")]
    capped = sorted(n for n in arms_all if arm(n)["best_iteration"] >= 990)
    above = ("also above every E seed (lowest A_sub " + format(a_min, ".4f") + " versus highest E " + format(e_max, ".4f") + ")") if a_min > e_max else "not above every E seed"
    L.append("## How to read the verdicts\n")
    L.append("* The H2 rule compares `B_sub` with the recorded neural runs. The arm without graph features, `A_sub` (seeds " + ", ".join(format(v, ".4f") for v in ps["A_sub"])
             + "), is " + above + ". So the H2 outcome mostly reflects the strength of the tuned tabular XGBoost, not a specific benefit of the 40 graph features; the graph features' "
             "own contribution is the H1-direct row.\n"
             "* The subsampling recipe (`*_sub`) scores higher than the comparison-A recipe for both feature sets. Arms that stopped within ten trees of the 1,000-tree cap ("
             + (", ".join(capped) if capped else "none") + ") were still improving, so the cap may have limited them; the cap belongs to the unchanged comparison-A recipe and was not raised.\n")
    L.append("## Table (pooled VAL; cells with several runs show the mean, SD in the CSV)\n")
    L.append("| Cell | Model | Runs | PR-AUC | ROC-AUC | PR-AUC cold-start | PR-AUC warm | PR-AUC no-edge | P@1% | R@1% |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in table.itertuples():
        L.append(f"| {r.cell} | {r.model} | {r.runs} | {pm(r.pr_auc, r.pr_auc_sd) if r.runs > 1 else fmt(r.pr_auc)} | {fmt(r.roc_auc)} | {fmt(r.pr_auc_cold)} | "
                 f"{fmt(r.pr_auc_warm)} | {fmt(r.pr_auc_no_edge)} | {fmt(r.precision_at_1pct)} | {fmt(r.recall_at_1pct)} |")
    L.append("\nF rows (distinct evaluation basis) and the historical-feature reference line are in the Phase 12 table and are not repeated here.\n")
    L.append("## Relationship-level and temporal-feature ablations of B (descriptive; A recipe, seed 42)\n")
    L.append(f"Seed noise (sample SD of `B_sub` PR-AUC): {ab['seed_noise_sd_B_sub']:.4f}. B_full PR-AUC: {arm('B_full')['pr_auc']:.4f}.\n")
    L.append("| Ablation | PR-AUC | Change vs B_full | Larger than seed noise |\n|---|---|---|---|")
    for k, v in ab["rows"].items():
        L.append(f"| {k} | {v['pr_auc']:.4f} | {v['change_vs_B_full']:+.4f} | {'yes' if v['larger_than_seed_noise'] else 'no'} |")
    L.append("")
    L.append("Ablation changes are single fits at one seed: a change inside the noise band cannot be told from zero. These ablations apply to the "
             "handcrafted-feature model only; relationship-level ablations of the neural graph models were not run (each neural fit takes far longer "
             "and the GAT never converged).\n")
    L.append("## Comparability (labels from the pre-registration, section 8)\n")
    L.append("* B versus A: comparable (same rows, labels, metrics and recipe; only the features differ).\n* B versus C, D, E: comparable on the evaluation basis only; "
             "attribution is not supported (different encodings and training regimes; all tuned on VAL).\n* B versus F: not comparable.\n")
    L.append("## Limits\n")
    L.append("* One handcrafted feature set (40 features) fixed before results; a null or negative result does not speak for other handcrafted sets.\n"
             "* The B features use same-entity history with strictly earlier timestamps; the neural graph models see same-day transactions at day granularity.\n"
             "* Window features include the boundary second (a prior transaction exactly 86,400 s or 604,800 s earlier counts), an implementation choice made explicit after the "
             "pre-registration text said \"preceding\".\n"
             "* Everything is tuned on VAL; no held-out claim.\n"
             "* XGBoost is deterministic without subsampling, so seed variation exists only in the `*_sub` arms; those arms use a recipe that differs from comparison A.\n")
    md = ("\n".join(L) + "\n").encode("utf-8")
    h_md = write(OUT / "B_RESULTS.md", md)
    print("wrote table_with_B.csv", h_csv[:12], "B_RESULTS.md", h_md[:12])
    return 0


if __name__ == "__main__":
    sys.exit(main())
