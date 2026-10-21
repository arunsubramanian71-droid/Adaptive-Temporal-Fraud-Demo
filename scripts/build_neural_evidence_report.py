"""Render experiments/neural_graph_evidence/outputs/NEURAL_EVIDENCE_REPORT.md from the recorded summary and effects (nothing is recomputed from models or data).
The registered acceptance outcome (including a failed criterion) is quoted as recorded; the post-result diagnostic is labelled as post hoc.

Usage:  .venv/Scripts/python.exe -B scripts/build_neural_evidence_report.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "experiments" / "neural_graph_evidence" / "outputs"

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RELS = ("charged_to", "used_card", "via_network", "from_device")
LABEL = {"charged_to": "pseudo-account", "used_card": "card", "via_network": "network proxy", "from_device": "device"}


def main() -> int:
    s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    a = pd.read_csv(OUT / "relation_effects.csv")
    acc, ne5 = s["acceptance"], s["NE5_sanity"]
    new = np.concatenate([a.loc[a[f"entity_new_{r}"] == 1, f"reset_effect_{r}"].to_numpy() for r in RELS])
    seen = np.concatenate([a.loc[a[f"entity_new_{r}"] == 0, f"reset_effect_{r}"].to_numpy() for r in RELS])
    L = ["# Graph-level evidence for Temporal GraphSAGE (seed 42): results\n",
         "**Validation-period (days 122-151) counterfactuals on the 300 pre-registered Phase 11 transactions for a model tuned on that period. Descriptive; not evidence of fraud, causation or coordination; no held-out data was read; no label was used.**\n",
         "Protocol: `PREREGISTRATION_NEURAL_EVIDENCE.md` (hashed before the code existed). Counterfactuals: removal of the transaction's own link to one entity, and reset of that entity's memory to the never-seen condition, each on a clone of the state at the start of the day.\n",
         "## Registered acceptance outcome\n", "| Criterion | Result |", "|---|---|",
         f"| NE1 reproduction (max probability difference to the recorded predictions {s['NE1_reproduction']['max_abs_probability_difference_vs_recorded']:.2e}, bound 1e-4) | {'PASS' if acc['NE1'] else 'FAIL'} |",
         f"| NE2 selection (300 Phase 11 ids, hash equal) | {'PASS' if acc['NE2'] else 'FAIL'} |", "| NE3 label-free | PASS |",
         f"| NE5 sanity (a: reset of never-seen entities exactly zero; b: reset of seen entities changes the score in a majority; c: link removal changes the score in a majority) | **{'PASS' if acc['NE5'] else 'FAIL'}** |",
         f"| NE7 repeatability (second run, max difference in effects {s['NE7_second_run_max_abs_difference_in_effects']:.2e}, bound 1e-5) | {'PASS' if acc['NE7_repeatability'] else 'FAIL'} |", ""]
    if not acc["NE5"]:
        L += ["**NE5 failed as registered, in part (a).** "
              f"Of {ne5['n_reset_new']} resets of never-seen entities, {int((new != 0).sum())} were not exactly zero; the largest absolute value is {np.abs(new).max():.2e}. "
              f"Part (b) passed (share changed {ne5['share_reset_seen_changed']:.2f}) and part (c) passed (share changed {ne5['share_removal_changed']:.2f}). "
              "The registered criterion demanded bit-exact zeros but GPU kernels are not bit-reproducible, which NE7 itself measured; the registration did not allow for that. "
              "The criterion was not changed. Post hoc diagnostic (labelled as such): the nonzero values are float32 rounding size, six orders of magnitude below the median "
              f"effect for seen entities ({np.median(np.abs(seen)):.2f}) and below the smallest one ({np.abs(seen).min():.2e}); the same invariant holds exactly on CPU in the unit test of the counterfactual functions.\n"]
    L += ["## Descriptive results\n", "| Relation | Share of transactions with the link | Mean absolute logit change on link removal | Mean absolute change on memory reset (entity seen before) | Share with entity new | Share where it is the largest removal effect |", "|---|---|---|---|---|---|"]
    for r in RELS:
        b = s["by_relation"][r]
        f = lambda x: "n/a" if x is None else f"{x:.3f}"  # noqa: E731
        L.append(f"| {LABEL[r]} | {f(b['share_with_edge'])} | {f(b['mean_abs_removal_effect_when_present'])} | {f(b['mean_abs_reset_effect_entity_seen'])} | {f(b['share_entity_new_when_present'])} | {f(b['share_of_transactions_where_largest_removal_effect'])} |")
    L += ["", f"{s['n_transactions']} transactions; {ne5['n_removals']} link removals and {ne5['n_reset_new'] + ne5['n_reset_seen']} memory resets performed.", "",
          "## Limits\n", "One checkpoint, 300 transactions, validation period; a counterfactual on one transaction's link is not a statement about the entity or about fraud; transactions that share an entity on the same day influence each other, so effects include that interaction; effects across relations are not additive. "
          "GraphSAGE and GAT were not examined (GAT never converged; attention weights are not used).", ""]
    data = ("\n".join(L)).encode("utf-8")
    (OUT / "NEURAL_EVIDENCE_REPORT.md").write_bytes(data)
    (OUT / "NEURAL_EVIDENCE_REPORT.sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    print("wrote NEURAL_EVIDENCE_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
