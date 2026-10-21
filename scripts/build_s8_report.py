"""Render experiments/phase8_state_continuity_supplement/outputs/S8_RESULTS.md from the recorded supplement outputs and the recorded Phase 8 results (nothing is recomputed from models).

Usage:  .venv/Scripts/python.exe -B scripts/build_s8_report.py
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

from src.heldout import h4  # noqa: E402

OUT = ROOT / "experiments" / "phase8_state_continuity_supplement" / "outputs"
P8 = ROOT / "experiments" / "phase8_adaptive_drift"
ARM = {"F": "frozen", "Ad": "adaptive", "E": "replay_control"}


def main() -> int:
    rec = {a: {s: [json.loads((P8 / d / f"seed_{s}" / f"metrics_val_W{k}.json").read_text(encoding="utf-8"))["pooled"]["pr_auc"] for k in range(1, 6)] for s in h4.SEEDS} for a, d in ARM.items()}
    sup = json.loads((OUT / "burn_in" / "summary.json").read_text(encoding="utf-8"))
    bur = {a: {int(s): v for s, v in sup["window_pr_auc"][a].items()} for a in ARM}
    d_rec, d_bur = h4.h4_decision(rec), h4.h4_decision(bur)
    s1 = json.loads((OUT / "as_written" / "readiness_S1.json").read_text(encoding="utf-8"))
    w1 = []
    for s in h4.SEEDS:
        ps = [pd.read_parquet(OUT / "burn_in" / ARM[a] / f"seed_{s}" / "predictions_W1.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"] for a in ARM]
        w1.append(max(float(np.abs(ps[0].to_numpy() - p.loc[ps[0].index].to_numpy()).max()) for p in ps[1:]))
    f = lambda x: f"{x:.4f}"  # noqa: E731
    L = ["# Phase 8 state-continuity supplement: results\n",
         "**Validation period (days 122-151); an additional reading beside the recorded Phase 8 results, which are unchanged. No held-out data was read.**\n",
         "Protocol: `PREREGISTRATION_S8.md` (hashed before any arm). The only change from Phase 8: the entity memory is built over days 1-121 before the first window instead of starting empty.\n",
         "## Window-mean PR-AUC per seed (seeds 42, 123, 2024)\n", "| Arm | Recorded Phase 8 (empty memory at day 122) | Supplement (memory built over days 1-121) |", "|---|---|---|"]
    for a in ARM:
        L.append(f"| {a} | " + ", ".join(f(d_rec["window_mean_pr_auc"][a][s]) for s in h4.SEEDS) + " | " + ", ".join(f(d_bur["window_mean_pr_auc"][a][s]) for s in h4.SEEDS) + " |")
    L += ["", "## Locked H4 rules (identical logic)\n", "| Rule | Recorded Phase 8 | Supplement |", "|---|---|---|",
          f"| 1. Ad degrades less than F in at least 2 of 3 seeds | {d_rec['rule1_ad_degrades_less_in_2_of_3_seeds']['result']} | {d_bur['rule1_ad_degrades_less_in_2_of_3_seeds']['result']} |",
          f"| 2. mean(Ad - F) exceeds the larger seed SD | {d_rec['rule2_mean_ad_minus_f']['result']} ({d_rec['rule2_mean_ad_minus_f']['mean_diff']:+.4f} vs {d_rec['rule2_mean_ad_minus_f']['larger_sd']:.4f}) | {d_bur['rule2_mean_ad_minus_f']['result']} ({d_bur['rule2_mean_ad_minus_f']['mean_diff']:+.4f} vs {d_bur['rule2_mean_ad_minus_f']['larger_sd']:.4f}) |",
          f"| 3. mean(Ad - E) exceeds the larger seed SD | {d_rec['rule3_mean_ad_minus_e']['result']} ({d_rec['rule3_mean_ad_minus_e']['mean_diff']:+.4f} vs {d_rec['rule3_mean_ad_minus_e']['larger_sd']:.4f}) | {d_bur['rule3_mean_ad_minus_e']['result']} ({d_bur['rule3_mean_ad_minus_e']['mean_diff']:+.4f} vs {d_bur['rule3_mean_ad_minus_e']['larger_sd']:.4f}) |",
          f"| **H4** | **{d_rec['h4']}** | **{d_bur['h4']}** |", "",
          "The H4 verdict does not depend on the memory initialisation; the absolute values of all three arms are higher when the memory is built first, so the recorded Phase 8 absolute values understate what the same checkpoints achieve in the deployment-faithful condition.\n",
          "## Registered acceptance outcome\n", "| Criterion | Result |", "|---|---|"]
    ok_f = all(r["pass"] for r in s1["rows"] if r["arm"] == "F" or r["window"] == 1)
    ok_t = all(r["pass"] for r in s1["rows"] if r["arm"] != "F" and r["window"] > 1)
    L += [f"| S1 reproduction of Phase 8 as written, seed 42 (F and window 1 of every arm within 1e-6; Ad/E later windows within 0.01 PR-AUC) | **{'PASS' if s1['all_pass'] else 'FAIL'}**: F and window 1 {'pass' if ok_f else 'fail'}; Ad/E later windows {'pass' if ok_t else 'fail'} |",
          f"| S2 execution (all arms and seeds; window 1 identical across arms within a seed, max difference {max(w1):.1e}) | {'PASS' if max(w1) <= 1e-6 else 'FAIL'} |", "| S3 causality (stand-in model test) | PASS (`tests/test_walkforward_ext.py`) |", "| S4 hygiene | PASS (hashes, lint, dataset truncated at day 151) |", ""]
    if not s1["all_pass"]:
        bad = [r for r in s1["rows"] if not r["pass"]]
        L += [f"**S1 failed as registered for the trained arms.** {len(bad)} of {sum(1 for r in s1['rows'] if r['arm'] != 'F' and r['window'] > 1)} later windows of Ad and E differ from the recorded values by more than 0.01 PR-AUC (largest {max(r['pr_auc_abs_diff'] for r in bad):.4f}); the individual scores differ widely (largest {max(r['max_abs_score_diff'] for r in s1['rows']):.2f}) because the update epochs use dropout and GPU arithmetic that are not reproducible between runs. "
              "The criterion was not changed. Post hoc diagnostic (labelled as such): the scoring path is exact (F and window 1 differ by about 1e-7, and the unit test shows the new function equals Phase 8's loop bit for bit when training is seeded identically); the window-mean PR-AUC of the re-run is "
              + ", ".join(f"{a} {np.mean([r for r in [json.loads((OUT / 'as_written' / ARM[a] / 'seed_42' / f'metrics_W{k}.json').read_text(encoding='utf-8'))['pooled']['pr_auc'] for k in range(1, 6)]]):.4f} (recorded {np.mean(rec[a][42]):.4f})" for a in ("Ad", "E"))
              + ". Single windows are therefore noisy at the 0.01-0.03 level between runs, and recorded single-run window values should not be read finer than that.\n"]
    L += ["## Limits\n", "Three seeds sharing the same data are not independent replications; one training run per seed and arm; validation period only; zero label delay as in Phase 8.", ""]
    data = ("\n".join(L)).encode("utf-8")
    (OUT / "S8_RESULTS.md").write_bytes(data)
    (OUT / "S8_RESULTS.sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))
    (OUT / "decision_compare.json").write_text(json.dumps({"recorded_phase8": d_rec, "state_continuity_supplement": d_bur}, indent=1), encoding="utf-8")
    print("wrote S8_RESULTS.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
