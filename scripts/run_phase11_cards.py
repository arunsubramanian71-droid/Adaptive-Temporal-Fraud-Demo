"""Phase 11 first pass: aggregate-only investigation cards for Phase 9 candidate communities (W = 7).

Reads only the existing label-free Tier 0 community-day CSVs and the TRAIN-calibrated tau locks / E0 results (read-only). Discovery is not
re-run, no label or risk score is read, nothing is installed, TEST is not touched. Full cards for seed 42; selection tables for seeds
123 and 2024 as an appendix. Scope record: experiments/phase11_explainable_evidence/SCOPE_AND_ACCEPTANCE.md (P5, P6).
Usage:  .venv/Scripts/python.exe -B scripts/run_phase11_cards.py
"""
from __future__ import annotations

import datetime
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.explain import cards as C
from src.explain import outputs as O

E = ROOT / "experiments" / "phase9_coordinated_fraud" / "exploratory_realdata"
OUT = ROOT / "experiments" / "phase11_explainable_evidence" / "community_cards"
SEEDS = (42, 123, 2024)


def main() -> int:
    manifest_inputs, hashes, tables, meta_by_seed = {}, {}, [], {}
    a2_diff = 0.0
    for S in SEEDS:
        csv = E / f"tier0_community_days_W{C.W_PRIMARY}_seed{S}.csv"
        lock = E / f"e0_tau_seed{S}.json"
        tl = json.loads(lock.read_text(encoding="utf-8"))
        assert tl["W"] == C.W_PRIMARY and tl["seed"] == S and tl["feasible"] is True, (S, tl["W"])
        tau = float(tl["tau"])
        df = C.load_tier0(csv)
        a2_diff = max(a2_diff, C.a2_max_abs_difference(df))
        groups = C.select_groups(df, tau, S)
        meta_by_seed[S] = {"tau": tau, "groups": groups, "n_tracks": int(df["track"].nunique()), "n_alerted_tracks": len(C.alerted_tracks(df, tau))}
        tables.append(C.selection_rows(df, groups, tau, S))
        for p in (csv, lock):
            manifest_inputs[str(p.relative_to(ROOT)).replace("\\", "/")] = O.sha256_file(p)
        if S == 42:
            m = C.memberships(groups)
            cards = [C.build_card(df, tr, gs, tau) for tr, gs in m.items()]
            e0 = json.loads((E / "e0_results.json").read_text(encoding="utf-8"))["results"]["42"]
            assert abs(e0["tau"] - tau) == 0.0
            meta = {"seed": S, "W": C.W_PRIMARY, "tau": tau, "groups": groups}
            md = C.render_markdown(cards, meta)
            md += ("## E0 descriptive reference (day-level counts only; seed 42, W = 7)\n\n"
                   f"Observed alerted community-days over days 122-151: {e0['A_obs']}. Degree-preserving null (K = {e0['K']} histories): "
                   f"mean {e0['null_summary']['mean']}, SD {e0['null_summary']['sd']:.3f}, min {e0['null_summary']['min']}, "
                   f"median {e0['null_summary']['q50']}, 95th percentile {e0['null_summary']['q95']}, max {e0['null_summary']['max']}; "
                   f"descriptive reference tail proportion {e0['tail']['tail_proportion']:.4f} (minimum attainable "
                   f"{e0['tail']['min_attainable']:.4f}). This is a descriptive reference distribution: it is not exact, not a "
                   "significance test, not a per-community null match, and not evidence of fraud or coordination.\n\n")
            hashes["cards_W7_seed42.json"] = O.write_json(OUT / "cards_W7_seed42.json",
                                                                    {"standing_statement": C.STANDING, "not_established": C.NOT_ESTABLISHED,
                                                                     "meta": meta, "cards": cards})
            hashes["cards_W7_seed42_report.md"] = O.write_text(OUT / "cards_W7_seed42_report.md", md)
            n_cards = len(cards)
    import pandas as pd
    hashes["appendix_selection_all_seeds_W7.csv"] = O.write_csv(OUT / "appendix_selection_all_seeds_W7.csv", pd.concat(tables, ignore_index=True))
    summary = {"scope_record": "experiments/phase11_explainable_evidence/SCOPE_AND_ACCEPTANCE.md", "standing_statement": C.STANDING,
               "W": C.W_PRIMARY, "seeds": list(SEEDS), "full_cards_for_seed": 42, "n_cards_seed_42": n_cards,
               "per_seed": {str(s): v for s, v in meta_by_seed.items()},
               "a2_score_equals_mean_of_u_s_and_u_t_max_abs_difference": a2_diff,
               "unavailable_fields": list(C.UNAVAILABLE_FIELDS), "discovery_rerun": False, "labels_or_risk_used": False}
    hashes["cards_summary.json"] = O.write_json(OUT / "cards_summary.json", summary)
    manifest = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(),
                "inputs_sha256": {**manifest_inputs, str((E / "e0_results.json").relative_to(ROOT)).replace("\\", "/"): O.sha256_file(E / "e0_results.json"),
                                  "experiments/phase11_explainable_evidence/SCOPE_AND_ACCEPTANCE.md":
                                      O.sha256_file(ROOT / "experiments/phase11_explainable_evidence/SCOPE_AND_ACCEPTANCE.md")},
                "outputs_sha256": {f"experiments/phase11_explainable_evidence/community_cards/{k}": v for k, v in hashes.items()},
                "discovery_rerun": False, "labels_or_risk_used": False, "dependencies_installed": False}
    O.write_json(OUT / "manifest_cards.json", manifest)
    print("written:", sorted(hashes), "| a2 diff", a2_diff, "| cards", n_cards)
    return 0


if __name__ == "__main__":
    sys.exit(main())
