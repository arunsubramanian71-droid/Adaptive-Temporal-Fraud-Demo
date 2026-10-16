"""Renders TIER0_REPORT.md from tier0_summary.json (exploratory real-data scope deviation) and lints it. No new computation."""
import hashlib
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.coordination import frozen as F
from src.coordination.lint import lint_text
from src.coordination.realrun import BANNER, out_dir


def f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def main() -> int:
    cfg = load_config()
    od = out_dir(cfg)
    s = json.load(open(od / "tier0_summary.json", encoding="utf-8"))
    L = [f"> {BANNER}", "", "# Tier 0 — descriptive discovery on clean real data (days 122-151), exploratory",
         "", "Generated from `tier0_summary.json` by `scripts/make_phase9_exploratory_report.py`. Everything below is **descriptive**; no statistical test was run "
         "in Tier 0 and nothing here is a statistically supported finding. Label-free: no fraud labels, no model-risk scores, no R_p / R_h, no risk-based baselines. "
         "No alert threshold exists, so these are counts of eligible candidate communities (>= 3 card tokens, >= 1 infrastructure entity), not alerts.",
         "Every community is a *candidate coordinated community; not a confirmed fraud ring*. This does not establish that IEEE-CIS contains fraud rings.", "",
         "## Frozen settings used",
         f"Locked gamma per W = {s['locked_gamma_per_W']} (primary W = {s['primary_W']}; W = 3, 14 pre-registered sensitivity); Louvain seed = S in {{42, 123, 2024}}; "
         "null tables hash-verified against the locked configuration; shared tracker started at day 122; m_min = 3; edge-weight cap 5.", "",
         "## Population at the top of the table (VAL days 122-151)",
         f"{s['population_VAL_n_transactions']:,} transactions; {100*s['population_VAL_share_without_infrastructure_edge']:.1f}% have no netproxy/device edge "
         "(structurally unobservable to infrastructure-based discovery). Fraud base rate of this population is **not computed** (requires labels; outside this authorization).", "",
         "| segment (never merged) | no infrastructure edge | infrastructure edge |", "|---|---|---|"]
    for k, v in s["population_VAL_segment_x_infra_observed"].items():
        L.append(f"| {k} | {v['no_infrastructure_edge']:,} | {v['infrastructure_edge']:,} |")
    L += ["", "## Per configuration and seed",
          "| W | seed | community-day records | candidates/day (median; q05-q95) | cards per candidate (median / q95 / max) | records with >= 17 cards | tracks | track age (median / q95 / max) |",
          "|---|---|---|---|---|---|---|---|"]
    for key, v in s["configs"].items():
        W, S = key.split("_")
        bs = v["size_bucket_counts"]
        big = bs[list(bs)[-1]] / v["n_community_day_records"]
        L.append(f"| {W[1:]} | {S[4:]} | {v['n_community_day_records']} | {v['candidates_per_day']['q50']:.1f}; {v['candidates_per_day']['q05']:.1f}-{v['candidates_per_day']['q95']:.1f} | "
                 f"{v['cards_per_candidate']['q50']:.0f} / {v['cards_per_candidate']['q95']:.0f} / {v['cards_per_candidate']['max']:.0f} | {100*big:.0f}% | {v['n_tracks']} | "
                 f"{v['track_age_days']['q50']:.0f} / {v['track_age_days']['q95']:.1f} / {v['track_age_days']['max']:.0f} |")
    L += ["", "### Null-calibrated component percentiles of the observed candidates (descriptive; NOT a null test)",
          "| W | seed | u_S mean (q05 / q50 / q95) | u_T mean (q05 / q50 / q95) | S+T mean score (q50 / q95) |", "|---|---|---|---|---|"]
    for key, v in s["configs"].items():
        W, S = key.split("_")
        a, b, c = v["score_u_s"], v["score_u_t"], v["score_st"]
        L.append(f"| {W[1:]} | {S[4:]} | {a['mean']:.3f} ({a['q05']:.2f} / {a['q50']:.2f} / {a['q95']:.2f}) | {b['mean']:.3f} ({b['q05']:.2f} / {b['q50']:.2f} / {b['q95']:.2f}) | "
                 f"{c['q50']:.3f} / {c['q95']:.3f} |")
    L += ["", "### Hub, density and conductance diagnostics (descriptive)",
          "| W | seed | hub share mean (q50 / q95) | density mean (q50) | conductance mean (q50) | Q recomputation max abs diff |", "|---|---|---|---|---|---|"]
    for key, v in s["configs"].items():
        W, S = key.split("_")
        h, d, c = v["hub_share"], v["density"], v["conductance"]
        L.append(f"| {W[1:]} | {S[4:]} | {h['mean']:.3f} ({h['q50']:.2f} / {h['q95']:.2f}) | {d['mean']:.3f} ({d['q50']:.2f}) | {c['mean']:.3f} ({c['q50']:.2f}) | {v['q_consistency_max_abs_diff']:.1e} |")
    L += ["", "### Lifecycle (shared tracker; tracks starting on day 122 are left-censored, tracks alive on day 151 right-censored)",
          "| W | seed | tracks | left-censored | right-censored | observable disappearances (last day <= 148) | split flags | merge flags | consecutive-day Jaccard, same track (mean / q50) | records on tracks aged >= 2 days |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for key, v in s["configs"].items():
        W, S = key.split("_")
        j = v["consecutive_day_jaccard_same_track"]
        L.append(f"| {W[1:]} | {S[4:]} | {v['n_tracks']} | {v['tracks_left_censored_first_day_122']} | {v['tracks_right_censored_alive_on_151']} | "
                 f"{v['tracks_with_observable_disappearance']} | {v['tracks_with_split_flag']} | {v['tracks_with_merge_flag']} | {j['mean']:.3f} / {j['q50']:.3f} | {100*v['share_records_on_tracks_age_ge2']:.0f}% |")
    L += ["", "Zero split/merge flags are structural, not a finding: with the frozen Jaccard >= 0.5 continuation rule and disjoint communities on each day, two children can both "
          "reach Jaccard >= 0.5 with one parent only in the degenerate exact-halves case.", "",
          "### Member-transaction composition: segment x infra_observed (transaction-day incidences: a transaction is counted once per day its card is in a community's window)",
          "| W | seed | member txn-days | cold, no infra | cold, infra | warm, no infra | warm, infra | no-edge, no infra | no-edge, infra |", "|---|---|---|---|---|---|---|---|---|"]
    for key, v in s["configs"].items():
        W, S = key.split("_")
        c = v["composition_member_transactions"]
        L.append(f"| {W[1:]} | {S[4:]} | {c['n_member_transactions']:,} | " + " | ".join(f"{100*c[k]:.1f}%" for k in (
            "has_edge_cold__noinfra", "has_edge_cold__infra", "has_edge_warm__noinfra", "has_edge_warm__infra", "no_pseudo_account_edge__noinfra", "no_pseudo_account_edge__infra")) + " |")
    L += ["", "## Reading these numbers (descriptive observations only)",
          "* The frozen pipeline produces roughly 11 eligible candidate communities per day at W = 7 (about 17 at W = 14); about 40-50% of the W = 7 records are communities of 17 or more card tokens "
          "(up to about 500), consistent with the design's recorded base-rate/giant-component facts (F6, F7) that infrastructure sharing is the norm. Their meaning is unknown.",
          "* Most tracks last one day (median age 1 day at every W); a few tracks persist for the whole 30-day period (maximum age 30). Day-to-day identity is therefore unstable for most candidates.",
          "* Observed candidates have mean null-calibrated u_S of about 0.62-0.74 and u_T of about 0.58-0.62 (above 0.5). That is a descriptive hint only: it is **not** E0, uses no permutation test, "
          "depends on the size-bucketed calibration tables (some buckets thin; nearest-bucket fallback), and says nothing about whether the structure is fraudulent.",
          "* Member-transaction composition is not the population composition: no-edge (missing pseudo-account key) transactions are over-represented among communities' transactions "
          "relative to the VAL population, because they almost always carry an infrastructure edge. The three segments are reported separately and were never merged.",
          "* No claim of validity, no recovery metric, no precision/recall, and no statement about H5a/H5b/H5c is made or implied.", "",
          "## Limitations",
          "Single real dataset; seeds vary only the Louvain seed (real data identical across seeds); Louvain community definitions are algorithm-dependent; tracker identity and lifecycle statistics "
          "are censored at both ends of a 30-day window; infrastructure fields are project-defined proxies with unverified semantics; 83% of VAL transactions carry no infrastructure edge.", ""]
    txt = "\n".join(L)
    bad = lint_text(txt)
    assert not bad, bad
    (od / "TIER0_REPORT.md").write_text(txt, encoding="utf-8")
    print("wrote TIER0_REPORT.md; lint clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
