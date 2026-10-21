"""Phase 11 first pass: aggregate-only investigation cards for Phase 9 candidate communities (scope record P5, P6).

Inputs are the existing, label-free Tier 0 community-day CSVs and the TRAIN-calibrated tau locks. Discovery is not re-run and no missing
field is fabricated: every card lists what the aggregates cannot provide. Every card carries the standing statement.
Candidate coordinated community; not a confirmed fraud ring (wording rule of the project; see the scope record).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

STANDING = "candidate coordinated community; not a confirmed fraud ring"
NOT_ESTABLISHED = "This does not establish that IEEE-CIS contains fraud rings."
DAYS = (122, 151)
W_PRIMARY = 7
EXPECTED_COLUMNS = [
    "day", "track", "n_cards", "n_infra", "q", "q_recomputed", "w_in", "edges_in", "density", "hub_share", "conductance", "u_s",
    "t_raw", "u_t", "score_st", "n_member_txn", "p90_window_degree",
    "txn_has_edge_cold__noinfra", "txn_has_edge_cold__infra", "txn_has_edge_warm__noinfra", "txn_has_edge_warm__infra",
    "txn_no_pseudo_account_edge__noinfra", "txn_no_pseudo_account_edge__infra",
]
SEGMENT_COLUMNS = EXPECTED_COLUMNS[-6:]
TIMELINE_COLUMNS = ["day", "n_cards", "n_infra", "density", "hub_share", "conductance", "u_s", "u_t", "score_st", "n_member_txn"]
GROUPS = ("a_top_peak_score", "b_largest", "c_alerted_sample", "d_hub_dominated_high_s")
UNAVAILABLE_FIELDS = (
    "member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run)",
    "new-member counts and daily active members",
    "pseudo-account counts",
    "descriptive attributes (ProductCD, card4, card6, email domain)",
    "risk percentile distribution and learned-state signature (no risk stream was created; not authorized)",
    "any label-based column (retrospective label rate is not authorized)",
    "per-community null-matched comparison (E0 gives day-level reference counts only)",
)
SAMPLE_SEED_BASE = 40000        # Phase 9 case-study sampling seed 40000 + S


def load_tier0(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected Tier 0 columns: {list(df.columns)}")
    if df["day"].min() < DAYS[0] or df["day"].max() > DAYS[1]:
        raise ValueError("Tier 0 file contains days outside 122-151")
    if df.duplicated(["day", "track"]).any():
        raise ValueError("duplicate (day, track) rows")
    return df


def a2_max_abs_difference(df) -> float:
    """score_st is the A2 score: the mean of u_s and u_t (checked, not assumed)."""
    return float(np.abs(df["score_st"] - (df["u_s"] + df["u_t"]) / 2.0).max())


def track_table(df) -> pd.DataFrame:
    g = df.groupby("track")
    t = pd.DataFrame({
        "first_day": g["day"].min(), "last_day": g["day"].max(), "n_days": g["day"].count(),
        "peak_score_st": g["score_st"].max(), "peak_n_cards": g["n_cards"].max(),
        "peak_u_s": g["u_s"].max(), "peak_hub_share": g["hub_share"].max(),
    }).reset_index()
    return t.sort_values("track").reset_index(drop=True)


def _rank_position(values, tracks):
    """Ordinal rank 0..n-1 by value descending, ties by track id ascending."""
    order = np.lexsort((np.asarray(tracks), -np.asarray(values, dtype=np.float64)))
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    return rank


def alerted_tracks(df, tau) -> list[int]:
    """Tracks with at least one community-day with score_st >= tau (ties at tau are alerts, as in E0)."""
    return sorted(int(x) for x in df.loc[df["score_st"] >= tau, "track"].unique())


def select_groups(df, tau, seed) -> dict:
    """Section 16 groups (a)-(d). Returns {group: [track ids]}. Deterministic; uses no label and no model risk."""
    t = track_table(df)
    tr = t["track"].to_numpy()
    a = t.sort_values(["peak_score_st", "track"], ascending=[False, True])["track"].head(10).tolist()
    b = t.sort_values(["peak_n_cards", "track"], ascending=[False, True])["track"].head(5).tolist()
    alerted = alerted_tracks(df, tau)
    if alerted:
        rng = np.random.default_rng(SAMPLE_SEED_BASE + int(seed))
        pick = rng.choice(len(alerted), size=min(5, len(alerted)), replace=False)
        c = sorted(alerted[i] for i in pick)
    else:
        c = []
    key = _rank_position(t["peak_u_s"], tr) + _rank_position(t["peak_hub_share"], tr)
    d = t.assign(_k=key).sort_values(["_k", "track"])["track"].head(5).tolist()
    return {"a_top_peak_score": [int(x) for x in a], "b_largest": [int(x) for x in b],
            "c_alerted_sample": [int(x) for x in c], "d_hub_dominated_high_s": [int(x) for x in d]}


def memberships(groups) -> dict:
    out: dict = {}
    for g in GROUPS:
        for tr in groups[g]:
            out.setdefault(int(tr), []).append(g)
    return {k: out[k] for k in sorted(out)}


def build_card(df, track, member_groups, tau) -> dict:
    rows = df[df["track"] == track].sort_values("day")
    first, last = int(rows["day"].min()), int(rows["day"].max())
    timeline = []
    for r in rows.to_dict("records"):
        timeline.append({**{c: (int(r[c]) if c in ("day", "n_cards", "n_infra", "n_member_txn") else float(r[c])) for c in TIMELINE_COLUMNS},
                         "alerted": bool(r["score_st"] >= tau)})
    comp = {c: int(rows[c].sum()) for c in SEGMENT_COLUMNS}
    return {
        "track": int(track), "groups": list(member_groups), "standing_statement": STANDING,
        "first_observed_day": first, "last_observed_day": last, "n_observed_days": int(len(rows)),
        "left_censored": first == DAYS[0], "right_censored": last == DAYS[1],
        "peak": {"score_st": float(rows["score_st"].max()), "u_s": float(rows["u_s"].max()), "u_t": float(rows["u_t"].max()),
                 "n_cards": int(rows["n_cards"].max()), "n_infra": int(rows["n_infra"].max()), "hub_share": float(rows["hub_share"].max())},
        "n_alerted_days": int(sum(x["alerted"] for x in timeline)),
        "member_transaction_composition_txn_days": comp,
        "timeline": timeline,
        "unavailable_fields": list(UNAVAILABLE_FIELDS),
    }


def selection_rows(df, groups, tau, seed) -> pd.DataFrame:
    t = track_table(df).set_index("track")
    rows = []
    for tr, gs in memberships(groups).items():
        r = t.loc[tr]
        rows.append({"seed": int(seed), "track": tr, "groups": "+".join(gs), "first_day": int(r["first_day"]),
                     "last_day": int(r["last_day"]), "n_days": int(r["n_days"]), "peak_score_st": float(r["peak_score_st"]),
                     "peak_n_cards": int(r["peak_n_cards"]), "peak_u_s": float(r["peak_u_s"]),
                     "peak_hub_share": float(r["peak_hub_share"])})
    return pd.DataFrame(rows)


def render_markdown(cards, meta) -> str:
    L = [f"# Candidate-community investigation cards (seed {meta['seed']}, W = {meta['W']})", "",
         f"**{STANDING}.** {NOT_ESTABLISHED}", "",
         "Descriptive, aggregate-only cards built from the existing Phase 9 Tier 0 community-day CSV. No discovery was re-run; "
         "no label, risk score or model output was used to select or rank anything. Tier 0 identity is unstable for most tracks "
         "and censored at both ends of the 30-day window (days 122-151).", "",
         f"Alert flag: `score_st >= tau` with tau = {meta['tau']!r} (TRAIN-calibrated, exploratory E0 lock). "
         "E0 comparisons are a descriptive reference, not exact inference and not a significance test.", "",
         "Selection groups (Phase 9 design section 16): " + "; ".join(f"`{g}` {meta['groups'][g]}" for g in GROUPS), ""]
    for c in cards:
        L += [f"## Track {c['track']}  ({', '.join(c['groups'])})", "",
              f"Observed days {c['first_observed_day']}-{c['last_observed_day']} ({c['n_observed_days']} days); "
              f"left-censored: {c['left_censored']}; right-censored: {c['right_censored']}; alerted days: {c['n_alerted_days']}.",
              f"Peak: score_st {c['peak']['score_st']:.4f}, u_s {c['peak']['u_s']:.4f}, u_t {c['peak']['u_t']:.4f}, "
              f"cards {c['peak']['n_cards']}, infrastructure entities {c['peak']['n_infra']}, hub share {c['peak']['hub_share']:.3f}.",
              "Member-transaction composition (transaction-day incidences): " +
              ", ".join(f"{k.replace('txn_', '').replace('__', ' / ')} {v}" for k, v in c["member_transaction_composition_txn_days"].items()) + ".",
              "", "| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in c["timeline"]:
            L.append(f"| {r['day']} | {r['n_cards']} | {r['n_infra']} | {r['density']:.4f} | {r['hub_share']:.3f} | "
                     f"{r['conductance']:.3f} | {r['u_s']:.3f} | {r['u_t']:.3f} | {r['score_st']:.3f} | {r['n_member_txn']} | "
                     f"{'yes' if r['alerted'] else 'no'} |")
        L += ["", "Not available from the aggregates (not fabricated): " + "; ".join(c["unavailable_fields"]) + ".",
              f"*{STANDING}.*", ""]
    return "\n".join(L) + "\n"
