"""E0 (exploratory, post-Tier-0 specification; see exploratory_realdata/E0_ADDENDUM_post_Tier0_exploratory.md).

Degree-preserving null ONLY (the time-shuffled null is omitted). Everything here is label-free, risk-free and uses the validated
`realrun.run_discovery` for every discovery run. The reference tail proportion is DESCRIPTIVE: it is not an exact or calibrated p-value
because the arrival table is held at its observed value for every null history (addendum section 3).
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import numpy as np

from src.coordination import frozen as F
from src.coordination.data import Arrays
from src.coordination.graphs import WindowEdges, rewire
from src.coordination.nullmodel import calibration_days
from src.coordination.pipeline import record_score
from src.coordination.realrun import BANNER, run_discovery

ROOT = pathlib.Path(__file__).resolve().parents[2]
W_E0 = 7
GAMMA_E0 = 0.5
K_E0 = 100
NULL_ID_DEGREE = 1
EVAL_DAYS = tuple(range(F.EVAL_PERIOD[0], F.EVAL_PERIOD[1] + 1))
REHIST_START = F.EVAL_PERIOD[0] - W_E0 + 1          # first day of any window used by E0 (116)
TAU_TARGET = F.ALERT_TARGET_PER_DAY
TAU_N_DAYS = 30
VALIDATION_K0 = 1_000_000                            # validation-only replicate keys; the experiment uses 0..K_E0-1
ADDENDUM = "experiments/phase9_coordinated_fraud/exploratory_realdata/E0_ADDENDUM_post_Tier0_exploratory.md"
SOURCE_FILES = ("src/coordination/e0.py", "src/coordination/realrun.py", "src/coordination/graphs.py", "src/coordination/scoring.py",
                "src/coordination/pipeline.py", "src/coordination/nullmodel.py", "src/coordination/tracking.py",
                "src/coordination/frozen.py", "src/coordination/data.py")
TAIL_LABEL = "descriptive reference tail proportion (not an exact or calibrated p-value)"
BANNED_REPORT_WORDS = ("significant", "reject", "alpha", "α", "p <", "p=", "inconsistent", "consistent with excess")


# ----------------------------------------------------------------------------- threshold (TRAIN-only)
def tau_days() -> list[int]:
    """TRAIN days 89..121 excluding the null-calibration days (the intersection with `nullmodel.calibration_days()`)."""
    cal = set(calibration_days())
    return [d for d in range(89, 122) if d not in cal]


def record_scores(out: dict) -> np.ndarray:
    """A2 score of every community-day record, exactly `pipeline.record_score(rec, ('s','t'))`."""
    return np.array([record_score(r, ("s", "t")) for r in out["comm"]], dtype=np.float64)


def select_tau(scores, n_days: int, target: float = TAU_TARGET) -> dict:
    """tau = the smallest DISTINCT observed score s with (number of records with score >= s) / n_days <= target.
    Alert rule: score >= tau (ties at tau are alerts). Infeasible (no invention of a replacement) if there are no records or even the
    highest distinct score level carries more than target * n_days records."""
    s = np.sort(np.asarray(scores, dtype=np.float64))
    if s.size == 0:
        return {"feasible": False, "reason": "no records", "n_records": 0, "n_days": n_days, "target": target}
    vals = np.unique(s)
    n_ge = s.size - np.searchsorted(s, vals, side="left")
    ok = np.nonzero(n_ge <= target * n_days)[0]
    base = {"n_records": int(s.size), "n_days": int(n_days), "target": float(target), "n_distinct_scores": int(vals.size),
            "max_score": float(vals[-1]), "n_records_at_max_score": int(n_ge[-1])}
    if ok.size == 0:
        return {**base, "feasible": False, "reason": "even the highest distinct score level exceeds the target rate"}
    i = int(ok[0])
    tau = float(vals[i])
    return {**base, "feasible": True, "tau": tau, "n_records_ge_tau": int(n_ge[i]), "realized_rate_per_day": float(n_ge[i]) / n_days,
            "n_records_tied_at_tau": int((s == tau).sum())}


def alert_counts(out: dict, tau: float, days=EVAL_DAYS) -> np.ndarray:
    """Number of alerted community-days (score >= tau) on each listed day."""
    by_day = {d: 0 for d in days}
    for r in out["comm"]:
        if r["day"] in by_day and record_score(r, ("s", "t")) >= tau:
            by_day[r["day"]] += 1
    return np.array([by_day[d] for d in days], dtype=np.int64)


# ----------------------------------------------------------------------------- degree-preserving history (primary null)
def rewired_arrays(real: Arrays, S: int, k: int, d_lo: int = REHIST_START, d_hi: int = F.MAX_DAY) -> Arrays:
    """One coherent randomized history: for each day d in [d_lo, d_hi] the infrastructure endpoint is shuffled by the existing `rewire`
    among that day's card-bearing rows (netproxy and device separately). Stream = SeedSequence([50000 + S, 1, k, d]) per day. The row
    index travels in the `card` field of WindowEdges so the permuted infrastructure can be written back by position."""
    net, dev = real.net.copy(), real.dev.copy()
    n_net = int(real.n_entities["netproxy"])
    in_range = np.nonzero((real.day >= d_lo) & (real.day <= d_hi) & (real.card >= 0))[0]
    day_in = real.day[in_range]
    for d in range(d_lo, d_hi + 1):
        rows = in_range[day_in == d]
        if rows.size == 0:
            continue
        rn, rd = rows[real.net[rows] >= 0], rows[real.dev[rows] >= 0]
        e = WindowEdges(card=np.concatenate([rn, rd]).astype(np.int64),
                        infra=np.concatenate([real.net[rn], real.dev[rd] + n_net]).astype(np.int64),
                        day=np.full(len(rn) + len(rd), d, dtype=np.int64),
                        kind=np.concatenate([np.zeros(len(rn), dtype=np.int8), np.ones(len(rd), dtype=np.int8)]))
        rng = np.random.default_rng(np.random.SeedSequence([F.PERM_SEED_BASE + S, NULL_ID_DEGREE, int(k), int(d)]))
        e2 = rewire(e, rng)
        net[rn] = e2.infra[:len(rn)]
        dev[rd] = e2.infra[len(rn):] - n_net
    return Arrays(day=real.day, txn_id=real.txn_id, card=real.card, pa=real.pa, net=net, dev=dev, n_entities=dict(real.n_entities), y=None, X=None)


def run_observed(real: Arrays, nt, S: int, table=None, days=EVAL_DAYS) -> dict:
    return run_discovery(real, nt, W_E0, GAMMA_E0, S, days, table=table)


def run_null(real: Arrays, nt, S: int, k: int, table, days=EVAL_DAYS) -> dict:
    """Null history k scored with the OBSERVED arrival table (addendum section 3: this is why the result is descriptive, not exact)."""
    return run_discovery(rewired_arrays(real, S, k), nt, W_E0, GAMMA_E0, S, days, table=table)


# ----------------------------------------------------------------------------- statistic and descriptive reference tail proportion
def tail_proportion(a_obs: int, a_null) -> dict:
    a = np.asarray(a_null, dtype=np.int64)
    n_ge = int((a >= a_obs).sum())
    return {"n_null": int(a.size), "n_null_ge_obs": n_ge, "tail_proportion": (1.0 + n_ge) / (1.0 + a.size), "min_attainable": 1.0 / (1.0 + a.size)}


def run_seed(real: Arrays, nt, S: int, tau: float, table, K: int = K_E0, k_start: int = 0, days=EVAL_DAYS) -> dict:
    """One seed: observed counts, K null histories, A and the descriptive reference tail proportion. Used by the E0 driver; exercised on a
    synthetic toy dataset by the validation (never on the real data before authorization)."""
    obs = run_observed(real, nt, S, table, days)
    obs_counts = alert_counts(obs, tau, days)
    null_counts = np.zeros((K, len(days)), dtype=np.int64)
    for j in range(K):
        null_counts[j] = alert_counts(run_null(real, nt, S, k_start + j, table, days), tau, days)
    a_obs, a_null = int(obs_counts.sum()), null_counts.sum(1)
    track_alerts: dict = {}
    for r in obs["comm"]:
        if record_score(r, ("s", "t")) >= tau:
            track_alerts[int(r["track"])] = track_alerts.get(int(r["track"]), 0) + 1
    top = sorted(track_alerts.values(), reverse=True)
    return {"S": S, "tau": float(tau), "days": list(days), "K": K, "k_start": k_start, "A_obs": a_obs, "A_null": a_null.tolist(),
            "obs_counts_per_day": obs_counts.tolist(), "null_mean_counts_per_day": null_counts.mean(0).tolist(),
            "null_summary": {"mean": float(a_null.mean()), "sd": float(a_null.std(ddof=1)) if K > 1 else None, "min": int(a_null.min()),
                             "q05": float(np.quantile(a_null, 0.05)), "q50": float(np.quantile(a_null, 0.5)),
                             "q95": float(np.quantile(a_null, 0.95)), "max": int(a_null.max())},
            "tail": tail_proportion(a_obs, a_null), "obs_alerted_tracks": len(track_alerts),
            "obs_top1_track_share": (top[0] / a_obs) if (top and a_obs) else None,
            "obs_top3_track_share": (sum(top[:3]) / a_obs) if (top and a_obs) else None}


def render_report(results: dict, taus: dict) -> str:
    """Descriptive per-seed report. No threshold, significance language or verdict (addendum section 4)."""
    L = [f"> {BANNER}", "", "# E0 (degree-preserving null only) - descriptive per-seed results",
         "Specification: post-Tier-0 exploratory addendum (not a preregistration). The time-shuffled null is omitted. Alerts are community-day records with "
         "A2 score >= tau_S (TRAIN-only). The null histories use the observed arrival table, so the reference distribution is descriptive and the tail "
         f"proportion is a {TAIL_LABEL}. Seeds share identical real data and differ only in Louvain seed and permutation streams; they are not independent "
         "replications and are not pooled.", "",
         "A difference from this degree-preserving null is not evidence of fraud and not evidence of confirmed coordination; ordinary benign persistence "
         "(households, shared egress, repeated device use) is enough to produce one. Candidate coordinated community; not a confirmed fraud ring. "
         "This does not establish that IEEE-CIS contains fraud rings.", "",
         "| seed | tau_S | TRAIN rate at tau_S (records/day) | records tied at tau_S | A observed | null A mean (SD) | null min / q05 / q50 / q95 / max | null histories with A >= observed | reference tail proportion |",
         "|---|---|---|---|---|---|---|---|---|"]
    for S, r in results.items():
        t, ns, tl = taus[S], r["null_summary"], r["tail"]
        L.append(f"| {S} | {t['tau']:.6f} | {t['realized_rate_per_day']:.3f} | {t['n_records_tied_at_tau']} | {r['A_obs']} | {ns['mean']:.1f} ({ns['sd']:.1f}) | "
                 f"{ns['min']} / {ns['q05']:.0f} / {ns['q50']:.0f} / {ns['q95']:.0f} / {ns['max']} | {tl['n_null_ge_obs']} of {tl['n_null']} | {tl['tail_proportion']:.4f} |")
    L += ["", f"With K = {next(iter(results.values()))['K']} null histories the smallest attainable tail proportion is {next(iter(results.values()))['tail']['min_attainable']:.4f}.",
          "", "## Concentration of observed alerts (descriptive)", "| seed | alerted tracks | top-1 track share of alerted community-days | top-3 share |", "|---|---|---|---|"]
    for S, r in results.items():
        f = lambda x: "n/a" if x is None else f"{x:.3f}"
        L.append(f"| {S} | {r['obs_alerted_tracks']} | {f(r['obs_top1_track_share'])} | {f(r['obs_top3_track_share'])} |")
    L += ["", "## Per-day counts (observed vs mean over null histories)", "| day | " + " | ".join(f"obs {S} / null mean {S}" for S in results) + " |",
          "|---|" + "---|" * len(results)]
    days = next(iter(results.values()))["days"]
    for i, d in enumerate(days):
        L.append(f"| {d} | " + " | ".join(f"{r['obs_counts_per_day'][i]} / {r['null_mean_counts_per_day'][i]:.2f}" for r in results.values()) + " |")
    L += ["", "No threshold or formal verdict is applied. Limitations: the arrival table is held at its observed value in every null history; the 30 TRAIN windows for tau "
          "are serially dependent; the null randomizes card-infrastructure pairing only; alert counts are counts of community-days, so persistent communities count repeatedly.", ""]
    return "\n".join(L)


# ----------------------------------------------------------------------------- gates for later E0 steps
def sha256_file(p) -> str:
    return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()


def source_hashes() -> dict:
    return {f: sha256_file(ROOT / f) for f in SOURCE_FILES}


def require_e0_gates(od: pathlib.Path) -> dict:
    """The addendum hash, a passing E0 validation record, and the sources recorded in it must all still verify."""
    add = ROOT / ADDENDUM
    if sha256_file(add) != (od / "E0_ADDENDUM_post_Tier0_exploratory.sha256").read_text().strip():
        raise RuntimeError("E0 addendum was modified after hashing")
    vp = od / "validation_E0.json"
    if not vp.exists():
        raise RuntimeError("E0 validation record missing")
    rec = json.loads(vp.read_text(encoding="utf-8"))
    if not rec.get("all_passed"):
        raise RuntimeError("E0 validation did not pass")
    if rec["addendum_sha256"] != sha256_file(add):
        raise RuntimeError("validation record refers to a different addendum")
    now = source_hashes()
    changed = [f for f, h in rec["source_sha256"].items() if now.get(f) != h]
    if changed:
        raise RuntimeError(f"sources changed since validation: {changed}")
    return rec
