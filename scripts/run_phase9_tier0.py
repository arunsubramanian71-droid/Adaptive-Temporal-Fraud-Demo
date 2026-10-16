"""Phase 9 EXPLORATORY Tier 0 (descriptive discovery, design 10 row 0) on clean real data, days 122-151.

  Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 protocol halted at S6.
  Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under the original protocol.

Label-free, risk-free, benchmark-free. Refuses to run unless the S2/S3/S4 validation record exists and passed, G0 + timing pilot passed, the
locked configuration and null tables are unmodified, and the pre-run record's hash matches. Uses `realrun.run_discovery` (the validated
function). No alert threshold exists yet, so NO alert counts are produced; counts are eligible candidate communities.
"""
import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from src.config import load_config
from src.coordination import frozen as F
from src.coordination.data import get_real
from src.coordination.lint import lint_text
from src.coordination.realdesc import SEG_NAMES, describe_day, infra_observed, segment_codes
from src.coordination.realrun import BANNER, hash_output, load_locked_context, out_dir, run_discovery
from src.coordination.scoring import ArrivalTable
from src.coordination.tracking import lifecycle_table

DAYS = list(range(F.EVAL_PERIOD[0], F.EVAL_PERIOD[1] + 1))
QS = (0.05, 0.25, 0.5, 0.75, 0.95)


def qdict(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {}
    return {"n": int(len(x)), "mean": float(x.mean()), "min": float(x.min()), "max": float(x.max()), **{f"q{int(q*100):02d}": float(np.quantile(x, q)) for q in QS}}


def jaccard_consecutive(recs):
    by_track = {}
    for r in recs:
        by_track.setdefault(r["track"], {})[r["day"]] = set(int(c) for c in r["cards"])
    js = []
    for tr in by_track.values():
        for d, s in tr.items():
            p = tr.get(d - 1)
            if p is not None:
                js.append(len(s & p) / len(s | p))
    return js


def main() -> int:
    cfg = load_config()
    od = out_dir(cfg)
    vp = od / "validation_S2_S3_S4.json"
    if not vp.exists() or not json.load(open(vp, encoding="utf-8")).get("all_passed"):
        raise RuntimeError("S2/S3/S4 validation record missing or not passed: no Tier 0 findings may be produced")
    pre = (od / "PRE_RUN_RECORD.md")
    if hashlib.sha256(pre.read_bytes()).hexdigest() != (od / "PRE_RUN_RECORD.sha256").read_text().strip():
        raise RuntimeError("pre-run record was modified after hashing")
    locked, tables, gates = load_locked_context(cfg)
    real = get_real(cfg)
    v = real.view()
    table = ArrivalTable(v)
    seg = segment_codes(real.day, real.pa)
    iobs = infra_observed(v)
    t0 = time.time()
    summary = {"banner": BANNER, "scope": "Tier 0 descriptive discovery only; no labels, no risk, no R_p/R_h, no alerts (tau does not exist)",
               "days": [DAYS[0], DAYS[-1]], "locked_gamma_per_W": locked["gamma_per_W"], "primary_W": F.PRIMARY_W,
               "validation_record_sha256": hashlib.sha256(vp.read_bytes()).hexdigest(), "gates": gates, "configs": {}}
    # structurally unobservable population on VAL days (count and segment composition only; fraud base rate is NOT computed - labels)
    val = (real.day >= DAYS[0]) & (real.day <= DAYS[-1])
    ct = np.zeros((3, 2), dtype=np.int64)
    np.add.at(ct, (seg[val], iobs[val].astype(np.int64)), 1)
    summary["population_VAL_segment_x_infra_observed"] = {SEG_NAMES[i]: {"no_infrastructure_edge": int(ct[i, 0]), "infrastructure_edge": int(ct[i, 1])} for i in range(3)}
    summary["population_VAL_n_transactions"] = int(val.sum())
    summary["population_VAL_share_without_infrastructure_edge"] = float(ct[:, 0].sum() / ct.sum())
    for W in (F.PRIMARY_W,) + tuple(w for w in F.WINDOWS if w != F.PRIMARY_W):
        g = float(locked["gamma_per_W"][str(W)])
        for S in F.SEEDS:
            out = run_discovery(real, tables[(W, g)], W, g, S, DAYS, table=table)
            recs = out["comm"]
            rows = []
            for t in DAYS:
                dr = [r for r in recs if r["day"] == t]
                rows += describe_day(v, t, W, dr, seg, iobs)
            df = pd.DataFrame(rows)
            df.to_csv(od / f"tier0_community_days_W{W}_seed{S}.csv", index=False)
            life = lifecycle_table(out["_trackers"]["comm"], DAYS[-1])
            ages = [l["age_days"] for l in life]
            per_day = df.groupby("day").size().reindex(DAYS, fill_value=0)
            comp_cols = [c for c in df.columns if c.startswith("txn_")]
            comp_tot = df[comp_cols].sum()
            tot = float(comp_tot.sum())
            ended_obs = [l for l in life if l["ended"] and l["last_day"] <= DAYS[-1] - F.G_DISAPPEAR]
            rec = {
                "output_hash": hash_output(out),
                "n_community_day_records": int(len(df)),
                "candidates_per_day": qdict(per_day.values),
                "cards_per_candidate": qdict(df["n_cards"]), "infra_per_candidate": qdict(df["n_infra"]),
                "size_bucket_counts": {f"{lo}-{hi if hi < 10**9 else '+'}": int(((df["n_cards"] >= lo) & (df["n_cards"] <= hi)).sum()) for lo, hi in F.SIZE_BUCKETS},
                "score_u_s": qdict(df["u_s"]), "score_u_t": qdict(df["u_t"]), "score_st": qdict(df["score_st"]),
                "hub_share": qdict(df["hub_share"]), "density": qdict(df["density"]), "conductance": qdict(df["conductance"]),
                "q_consistency_max_abs_diff": float(np.nanmax(np.abs(df["q"] - df["q_recomputed"]))),
                "n_tracks": len(life), "track_age_days": qdict(ages),
                "tracks_left_censored_first_day_122": int(sum(l["first_day"] == DAYS[0] for l in life)),
                "tracks_right_censored_alive_on_151": int(sum(not l["ended"] for l in life)),
                "tracks_with_observable_disappearance": len(ended_obs),
                "tracks_with_split_flag": int(sum(any(f[0] == "split" for f in l["flags"]) for l in life)),
                "tracks_with_merge_flag": int(sum(any(f[0] == "merged" for f in l["flags"]) for l in life)),
                "share_records_on_tracks_age_ge2": float(np.mean([1.0 if (r["day"] > DAYS[0] and out["_trackers"]["comm"].tracks[r["track"]].first_day < r["day"]) else 0.0 for r in recs])) if recs else None,
                "consecutive_day_jaccard_same_track": qdict(jaccard_consecutive(recs)),
                "composition_member_transactions": {
                    "n_member_transactions": int(tot),
                    **{c.replace("txn_", ""): float(comp_tot[c] / tot) if tot else None for c in comp_cols}},
            }
            summary["configs"][f"W{W}_seed{S}"] = rec
            print(f"W={W} seed={S}: {len(df)} records, {len(life)} tracks, candidates/day median {rec['candidates_per_day'].get('q50')}, {time.time()-t0:.0f}s", flush=True)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    with open(od / "tier0_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    bad = lint_text(json.dumps(summary))
    print("terminology lint:", bad or "clean")
    return 0 if not bad else 5


if __name__ == "__main__":
    sys.exit(main())
