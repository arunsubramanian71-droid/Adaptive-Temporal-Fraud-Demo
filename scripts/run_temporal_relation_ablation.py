"""Relationship-level ablation of Temporal GraphSAGE (experiments/relation_ablation_temporal/PREREGISTRATION_RELATION_ABLATION.md).

  .venv/Scripts/python.exe -B scripts/run_temporal_relation_ablation.py              # 4 relations x 3 seeds + the seed-42 full-model noise rerun
  .venv/Scripts/python.exe -B scripts/run_temporal_relation_ablation.py --report     # evaluate the registered rule from the recorded outputs

Training is Phase 7's `run_seed` (scripts/train_temporal_graphsage.py), unchanged; the only change is that the edges of one relation are removed on
every day before training and scoring. Days 1-151 only. Refuses to run unless the pre-registration matches its hashed sidecar; existing arm folders
are never overwritten.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import time

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402

EXP = ROOT / "experiments" / "relation_ablation_temporal"
PRE = EXP / "PREREGISTRATION_RELATION_ABLATION.md"
FULL = ROOT / "experiments" / "temporal_graphsage"
RELATIONS = ("charged_to", "used_card", "via_network", "from_device")
SEEDS = (42, 123, 2024)
SEGMENTS = ("pooled", "has_edge_cold", "has_edge_warm", "no_pseudo_account_edge")


def write(path: pathlib.Path, data: bytes):
    path.write_bytes(data)
    path.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))


def drop_relation(ds, rel: str):
    """A shallow copy of the dataset whose relation `rel` has no edges on any day (other relations untouched)."""
    import copy
    import torch
    from src.graph.temporal import DayRel
    out = copy.copy(ds)
    out.rel = {}
    for d, per in ds.rel.items():
        p = dict(per)
        dev = per[rel].src_local.device
        empty = torch.zeros(0, dtype=torch.long, device=dev)
        p[rel] = DayRel(empty, empty.clone(), empty.clone())
        out.rel[d] = p
    return out


def metrics(path: pathlib.Path) -> dict:
    m = json.loads(path.read_text(encoding="utf-8"))
    return {s: m[s]["pr_auc"] for s in SEGMENTS if s in m and isinstance(m[s], dict) and "pr_auc" in m[s]}


def report() -> int:
    full = {S: metrics(FULL / f"seed_{S}" / "metrics_val.json") for S in SEEDS}
    rows, rec = {}, {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "preregistration_sha256": PRE.with_suffix(".sha256").read_text().strip()}
    for r in RELATIONS:
        arm = {S: metrics(EXP / f"drop_{r}" / f"seed_{S}" / "metrics_val.json") for S in SEEDS}
        f = np.array([full[S]["pooled"] for S in SEEDS])
        a = np.array([arm[S]["pooled"] for S in SEEDS])
        d = f - a
        bound = max(float(np.std(f, ddof=1)), float(np.std(a, ddof=1)))
        verdict = (f"{r} contributes" if (d > 0).sum() >= 2 and d.mean() > bound else
                   f"removing {r} does not hurt" if d.mean() <= 0 else "not established")
        rows[r] = {"pooled_pr_auc_full": f.tolist(), "pooled_pr_auc_without": a.tolist(), "differences_full_minus_without": d.tolist(),
                   "mean_difference": float(d.mean()), "seeds_where_full_is_higher": int((d > 0).sum()), "larger_sd": bound, "verdict": verdict,
                   "segments_without_mean": {s: float(np.mean([arm[S][s] for S in SEEDS])) for s in SEGMENTS if all(s in arm[S] for S in SEEDS)},
                   "segments_full_mean": {s: float(np.mean([full[S][s] for S in SEEDS])) for s in SEGMENTS if all(s in full[S] for S in SEEDS)}}
    noise = EXP / "full_rerun" / "seed_42" / "metrics_val.json"
    rec["full_model_rerun_seed42"] = {"recorded": full[42]["pooled"], "rerun": metrics(noise)["pooled"], "abs_difference": abs(full[42]["pooled"] - metrics(noise)["pooled"])} if noise.exists() else None
    rec["relations"] = rows
    rec["test_data_read"] = False
    write(EXP / "results.json", (json.dumps(rec, indent=1) + "\n").encode("utf-8"))
    L = ["# Relationship-level ablation of Temporal GraphSAGE: results\n",
         "**Validation period (days 122-151); recorded Phase 7 full model as reference; pre-registration `PREREGISTRATION_RELATION_ABLATION.md` (hashed before any arm). No held-out data was read.**\n",
         "| Relation removed | Pooled PR-AUC without it (seeds 42, 123, 2024) | Full model (recorded) | Mean difference (full - without) | Seeds where full is higher | Larger seed SD | Verdict |", "|---|---|---|---|---|---|---|"]
    for r, x in rows.items():
        L.append(f"| {r} | {', '.join(f'{v:.4f}' for v in x['pooled_pr_auc_without'])} | {', '.join(f'{v:.4f}' for v in x['pooled_pr_auc_full'])} | {x['mean_difference']:+.4f} | {x['seeds_where_full_is_higher']} of 3 | {x['larger_sd']:.4f} | **{x['verdict']}** |")
    if rec["full_model_rerun_seed42"]:
        n = rec["full_model_rerun_seed42"]
        L += ["", f"Run-to-run noise reference: retraining the full model with seed 42 by the same code gives pooled PR-AUC {n['rerun']:.4f} against the recorded {n['recorded']:.4f} (absolute difference {n['abs_difference']:.4f}); differences of that size are not effects."]
    L += ["", "Segment PR-AUC (means over seeds) are in `results.json`. Three seeds sharing the same data are not independent replications; validation was used for early stopping and thresholds, as in Phase 7.", ""]
    write(EXP / "RESULTS_RELATION_ABLATION.md", "\n".join(L).encode("utf-8"))
    print(json.dumps({r: (x["mean_difference"], x["verdict"]) for r, x in rows.items()}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args(argv)
    if hashlib.sha256(PRE.read_bytes()).hexdigest() != PRE.with_suffix(".sha256").read_text().strip():
        print("error: the pre-registration does not match its hashed sidecar", file=sys.stderr)
        return 2
    if a.report:
        return report()
    import pandas as pd
    import torch
    import train_temporal_graphsage as T
    from src.config import load_config
    from src.graph.schema import compute_pseudo_account_segments
    from src.graph.temporal import build_temporal_dataset, load_raw_arrays
    cfg = load_config()
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    raw, graph, _scaler = load_raw_arrays(cfg)
    ds = build_temporal_dataset(raw, cfg.split.val_day_max, dev, cfg)
    assert ds.max_day == 151
    segment_by_id = pd.Series(compute_pseudo_account_segments(graph), index=graph["transaction"].transaction_id.numpy())
    jobs = [("full_rerun", None, 42)] + [(f"drop_{r}", r, S) for r in RELATIONS for S in SEEDS]
    t0 = time.time()
    for name, rel, S in jobs:
        out = EXP / name / f"seed_{S}"
        if (out / "metrics_val.json").exists():
            print(f"[skip] {name} seed {S}: already recorded (never overwritten)", flush=True)
            continue
        dsr = ds if rel is None else drop_relation(ds, rel)
        t1 = time.time()
        res = T.run_seed(S, dsr, raw, segment_by_id, cfg, dev, out, cfg.split.train_day_max, cfg.split.val_day_min, cfg.split.val_day_max,
                         T.MAX_EPOCHS, T.PATIENCE, pilot=False)
        print(f"[done] {name} seed {S}: pooled PR-AUC {res['metrics']['pooled']['pr_auc']:.4f} ({time.time() - t1:.0f}s; total {time.time() - t0:.0f}s)", flush=True)
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return report()


if __name__ == "__main__":
    sys.exit(main())
