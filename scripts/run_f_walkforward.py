"""Walk-forward runs of the adaptive system (arms F, Ad, E) with the Phase 8 shared checkpoints.

  validation, as written (readiness / reproduction of Phase 8, seed 42 by default):
      .venv/Scripts/python.exe -B scripts/run_f_walkforward.py --split val --variant as_written --seeds 42
  validation, state burn-in (the Phase 8 state-continuity supplement):
      .venv/Scripts/python.exe -B scripts/run_f_walkforward.py --split val --variant burn_in
  held-out (amendment 1): REFUSED unless the user's AUTHORIZATION.txt quotes the protocol and amendment hashes (see src/heldout/evaluator.py) and Module F's
      readiness gate has passed; arms F, Ad, E and seeds 42, 123, 2024 together (Amendment 1 section 3); one-shot: refused once the output folder holds any file.

Windows: validation 122-127 ... 146-151 (Phase 8); held-out 152-157, 158-163, 164-169, 170-175, 176-181 (five 6-day windows; day 182 is not scored for these arms).
Burn-in (variant burn_in and always for held-out): the state is first streamed over days 1..(first window - 1) unlabeled. Outputs: experiments/phase8_state_continuity_supplement/outputs/<variant>/
(validation; --out-name chooses another folder) or experiments/final_heldout_protocol/heldout_results_f/ (held-out). A folder holding a completed run is never
overwritten (the registered readiness result and the reference runs are preserved records).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import pathlib
import sys
import time
import warnings

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src.adaptive.windows import build_windows  # noqa: E402
from src.config import load_config  # noqa: E402
from src.graph.schema import compute_pseudo_account_segments  # noqa: E402
from src.graph.temporal import build_temporal_dataset, load_raw_arrays  # noqa: E402
from src.heldout import amendment as A  # noqa: E402
from src.heldout import evaluator as E  # noqa: E402
from src.heldout import h4  # noqa: E402
from src.heldout import oneshot as OS  # noqa: E402
from src.heldout.walkforward import run_walk_forward_ext  # noqa: E402
from src.models.metrics import compute_metrics  # noqa: E402
from src.models.temporal_graphsage import TemporalGraphSAGE  # noqa: E402

P8 = ROOT / "experiments" / "phase8_adaptive_drift"
ARM_DIR = {"F": "frozen", "Ad": "adaptive", "E": "replay_control"}
HELDOUT_WINDOWS = [(152, 157), (158, 163), (164, 169), (170, 175), (176, 181)]
HELDOUT_REPLAY_WINDOWS = [(122, 157), (158, 163), (164, 169), (170, 175), (176, 181)]     # E draws as many TRAIN days as Ad adds: 36, then 6 per round
SEG = ("has_edge_warm", "has_edge_cold", "no_pseudo_account_edge")
HELDOUT_OUT = ROOT / "experiments" / "final_heldout_protocol" / "heldout_results_f"


def write(path: pathlib.Path, data: bytes):
    path.write_bytes(data)
    path.with_suffix(".sha256").write_bytes(hashlib.sha256(data).hexdigest().encode("ascii"))


def amendment_readiness(out_root: pathlib.Path, per_arm: dict, seeds, arms) -> int:
    """Amendment 1 readiness of module F on validation: the burn-in run is repeated and compared with the recorded supplement run (outputs/burn_in).
    F (no training): every score within 1e-6. Every arm, window 1: within 1e-6. Ad and E (trained, GPU/dropout not bit-reproducible): window-mean PR-AUC within 0.01 per seed.
    The 0.01 tolerance is set in AMENDMENT_1.md after the supplement showed run-to-run window-mean differences of about 0.003; the code-equivalence of the update path is proved by tests/test_walkforward_ext.py."""
    ref_root = out_root.parent / "burn_in"
    ok, rows = True, []
    for seed in seeds:
        for arm in arms:
            for k in range(1, 6):
                rec = pd.read_parquet(ref_root / ARM_DIR[arm] / f"seed_{seed}" / f"predictions_W{k}.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
                new = pd.read_parquet(out_root / ARM_DIR[arm] / f"seed_{seed}" / f"predictions_W{k}.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
                mx = float(np.abs(new.loc[rec.index].to_numpy() - rec.to_numpy()).max())
                if arm == "F" or k == 1:
                    good = mx <= 1e-6
                    ok &= good
                    rows.append({"arm": arm, "seed": seed, "window": k, "check": "scores within 1e-6", "max_abs_score_diff": mx, "pass": bool(good)})
            if arm != "F":
                ref_mean = float(np.mean([json.loads((ref_root / ARM_DIR[arm] / f"seed_{seed}" / f"metrics_W{k}.json").read_text(encoding="utf-8"))["pooled"]["pr_auc"] for k in range(1, 6)]))
                d = abs(float(np.mean(per_arm[arm][seed])) - ref_mean)
                good = d <= 0.01
                ok &= good
                rows.append({"arm": arm, "seed": seed, "window": "mean", "check": "window-mean PR-AUC within 0.01", "abs_diff": d, "pass": bool(good)})
    write(out_root / "readiness_F.json", (json.dumps({"all_pass": bool(ok), "rows": rows}, indent=1) + chr(10)).encode("utf-8"))
    print("F readiness (amendment 1):", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "heldout"], required=True)
    ap.add_argument("--variant", choices=["as_written", "burn_in", "readiness"], default="burn_in")
    ap.add_argument("--seeds", default="42,123,2024")
    ap.add_argument("--arms", default="F,Ad,E")
    ap.add_argument("--out-name", default=None, help="validation only: output folder name under the supplement outputs (default: the variant). A folder holding a completed run is never overwritten.")
    a = ap.parse_args(argv)
    seeds = [int(s) for s in a.seeds.split(",")]
    arms = a.arms.split(",")
    if a.split == "heldout":
        if a.variant != "burn_in":
            print("error: the held-out amendment fixes the burn-in variant", file=sys.stderr)
            return 2
        if sorted(seeds) != sorted(h4.SEEDS) or len(seeds) != len(h4.SEEDS) or sorted(arms) != sorted(ARM_DIR) or len(arms) != len(ARM_DIR):
            print(f"error: Amendment 1 section 3 fixes arms {','.join(ARM_DIR)} and seeds {','.join(map(str, h4.SEEDS))} for the held-out run; a subset would spend the "
                  "one-shot period for the rest. Nothing was read.", file=sys.stderr)
            return 2
        out_root = HELDOUT_OUT
        try:
            OS.require_unused(out_root, "the held-out evaluation of the adaptive arms")    # Amendment 1 section 7
            auth = A.require_amendment_authorization("heldout", ("F",))
            auth["readiness"] = A.require_amendment_readiness(("F",))           # refuses while the registered gate has not passed
        except (E.AuthorizationError, A.ReadinessError, OS.OneShotError) as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        windows, replay_windows, max_day = HELDOUT_WINDOWS, HELDOUT_REPLAY_WINDOWS, 181
    else:
        name = a.out_name or a.variant
        if not name.replace("_", "").isalnum():
            print(f"error: --out-name must be a plain folder name, got {name!r}", file=sys.stderr)
            return 2
        out_root = ROOT / "experiments" / "phase8_state_continuity_supplement" / "outputs" / name
        if (out_root / "summary.json").exists():
            print(f"error: {out_root} holds a completed run, which is a preserved record (the registered readiness result and the reference runs live here); "
                  f"choose a new --out-name.", file=sys.stderr)
            return 2
        auth = {"split": "val"}
        windows, replay_windows, max_day = build_windows(122, 151), None, 151
    out_root.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    raw, graph, _ = load_raw_arrays(cfg)
    seg_all = pd.Series(compute_pseudo_account_segments(graph), index=graph["transaction"].transaction_id.numpy())
    y_all = pd.Series(raw["y"], index=raw["txn_id"])
    ds = build_temporal_dataset(raw, max_day, dev, cfg, allow_test=(a.split == "heldout"))
    assert ds.max_day == max_day
    tr = ds.txn_day <= cfg.split.train_day_max
    y_tr = raw["y"][:len(tr)][tr]
    pos_weight = torch.tensor(float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), device=dev)
    burn = windows[0][0] - 1 if a.variant in ("burn_in", "readiness") else None
    per_arm = {arm: {} for arm in arms}
    for seed in seeds:
        hp = json.loads((P8 / "shared_init" / f"hparams_seed{seed}.json").read_text(encoding="utf-8"))
        for arm in arms:
            torch.manual_seed(seed)
            model = TemporalGraphSAGE(raw["X"].shape[1]).to(dev)
            model.load_state_dict(torch.load(P8 / "shared_init" / f"model_checkpoint_initial_seed{seed}.pt", map_location=dev))
            opt = torch.optim.Adam(model.parameters(), lr=hp["fine_tune_lr"], weight_decay=1e-5) if arm != "F" else None
            ta = time.time()
            res = run_walk_forward_ext(arm, model, ds, windows, opt=opt, pos_weight=pos_weight if arm != "F" else None, fine_tune_epochs=hp["fine_tune_epochs"] if arm != "F" else 0,
                                       base_seed=seed, train_day_min=cfg.split.train_day_min, train_day_max=cfg.split.train_day_max, burn_in_through=burn, replay_windows=replay_windows)
            d = out_root / ARM_DIR[arm] / f"seed_{seed}"
            d.mkdir(parents=True, exist_ok=True)
            prs = []
            for k, (ids, scores, days) in enumerate(res, start=1):
                y = y_all.loc[ids].to_numpy()
                seg = seg_all.loc[ids].to_numpy()
                met = {"pooled": compute_metrics(y, scores, threshold=hp["threshold"])}
                for name in SEG:
                    m = seg == name
                    met[name] = compute_metrics(y[m], scores[m], threshold=hp["threshold"]) if 0 < y[m].sum() < m.sum() else {"n_rows": int(m.sum()), "note": "insufficient class balance"}
                (d / f"metrics_W{k}.json").write_text(json.dumps(met, indent=1), encoding="utf-8")
                pd.DataFrame({"TransactionID": ids, "isFraud": y, "y_score": scores, "day": days, "window": k, "segment": seg}).to_parquet(d / f"predictions_W{k}.parquet", index=False)
                prs.append(met["pooled"]["pr_auc"])
            per_arm[arm][seed] = prs
            print(f"[done] {a.split}/{a.variant} {arm} seed {seed}: window PR-AUC {[round(p, 4) for p in prs]} mean {np.mean(prs):.4f} ({time.time() - ta:.0f}s)", flush=True)
    summary = {"created_local": datetime.datetime.now().astimezone().replace(microsecond=0).isoformat(), "split": a.split, "variant": a.variant, "windows": windows, "burn_in_through": burn,
               "seeds": seeds, "arms": arms, "window_pr_auc": {arm: {str(s): v for s, v in per_arm[arm].items()} for arm in arms}, "authorization": auth, "seconds": round(time.time() - t0, 1)}
    if set(arms) == {"F", "Ad", "E"} and set(seeds) == set(h4.SEEDS):
        summary["h4_locked_rules"] = h4.h4_decision(per_arm)
    write(out_root / "summary.json", (json.dumps(summary, indent=1) + "\n").encode("utf-8"))
    if a.split == "heldout":
        (out_root / ".completed").write_text(summary["created_local"], encoding="utf-8")
    if a.split == "val" and a.variant == "readiness":
        return amendment_readiness(out_root, per_arm, seeds, arms)
    if a.split == "val" and a.variant == "as_written":
        ok = True
        rows = []
        for seed in seeds:
            for arm in arms:
                for k in range(1, 6):
                    rec = pd.read_parquet(P8 / ARM_DIR[arm] / f"seed_{seed}" / f"predictions_val_W{k}.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
                    new = pd.read_parquet(out_root / ARM_DIR[arm] / f"seed_{seed}" / f"predictions_W{k}.parquet", columns=["TransactionID", "y_score"]).set_index("TransactionID")["y_score"]
                    mx = float(np.abs(new.loc[rec.index].to_numpy() - rec.to_numpy()).max())
                    prd = abs(per_arm[arm][seed][k - 1] - json.loads((P8 / ARM_DIR[arm] / f"seed_{seed}" / f"metrics_val_W{k}.json").read_text(encoding="utf-8"))["pooled"]["pr_auc"])
                    good = (mx <= 1e-6) if arm == "F" or k == 1 else (prd <= 0.01)
                    ok &= good
                    rows.append({"arm": arm, "seed": seed, "window": k, "max_abs_score_diff": mx, "pr_auc_abs_diff": prd, "pass": bool(good)})
        write(out_root / "readiness_S1.json", (json.dumps({"all_pass": bool(ok), "rule": "F and window 1 of every arm: max score difference <= 1e-6; Ad/E later windows: PR-AUC difference <= 0.01", "rows": rows}, indent=1) + "\n").encode("utf-8"))
        print("S1 reproduction of Phase 8 as written:", "PASS" if ok else "FAIL")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
