"""Phase 7: Temporal GraphSAGE (per-entity GRU memory) — pilot and full run.

  python -u scripts/train_temporal_graphsage.py --pilot     # feasibility pilot ONLY (never a result)
  python -u scripts/train_temporal_graphsage.py             # full 3-seed experiment

Protocol is the established one: TRAIN days 1-121 fit, VALIDATION days
122-151 for early stopping / threshold, TEST never loaded (the dataset
builder refuses TEST days). Loss: BCEWithLogits with pos_weight = TRAIN
neg:pos ratio. max_epochs=150, patience=20, seeds 42/123/2024.

Documented training-schedule difference from the static model (required by
the temporal architecture, not a tuning choice): truncated BPTT with a
7-day window (~17 optimizer steps per epoch instead of 1) and gradient-norm
clipping at 1.0 (standard for recurrent models). Epoch cap and patience are
unchanged.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config import load_config
from src.graph import temporal_checks as TC
from src.graph.schema import compute_pseudo_account_segments
from src.graph.temporal import build_temporal_dataset, load_raw_arrays, snapshot_stats
from src.models.metrics import compute_metrics, select_best_f1_threshold
from src.models.temporal_graphsage import TemporalGraphSAGE, rolling_temporal_predict

SEEDS = [42, 123, 2024]
HIDDEN = 64
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 150
PATIENCE = 20
TBPTT_DAYS = 7
GRAD_CLIP = 1.0
EXPERIMENT_NAME = "temporal_graphsage"


def train_epoch(model, ds, opt, train_last: int, pos_weight, window: int = TBPTT_DAYS) -> float:
    model.train()
    state = model.init_state(ds)
    days = [d for d in ds.days if d <= train_last]
    tot, n_tot = 0.0, 0
    for w0 in range(0, len(days), window):
        lgs, ys = [], []
        for d in days[w0:w0 + window]:
            lgs.append(model.step(state, ds, d, predict=True))
            ys.append(ds.labels(d))
        lg, y = torch.cat(lgs), torch.cat(ys)
        loss = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pos_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        model.detach_state(state)  # cut the graph; state VALUES carry forward, never reset
        tot += float(loss.item()) * y.numel()
        n_tot += y.numel()
    return tot / n_tot


def run_seed(seed, ds, raw, segment_by_id, cfg, dev, out_dir, train_last, first_pred, last_pred, max_epochs, patience, pilot):
    tag = "PILOT " if pilot else ""
    print(f"\n{'='*70}\n{tag}SEED {seed}  (train days 1-{train_last}, eval days {first_pred}-{last_pred}, "
          f"max_epochs={max_epochs}, patience={patience})\n{'='*70}", flush=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = TemporalGraphSAGE(raw["X"].shape[1], hidden=HIDDEN, dropout=DROPOUT).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    y_all = pd.Series(raw["y"], index=raw["txn_id"])
    tr_rows = (ds.txn_day <= train_last)
    y_tr = raw["y"][:len(tr_rows)][tr_rows]
    scale_pos_weight = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
    pos_weight = torch.tensor(scale_pos_weight, device=dev)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_pr, best_epoch, patience_left, best_state = -1.0, -1, patience, None
    history, stopped_early = [], False
    t0 = time.time()
    for epoch in range(1, max_epochs + 1):
        te = time.time()
        loss = train_epoch(model, ds, opt, train_last, pos_weight)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t_train = time.time() - te
        tv = time.time()
        ids, scores, _ = rolling_temporal_predict(model, ds, first_pred, last_pred)
        y_true = y_all.loc[ids].to_numpy()
        pr, roc = average_precision_score(y_true, scores), roc_auc_score(y_true, scores)
        t_eval = time.time() - tv
        history.append({"epoch": epoch, "train_loss": loss, "val_pr_auc": float(pr), "val_roc_auc": float(roc),
                        "epoch_seconds": round(time.time() - te, 3), "train_seconds": round(t_train, 3),
                        "eval_seconds": round(t_eval, 3)})
        improved = pr > best_pr + 1e-4
        if epoch % 5 == 0 or improved or epoch <= 2:
            print(f"  epoch {epoch:>3}: loss={loss:.4f}  {'PILOT-' if pilot else ''}VAL PR-AUC={pr:.4f}  ROC-AUC={roc:.4f}  "
                  f"(train {t_train:.1f}s + eval {t_eval:.1f}s){' *' if improved else ''}", flush=True)
        if improved:
            best_pr, best_epoch, patience_left = pr, epoch, patience
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  EARLY STOPPING at epoch {epoch} (best epoch {best_epoch}, PR-AUC {best_pr:.4f})", flush=True)
                stopped_early = True
                break
    wall = time.time() - t0
    model.load_state_dict(best_state)
    print(f"  done in {wall:.1f}s: {len(history)} epochs ({'early-stopped' if stopped_early else 'hit max_epochs — not converged'}), "
          f"best epoch {best_epoch}", flush=True)

    resource = {"parameter_count": n_params, "training_wall_seconds": round(wall, 2), "epochs_run": len(history),
                "best_epoch": best_epoch, "stopped_early": stopped_early,
                "mean_seconds_per_epoch": round(wall / len(history), 3)}
    if dev.type == "cuda":
        resource["peak_gpu_memory_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        resource["peak_gpu_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    with open(out_dir / "resource_measurements.json", "w", encoding="utf-8") as fh:
        json.dump(resource, fh, indent=1)
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=1)
    torch.save(best_state, out_dir / "model_checkpoint.pt")

    result = {"seed": seed, "history": history, "resource": resource, "model": model}
    if pilot:
        return result

    ids, scores, days_arr = rolling_temporal_predict(model, ds, first_pred, last_pred)
    y_true = y_all.loc[ids].to_numpy()
    seg = segment_by_id.loc[ids].to_numpy()
    th = select_best_f1_threshold(y_true, scores)
    m_pooled = compute_metrics(y_true, scores, threshold=th["threshold"])
    m_pooled["threshold_selection"] = th
    metrics = {"pooled": m_pooled}
    for name in ["has_edge_warm", "has_edge_cold", "no_pseudo_account_edge"]:
        m = seg == name
        yt, sc = y_true[m], scores[m]
        metrics[name] = (compute_metrics(yt, sc, threshold=th["threshold"]) if 0 < yt.sum() < len(yt)
                         else {"n_rows": int(len(yt)), "n_fraud": int(yt.sum()), "note": "insufficient class balance"})
    print(f"  FINAL: pooled PR-AUC={m_pooled['pr_auc']:.4f}  cold={metrics['has_edge_cold']['pr_auc']:.4f}  "
          f"warm={metrics['has_edge_warm']['pr_auc']:.4f}  no-edge={metrics['no_pseudo_account_edge']['pr_auc']:.4f}", flush=True)
    with open(out_dir / "metrics_val.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=1)
    pd.DataFrame({"TransactionID": ids, "isFraud": y_true, "y_score": scores, "day": days_arr,
                  "y_pred": (scores >= th["threshold"]).astype(int), "segment": seg}).to_parquet(out_dir / "predictions_val.parquet", index=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR, "weight_decay": WEIGHT_DECAY,
                   "max_epochs": max_epochs, "patience": patience, "tbptt_days": TBPTT_DAYS, "grad_clip": GRAD_CLIP,
                   "scale_pos_weight": scale_pos_weight, "parameter_count": n_params}, fh, indent=1)
    result["metrics"] = metrics
    return result


def run_pilot(cfg, raw, graph, dev, base_dir):
    print("=== FEASIBILITY PILOT (within the TRAIN period only; NOT a scientific result) ===", flush=True)
    PILOT_MAX_DAY, PILOT_TRAIN_LAST, PILOT_FIRST = 35, 28, 29
    ds = build_temporal_dataset(raw, PILOT_MAX_DAY, dev, cfg)
    out_dir = base_dir / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PILOT_NOT_A_RESULT.txt").write_text(
        "Feasibility pilot: train days 1-28, eval days 29-35 (all inside the TRAIN period), 3 epochs. "
        "Exists only to verify execution, state behaviour, memory, runtime and leakage checks. "
        "Its metrics are NOT scientific results and must not be cited.\n", encoding="utf-8")
    res = run_seed(42, ds, raw, None, cfg, dev, out_dir, PILOT_TRAIN_LAST, PILOT_FIRST, PILOT_MAX_DAY, 3, 20, pilot=True)
    model = res["model"]
    h = res["history"]
    n_train_days, n_eval_days = PILOT_TRAIN_LAST, PILOT_MAX_DAY
    tr_per_day = np.mean([x["train_seconds"] for x in h[1:] or h]) / n_train_days
    ev_per_day = np.mean([x["eval_seconds"] for x in h[1:] or h]) / n_eval_days
    full_epoch = tr_per_day * 121 + ev_per_day * 151
    print("\n--- pilot resource findings ---", flush=True)
    print(f"  train step: {tr_per_day*1000:.1f} ms/day (fwd+bwd+opt) | eval step: {ev_per_day*1000:.1f} ms/day", flush=True)
    print(f"  extrapolated full epoch: {full_epoch:.1f}s -> worst case 150 epochs x 3 seeds = {full_epoch*150*3/60:.0f} min", flush=True)
    if dev.type == "cuda":
        print(f"  pilot peak GPU memory reserved: {res['resource']['peak_gpu_memory_reserved_gib']} GiB "
              f"(of 15.98; static GraphSAGE ~5 GiB, GAT 12.3 GiB)", flush=True)
    print(f"  loss by epoch: {[round(x['train_loss'],4) for x in h]} (should trend down)", flush=True)

    print("\n--- pilot leakage/causality smoke checks (trained pilot model, days <= 35 only) ---", flush=True)
    checks = [
        ("future_invariance", lambda: TC.check_future_invariance(model, raw, dev, cfg, PILOT_MAX_DAY, PILOT_FIRST, [30, 32])),
        ("past_sensitivity", lambda: TC.check_past_sensitivity(model, raw, dev, cfg, PILOT_MAX_DAY, PILOT_FIRST)),
        ("boundary_continuity", lambda: TC.check_boundary_continuity(model, raw, dev, cfg, PILOT_MAX_DAY, PILOT_TRAIN_LAST)),
        ("access_pattern", lambda: TC.check_access_pattern(model, raw, dev, cfg, PILOT_MAX_DAY, PILOT_FIRST)),
        ("labels_not_used", lambda: TC.check_labels_not_used(model, raw, dev, cfg, PILOT_MAX_DAY, PILOT_FIRST)),
        ("edge_consumption", lambda: TC.check_edge_consumption(raw, dev, cfg, PILOT_MAX_DAY)),
    ]
    all_ok = True
    for name, fn in checks:
        ok, detail = fn()
        all_ok &= ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    snap = snapshot_stats(ds)
    verdict = {"pilot_checks_all_passed": bool(all_ok), "extrapolated_full_epoch_seconds": round(float(full_epoch), 1),
               "extrapolated_worst_case_minutes_all_seeds": round(float(full_epoch * 150 * 3 / 60), 1),
               "pilot_peak_gpu_memory_reserved_gib": res["resource"].get("peak_gpu_memory_reserved_gib"),
               "parameter_count": res["resource"]["parameter_count"]}
    with open(out_dir / "pilot_verdict.json", "w", encoding="utf-8") as fh:
        json.dump(verdict, fh, indent=1)
    print(f"\nPILOT VERDICT: {'SAFE — proceed to the full experiment' if all_ok else 'CHECKS FAILED — do NOT proceed'}", flush=True)
    sys.exit(0 if all_ok else 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    cfg = load_config()
    base_dir = cfg.experiments_dir / EXPERIMENT_NAME
    base_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    print("=== Loading Phase 4 graph (read-only) + TRAIN-only feature scaling ===", flush=True)
    raw, graph, scaler = load_raw_arrays(cfg)
    joblib.dump(scaler, base_dir / "feature_scaler.joblib")
    print(f"  device={dev}", flush=True)
    if args.pilot:
        run_pilot(cfg, raw, graph, dev, base_dir)
        return

    ds = build_temporal_dataset(raw, cfg.split.val_day_max, dev, cfg)
    print(f"  dataset: {ds.X.shape[0]:,} transactions (days 1-{ds.max_day}); TEST rows never loaded", flush=True)
    snap = snapshot_stats(ds)
    with open(base_dir / "snapshot_stats.json", "w", encoding="utf-8") as fh:
        json.dump(snap, fh, indent=1)
    print(f"  snapshot sizes (per day): mean {snap['mean']['transactions']:.0f} txns, "
          f"max {snap['max']['transactions']} txns", flush=True)

    seg_all = compute_pseudo_account_segments(graph)
    segment_by_id = pd.Series(seg_all, index=graph["transaction"].transaction_id.numpy())

    results = []
    for seed in SEEDS:
        results.append(run_seed(seed, ds, raw, segment_by_id, cfg, dev, base_dir / f"seed_{seed}",
                                cfg.split.train_day_max, cfg.split.val_day_min, cfg.split.val_day_max,
                                MAX_EPOCHS, PATIENCE, pilot=False))
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*70}\n=== Cross-seed aggregation ===\n{'='*70}", flush=True)
    agg = {}
    for key, (seg, m) in {"pooled_pr_auc": ("pooled", "pr_auc"), "pooled_roc_auc": ("pooled", "roc_auc"),
                          "cold_pr_auc": ("has_edge_cold", "pr_auc"), "cold_roc_auc": ("has_edge_cold", "roc_auc"),
                          "warm_pr_auc": ("has_edge_warm", "pr_auc"), "warm_roc_auc": ("has_edge_warm", "roc_auc"),
                          "no_edge_pr_auc": ("no_pseudo_account_edge", "pr_auc"),
                          "no_edge_roc_auc": ("no_pseudo_account_edge", "roc_auc")}.items():
        v = [r["metrics"][seg][m] for r in results]
        agg[key] = {"values": v, "mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)),
                    "min": float(np.min(v)), "max": float(np.max(v))}
        print(f"  {key:<16}: {[round(x,4) for x in v]}  mean={agg[key]['mean']:.4f} std={agg[key]['std']:.4f} "
              f"min={agg[key]['min']:.4f} max={agg[key]['max']:.4f}", flush=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    col = {42: "tab:blue", 123: "tab:orange", 2024: "tab:green"}
    for r in results:
        h = pd.DataFrame(r["history"])
        ax[0].plot(h.epoch, h.train_loss, color=col[r["seed"]], label=f"seed {r['seed']}")
        ax[1].plot(h.epoch, h.val_pr_auc, color=col[r["seed"]], label=f"seed {r['seed']}")
        ax[1].axvline(r["resource"]["best_epoch"], color=col[r["seed"]], ls=":", alpha=.5)
        ax[2].plot(h.epoch, h.val_roc_auc, color=col[r["seed"]], label=f"seed {r['seed']}")
    ax[0].set_title("Temporal GraphSAGE training loss"); ax[1].set_title("Validation PR-AUC (causal rolling, pooled)")
    ax[2].set_title("Validation ROC-AUC (causal rolling, pooled)")
    for a in ax:
        a.set_xlabel("epoch"); a.legend()
    plt.tight_layout(); plt.savefig(base_dir / "convergence_curves.png", dpi=120); plt.close()

    summary = {"seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "tbptt_days": TBPTT_DAYS,
               "per_seed": [{"seed": r["seed"], **r["resource"],
                             "pooled_pr_auc": r["metrics"]["pooled"]["pr_auc"], "pooled_roc_auc": r["metrics"]["pooled"]["roc_auc"],
                             "cold_pr_auc": r["metrics"]["has_edge_cold"]["pr_auc"], "warm_pr_auc": r["metrics"]["has_edge_warm"]["pr_auc"],
                             "no_edge_pr_auc": r["metrics"]["no_pseudo_account_edge"]["pr_auc"]} for r in results],
               "aggregate": agg, "total_script_seconds": round(time.time() - t_start, 2)}
    with open(base_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    with open(base_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump({"timestamp_utc": pd.Timestamp.now("UTC").isoformat(), "python": platform.python_version(),
                   "torch": torch.__version__, "torch_geometric": pyg.__version__, "device": str(dev),
                   "state_handling": {"initial_state": "learned per-type h0 for never-seen entities, zeros buffer otherwise; day-0 start",
                                      "train_to_val_boundary": "state carried forward unchanged (no reset); eval rebuilds state from day 1 with current parameters in eval mode",
                                      "val_to_test_boundary": "not executed in Phase 7 (TEST sealed); designated procedure: continue the same stream from day 151 with final parameters, never reset, never using labels",
                                      "labels_in_state": False},
                   "total_script_seconds": round(time.time() - t_start, 2)}, fh, indent=1)
    print(f"\nTotal runtime {time.time()-t_start:.1f}s. TEST split was not loaded or touched.", flush=True)


if __name__ == "__main__":
    main()
