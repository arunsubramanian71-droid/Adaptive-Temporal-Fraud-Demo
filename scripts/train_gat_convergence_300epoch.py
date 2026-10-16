"""Phase 6C, Option A: GAT convergence-resolution experiment.

Identical Phase 6B architecture, graph, features, split, rolling per-day
temporal protocol, leakage rules, and 3-way stratification. The ONLY
change: max_epochs 150->300, patience 20->30, same optimizer/lr, to
determine whether GAT's non-convergence in Phase 6B was a training-budget
artifact rather than a genuine architectural ceiling.

Resource safety (checked before this run): Phase 6B's actual 150-epoch
training runs measured peak GPU memory of 12.10-12.32 GiB across all 3
seeds, matching the single-pass feasibility estimate (12.26 GiB) closely —
no memory growth across epochs, so extending the epoch count is not
expected to raise memory (confirmed empirically below, not just assumed).
hidden=64, heads=4 unchanged; seeds run strictly sequentially, one process.

Writes to experiments/gat_300epoch/ — does not touch experiments/gat/
(Phase 6B), experiments/graphsage_robust/ (Phase 6A), or any earlier
artifact.
"""
from __future__ import annotations

import json
import pathlib
import platform
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric as pyg
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from src.config import load_config
from src.graph.build import apply_time_mask
from src.graph.rolling_eval import rolling_predict
from src.graph.schema import compute_pseudo_account_segments
from src.models.gat import HeteroGAT
from src.models.metrics import compute_metrics, select_best_f1_threshold

SEEDS = [42, 123, 2024]
HIDDEN = 64
HEADS = 4
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 300
PATIENCE = 30
EXPERIMENT_NAME = "gat_300epoch"
GPU_MEMORY_SAFETY_LIMIT_GIB = 14.5  # abort a seed if exceeded, rather than risk an OOM crash mid-sweep


def build_scaled_graph(cfg):
    graph = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    day = graph["transaction"].day.numpy()
    train_mask_np = (day >= cfg.split.train_day_min) & (day <= cfg.split.train_day_max)
    x = graph["transaction"].x.numpy()
    scaler = StandardScaler().fit(x[train_mask_np])
    graph["transaction"].x = torch.tensor(scaler.transform(x), dtype=torch.float32)
    return graph, scaler


def run_one_seed(seed: int, graph, cfg, dev, out_dir: pathlib.Path) -> dict:
    print(f"\n{'='*70}\nSEED {seed}  (max_epochs={MAX_EPOCHS}, patience={PATIENCE})\n{'='*70}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    model = HeteroGAT(graph.metadata(), feature_dims, hidden=HIDDEN, heads=HEADS, dropout=DROPOUT).to(dev)
    n_params = sum(p.numel() for p in model.parameters())

    train_idx = (graph["transaction"].day <= cfg.split.train_day_max).nonzero(as_tuple=True)[0]
    y_full = graph["transaction"].y
    scale_pos_weight = float((y_full[train_idx] == 0).sum() / max((y_full[train_idx] == 1).sum(), 1))
    pos_weight = torch.tensor(scale_pos_weight, device=dev)
    train_graph = apply_time_mask(graph, cfg.split.train_day_max)

    val_days = list(range(cfg.split.val_day_min, cfg.split.val_day_max + 1))
    y_true_by_id = pd.Series(y_full.cpu().numpy(), index=graph["transaction"].transaction_id.cpu().numpy())

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    best_pr_auc, best_epoch, patience_left, best_state = -1.0, -1, PATIENCE, None
    history = []
    t_train0 = time.time()
    stopped_early = False
    aborted_for_memory = False

    for epoch in range(1, MAX_EPOCHS + 1):
        t_ep0 = time.time()
        model.train()
        opt.zero_grad(set_to_none=True)
        logits = model(train_graph.x_dict, train_graph.edge_index_dict)
        loss = F.binary_cross_entropy_with_logits(logits[train_idx], y_full[train_idx], pos_weight=pos_weight)
        loss.backward()
        opt.step()

        ids, scores, _ = rolling_predict(model, graph, val_days)
        y_true = y_true_by_id.loc[ids].to_numpy()
        val_pr_auc = average_precision_score(y_true, scores)
        val_roc_auc = roc_auc_score(y_true, scores)
        t_ep = time.time() - t_ep0

        cur_mem = torch.cuda.max_memory_reserved() / 2**30 if dev.type == "cuda" else 0.0
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_pr_auc": float(val_pr_auc),
                        "val_roc_auc": float(val_roc_auc), "epoch_seconds": round(t_ep, 3),
                        "gpu_mem_reserved_gib": round(cur_mem, 3)})
        improved = val_pr_auc > best_pr_auc + 1e-4
        if epoch % 20 == 0 or improved or epoch == 1:
            print(f"  epoch {epoch:>3}: loss={loss.item():.4f}  VAL PR-AUC={val_pr_auc:.4f}  "
                  f"VAL ROC-AUC={val_roc_auc:.4f}  mem={cur_mem:.2f}GiB  ({t_ep:.2f}s){' *' if improved else ''}")
        if improved:
            best_pr_auc, best_epoch, patience_left = val_pr_auc, epoch, PATIENCE
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  EARLY STOPPING at epoch {epoch} (best epoch {best_epoch}, PR-AUC {best_pr_auc:.4f})")
                stopped_early = True
                break

        if dev.type == "cuda" and cur_mem > GPU_MEMORY_SAFETY_LIMIT_GIB:
            print(f"  ABORTING seed {seed} at epoch {epoch}: GPU memory {cur_mem:.2f} GiB exceeded safety "
                  f"limit {GPU_MEMORY_SAFETY_LIMIT_GIB} GiB. Reporting as unsafe rather than risking an OOM crash.")
            aborted_for_memory = True
            break

    t_train = time.time() - t_train0
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  training complete in {t_train:.1f}s, {len(history)} epochs run "
          f"({'early-stopped' if stopped_early else ('ABORTED (memory)' if aborted_for_memory else 'hit max_epochs — NOT converged')}), "
          f"best epoch={best_epoch} (VAL PR-AUC {best_pr_auc:.4f})")

    if aborted_for_memory:
        resource = {"parameter_count": n_params, "training_wall_seconds": round(t_train, 2),
                    "epochs_run": len(history), "best_epoch": best_epoch, "stopped_early": False,
                    "aborted_for_memory": True}
        with open(out_dir / "resource_measurements.json", "w", encoding="utf-8") as fh:
            json.dump(resource, fh, indent=1)
        with open(out_dir / "training_history.json", "w", encoding="utf-8") as fh:
            json.dump(history, fh, indent=1)
        return {"seed": seed, "aborted_for_memory": True, "history": history, "resource": resource}

    ids, scores, days_arr = rolling_predict(model, graph, val_days)
    y_true = y_true_by_id.loc[ids].to_numpy()
    thresh_info = select_best_f1_threshold(y_true, scores)
    metrics_pooled = compute_metrics(y_true, scores, threshold=thresh_info["threshold"])
    metrics_pooled["threshold_selection"] = thresh_info

    segment_all = compute_pseudo_account_segments(graph)
    txn_id_all = graph["transaction"].transaction_id.cpu().numpy()
    segment_by_id = pd.Series(segment_all, index=txn_id_all)
    seg = segment_by_id.loc[ids].to_numpy()

    metrics_by_segment = {"pooled": metrics_pooled}
    for seg_name in ["has_edge_warm", "has_edge_cold", "no_pseudo_account_edge"]:
        m = seg == seg_name
        yt, sc = y_true[m], scores[m]
        if len(yt) == 0 or yt.sum() == 0 or yt.sum() == len(yt):
            metrics_by_segment[seg_name] = {"n_rows": int(len(yt)), "n_fraud": int(yt.sum()), "note": "insufficient class balance"}
            continue
        metrics_by_segment[seg_name] = compute_metrics(yt, sc, threshold=thresh_info["threshold"])

    print(f"  FINAL: pooled PR-AUC={metrics_pooled['pr_auc']:.4f}  "
          f"cold={metrics_by_segment['has_edge_cold']['pr_auc']:.4f}  "
          f"warm={metrics_by_segment['has_edge_warm']['pr_auc']:.4f}  "
          f"no-edge={metrics_by_segment['no_pseudo_account_edge']['pr_auc']:.4f}")

    resource = {"parameter_count": n_params, "training_wall_seconds": round(t_train, 2),
                "epochs_run": len(history), "best_epoch": best_epoch, "stopped_early": stopped_early,
                "aborted_for_memory": False}
    if dev.type == "cuda":
        resource["peak_gpu_memory_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        resource["peak_gpu_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)

    torch.save(best_state, out_dir / "model_checkpoint.pt")
    with open(out_dir / "metrics_val.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_by_segment, fh, indent=1)
    with open(out_dir / "training_history.json", "w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=1)
    with open(out_dir / "resource_measurements.json", "w", encoding="utf-8") as fh:
        json.dump(resource, fh, indent=1)
    preds = pd.DataFrame({"TransactionID": ids, "isFraud": y_true, "y_score": scores, "day": days_arr,
                          "y_pred": (scores >= thresh_info["threshold"]).astype(int), "segment": seg})
    preds.to_parquet(out_dir / "predictions_val.parquet", index=False)
    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump({"seed": seed, "hidden": HIDDEN, "heads": HEADS, "dropout": DROPOUT, "lr": LR,
                   "weight_decay": WEIGHT_DECAY, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
                   "scale_pos_weight": scale_pos_weight, "parameter_count": n_params}, fh, indent=1)

    return {"seed": seed, "aborted_for_memory": False, "metrics": metrics_by_segment, "history": history, "resource": resource}


def main() -> None:
    t_start = time.time()
    cfg = load_config()
    base_dir = cfg.experiments_dir / EXPERIMENT_NAME
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading + scaling graph (identical to Phase 6B; Phase 4 artifact untouched) ===")
    graph, scaler = build_scaled_graph(cfg)
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    graph = graph.to(dev)
    print(f"  device: {dev}")

    print("\n=== 2. Resource safety check (before committing to the full sweep) ===")
    print(f"  Phase 6B measured sustained training peak: 12.10-12.32 GiB across 150 epochs, 3 seeds — "
          f"stable, no growth over epochs. Safety abort threshold for this run: {GPU_MEMORY_SAFETY_LIMIT_GIB} GiB "
          f"(headroom to the 15.98 GiB card).")
    print(f"  max_epochs={MAX_EPOCHS} (2x Phase 6B), patience={PATIENCE} (1.5x Phase 6B), same lr/optimizer, "
          f"seeds={SEEDS}, run strictly sequentially in this one process.")
    print(f"  Phase 6B measured ~5.1-5.3s/epoch -> up to {MAX_EPOCHS}*5.3={MAX_EPOCHS*5.3:.0f}s per seed worst "
          f"case, x{len(SEEDS)} seeds -> up to ~{MAX_EPOCHS*5.3*len(SEEDS)/60:.0f} min worst case if no seed "
          f"early-stops. Proceeding automatically per the Phase 6C execution policy.")

    seed_results = []
    for seed in SEEDS:
        out_dir = base_dir / f"seed_{seed}"
        result = run_one_seed(seed, graph, cfg, dev, out_dir)
        seed_results.append(result)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    any_aborted = any(r.get("aborted_for_memory") for r in seed_results)
    if any_aborted:
        print("\n" + "=" * 70)
        print("ONE OR MORE SEEDS WERE ABORTED FOR MEMORY SAFETY. See per-seed resource_measurements.json.")
        print("=" * 70)

    converged_results = [r for r in seed_results if not r.get("aborted_for_memory")]
    n_early_stopped = sum(1 for r in converged_results if r["resource"]["stopped_early"])

    print(f"\n{'='*70}\n=== 3. Cross-seed aggregation ({len(converged_results)} of {len(SEEDS)} seeds completed) ===\n{'='*70}")
    agg = {}
    if converged_results:
        for metric_key, path in [("pooled_pr_auc", ("pooled", "pr_auc")), ("pooled_roc_auc", ("pooled", "roc_auc")),
                                  ("cold_pr_auc", ("has_edge_cold", "pr_auc")), ("cold_roc_auc", ("has_edge_cold", "roc_auc")),
                                  ("warm_pr_auc", ("has_edge_warm", "pr_auc")), ("warm_roc_auc", ("has_edge_warm", "roc_auc")),
                                  ("no_edge_pr_auc", ("no_pseudo_account_edge", "pr_auc")),
                                  ("no_edge_roc_auc", ("no_pseudo_account_edge", "roc_auc"))]:
            vals = [r["metrics"][path[0]][path[1]] for r in converged_results]
            agg[metric_key] = {"values": vals, "mean": float(np.mean(vals)),
                              "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                              "min": float(np.min(vals)), "max": float(np.max(vals))}
            print(f"  {metric_key:<16}: {[round(v,4) for v in vals]}  mean={agg[metric_key]['mean']:.4f}  "
                  f"std={agg[metric_key]['std']:.4f}  min={agg[metric_key]['min']:.4f}  max={agg[metric_key]['max']:.4f}")

    print(f"\n  {n_early_stopped}/{len(converged_results)} seeds triggered genuine early stopping "
          f"(patience={PATIENCE} exhausted after peaking) — the rest hit max_epochs={MAX_EPOCHS} without converging.")

    print("\n=== 4. Generating convergence plots ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {42: "tab:blue", 123: "tab:orange", 2024: "tab:green"}
    for r in seed_results:
        h = pd.DataFrame(r["history"])
        c = colors[r["seed"]]
        axes[0].plot(h["epoch"], h["train_loss"], label=f"seed {r['seed']}", color=c)
        if "val_pr_auc" in h.columns:
            axes[1].plot(h["epoch"], h["val_pr_auc"], label=f"seed {r['seed']}", color=c)
            if not r.get("aborted_for_memory"):
                axes[1].axvline(r["resource"]["best_epoch"], color=c, linestyle=":", alpha=0.5)
            axes[2].plot(h["epoch"], h["val_roc_auc"], label=f"seed {r['seed']}", color=c)
    axes[0].set_title("GAT (300-epoch) training loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss"); axes[0].legend()
    axes[1].set_title("GAT (300-epoch) validation PR-AUC (rolling, pooled)"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].set_title("GAT (300-epoch) validation ROC-AUC (rolling, pooled)"); axes[2].set_xlabel("epoch"); axes[2].legend()
    plt.tight_layout()
    plot_path = base_dir / "convergence_curves.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  wrote {plot_path}")

    summary = {
        "seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "heads": HEADS,
        "n_seeds_early_stopped": n_early_stopped, "n_seeds_completed": len(converged_results),
        "any_aborted_for_memory": any_aborted,
        "per_seed": [{"seed": r["seed"], "aborted_for_memory": r.get("aborted_for_memory", False),
                     **({"best_epoch": r["resource"]["best_epoch"], "epochs_run": r["resource"]["epochs_run"],
                         "stopped_early": r["resource"]["stopped_early"],
                         "pooled_pr_auc": r["metrics"]["pooled"]["pr_auc"], "pooled_roc_auc": r["metrics"]["pooled"]["roc_auc"],
                         "cold_pr_auc": r["metrics"]["has_edge_cold"]["pr_auc"], "warm_pr_auc": r["metrics"]["has_edge_warm"]["pr_auc"],
                         "no_edge_pr_auc": r["metrics"]["no_pseudo_account_edge"]["pr_auc"],
                         "training_wall_seconds": r["resource"]["training_wall_seconds"],
                         "peak_gpu_memory_reserved_gib": r["resource"].get("peak_gpu_memory_reserved_gib")}
                        if not r.get("aborted_for_memory") else {})} for r in seed_results],
        "aggregate": agg,
        "total_script_seconds": round(time.time() - t_start, 2),
    }
    with open(base_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)

    metadata = {
        "timestamp_utc": pd.Timestamp.now("UTC").isoformat(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_geometric_version": pyg.__version__,
        "device": str(dev),
        "total_script_seconds": round(time.time() - t_start, 2),
    }
    with open(base_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=1)

    print(f"\nWrote {base_dir / 'summary.json'} and {base_dir / 'metadata.json'}")
    print(f"Total runtime: {time.time()-t_start:.1f}s across {len(SEEDS)} seeds. TEST split was not loaded or touched.")
    print(f"\nCONVERGENCE VERDICT: {n_early_stopped}/{len(SEEDS)} seeds converged (genuine early stop). "
          f"{'ALL SEEDS CONVERGED -> use this as the resolution result.' if n_early_stopped == len(SEEDS) else 'NOT all seeds converged -> Option B (LR schedule) should be considered.'}")


if __name__ == "__main__":
    main()
