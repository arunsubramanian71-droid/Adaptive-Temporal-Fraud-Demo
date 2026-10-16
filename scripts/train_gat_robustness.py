"""Phase 6B: the GAT experiment.

Exact Phase 4/5/6A methodology — same graph, features, split, rolling
per-day temporal protocol, leakage rules, 3-way stratification, class-
imbalance treatment, 150-epoch/patience-20 framework, and 3 seeds (42, 123,
2024). The only change from Phase 6A's GraphSAGE sweep is the model:
HeteroGAT (attention) instead of HeteroSAGE (mean/sum pooling) — see
src/models/gat.py for why the two are dimension-matched for a fair test.

A single feasibility pass (forward+backward, real measurement) is run
before the full sweep, mirroring Phase 4's own resource-discipline
practice, since GAT's per-edge attention coefficients give it a different
memory profile than SAGEConv and a real number is better than a guess
before committing to 3 x up to 150 epochs.

Writes to experiments/gat/ — does not touch experiments/graphsage_robust/,
experiments/graphsage/, experiments/graphsage_control_mlp/, or any Phase 3
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
from src.graph.schema import compute_pseudo_account_segments, load_graph_config
from src.models.gat import HeteroGAT
from src.models.metrics import compute_metrics, select_best_f1_threshold

SEEDS = [42, 123, 2024]
HIDDEN = 64
HEADS = 4
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-5
MAX_EPOCHS = 150
PATIENCE = 20
EXPERIMENT_NAME = "gat"


def build_scaled_graph(cfg):
    """Identical to Phase 6A's build_scaled_graph — same TRAIN-only
    StandardScaler, same in-memory-only modification. Phase 4's saved
    hetero_graph.pt is never touched."""
    graph = torch.load(cfg.processed_dir / "graph" / "hetero_graph.pt", weights_only=False)
    day = graph["transaction"].day.numpy()
    train_mask_np = (day >= cfg.split.train_day_min) & (day <= cfg.split.train_day_max)
    x = graph["transaction"].x.numpy()
    scaler = StandardScaler().fit(x[train_mask_np])
    graph["transaction"].x = torch.tensor(scaler.transform(x), dtype=torch.float32)
    return graph, scaler


def feasibility_pass(graph, cfg, dev) -> dict:
    """One real forward+backward pass on the TRAIN-masked graph, measuring
    actual GPU memory/time before committing to the full sweep — the same
    discipline Phase 4 used before its own GraphSAGE feasibility check."""
    feature_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    torch.manual_seed(0)
    model = HeteroGAT(graph.metadata(), feature_dims, hidden=HIDDEN, heads=HEADS, dropout=DROPOUT).to(dev)
    train_graph = apply_time_mask(graph, cfg.split.train_day_max)
    train_idx = (graph["transaction"].day <= cfg.split.train_day_max).nonzero(as_tuple=True)[0]
    y = graph["transaction"].y
    pos_weight = torch.tensor(27.36, device=dev)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    opt.zero_grad(set_to_none=True)
    logits = model(train_graph.x_dict, train_graph.edge_index_dict)
    loss = F.binary_cross_entropy_with_logits(logits[train_idx], y[train_idx], pos_weight=pos_weight)
    loss.backward()
    opt.step()
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    result = {"time_seconds": round(dt, 3), "loss": float(loss.item()),
              "parameter_count": sum(p.numel() for p in model.parameters())}
    if dev.type == "cuda":
        result["peak_gpu_memory_reserved_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 3)
        result["peak_gpu_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    del model, opt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return result


def run_one_seed(seed: int, graph, cfg, dev, out_dir: pathlib.Path) -> dict:
    print(f"\n{'='*70}\nSEED {seed}\n{'='*70}")
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

        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_pr_auc": float(val_pr_auc),
                        "val_roc_auc": float(val_roc_auc), "epoch_seconds": round(t_ep, 3)})
        improved = val_pr_auc > best_pr_auc + 1e-4
        if epoch % 10 == 0 or improved or epoch == 1:
            print(f"  epoch {epoch:>3}: loss={loss.item():.4f}  VAL PR-AUC={val_pr_auc:.4f}  "
                  f"VAL ROC-AUC={val_roc_auc:.4f}  ({t_ep:.2f}s){' *' if improved else ''}")
        if improved:
            best_pr_auc, best_epoch, patience_left = val_pr_auc, epoch, PATIENCE
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"  EARLY STOPPING at epoch {epoch} (best epoch {best_epoch}, PR-AUC {best_pr_auc:.4f})")
                stopped_early = True
                break

    t_train = time.time() - t_train0
    model.load_state_dict(best_state)
    print(f"  training complete in {t_train:.1f}s, {len(history)} epochs run "
          f"({'early-stopped' if stopped_early else 'hit max_epochs — NOT converged, reported as-is'}), "
          f"best epoch={best_epoch} (VAL PR-AUC {best_pr_auc:.4f})")

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
                "epochs_run": len(history), "best_epoch": best_epoch, "stopped_early": stopped_early}
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

    return {"seed": seed, "metrics": metrics_by_segment, "history": history, "resource": resource}


def main() -> None:
    t_start = time.time()
    cfg = load_config()
    base_dir = cfg.experiments_dir / EXPERIMENT_NAME
    base_dir.mkdir(parents=True, exist_ok=True)

    print("=== 1. Loading + scaling graph (identical to Phase 6A; Phase 4 artifact untouched) ===")
    graph, scaler = build_scaled_graph(cfg)
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    graph = graph.to(dev)
    print(f"  device: {dev}")

    print("\n=== 2. Architecture ===")
    feature_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    tmp = HeteroGAT(graph.metadata(), feature_dims, hidden=HIDDEN, heads=HEADS, dropout=DROPOUT)
    n_params = sum(p.numel() for p in tmp.parameters())
    del tmp
    print(f"  2 HeteroConv/GATConv layers, hidden={HIDDEN}, heads={HEADS} (concat layer 1 -> {HIDDEN}, "
          f"average layer 2 -> {HIDDEN}), dropout={DROPOUT}, ReLU activation, Linear(64,1) classifier head")
    print(f"  parameter count: {n_params:,} (GraphSAGE reference: 162,817)")

    print("\n=== 3. Resource discipline: real feasibility pass BEFORE the full sweep ===")
    feas = feasibility_pass(graph, cfg, dev)
    print(f"  single forward+backward pass: {feas}")
    est_total = MAX_EPOCHS * max(feas["time_seconds"], 1.0) * 1.3 * len(SEEDS)  # +30% for the 30 rolling VAL passes/epoch
    print(f"  estimated worst case (no early stopping, any seed): ~{est_total/60:.1f} min for all {len(SEEDS)} seeds")
    if dev.type == "cuda":
        print(f"  measured peak GPU memory on this single pass: {feas.get('peak_gpu_memory_reserved_gib')} GiB "
              f"reserved (of 15.98 GiB total) — {'SAFE, proceeding automatically' if feas.get('peak_gpu_memory_reserved_gib', 99) < 12 else 'CLOSE TO LIMIT'}")

    seed_results = []
    for seed in SEEDS:
        out_dir = base_dir / f"seed_{seed}"
        result = run_one_seed(seed, graph, cfg, dev, out_dir)
        seed_results.append(result)
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\n{'='*70}\n=== 4. Cross-seed aggregation ===\n{'='*70}")
    agg = {}
    for metric_key, path in [("pooled_pr_auc", ("pooled", "pr_auc")), ("pooled_roc_auc", ("pooled", "roc_auc")),
                              ("cold_pr_auc", ("has_edge_cold", "pr_auc")), ("cold_roc_auc", ("has_edge_cold", "roc_auc")),
                              ("warm_pr_auc", ("has_edge_warm", "pr_auc")), ("warm_roc_auc", ("has_edge_warm", "roc_auc")),
                              ("no_edge_pr_auc", ("no_pseudo_account_edge", "pr_auc")),
                              ("no_edge_roc_auc", ("no_pseudo_account_edge", "roc_auc"))]:
        vals = [r["metrics"][path[0]][path[1]] for r in seed_results]
        agg[metric_key] = {"values": vals, "mean": float(np.mean(vals)), "std": float(np.std(vals, ddof=1)),
                          "min": float(np.min(vals)), "max": float(np.max(vals))}
        print(f"  {metric_key:<16}: {[round(v,4) for v in vals]}  mean={agg[metric_key]['mean']:.4f}  "
              f"std={agg[metric_key]['std']:.4f}  min={agg[metric_key]['min']:.4f}  max={agg[metric_key]['max']:.4f}")

    print("\n=== 5. Generating convergence plots ===")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    colors = {42: "tab:blue", 123: "tab:orange", 2024: "tab:green"}
    for r in seed_results:
        h = pd.DataFrame(r["history"])
        c = colors[r["seed"]]
        axes[0].plot(h["epoch"], h["train_loss"], label=f"seed {r['seed']}", color=c)
        axes[1].plot(h["epoch"], h["val_pr_auc"], label=f"seed {r['seed']}", color=c)
        axes[1].axvline(r["resource"]["best_epoch"], color=c, linestyle=":", alpha=0.5)
        axes[2].plot(h["epoch"], h["val_roc_auc"], label=f"seed {r['seed']}", color=c)
    axes[0].set_title("GAT training loss"); axes[0].set_xlabel("epoch"); axes[0].set_ylabel("BCE loss"); axes[0].legend()
    axes[1].set_title("GAT validation PR-AUC (rolling, pooled)"); axes[1].set_xlabel("epoch"); axes[1].legend()
    axes[2].set_title("GAT validation ROC-AUC (rolling, pooled)"); axes[2].set_xlabel("epoch"); axes[2].legend()
    plt.tight_layout()
    plot_path = base_dir / "convergence_curves.png"
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  wrote {plot_path}")

    summary = {
        "seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "heads": HEADS,
        "feasibility_pass": feas,
        "per_seed": [{"seed": r["seed"], "best_epoch": r["resource"]["best_epoch"],
                     "epochs_run": r["resource"]["epochs_run"], "stopped_early": r["resource"]["stopped_early"],
                     "pooled_pr_auc": r["metrics"]["pooled"]["pr_auc"], "pooled_roc_auc": r["metrics"]["pooled"]["roc_auc"],
                     "cold_pr_auc": r["metrics"]["has_edge_cold"]["pr_auc"], "warm_pr_auc": r["metrics"]["has_edge_warm"]["pr_auc"],
                     "no_edge_pr_auc": r["metrics"]["no_pseudo_account_edge"]["pr_auc"],
                     "training_wall_seconds": r["resource"]["training_wall_seconds"],
                     "peak_gpu_memory_reserved_gib": r["resource"].get("peak_gpu_memory_reserved_gib")} for r in seed_results],
        "aggregate": agg,
        "parameter_count": n_params,
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


if __name__ == "__main__":
    main()
