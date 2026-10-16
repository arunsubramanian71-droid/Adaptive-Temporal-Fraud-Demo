"""Graph feasibility benchmark — Phase 4 requirement 15.

NOT the GraphSAGE experiment. This is a single forward + backward pass on
the REAL constructed graph (real features, real edges, real time-masking),
to confirm the graph actually fits and trains within budget on this GPU,
and to replace the scaled *estimate* from scripts/estimate_graph_schema.py
with a *measured* number. No training loop, no epochs, no metrics, no
model comparison.
"""
from __future__ import annotations

import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv

from src.config import load_config
from src.graph.build import apply_time_mask

HIDDEN = 64  # matches the size already verified safe in the Phase 2 GPU capability probe


class HeteroGNN(torch.nn.Module):
    def __init__(self, metadata, feature_dims, hidden=HIDDEN):
        super().__init__()
        self.proj = torch.nn.ModuleDict({nt: torch.nn.Linear(feature_dims[nt], hidden) for nt in metadata[0]})
        self.convs = torch.nn.ModuleList([
            HeteroConv({et: SAGEConv((hidden, hidden), hidden) for et in metadata[1]}, aggr="sum")
            for _ in range(2)
        ])
        self.head = torch.nn.Linear(hidden, 1)

    def forward(self, x_dict, edge_index_dict):
        h = {k: self.proj[k](v).relu() for k, v in x_dict.items()}
        for conv in self.convs:
            h = {k: v.relu() for k, v in conv(h, edge_index_dict).items()}
        return self.head(h["transaction"]).squeeze(-1)


def run_pass(graph, dev, do_backward: bool, label: str) -> dict:
    feature_dims = {nt: graph[nt].x.shape[1] for nt in graph.node_types}
    torch.manual_seed(42)
    model = HeteroGNN(graph.metadata(), feature_dims).to(dev)
    g = graph.to(dev)

    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    t0 = time.time()

    if do_backward:
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        logits = model(g.x_dict, g.edge_index_dict)
        pos_weight = torch.tensor(27.58, device=dev)
        loss = F.binary_cross_entropy_with_logits(logits, g["transaction"].y, pos_weight=pos_weight)
        loss.backward()
        opt.step()
        result_val = float(loss.item())
    else:
        model.eval()
        with torch.no_grad():
            logits = model(g.x_dict, g.edge_index_dict)
        result_val = float(logits.mean().item())

    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.time() - t0

    mem = {}
    if dev.type == "cuda":
        mem = {"peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
               "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 3)}
    del model, g
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    n_edges = sum(graph[et].edge_index.shape[1] for et in graph.edge_types)
    print(f"  [{label}] device={dev.type} nodes={sum(graph[nt].num_nodes for nt in graph.node_types):,} "
          f"edges={n_edges:,} time={dt*1000:.1f}ms {mem} sample_value={result_val:.4f}")
    return {"label": label, "device": dev.type, "n_edges": n_edges, "time_ms": round(dt * 1000, 1), **mem}


def main() -> None:
    cfg = load_config()
    graph_dir = cfg.processed_dir / "graph"
    graph = torch.load(graph_dir / "hetero_graph.pt", weights_only=False)

    print("=== Graph feasibility benchmark (single forward/backward pass, NOT training) ===")
    print(f"hidden_dim={HIDDEN}, 2-layer HeteroConv+SAGEConv, real graph, real time-masking")

    results = []
    dev_cuda = torch.device("cuda:0") if torch.cuda.is_available() else None
    if dev_cuda is None:
        print("  CUDA/ROCm not available in this process — CPU-only benchmark")

    print("\n-- TRAIN-masked view (cutoff day=121): forward + backward --")
    train_graph = apply_time_mask(graph, cfg.split.train_day_max)
    if dev_cuda:
        results.append(run_pass(train_graph, dev_cuda, do_backward=True, label="train_masked_gpu_fwd_bwd"))
    results.append(run_pass(train_graph, torch.device("cpu"), do_backward=True, label="train_masked_cpu_fwd_bwd"))

    print("\n-- VAL-masked view (cutoff day=151): forward only (inference) --")
    val_graph = apply_time_mask(graph, cfg.split.val_day_max)
    if dev_cuda:
        results.append(run_pass(val_graph, dev_cuda, do_backward=False, label="val_masked_gpu_fwd_only"))

    out_path = graph_dir / "feasibility_benchmark.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nWrote {out_path}")
    print("This was a single-pass FEASIBILITY CHECK only — no training loop, no metrics, no GraphSAGE experiment.")


if __name__ == "__main__":
    main()
