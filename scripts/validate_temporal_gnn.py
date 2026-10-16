"""Phase 7 temporal-leakage validation suite.

1. Re-runs the complete Phase 4 graph validation suite (scripts/validate_graph.py).
2. Runs the temporal-state checks (src/graph/temporal_checks.py) on the
   TRAINED checkpoints of all three seeds, at validation scale (days 1-151).

TEST days are never loaded: the dataset builder refuses them, and check
`no_test_data` verifies that.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.config import load_config
from src.graph import temporal_checks as TC
from src.graph.temporal import build_temporal_dataset, load_raw_arrays
from src.models.temporal_graphsage import TemporalGraphSAGE

SEEDS = [42, 123, 2024]
CUTOFFS = [122, 130, 140, 150]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu",
                    help="cpu = deterministic execution (exact bitwise equality required); cuda = noise-floor tolerance")
    args = ap.parse_args()
    cfg = load_config()
    base = cfg.experiments_dir / "temporal_graphsage"
    out_dir = base / "leakage_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    print(f"device for temporal-state checks: {dev} ({'deterministic - exact equality' if dev.type == 'cpu' else 'GPU - run-to-run noise-floor tolerance'})", flush=True)
    results = []

    print("=== 1. Phase 4 graph validation suite (unchanged, re-run) ===", flush=True)
    proc = subprocess.run([sys.executable, "-u", "scripts/validate_graph.py"], capture_output=True, text=True,
                          cwd=str(pathlib.Path(__file__).resolve().parent.parent))
    text = "\n".join(l for l in (proc.stdout + proc.stderr).splitlines() if "offload-arch" not in l)
    (out_dir / f"graph_validation_output_{args.device}.txt").write_text(text, encoding="utf-8")
    print(text, flush=True)
    graph_ok = proc.returncode == 0
    results.append(("graph_validation_suite_8_of_8", graph_ok, "scripts/validate_graph.py exit code 0" if graph_ok else "FAILED"))

    print("\n=== 2. Temporal-state checks on trained models ===", flush=True)
    raw, graph, scaler = load_raw_arrays(cfg)
    max_day, first_pred = cfg.split.val_day_max, cfg.split.val_day_min

    ds = build_temporal_dataset(raw, max_day, dev, cfg)
    for name, fn in [("no_test_data_loaded", lambda: TC.check_no_test_data(ds, raw, cfg)),
                     ("scaler_fit_on_train_only", lambda: TC.check_scaler_train_only(raw, cfg)),
                     ("edge_consumption_no_future_edges", lambda: TC.check_edge_consumption(raw, dev, cfg, max_day))]:
        ok, detail = fn()
        results.append((name, ok, detail))
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
    del ds

    for seed in SEEDS:
        ckpt = base / f"seed_{seed}" / "model_checkpoint.pt"
        model = TemporalGraphSAGE(raw["X"].shape[1]).to(dev)
        model.load_state_dict(torch.load(ckpt, map_location=dev))
        model.eval()
        print(f"\n  --- seed {seed} (trained checkpoint) ---", flush=True)
        checks = [
            ("future_invariance", lambda: TC.check_future_invariance(model, raw, dev, cfg, max_day, first_pred, CUTOFFS)),
            ("past_sensitivity_recurrent_memory_used", lambda: TC.check_past_sensitivity(model, raw, dev, cfg, max_day, first_pred)),
            ("train_val_boundary_state_continuity", lambda: TC.check_boundary_continuity(model, raw, dev, cfg, max_day, cfg.split.train_day_max)),
            ("per_step_access_only_own_day", lambda: TC.check_access_pattern(model, raw, dev, cfg, max_day, first_pred)),
            ("labels_never_reach_model_or_state", lambda: TC.check_labels_not_used(model, raw, dev, cfg, max_day, first_pred)),
        ]
        for name, fn in checks:
            ok, detail = fn()
            results.append((f"seed{seed}:{name}", ok, detail))
            print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}", flush=True)
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    n_pass = sum(1 for _, ok, _ in results if ok)
    print(f"\n{n_pass}/{len(results)} checks passed.", flush=True)
    with open(out_dir / f"temporal_leakage_validation_{args.device}.json", "w", encoding="utf-8") as fh:
        json.dump([{"check": n, "passed": bool(ok), "detail": d} for n, ok, d in results], fh, indent=1)
    print(f"Wrote {out_dir / f'temporal_leakage_validation_{args.device}.json'}", flush=True)
    return 0 if n_pass == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
