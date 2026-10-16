"""Phase 8 — shared initial checkpoint + pre-registered hyperparameters.

NOT PART OF THE ORIGINAL DESIGN-DOCUMENT FILE LIST (Section 12 lists train_phase8_frozen/adaptive/
replay_control + measure_drift/validate_leakage/compare). This script is a small, disclosed
ADDITION, not a design change: it factors out "fit the shared initial TRAIN-only checkpoint and
lock K/LR/threshold" into ONE script, called once per seed, so that F/Ad/E's scripts can each
*load* the identical saved checkpoint rather than independently re-fitting and hoping three
separate runs coincide bit-for-bit. This is how check #7 ("F, Ad, and E begin from the same
initial TRAIN-only checkpoint") is satisfied BY CONSTRUCTION rather than by chance. The design's
required outputs and comparisons are unaffected; only how the code guarantees one of the design's
own stated requirements is implemented here.

  python -u scripts/train_phase8_init.py --seed 42     # writes the shared init artifacts for seed 42

THIS SCRIPT PERFORMS REAL TRAINING (TRAIN-only, days 1-121, via a TRAIN-internal holdout) WHEN
RUN. It is written now, as implementation, and is NOT executed by this implementation pass.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import torch

from src.adaptive.hparams import fit_initial_model, select_hyperparameters
from src.config import load_config
from src.graph.temporal import load_raw_arrays

EXPERIMENT = "phase8_adaptive_drift/shared_init"
CHECKPOINT_NAME_TEMPLATE = "model_checkpoint_initial_seed{seed}.pt"  # must match train_phase8_{frozen,adaptive,replay_control}.py's init_ckpt path exactly


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    args = ap.parse_args()
    cfg = load_config()
    out_dir = cfg.experiments_dir / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    raw, graph, scaler = load_raw_arrays(cfg)  # TRAIN-only scaler, same as every prior phase
    hp = select_hyperparameters(raw, cfg, dev, base_seed=args.seed)  # TRAIN-only holdout; see src/adaptive/hparams.py

    with open(out_dir / f"hparams_seed{args.seed}.json", "w", encoding="utf-8") as fh:
        json.dump({"fine_tune_epochs": hp.fine_tune_epochs, "fine_tune_lr": hp.fine_tune_lr,
                   "threshold": hp.threshold, "base_seed": hp.base_seed, "selected_from": hp.selected_from,
                   "holdout_fit_end": hp.holdout_fit_end, "holdout_select_start": hp.holdout_select_start,
                   "holdout_select_end": hp.holdout_select_end, "locked": hp.locked, "extra": hp.extra}, fh, indent=1)
    print(f"seed {args.seed}: K={hp.fine_tune_epochs} lr={hp.fine_tune_lr} threshold={hp.threshold:.4f} "
          f"(holdout PR-AUC {hp.extra.get('holdout_pr_auc')}) -- locked, will not be revisited using VAL", flush=True)

    # The shared initial checkpoint: ONE clean re-fit, on TRAIN days 1..hp.holdout_fit_end only,
    # for exactly hp.fine_tune_epochs epochs at hp.fine_tune_lr, reusing Phase 7's train_epoch
    # directly (src.adaptive.hparams.fit_initial_model). F, Ad, and E all load this exact file --
    # see CHECKPOINT_NAME_TEMPLATE above and each arm script's `init_ckpt` path, which must stay
    # byte-identical to this one (checked statically in tests/test_phase8_adaptive.py).
    model = fit_initial_model(raw, cfg, dev, args.seed, hp)
    ckpt_path = out_dir / CHECKPOINT_NAME_TEMPLATE.format(seed=args.seed)
    torch.save(model.state_dict(), ckpt_path)
    print(f"seed {args.seed}: wrote shared initial checkpoint to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
