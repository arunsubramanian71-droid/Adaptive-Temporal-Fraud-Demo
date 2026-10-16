"""TRAIN-only hyperparameter pre-registration (Phase 8 design Section 8 / 14.3).

`K` (fine-tune epochs per update), the fine-tune learning rate, and the classification threshold
are each selected ONCE, before any VAL window is scored, using only a TRAIN-internal holdout
(fit on TRAIN days 1-111, select on TRAIN days 112-121). No function in this module ever reads a
VAL (122-151) or TEST (152-182) day. `select_hyperparameters` performs real training on TRAIN-only
data (forward+backward passes, GPU/CPU) — it is written here so Phase 8 execution can call it, but
it is NOT invoked anywhere in this implementation pass. The only thing exercised by the lightweight
test suite is the pure, training-free validation guard on `Phase8Hyperparams`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Pre-registered candidate grid (small and cheap; NOT tuned against VAL — selected once, below,
# using only the TRAIN-internal holdout days 112-121).
K_CANDIDATES: tuple[int, ...] = (5, 10, 15)
LR_CANDIDATES: tuple[float, ...] = (1e-3, 3e-4, 1e-4)

HOLDOUT_FIT_END = 111      # fit on TRAIN days 1..HOLDOUT_FIT_END
HOLDOUT_SELECT_START = 112  # select K/LR/threshold on TRAIN days HOLDOUT_SELECT_START..HOLDOUT_SELECT_END
HOLDOUT_SELECT_END = 121


@dataclass(frozen=True)
class Phase8Hyperparams:
    """A locked, pre-registered hyperparameter set. Once constructed, these values are used
    unchanged for every window, every seed, every arm (Ad and E share the same K/LR; the
    threshold is shared by F/Ad/E) — Section 14.3 of the locked design."""
    fine_tune_epochs: int
    fine_tune_lr: float
    threshold: float
    base_seed: int
    selected_from: str = "TRAIN_internal_holdout"
    holdout_fit_end: int = HOLDOUT_FIT_END
    holdout_select_start: int = HOLDOUT_SELECT_START
    holdout_select_end: int = HOLDOUT_SELECT_END
    locked: bool = True
    extra: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Pure, training-free structural guards: fine_tune_epochs/lr/threshold must be in sane
        # ranges, and the fit/selection holdout ranges must not overlap. This is what the
        # lightweight test suite actually exercises; the VAL-related guard is a separate method
        # (`assert_holdout_precedes_val`) since it needs the project's split config to check.
        if self.fine_tune_epochs <= 0:
            raise ValueError(f"fine_tune_epochs must be positive, got {self.fine_tune_epochs}")
        if not (0.0 <= self.threshold <= 1.0):
            raise ValueError(f"threshold must be a probability in [0, 1], got {self.threshold}")
        if self.fine_tune_lr <= 0:
            raise ValueError(f"fine_tune_lr must be positive, got {self.fine_tune_lr}")
        if self.holdout_fit_end >= self.holdout_select_start:
            raise ValueError("holdout fit range and selection range must not overlap "
                              f"(fit ends {self.holdout_fit_end}, selection starts {self.holdout_select_start})")

    def assert_holdout_precedes_val(self, val_day_min: int) -> None:
        """The one check that actually matters for leakage: the selection holdout must end
        strictly before VAL begins. Call this with the project's real val_day_min (122) before
        trusting any Phase8Hyperparams instance."""
        if self.holdout_select_end >= val_day_min:
            raise ValueError(f"holdout selection window ends on day {self.holdout_select_end}, which is not "
                              f"strictly before VAL (starts day {val_day_min}) — this would leak VAL information "
                              f"into hyperparameter selection and must never be constructed")


def select_hyperparameters(raw, cfg, dev, base_seed: int,
                            k_candidates: tuple[int, ...] = K_CANDIDATES,
                            lr_candidates: tuple[float, ...] = LR_CANDIDATES) -> Phase8Hyperparams:
    """Runs the TRAIN-only internal-holdout selection procedure for real: fits the Phase 7
    Temporal GraphSAGE architecture on TRAIN days 1..HOLDOUT_FIT_END, evaluates every
    (K, lr) candidate's effect via a short fine-tune continuation scored on TRAIN days
    HOLDOUT_SELECT_START..HOLDOUT_SELECT_END (never touching VAL/TEST), picks the best pooled
    PR-AUC candidate, and derives the classification threshold from that same holdout's scores
    via `select_best_f1_threshold`.

    THIS FUNCTION PERFORMS REAL TRAINING (forward+backward passes). It is written here so an
    authorized Phase 8 execution run can call it as its first step, but it is NOT invoked by
    this implementation pass or by the lightweight test suite — see
    experiments/phase8_adaptive_drift/DESIGN_BEFORE_EXECUTION.md Section 14.3 and the review
    instructions this module was written under.
    """
    import sys
    import pathlib

    import torch
    import torch.nn.functional as F
    from sklearn.metrics import average_precision_score

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "scripts"))
    from train_temporal_graphsage import GRAD_CLIP, TBPTT_DAYS  # Phase 7 artifact, imported unmodified

    from src.graph.temporal import build_temporal_dataset
    from src.models.metrics import select_best_f1_threshold
    from src.models.temporal_graphsage import TemporalGraphSAGE, rolling_temporal_predict

    assert HOLDOUT_SELECT_END < cfg.split.val_day_min, (
        "the holdout selection window must end strictly before VAL — refusing to proceed")

    ds = build_temporal_dataset(raw, HOLDOUT_SELECT_END, dev, cfg)  # only TRAIN days ever loaded here
    y_all = {int(i): float(y) for i, y in zip(raw["txn_id"], raw["y"])}

    best = {"pr_auc": -1.0, "k": None, "lr": None}
    for lr in lr_candidates:
        torch.manual_seed(base_seed)
        model = TemporalGraphSAGE(raw["X"].shape[1]).to(dev)
        tr_rows = ds.txn_day <= HOLDOUT_FIT_END
        y_tr = raw["y"][:len(tr_rows)][tr_rows]
        pos_weight = torch.tensor(float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), device=dev)
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

        for k in k_candidates:
            # fresh copy of the model's state so each (lr, k) candidate is evaluated independently
            model_k = TemporalGraphSAGE(raw["X"].shape[1]).to(dev)
            model_k.load_state_dict(model.state_dict())
            opt_k = torch.optim.Adam(model_k.parameters(), lr=lr, weight_decay=1e-5)
            for _ in range(k):
                state = model_k.init_state(ds)
                days = [d for d in ds.days if d <= HOLDOUT_FIT_END]
                for w0 in range(0, len(days), TBPTT_DAYS):
                    lgs, ys = [], []
                    for d in days[w0:w0 + TBPTT_DAYS]:
                        lgs.append(model_k.step(state, ds, d, predict=True))
                        ys.append(ds.labels(d))
                    lg, y = torch.cat(lgs), torch.cat(ys)
                    loss = F.binary_cross_entropy_with_logits(lg, y, pos_weight=pos_weight)
                    opt_k.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model_k.parameters(), GRAD_CLIP)
                    opt_k.step()
                    model_k.detach_state(state)
            ids, scores, _ = rolling_temporal_predict(model_k, ds, HOLDOUT_SELECT_START, HOLDOUT_SELECT_END)
            y_true = [y_all[int(i)] for i in ids]
            pr = float(average_precision_score(y_true, scores))
            if pr > best["pr_auc"]:
                best.update(pr_auc=pr, k=k, lr=lr, scores=scores, y_true=y_true)

    th = select_best_f1_threshold(best["y_true"], best["scores"])
    hp = Phase8Hyperparams(fine_tune_epochs=best["k"], fine_tune_lr=best["lr"],
                           threshold=th["threshold"], base_seed=base_seed,
                           extra={"holdout_pr_auc": best["pr_auc"]})
    hp.assert_holdout_precedes_val(cfg.split.val_day_min)
    return hp


def fit_initial_model(raw, cfg, dev, seed: int, hp: Phase8Hyperparams):
    """The shared initial checkpoint (Section 14.1/14.5/the Phase 8 audit's Part A): ONE clean
    fit on TRAIN days 1..hp.holdout_fit_end, for exactly hp.fine_tune_epochs epochs at
    hp.fine_tune_lr, reusing Phase 7's `train_epoch` directly (not reimplementing its loop --
    `select_hyperparameters` above is left completely unmodified by this addition; its own
    candidate-sweep loop is untouched, so K/LR/threshold selection behavior is unchanged).

    Builds its OWN dataset with `max_day=hp.holdout_fit_end` (111, not HOLDOUT_SELECT_END=121
    and not cfg.split.val_day_max=151) -- stricter than select_hyperparameters's dataset, so
    VAL (122-151) and TEST (152-182) are structurally unreachable from this function: nothing
    past day 111 is ever loaded into `ds` for this call, regardless of what `hp` or `cfg` say.

    Reproducibility: `torch.manual_seed(seed)` immediately before constructing the model,
    exactly mirroring `select_hyperparameters`'s own seeding of its per-lr "shared init" (seed,
    then construct, with nothing else consuming RNG state in between) -- so this reproduces the
    identical initial weights that candidate evaluation was performed from, for the given seed.

    THIS FUNCTION PERFORMS REAL TRAINING (forward+backward passes). It is written here so an
    authorized Phase 8 execution run (scripts/train_phase8_init.py) can call it, but it is NOT
    invoked by this implementation pass or by the lightweight test suite.
    """
    import sys
    import pathlib

    import torch

    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent.parent / "scripts"))
    from train_temporal_graphsage import TBPTT_DAYS, train_epoch  # Phase 7 artifact, reused unmodified

    from src.graph.temporal import build_temporal_dataset
    from src.models.temporal_graphsage import TemporalGraphSAGE

    fit_end = hp.holdout_fit_end
    ds = build_temporal_dataset(raw, fit_end, dev, cfg)  # max_day=111: VAL/TEST structurally unreachable

    torch.manual_seed(seed)
    model = TemporalGraphSAGE(raw["X"].shape[1]).to(dev)

    tr_rows = ds.txn_day <= fit_end
    y_tr = raw["y"][:len(tr_rows)][tr_rows]
    pos_weight = torch.tensor(float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1)), device=dev)
    opt = torch.optim.Adam(model.parameters(), lr=hp.fine_tune_lr, weight_decay=1e-5)

    for _ in range(hp.fine_tune_epochs):
        train_epoch(model, ds, opt, train_last=fit_end, pos_weight=pos_weight, window=TBPTT_DAYS)
    return model
