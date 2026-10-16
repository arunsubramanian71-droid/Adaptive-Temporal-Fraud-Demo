"""Lightweight, CPU/unit-level tests for the Phase 8 implementation (design Section 12/13 of the
code-review instructions). Every test here is pure logic over small synthetic inputs: no raw
data is loaded, no graph is built, no model is instantiated, no forward or backward pass ever
runs. This is deliberate -- these tests verify the window/replay/hparams/leakage *logic* the
real Phase 8 training scripts depend on, without training anything, per the review's testing
boundary ("verify window boundaries", "verify replay sampler uses TRAIN only", etc.)."""
from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from src.adaptive.hparams import HOLDOUT_FIT_END, HOLDOUT_SELECT_END, HOLDOUT_SELECT_START, Phase8Hyperparams
from src.adaptive.leakage import (
    check_holdout_precedes_val,
    check_identical_initial_checkpoint,
    check_replay_pool_excludes_val_and_test,
    check_state_dicts_equal,
    check_update_boundaries_equal_window_ends,
    check_windows_cover_val_exactly,
    replay_control_never_touches_val_check,
)
from src.adaptive.replay import REPLAY_SEED_MULTIPLIER, ReplayPoolBuilder, replay_rng, sample_replay_days
from src.adaptive.windows import N_WINDOWS, WINDOW_LEN_DAYS, assert_disjoint_chronological, boundary_days, \
    build_windows, days_in_window, window_index_for_day
from src.config import load_config

VAL_DAY_MIN, VAL_DAY_MAX = 122, 151       # this project's real config, used only as test fixtures
TRAIN_DAY_MIN, TRAIN_DAY_MAX = 1, 121
TEST_DAY_MIN = 152


# ------------------------------------------------------------------------- windows

def test_build_windows_matches_locked_design():
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    assert len(windows) == N_WINDOWS == 5
    assert windows == [(122, 127), (128, 133), (134, 139), (140, 145), (146, 151)]
    assert all(e - s + 1 == WINDOW_LEN_DAYS for s, e in windows)


def test_build_windows_rejects_non_dividing_span():
    try:
        build_windows(122, 150)  # 29 days, not divisible by 6
        raised = False
    except ValueError:
        raised = True
    assert raised, "build_windows must reject a VAL span that doesn't divide evenly"


def test_windows_are_disjoint_and_chronological():
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    assert_disjoint_chronological(windows)  # must not raise
    try:
        assert_disjoint_chronological([(122, 127), (127, 133)])  # overlapping
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_windows_cover_val_exactly_and_nothing_else():
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    ok, _ = check_windows_cover_val_exactly(windows, VAL_DAY_MIN, VAL_DAY_MAX)
    assert ok
    ok_bad, _ = check_windows_cover_val_exactly(windows, VAL_DAY_MIN, VAL_DAY_MAX + 1)
    assert not ok_bad


def test_boundary_days_equal_window_ends():
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    b = boundary_days(windows)
    assert b == [127, 133, 139, 145, 151]
    ok, _ = check_update_boundaries_equal_window_ends(windows, b)
    assert ok
    ok_bad, _ = check_update_boundaries_equal_window_ends(windows, [127, 133, 139, 145, 150])
    assert not ok_bad


def test_window_index_for_day():
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    assert window_index_for_day(122, windows) == 1
    assert window_index_for_day(127, windows) == 1
    assert window_index_for_day(128, windows) == 2
    assert window_index_for_day(151, windows) == 5
    assert window_index_for_day(121, windows) is None   # TRAIN day
    assert window_index_for_day(152, windows) is None   # TEST day


def test_days_in_window():
    assert days_in_window((128, 133)) == [128, 129, 130, 131, 132, 133]


def test_real_project_config_produces_five_six_day_windows():
    """Regression check against the project's actual configs/data.yaml split boundaries, not a
    hardcoded fixture -- catches drift if the split were ever (incorrectly) changed."""
    cfg = load_config()
    windows = build_windows(cfg.split.val_day_min, cfg.split.val_day_max)
    assert len(windows) == 5
    assert all(e - s + 1 == 6 for s, e in windows)
    assert windows[0][0] == cfg.split.val_day_min
    assert windows[-1][1] == cfg.split.val_day_max


# ------------------------------------------------------------------------- replay sampler

def test_sample_replay_days_only_train_range():
    for round_idx in range(1, 6):
        days = sample_replay_days(base_seed=42, window_index=round_idx, n_days=6,
                                  train_day_min=TRAIN_DAY_MIN, train_day_max=TRAIN_DAY_MAX)
        assert len(days) == 6
        assert all(TRAIN_DAY_MIN <= d <= TRAIN_DAY_MAX for d in days), f"round {round_idx}: {days}"
        assert all(d < VAL_DAY_MIN for d in days)
        assert all(d < TEST_DAY_MIN for d in days)
        assert days == sorted(days), "sample_replay_days must return ascending order"


def test_sample_replay_days_allows_duplicates_with_replacement():
    # A small range with many draws should produce at least one repeat almost certainly;
    # this directly checks "with replacement" rather than just trusting the implementation.
    days = sample_replay_days(base_seed=42, window_index=1, n_days=50, train_day_min=1, train_day_max=5)
    assert len(days) == 50
    assert len(set(days)) < 50, "expected duplicates when sampling 50 draws from only 5 possible days"
    assert all(1 <= d <= 5 for d in days)


def test_sample_replay_days_reproducible_for_same_seed_and_round():
    a = sample_replay_days(42, 3, 6, TRAIN_DAY_MIN, TRAIN_DAY_MAX)
    b = sample_replay_days(42, 3, 6, TRAIN_DAY_MIN, TRAIN_DAY_MAX)
    assert a == b, "same (base_seed, window_index) must reproduce the identical replay sample"


def test_sample_replay_days_differs_across_rounds_and_seeds():
    a = sample_replay_days(42, 1, 6, TRAIN_DAY_MIN, TRAIN_DAY_MAX)
    b = sample_replay_days(42, 2, 6, TRAIN_DAY_MIN, TRAIN_DAY_MAX)
    c = sample_replay_days(123, 1, 6, TRAIN_DAY_MIN, TRAIN_DAY_MAX)
    assert a != b or a != c, "different rounds/seeds should (almost certainly) draw different samples"


def test_replay_rng_matches_design_formula():
    rng1 = replay_rng(base_seed=42, window_index=3)
    rng2 = __import__("numpy").random.RandomState(REPLAY_SEED_MULTIPLIER * 42 + 3)
    assert rng1.randint(0, 1000, size=5).tolist() == rng2.randint(0, 1000, size=5).tolist()


def test_replay_pool_builder_volume_matches_ad_each_round():
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    builder = ReplayPoolBuilder(42, TRAIN_DAY_MIN, TRAIN_DAY_MAX, windows)
    n_train = TRAIN_DAY_MAX - TRAIN_DAY_MIN + 1
    for k in range(1, 6):
        pool = builder.pool_for_round(k)
        expected = n_train + sum(e - s + 1 for s, e in windows[:k])  # TRAIN + k windows' worth of resampled days
        assert len(pool) == expected, f"round {k}: pool size {len(pool)} != expected {expected}"
        ok, _ = check_replay_pool_excludes_val_and_test(pool, VAL_DAY_MIN)
        assert ok, f"round {k}: replay pool must never contain a VAL/TEST day"


def test_replay_pool_builder_rejects_out_of_order_rounds():
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    builder = ReplayPoolBuilder(42, TRAIN_DAY_MIN, TRAIN_DAY_MAX, windows)
    builder.pool_for_round(1)
    try:
        builder.pool_for_round(3)  # skipping round 2
        raised = False
    except ValueError:
        raised = True
    assert raised, "rounds must be requested strictly in order 1, 2, 3, ..."


def test_replay_pool_never_contains_val_or_test_days_across_all_seeds_and_rounds():
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    for seed in (42, 123, 2024):
        builder = ReplayPoolBuilder(seed, TRAIN_DAY_MIN, TRAIN_DAY_MAX, windows)
        for k in range(1, 6):
            pool = builder.pool_for_round(k)
            assert max(pool) <= TRAIN_DAY_MAX
            assert min(pool) >= TRAIN_DAY_MIN


def test_replay_control_never_touches_val_check_pure_logic():
    ok, _ = replay_control_never_touches_val_check(None, [1, 50, 100, 121], VAL_DAY_MIN)
    assert ok
    bad, _ = replay_control_never_touches_val_check(None, [1, 50, 122], VAL_DAY_MIN)
    assert not bad


# ------------------------------------------------------------------------- hyperparameter lock

def test_hyperparameters_holdout_precedes_val():
    assert HOLDOUT_SELECT_END < VAL_DAY_MIN
    ok, _ = check_holdout_precedes_val(HOLDOUT_SELECT_END, VAL_DAY_MIN)
    assert ok
    ok_bad, _ = check_holdout_precedes_val(VAL_DAY_MIN, VAL_DAY_MIN)  # would leak
    assert not ok_bad


def test_hyperparameters_fit_and_select_ranges_disjoint():
    assert HOLDOUT_FIT_END < HOLDOUT_SELECT_START <= HOLDOUT_SELECT_END


def test_phase8_hyperparams_validates_on_construction():
    hp = Phase8Hyperparams(fine_tune_epochs=10, fine_tune_lr=1e-3, threshold=0.5, base_seed=42)
    hp.assert_holdout_precedes_val(VAL_DAY_MIN)  # must not raise

    for bad_kwargs in [
        dict(fine_tune_epochs=0, fine_tune_lr=1e-3, threshold=0.5, base_seed=42),      # K must be positive
        dict(fine_tune_epochs=10, fine_tune_lr=-1.0, threshold=0.5, base_seed=42),      # lr must be positive
        dict(fine_tune_epochs=10, fine_tune_lr=1e-3, threshold=1.5, base_seed=42),      # threshold out of [0,1]
    ]:
        try:
            Phase8Hyperparams(**bad_kwargs)
            raised = False
        except ValueError:
            raised = True
        assert raised, f"Phase8Hyperparams accepted an invalid configuration: {bad_kwargs}"


def test_phase8_hyperparams_rejects_holdout_overlapping_val_if_misconfigured():
    # Simulates a misconfiguration where someone widened the holdout into VAL -- the guard
    # method, called with the real val_day_min, must catch it even though __post_init__ alone
    # (which doesn't know val_day_min) would not.
    hp = Phase8Hyperparams(fine_tune_epochs=10, fine_tune_lr=1e-3, threshold=0.5, base_seed=42,
                           holdout_select_end=130)  # overlaps VAL (122-151)
    try:
        hp.assert_holdout_precedes_val(VAL_DAY_MIN)
        raised = False
    except ValueError:
        raised = True
    assert raised


# ------------------------------------------------------------------------- state-dict equality

def test_check_state_dicts_equal_pure_tensor_comparison():
    # Tiny, untrained tensors -- object construction only, no forward pass, no model class.
    a = {"w": torch.zeros(3, 3), "b": torch.ones(3)}
    b = {"w": torch.zeros(3, 3), "b": torch.ones(3)}
    c = {"w": torch.ones(3, 3), "b": torch.ones(3)}
    ok, _ = check_state_dicts_equal(a, b)
    assert ok
    ok_bad, _ = check_state_dicts_equal(a, c)
    assert not ok_bad


def test_check_identical_initial_checkpoint_pure():
    sd = {"w": torch.zeros(2, 2)}
    sd_diff = {"w": torch.ones(2, 2)}
    ok, _ = check_identical_initial_checkpoint(sd, dict(sd), dict(sd))
    assert ok
    ok_bad, _ = check_identical_initial_checkpoint(sd, sd_diff, dict(sd))
    assert not ok_bad


# ------------------------------------------------------------------------- orchestration logic

def test_update_loop_module_requires_opt_for_adaptive_and_replay_arms():
    """`run_walk_forward` must refuse to run arm Ad/E without opt/pos_weight/fine_tune_epochs --
    this is checked without instantiating any model or dataset, by calling with `model=None,
    ds=None` and confirming the argument-validation ValueError fires before either is touched."""
    from src.adaptive.update_loop import run_walk_forward
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    for arm in ("Ad", "E"):
        try:
            run_walk_forward(arm, model=None, ds=None, windows=windows)  # no opt/pos_weight/epochs
            raised = False
        except ValueError:
            raised = True
        assert raised, f"arm {arm} must require opt/pos_weight/fine_tune_epochs"


def test_update_loop_rejects_unknown_arm_name():
    from src.adaptive.update_loop import run_walk_forward
    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    try:
        run_walk_forward("Z", model=None, ds=None, windows=windows)
        raised = False
    except ValueError:
        raised = True
    assert raised


# --------------------------------------------------------------- checkpoint-wiring static checks
# (added for the checkpoint-refit-and-save step; pure source-text/string inspection, no model, no
# training, no I/O beyond reading these .py files as text)

_SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
_CHECKPOINT_PATTERN = "model_checkpoint_initial_seed{seed}"  # the f-string template, braces-normalized below


def _extract_checkpoint_literal(source: str) -> str | None:
    """Pulls the literal `model_checkpoint_initial_seed...` f-string segment out of a script's
    source text, normalizing the variable name inside `{...}` to `{seed}` so e.g. `{seed}` and
    `{args.seed}` compare equal -- we only care that the surrounding literal text matches."""
    import re
    m = re.search(r'model_checkpoint_initial_seed\{[^}]+\}\.pt', source)
    if not m:
        return None
    return re.sub(r'\{[^}]+\}', '{seed}', m.group(0))


def test_checkpoint_filename_identical_across_init_and_all_three_arm_scripts():
    files = ["train_phase8_init.py", "train_phase8_frozen.py", "train_phase8_adaptive.py",
             "train_phase8_replay_control.py"]
    literals = {}
    for f in files:
        src = (_SCRIPTS_DIR / f).read_text(encoding="utf-8")
        lit = _extract_checkpoint_literal(src)
        assert lit is not None, f"{f}: no 'model_checkpoint_initial_seed{{...}}.pt' literal found"
        literals[f] = lit
    unique = set(literals.values())
    assert len(unique) == 1, f"checkpoint filename literal differs across scripts: {literals}"
    assert unique == {"model_checkpoint_initial_seed{seed}.pt"}


def test_checkpoint_directory_identical_across_init_and_all_three_arm_scripts():
    files = ["train_phase8_init.py", "train_phase8_frozen.py", "train_phase8_adaptive.py",
             "train_phase8_replay_control.py"]
    for f in files:
        src = (_SCRIPTS_DIR / f).read_text(encoding="utf-8")
        assert "shared_init" in src, f"{f}: does not reference the shared_init directory"


def test_fit_initial_model_source_never_references_a_val_or_test_day_constant():
    """Static/source-level guard (no import of src.adaptive.hparams beyond what's already
    imported above, no model instantiation): `fit_initial_model`'s source text must build its
    dataset from `hp.holdout_fit_end`, never from a VAL/TEST-range constant such as
    `cfg.split.val_day_max`, `HOLDOUT_SELECT_END`, or a literal day >= 122.

    Checks the function's CODE, not its docstring -- the docstring legitimately mentions
    `val_day_max`/`HOLDOUT_SELECT_END` in prose explaining what the function deliberately does
    NOT do, so the docstring text is stripped before scanning for forbidden identifiers."""
    import inspect

    from src.adaptive.hparams import fit_initial_model
    full_src = inspect.getsource(fit_initial_model)
    doc = fit_initial_model.__doc__ or ""
    code_only = full_src.replace(doc, "") if doc else full_src
    assert "build_temporal_dataset(raw, fit_end, dev, cfg)" in code_only
    assert "fit_end = hp.holdout_fit_end" in code_only
    for forbidden in ("val_day_max", "val_day_min", "HOLDOUT_SELECT_END", "test_day"):
        assert forbidden not in code_only, f"fit_initial_model's CODE must never reference {forbidden!r}"


def test_holdout_fit_end_precedes_val():
    cfg = load_config()
    assert HOLDOUT_FIT_END < cfg.split.val_day_min
    assert HOLDOUT_FIT_END < HOLDOUT_SELECT_START <= HOLDOUT_SELECT_END < cfg.split.val_day_min


# ------------------------------------------------------- mock-model walk-forward ordering check
# No real checkpoint, no real TemporalGraphSAGE, no GPU, no real backward pass: `train_epoch` is
# monkeypatched (in-memory, for the duration of this test only; no source file is modified) with
# a fake recorder, so arm Ad's update calls are observed without ever computing a real gradient.

def test_run_walk_forward_ordering_score_then_update_between_windows():
    """Verifies the causal sequence locked in DESIGN_BEFORE_EXECUTION.md Section 14.5:

        score W1 -> update -> score W2 -> update -> score W3 -> update -> score W4 -> update
        -> score W5 -> STOP

    i.e. an update happens strictly BETWEEN two scored windows, and NOT after the final window
    (there is no W6 for a post-W5 update to ever matter for) -- and there are exactly FOUR
    updates total for Ad/E, not five.

    This test previously (correctly) FAILED against an earlier version of `run_walk_forward`
    that updated after every window including W5; that off-by-one was fixed in
    src/adaptive/update_loop.py (gated on `k == n: break` before the update branch) and this
    test now passes against the fixed code.
    """
    import numpy as np
    import torch as _torch

    from src.adaptive import update_loop as _ul

    windows = build_windows(VAL_DAY_MIN, VAL_DAY_MAX)
    call_log = []

    class MockModel:
        def init_state(self, ds):
            return {}

        def eval(self):
            pass

        def step(self, state, ds, d, predict=True):
            call_log.append(("score", d))
            return _torch.zeros(2)

    class MockDS:
        def ids_for_day(self, d):
            return np.array([d * 10, d * 10 + 1])

    def fake_train_epoch(model, ds, opt, train_last, pos_weight, window=7):
        call_log.append(("update", train_last))
        return 0.0

    original = _ul.train_epoch
    _ul.train_epoch = fake_train_epoch
    try:
        _ul.run_walk_forward("Ad", MockModel(), MockDS(), windows, opt="dummy",
                             pos_weight=1.0, fine_tune_epochs=1)
    finally:
        _ul.train_epoch = original  # always restore, even if the assertion below fails

    b = boundary_days(windows)
    expected = []
    for k, (s, e) in enumerate(windows):
        expected += [("score", d) for d in range(s, e + 1)]
        if k < len(windows) - 1:          # an update is expected BETWEEN windows, not after the last one
            expected.append(("update", b[k]))
    assert call_log == expected, (
        f"walk-forward ordering mismatch:\nactual:   {call_log}\nexpected: {expected}")

    n_updates = sum(1 for kind, _ in call_log if kind == "update")
    assert n_updates == N_WINDOWS - 1 == 4, f"expected exactly 4 updates for 5 windows, got {n_updates}"
    assert call_log[-1] == ("score", windows[-1][1]), "the walk-forward must end on a score, not an update"


def _run_all():
    fns = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    passed, failed = 0, []
    for fn in fns:
        try:
            fn()
            passed += 1
            print(f"[PASS] {fn.__name__}")
        except Exception as e:
            failed.append((fn.__name__, e))
            print(f"[FAIL] {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} passed")
    return len(failed) == 0


if __name__ == "__main__":
    sys.exit(0 if _run_all() else 1)
