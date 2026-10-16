"""Frozen Phase 9 parameter registry (design section 15.1, approved 2026-10-07). Never tune after seeing results."""
from __future__ import annotations

M_MIN = 3
WINDOWS = (3, 7, 14)
PRIMARY_W = 7
GAMMAS = (0.5, 1.0, 2.0)
K_NULL = 10
N_NULL_WINDOWS = 8
ALERT_TARGET_PER_DAY = 2.0
GCAL_FACTOR = 1.5
BUDGETS = (5, 10, 20)
PRIMARY_BUDGET = 10
JACCARD_PRIMARY = 0.5
JACCARD_SENS = 0.3
PLANT_CAP = 0.05
G1_MAX_TOP1 = 0.03
G2_MIN_RATIO = 5.0
G4_MAX_DELTA_PCT_POINTS = 0.5
EDGE_WEIGHT_CAP = 5
W_SHORT = 3            # w for T / R_p / R_h windows
T_BASELINE_DAYS = 28
G_DISAPPEAR = 3
TRACK_JACCARD = 0.5
SIZE_BUCKETS = ((3, 4), (5, 8), (9, 16), (17, 10**9))
K_NULL_E0 = 100
SEEDS = (42, 123, 2024)
SPLIT = dict(train=(1, 121), val=(122, 151))
MAX_DAY = 151
DEV_PERIOD = (62, 121)
EVAL_PERIOD = (122, 151)
DEV_COUNTS = dict(groups=60, decoys=30)
EVAL_COUNTS = dict(groups=36, decoys=18)
GENERATOR_SEED_DEV = 10000
GENERATOR_SEED_EVAL = 20000
NULL_SEED_BASE = 30000
CASE_SEED_BASE = 40000
PERM_SEED_BASE = 50000


def size_bucket(n: int) -> int:
    for i, (lo, hi) in enumerate(SIZE_BUCKETS):
        if lo <= n <= hi:
            return i
    return 0
