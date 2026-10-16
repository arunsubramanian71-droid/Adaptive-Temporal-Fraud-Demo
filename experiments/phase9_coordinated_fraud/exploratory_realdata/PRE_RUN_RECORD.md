# Exploratory real-data continuation (Tier 0 and E0 only) — pre-run record

> Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 protocol halted at S6. Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under the original protocol.

Written 2026-10-10 BEFORE any Tier 0 or E0 result existed (before the validations finished and before any real-data discovery output on days 122-151 was produced). Its SHA-256 is recorded in `PRE_RUN_RECORD.sha256`.

## 1. Authorization and scope (user, 2026-10-10)
Explicit scope deviation authorising Tier 0 (descriptive discovery) and E0 (null-controlled structure) only. It does not amend, waive, reinterpret or reverse any benchmark validity gate. Excluded: E1 and any use of fraud labels, risk calibration / R_p / R_h / risk-based baselines, additional dev gates, benchmark regeneration, quantitative H5a/b/c evaluation, G4s, Phase 10, any Phase 1-8 change, any installation.
Standing facts: original G4 FAILED on seeds 42 (0.795) and 2024 (0.799), PASSED on seed 123 (0.462); H5a is NOT ESTABLISHED under the original protocol. Unresolved dev-gate ambiguity (G1/G2/G4 were never computed on the dev benchmark; no dev risk stream exists; design 11.7 does not say which benchmark): recorded, no further dev gate computed.

## 2. Validations (Step A) — definitions used
Taken from design 5 (leakage matrix), 19 and 20 (S2, S3, S4); implemented in `scripts/validate_phase9_leakage.py`, applied to `src/coordination/realrun.run_discovery`, the function that Tier 0 uses:
* S2 label invariance: bitwise-identical outputs for real labels / random bits / absent / a poisoned object that raises on any access, plus static no-label / no-risk / no-benchmark checks.
* S3 future invariance: bitwise-identical outputs for days <= c when all data after c is randomised, replaced by an adversarial burst, or deleted; c in {60, 100, 121, 136, 150}; a positive control (corrupting days inside the range) must change the hash.
* S4 determinism: same configuration twice in-process and once in a fresh interpreter with a random PYTHONHASHSEED.
Hash comparison only; no discovery content is inspected during validation.

## 3. Tier 0 — frozen settings and descriptor definitions (fixed now, before the run)
* Data: clean real arrays, days <= 151; reported days 122-151 (VAL); trailing windows reach into earlier days (past data only).
* Locked configuration (unmodified, hash-verified): gamma = 0.5 for W = 3, 7, 14; primary W = 7; W = 3, 14 pre-registered sensitivity. Null tables: `null_tables.pkl` (hash must match `locked_config.json`). Louvain seed = S for S in {42, 123, 2024}. m_min = 3, edge-weight cap 5, size buckets {3-4, 5-8, 9-16, 17+}.
* Score: A2 = unweighted mean of null-calibrated u_S and u_T. No R_p, R_h, risk, labels.
* Tracker: the shared design-6 tracker started at day 122 (tracks starting on 122 are left-censored; tracks alive on 151 are right-censored; disappearance is declared 3 days after the last active day and is therefore only observable for last_day <= 148).
* Descriptors (definitions in `src/coordination/realdesc.py`): counts/day, sizes, lifecycle statistics, segment (cold / warm / no-edge, never merged) x infra_observed composition, hub share (internal link mass on entities above the 90th percentile of window distinct-card degree), density, conductance, recomputed-Q consistency check.
* Fraud base rate of the structurally unobservable population (design 9) requires labels and is NOT computed (outside this authorization); its count and segment composition are reported.
* Tier 0 reports NO alert counts: the alert threshold tau does not exist yet (see 4). Candidate-community counts are counts of eligible candidates (>= 3 card tokens, >= 1 infrastructure entity), not alerts.
* No statistical test in Tier 0. Everything is descriptive.

## 4. Identified gaps in the frozen design for E0 (and for alert counts) — NOT resolved here
The design requires E0 as "observed counts of high-S/T communities vs counts under degree-preserving and time-shuffled nulls on the same real windows (permutation p-values, K_null >= 100 on VAL, seeded)" (10, 12, 15.1) and defines the alert threshold tau (7.4(5), 15.1: fixed on TRAIN background-only windows so the expected false-alert rate is <= 2 alerts/day, then locked). The following are required but not specified anywhere in the frozen design or locked artifacts:
1. tau: which TRAIN windows (days), how many, which W / configuration. (The only trace is the runtime-projection entry "tau_calibration_TRAIN_background, W7, 30 windows" in `timing_pilot.json`, which is not a specification of the calibration.) No tau exists on disk.
2. "High-S/T" has no numerical definition other than "alerted" (12), which depends on tau.
3. Which windows E0 uses ("the same real windows": all 30 VAL days or a subset; the timing projection used an unexplained "4 x K_NULL_E0 x 3 seeds").
4. How the S+T score is evaluated under each null and how the two nulls enter one test; the test statistic and aggregation across days; the permutation p-value convention (sidedness, +1 correction); which of W / seeds enter E0.
Per the authorization, E0 (and tau) are therefore NOT implemented or run; the specific gaps are reported for a decision. Proposed resolutions are listed in the completion report as proposals only, written before any Tier 0 output was seen.
