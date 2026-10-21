# Phase 9b: synthetic recovery benchmark with a corrected interference gate. Pre-registration

Dated 2026-10-22. **Written and hashed (`PREREGISTRATION_9B.sha256`) before any 9b benchmark is generated.** Days 1-151 only (the Phase 9 real-data cache is truncated at day 151); no held-out day is read. **Nothing in Phase 9 is changed:** its design, locked configuration, benchmarks, gate results and closure report stay as they are; the original G4 failure stays failed; **H5a stays NOT ESTABLISHED under the original protocol**. This is a new benchmark whose result is reported separately as **H5a-9b**; it cannot overturn, rescue or substitute for the original H5a verdict, and no Phase 9 artifact is regenerated or amended.

## 1. Why

The original brief asks, for the coordinated-community task: "If feasible, create a small controlled synthetic benchmark with planted coordinated patterns to evaluate whether the discovery method can recover known structures." Phase 9 built that benchmark, but its frozen validity gate G4 failed on two of three seeds, so the protocol stopped at S6 and the recovery evaluation never ran. The brief's question ("can the method recover known structures?") is therefore unanswered.

The closure report's read-only forensics (`experiments/phase9_coordinated_fraud/diagnostics/g4_diagnosis.json`) showed why G4 failed. The frozen formula measures the change in a real transaction's within-day risk percentile with planted rows **included in the ranking population**. Planted rows are about 4.8 % of each day and are ranked high (mean percentile 0.56 to 0.65), so every real row's percentile drops mechanically: the as-defined values were 0.7954, 0.4622 and 0.7994 points. With planted rows removed from the ranking population, the change was 0.0345, 0.0646 and 0.0671 points, and 97-98 % of real rows had bit-identical logits. G4's stated purpose is to ensure that planting does not distort the risk model's treatment of the real background ("G4 interference"); the formula conflated that with a denominator effect proportional to the planting share. The owner chose not to amend the original benchmark; this pre-registration does not amend it.

## 2. Design (fixed now; everything not listed is exactly the Phase 9 frozen registry, design section 15.1)

* **Generator, counts and periods:** `src/coordination/benchmark.plant` unchanged; evaluation period days 122-151 (formation day at most 146); pools from real days 112 to the day before each planted day; 36 coordinated groups (12 per camouflage regime alpha in {0, 0.5, 1}) plus 18 decoy groups, scaled to the 5 % planting cap.
* **New generator seeds:** 70000 + S for S in {42, 123, 2024}. These benchmarks have never been generated; the Phase 9 seeds (20000 + S) are not reused.
* **Locked configuration reused unchanged:** primary W = 7 with gamma = 0.5 and the locked null tables (`locked_config.json`, selected on the development benchmark before any evaluation benchmark existed; hashes verified at run time); Louvain seed S.
* **Risk stream:** the Phase 8 shared-init checkpoint for seed S (fit on TRAIN days 1-111), CPU, labels never loaded (`src/coordination/risk.risk_stream`). The clean-stream risk record of Phase 9 (`risk/risk_clean_seed{S}.npz`) is reused for the interference gate.
* **Row-level benchmark arrays are not written to disk;** only hashes of the manifest and summary results are saved (the generator is deterministic, so the benchmark is reproducible from its seed).

## 3. Validity gates (before any detector output is computed; any failure is a hard stop, no regeneration)

* **G1, G2, G3, G5:** exactly as Phase 9 (G1: alpha = 0 top-1 % risk fraction at most 3 %; G2: alpha = 1 fraction at least 5 times the alpha = 0 fraction; G3: planting fraction at most 5 %; G5: manifest consistency).
* **G4-9b (replaces G4 for this benchmark only):** mean absolute change in the within-day risk percentile of real transactions between the clean and the augmented stream, **with real rows ranked among real rows only** in both streams, at most **0.5** percentile points (threshold value unchanged).
  **Disclosure:** this definition follows the forensic decomposition in the Phase 9 closure report, where the same quantity was 0.03 to 0.07 on the original benchmark, so passing is expected. G4-9b checks only that planting does not disturb the model's treatment of real transactions; it is not a result. The original G4 quantity is also computed and reported, descriptively, and is expected to fail as before.

## 4. Methods and metrics

* **Methods:** A2 (S + T; label-free, risk-free), B1 (naive shared-infrastructure grouping, ranked by size), B2 (S only), B3a (risk-first grouping), B3b (R_p on the same candidates), B0 (size-matched random card sets). A5 (S + T + R_p) is computed **descriptively only**. All exactly as implemented in `src/coordination/pipeline.py`.
* **Primary metric:** track-level AUPRC at Jaccard at least 0.5 on the **alpha = 0** (camouflaged) coordinated groups (`src/coordination/metrics.track_ap`). Descriptive: the same metric for alpha = 0.5, alpha = 1 and all groups, for every method.

## 5. Decision rule for H5a-9b (the Phase 9 rule, design section 15, unchanged)

**H5a-9b is supported** only if all of these hold:
1. A2 beats **each** of B1, B2, B3a and B3b on alpha = 0 track-level AUPRC in at least 2 of 3 seeds.
2. For each of those four comparisons, the mean difference is positive and exceeds the larger of the two compared methods' across-seed SDs (ddof = 1).
3. **Guard G-cal holds:** on the clean real validation days 122-151, A2 at W = 7 and gamma = 0.5 produces at most 1.5 x 2 = 3 false alerts per day at the TRAIN-calibrated threshold tau_S. tau_S is the value already locked from TRAIN days only in `exploratory_realdata/e0_tau_seed{S}.json` (A2 score, W = 7, gamma = 0.5, target 2 per day); it is not recomputed or changed.

Otherwise the result is **not supported**, reported with which conditions held.

* **Reading note (from the original design):** at alpha = 0 the risk-based baselines B3a and B3b are blind by construction, so the discriminating comparisons are B1 and B2.
* **S8 (benchmark non-diagnostic):** the benchmark is reported as non-diagnostic, with no conclusion drawn, if either (a) B1 reaches AUPRC at least 0.95 in every regime, or (b) in every regime every method's mean AUPRC exceeds B0's mean by no more than the larger of the two across-seed SDs.

## 6. Limits stated in advance

* The real background is identical across seeds, so the three seeds capture generator, detector and risk-initialisation variance, not sampling variance of real data.
* Planted patterns test what the generator plants; a supported H5a-9b says the tested planted patterns are recoverable, not that the real data contains coordinated groups. Candidate coordinated community; not a confirmed fraud ring. This does not establish that IEEE-CIS contains fraud rings.
* H5b and H5c are not evaluated here.
* Validation-period days only; nothing here is a held-out result.
