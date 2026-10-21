# Proposed Amendment 2, revision 1 (A2-a-r1): Module F trained-arm readiness gate. PROPOSED, NOT ACTIVE

Dated 2026-10-22. **This is a proposal for the owner's decision. It is not in force and no code reads it.** It changes nothing in `HELDOUT_EVALUATION_PROTOCOL.md` (SHA-256 `59bf1565...c806`) or `AMENDMENT_1.md` (SHA-256 `37a1c057...7e6e`). The registered Module F gate result stays **FAIL as written**. The original proposal `AMENDMENT_2_PROPOSAL_F_GATE.md` (SHA-256 `55776392...a4c8c7`) is kept unchanged as a record; this revision corrects an error in it.

**What was used to prepare it:** no held-out data was read, and no new validation run was made. Before this revision was written, no Amendment 2 measurement had been made, so the outcome of the procedure below is unknown.

## 1. The error in the original proposal

The original A2-a passes a cell if |fresh - recorded| <= 3 x sqrt(2) x s_cell. Here s_cell is a standard deviation **estimated from five runs**. With an estimated SD, the ratio follows a t distribution with 4 degrees of freedom, not a normal distribution. The false-alarm rates stated in the original section 3 are therefore wrong:

| | stated in the original | correct (t, 4 df) |
|---|---|---|
| one cell, correct code | 0.3 % | **4.0 %** |
| any of six cells (independent) | 1.6 % | **21.7 %** |

So A2-a as written would fail a correct pipeline about one time in five. A2-b (0.0187) also rests on one SD estimated from five runs of one cell, and it was chosen after the failure was seen.

## 2. The revised criterion (exact)

1. **Measurement.** For each trained cell c (arms Ad and E, seeds 42, 123 and 2024: six cells), make n = 10 identical-seed validation runs of the burn-in walk-forward. Each run uses days 122-151 and the same checkpoint, hyperparameters, seeding and device type as the recorded supplement run. Each run is a separate process:
   `.venv/Scripts/python.exe -B scripts/diagnose_f_readiness.py --arm <A> --seed <S> --runs 1 --out-dir experiments/phase8_state_continuity_supplement/diagnostics/amendment2/<A>_seed<S>_run<NN>`, for NN = 01 to 10.
   That is 60 runs. Each run gives a window-mean PR-AUC x_1, ..., x_10.
2. **Reference.** r_c is the recorded supplement window mean: the mean of the pooled PR-AUC in `outputs/burn_in/<arm>/seed_<S>/metrics_W1..W5.json`, unchanged.
3. **Statistic.** Let m = mean(x), s = SD(x, ddof = 1) and T = (m - r_c) / (s x sqrt(1 + 1/n)). If the recorded run and the new runs come from the same run-to-run distribution, T follows a t distribution with 9 degrees of freedom. **A cell passes if |T| <= 4.094.** That is the two-sided 0.27 % point of t with 9 df: the same per-check false-alarm rate as a normal 3-SD rule, and about 1.6 % over six independent cells. If s = 0, the cell passes only if |m - r_c| <= 1e-6.
4. **Deterministic checks (tolerance 1e-6, unchanged):**
   * (a) In each of the 60 runs, the window-1 scores equal the recorded supplement window-1 scores within 1e-6.
   * (b) A fresh readiness run of all arms and seeds is written to a new folder (`scripts/run_f_walkforward.py --split val --variant readiness --out-name readiness_a2`; existing folders are refused by the runner). F in every window and window 1 of every arm must be within 1e-6: 21 checks. That run also writes the old 0.01 rows; they are recorded but are not the decision rule.
5. **Decision.** Module F is ready only if 4(a), 4(b) and all six cells under 3 pass. The decision goes to `experiments/phase8_state_continuity_supplement/outputs/readiness_a2/readiness_F_amendment2.json` before any held-out access. The record includes the 60 values, the statistics, the code fingerprint and the torch/ROCm versions and GPU name. The registered result "FAIL as written" stays in the record beside it.
6. **Fixed in advance:**
   * n = 10, the critical value 4.094 and the reference r_c.
   * No completed run is discarded. A crashed run is repeated and the crash is recorded.
   * Nothing changes after any of the 60 values is seen.
   * If a cell fails, Module F is not ready and is reported that way. No further amendment of this gate is proposed before the held-out run.

## 3. Relation to the held-out period

No held-out data has been read, and none can influence this criterion, which is applied before any held-out access. The design was made **after** the registered validation gate failed, and that is disclosed. The revision itself was prompted by a statistical error in the original proposal, not by any new result.

## 4. Sensitivity and limits

* With s about 0.0044 (the only measured cell, E seed 123), the tolerance on |m - r_c| is about 4.094 x 0.0044 x 1.049 = 0.019. That is the size the original A2-a intended (0.0187), now with the stated false-alarm rate.
* An update-path defect that moves the window-mean PR-AUC by less than about 0.02 would not be caught by this trained-arm gate. Defects in the deterministic path stay checked at 1e-6. Code equivalence of the update path with Phase 8 under identical seeding is tested separately (`tests/test_walkforward_ext.py`).
* **Assumptions:**
  * The recorded run and the new runs share one run-to-run distribution (same code, GPU, driver and library versions). Separate processes are used because the recorded runs came from separate processes; the three within-process repeats had SD 0.0031 against 0.0044 over all five runs, too few runs to tell whether this matters.
  * Window means are roughly normal.
  * Cells are independent, for the family-wise figure only.

  If the hardware or library stack differs from the recorded run, the first assumption may fail, and that would be reported.

## 5. Cost

60 GPU runs, each in its own process: about 1 to 2 minutes each including data loading. Plus one fresh readiness run of about 10 minutes. In total about 1.5 to 2.5 hours, on validation days 122-151 only.

## 6. Implementation needed after approval (not written before approval)

* A small evaluation script that reads the 60 records and the fresh readiness run, applies section 2 and writes the decision record.
* An additive output field in `scripts/diagnose_f_readiness.py`: each run's window-1 difference from the supplement.
* In `src/heldout/amendment.py`, accept Module F only if two things hold:
  * `AUTHORIZATION.txt` quotes this revision's SHA-256 (`AMENDMENT_2_PROPOSAL_F_GATE_REV1.sha256`) and the approval sentence below;
  * the decision record passed and fingerprints unchanged code.
* Tests for each of these.

## 7. Approval wording

> I approve Amendment 2 revision 1 (A2-a-r1) as the Module F trained-arm readiness gate.

Approval in chat authorizes the validation measurement in section 2. For the held-out run, `AUTHORIZATION.txt` must also quote this revision's SHA-256 and the sentence above, in addition to the base and Amendment 1 entries.
