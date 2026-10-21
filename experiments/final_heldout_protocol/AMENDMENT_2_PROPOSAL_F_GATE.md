# Proposed Amendment 2: Module F trained-arm readiness gate. PROPOSED, NOT ACTIVE

Dated 2026-10-21. **This is a proposal for the owner's decision. It is not in force, it changes nothing in `HELDOUT_EVALUATION_PROTOCOL.md` (SHA-256 `59bf1565...c806`) or in `AMENDMENT_1.md` (SHA-256 `37a1c057...7e6e`), no code reads it, and the registered Module F gate result stays FAIL as written.** No held-out data was read in preparing it. It becomes active only if the owner approves it explicitly and `AUTHORIZATION.txt` quotes its SHA-256 (in `AMENDMENT_2_PROPOSAL_F_GATE.sha256`) in addition to the base protocol and Amendment 1 hashes.

## 1. The problem it addresses

Amendment 1 section 3 registered, for the trained arms (Ad, E), "window-mean PR-AUC within 0.01 per seed" between a repeated validation run and one earlier run. The repeat failed in 1 of 6 cases (E, seed 123: 0.0104). The diagnosis (`experiments/phase8_state_continuity_supplement/diagnostics/`) shows the cause:
* the deterministic parts (arm F, window 1 of every arm, 21 checks) agree to 1e-7;
* identical-seed GPU runs of the same trained arm differ: window means 0.4330, 0.4273, 0.4281 (and 0.4367, 0.4263 for the two earlier runs), sample SD 0.0044 over the five runs; single scores in later windows differ by up to about 0.97;
* two identical-seed CPU runs are bitwise equal, so the variation comes from nondeterministic GPU kernels, not from the code or the seeding.

A difference between two runs therefore has an SD of about sqrt(2) x 0.0044 = 0.0062, and the registered 0.01 (1.6 SD of a difference, checked six times) has a family-wise false-alarm rate far above what a readiness gate should have.

## 2. The proposed criterion (exact)

**Option A2-a (recommended; prospective).**
1. For each trained arm (Ad, E) and seed (42, 123, 2024), run the burn-in validation walk-forward five times with identical seeding (`scripts/diagnose_f_readiness.py --arm <A> --seed <S> --runs 5`), giving 30 runs, and compute that cell's sample SD of the window-mean PR-AUC, s_cell (ddof = 1).
2. Then run one fresh readiness repeat (`scripts/run_f_walkforward.py --split val --variant readiness`).
3. The trained-arm gate passes for a cell if |fresh window mean - recorded supplement window mean| <= 3 x sqrt(2) x s_cell. The 21 deterministic checks keep their 1e-6 tolerance unchanged.
4. Module F is ready only if the deterministic checks and all six trained-arm cells pass.

**Option A2-b (not recommended; retrospective).** Apply 3 x sqrt(2) x 0.0044 = 0.0187 (the SD from the five E-seed-123 runs) to the existing readiness record. All six cells would pass (largest difference 0.0104). This tolerance was formed after the failure was seen, from one cell's runs, and would turn a failed gate into a pass. It must be read as post hoc if chosen.

## 3. Reasons independent of any held-out result

* A readiness gate exists to detect a code or data-path defect before the one-shot run. For trained arms, run-to-run GPU variation is a property of the hardware and kernels, measured here on validation data only. A tolerance below that variation tests luck, not code.
* 3 standard deviations of a two-run difference gives a per-check false-alarm rate of about 0.3 % and about 1.6 % over six checks, if the noise is roughly normal.
* Nothing about the held-out period was seen, and nothing in it can be seen before the gate is applied.

## 4. Evidence supporting the tolerance

Five identical-seed validation runs of E seed 123 (`F_readiness_diagnosis_E_seed123.json`) give an SD of 0.0044. Two identical CPU runs are bitwise equal (`F_readiness_diagnosis_E_seed123_cpu.json`). Recorded trained-arm differences in the existing readiness record are 0.0060, 0.0004, 0.0005, 0.0104, 0.0011 and 0.0012. **Limitation:** the SD comes from one arm and seed. A2-a removes this limitation by measuring every cell; A2-b does not.

## 5. Consequences for interpretation

* **Defect detection is still covered.** The trained-arm gate at about 0.019 could not, on its own, detect a defect as small as the Phase 8 memory-initialisation effect (0.013 on the frozen arm's window mean). Defects of that kind are caught by the deterministic checks, which stay at 1e-6 and already caught it.
* **Held-out F differences have a noise floor.** They cannot be read finer than this GPU variation, about 0.006 between two runs. The locked H4 rules compare seed means against seed SDs and are not changed by this amendment.
* **A2-b leaves a record.** If it were chosen, the registered result "FAIL as written" would remain in the record beside the post-hoc pass.
* **Without either option.** If neither is approved, Module F can only be authorized with the gate failed as written, or excluded (`submission/HELDOUT_DECISION_PACKAGE.md` section 4).

## 6. Cost and procedure if A2-a is approved

30 GPU runs of about 1-2 minutes each, plus one readiness repeat, all on validation days 122-151 only. The result, pass or fail, is recorded before any held-out access. If a cell fails A2-a, Module F is not ready, and that would be reported as such.
