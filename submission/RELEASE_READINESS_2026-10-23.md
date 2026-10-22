# Release readiness for the collaborator's laptop (2026-10-23)

Purpose: publish the project on GitHub so that a collaborator can clone it on another laptop and the project can be demonstrated from there. Nothing in this record publishes anything; every publication step needs the owner's approval.

## 1. State on 2026-10-23

| Item | State |
|---|---|
| Authoritative project | `Desktop/Adaptive-Temporal-Fraud`, branch `main`, `HEAD` = `11c7348` = GitHub `origin/main`, plus the prepared, uncommitted handoff changes (root README, quick-start script, guides, port fix) |
| Public repository | `github.com/arunsubramanian71-droid/Adaptive-Temporal-Fraud` (public, 0 forks, 0 stars), `main` = `11c7348`: no root README and no quick-start script yet |
| Second repository | `Adaptive-Temporal-Fraud-Research` (private, `master` = `93f02da`, made by the owner from the local copy `Desktop/Adaptive-Temporal-Fraud-Release`): `11c7348` plus the six transaction-level files at its tip |
| File-level comparison | `submission/RELEASE_MANIFEST.json` (`scripts/make_release_manifest.py`): against `origin/main`, 1,133 files unchanged, 20 modified, 9 added, 0 dropped; the withheld groups and how a clone obtains each |

## 2. The exposure that must be resolved before the release

* Six files, each about 300 validation transactions keyed by TransactionID: model scores and per-feature contributions (four files) and, in `top_features_B.json` and `top_features_val.json`, **raw feature values** of those transactions' top features. No fraud labels.
* They are untracked since commit `4356385`, but they are in the trees of 21 historical commits of the **public** repository. All six were downloadable on 2026-10-23 from commit `de16aa0` (HTTP 200 on header-only requests). A new commit does not remove them from earlier commits.
* Competition rule 7B forbids making Competition Data available to non-participants. Raw values of identified transactions most likely qualify.

## 3. Evidence prepared for the decision (nothing published)

* **History filter, dry run** on a throwaway clone (`git filter-branch --index-filter`, removing only the six paths): 34 of 34 commits kept, **0 occurrences of the six files** afterwards, the **root commit `e70238f` unchanged** (the AC12 Git-snapshot evidence depends on it), and the **current file tree byte-identical** (tree `4878678` before and after). Every commit from `b28a03b` onwards gets a new ID (for example `11c7348` → `fa2e974`); no test cites a commit ID; seven documents cite old IDs as history, so an old-to-new commit map would be published with the release. The Git-dependent tests pass on the filtered clone.
* **Research model regenerated** in an isolated clone from the TRAIN and VAL feature tables (`scripts/train_xgboost.py`, about 3.5 minutes on a CPU): the model file, its encoders, `metrics_val.json` and every validation prediction are byte-identical to the recorded ones; research-model serving passed 12 of 12 there.

## 4. Options (the owner decides)

| Option | What happens | Exposure afterwards | Irreversible effects |
|---|---|---|---|
| **A. New clean public repository (recommended)** | the filtered history plus the handoff commit and a commit map go to a **new, empty** public repository; the old public repository is made private (owner's setting) and kept as the full-history archive | the published repository never contained the six files; the old one is no longer public | none for the new repository; making the old one private is reversible; existing copies of the old public repository (0 forks known) cannot be recalled |
| B. Rewrite the existing public repository | the filtered history is force-pushed to `Adaptive-Temporal-Fraud` (same URL) | old commits can stay reachable by ID on GitHub until GitHub Support purges them; the owner should request that | force-push replaces public history; other clones diverge |
| C. Publish to the existing repository as it is | a normal push of the handoff commit | **unchanged: the six files stay downloadable from history** | none, but the exposure continues |
| D. Share the private Research repository | it holds the six files at its tip | the collaborator (not a competition participant) receives them | not acceptable as is |

## 5. Steps after approval, per option

* **A:** the owner creates an empty public repository (no README, licence or .gitignore). In a separate folder, I build the filtered history, commit the handoff changes, the commit map and an updated clone URL, push to the new repository, clone it into a fresh folder, and repeat the checks of section 6 there. The owner then sets the old repository to private.
* **B:** a backup bundle first, then the filter, the handoff commit and a force-push of `main`, then the section 6 checks from a fresh clone, then the owner's GitHub Support request.
* **C:** commit and a normal push, then the section 6 checks from a fresh clone.

## 5a. Decision (2026-10-23)

The owner chose **option A** and approved the commit, the push and the verification from a fresh clone. New public repository: `github.com/arunsubramanian71-droid/Adaptive-Temporal-Fraud-Demo`. The handoff changes were committed in the development repository without a push to the old public repository. The release was built from that commit in a separate folder with the filtered history.

## 6. Checks on the published repository (to be run after publication)

Fresh clone from GitHub into an empty folder, then: the branch, commit, README and file inventory; `scripts/quickstart_windows.ps1` from a new environment (setup check, core tests, demonstration); dashboard end to end (31 checks); the real-browser check; the README's API examples; stop and restart; the six files absent from every commit (options A and B).

## 7. What the collaborator's laptop needs

Windows, Python 3.12, Git, internet access for `pip install` (about 0.5 GB), and Chrome or Edge to view the dashboard. The supported demonstration (setup check, tests, synthetic end-to-end pipeline, dashboard with evidence, demo scoring, candidate communities and status) needs nothing else. The research-model tab additionally needs the IEEE-CIS data, which the collaborator must download from Kaggle after accepting the competition rules (it may not be passed on), followed by `scripts/build_processed_data.py` and `scripts/train_xgboost.py` (README section 7). That optional step does not prevent the supported demonstration.
