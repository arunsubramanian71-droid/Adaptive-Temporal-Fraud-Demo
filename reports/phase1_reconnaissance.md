# Phase 1 — Reconnaissance Report

**Project:** Adaptive Temporal Graph Intelligence for Fraud Detection & Coordinated Fraud-Ring Discovery
**Date:** 2026-09-14
**Scope:** Repository/environment inspection + dataset reconnaissance only. No models trained, no graph built, no raw data modified.

Every number below was measured directly from the files in `data/raw/`. Claims are tagged:

- **[FACT]** — measured from the data in this session.
- **[INFERENCE]** — reasoned conclusion from those measurements. Could be wrong.
- **[REC]** — recommendation / design proposal for approval.

---

## 1. Repository structure

**[FACT]** The repository was effectively empty apart from data. Not a git repository.

```
Adaptive-Temporal-Fraud/
├── data/
│   └── raw/            <- 5 CSVs, untouched
├── reports/            <- created by me (this file only)
└── .venv/              <- created by me (recon tooling only)
```

**[FACT]** Nothing was deleted, renamed, or modified in `data/raw/`. No source code, configs, notebooks, or models existed before this session.

### Environment

| Item | Value |
|---|---|
| OS | Windows 11 Enterprise 10.0.26200 |
| CPU | AMD Ryzen 7 7700X — 4 physical / 8 logical cores |
| RAM | 31.2 GB total, 21.6 GB free at inspection |
| GPU | **AMD Radeon RX 7900 GRE, 17.2 GB VRAM** |
| Disk free (C:) | 96 GB |
| Default `python` | 3.14.3 — only `pip`, `scapy` installed |
| Secondary | 3.12.0 — `torch 2.9.1+rocmsdk20260116`, `networkx 3.6.1`, `numpy 2.4.4`, `transformers` |
| GPU compute | `torch.cuda.is_available() == True` via **ROCm/HIP 7.2** |

**[FACT]** I created `.venv` from Python 3.12 with `--system-site-packages` and installed **only `pandas` and `pyarrow`** for this reconnaissance. No ML framework was installed.

**[INFERENCE] This is the single biggest engineering risk in the project, and it is not about the data.** The GPU is AMD/ROCm on Windows, not CUDA. PyTorch Geometric's optional compiled companions (`torch-scatter`, `torch-sparse`, `torch-cluster`) publish CUDA and CPU wheels — there is no ROCm-on-Windows wheel. PyG ≥ 2.3 runs its core layers (`SAGEConv`, `GATConv`, `HeteroConv`, `NeighborLoader`) on pure-PyTorch scatter ops without those extensions, so the plan is viable, but this must be verified empirically before we commit, not assumed. Python 3.14 is also ahead of the wheel-availability curve for XGBoost/PyG; the 3.12 venv is the right target.

**[REC]** First action of Phase 2 is a 30-minute *capability probe*, before any modelling code: install `torch-geometric` + `scikit-learn` + `xgboost` into `.venv`, and run a small `HeteroConv` forward/backward pass on `cuda:0`. If it fails, fall back to CPU-only PyG (feasible at this graph size — see §15) and record the decision. Do not discover this in week three.

---

## 2–5. Dataset files, sizes, row and column counts

**[FACT]** All five expected files are present.

| File | Bytes | On disk | Rows | Cols | In-RAM (pandas, deep) |
|---|---|---|---|---|---|
| `train_transaction.csv` | 683,351,067 | 652 MB | **590,540** | **394** | 1,879 MB |
| `train_identity.csv` | 26,529,680 | 26 MB | **144,233** | **41** | 59 MB |
| `test_transaction.csv` | 613,194,934 | 585 MB | **506,691** | **393** | 1,609 MB |
| `test_identity.csv` | 25,797,161 | 25 MB | **141,907** | **41** | 58 MB |
| `sample_submission.csv` | 6,080,314 | 5.9 MB | **506,691** | **2** | 8 MB |

**[FACT]** Dtypes as loaded: `train_transaction` = 376 float64, 14 str, 4 int64. `train_identity` = 23 float64, 17 str, 1 int64.

**[FACT]** `test_transaction` has 393 columns — identical to train minus `isFraud`. That is the only column difference.

---

## 6. Important columns

**[FACT]** Column inventory of `train_transaction` (394), by naming block:

| Block | Count | Range |
|---|---|---|
| `V*` | **339** | V1–V339 |
| `D*` | 15 | D1–D15 |
| `C*` | 14 | C1–C14 |
| `M*` | 9 | M1–M9 |
| `card*` | 6 | card1–card6 |
| `addr*` | 2 | addr1, addr2 |
| `dist*` | 2 | dist1, dist2 |
| singletons | 7 | `TransactionID`, `isFraud`, `TransactionDT`, `TransactionAmt`, `ProductCD`, `P_emaildomain`, `R_emaildomain` |

**[FACT]** `train_identity` (41) = `TransactionID` + `id_01`…`id_38` + `DeviceType` + `DeviceInfo`.

**[FACT]** **Column names differ between train and test identity files.** Train uses underscores (`id_01`); test uses hyphens (`id-01`). `list(train_identity.columns) == list(test_identity.columns)` is `False`.

**[REC]** Normalise `-` → `_` in the loader. This is a silent all-NaN-join bug waiting to happen.

### Categorical value inventories **[FACT]**

- `ProductCD` (5, 0% missing): `W, H, C, S, R`
- `card4` (4): `visa, mastercard, discover, american express`
- `card6` (4): `debit, credit, debit or credit, charge card`
- `M1`–`M3`, `M5`–`M9`: `T`/`F`. `M4`: `M0, M1, M2`
- `id_12/16/29`: `Found`/`NotFound`. `id_15/28`: `New`/`Found`(/`Unknown`)
- `id_34` (4): `match_status:-1, 0, 1, 2`
- **`id_23` (3): `IP_PROXY:TRANSPARENT`, `IP_PROXY:ANONYMOUS`, `IP_PROXY:HIDDEN`** — 99.1% missing

**[FACT]** `id_23` is a proxy *classification*, not an IP address. There is no IP field anywhere in the dataset.

### `TransactionAmt` **[FACT]**

min 0.251, max 31,937.391, mean 135.03, median 68.769, 20,902 unique, **0 nulls**. p99 = 1,104; p99.9 = 2,769.81.

**[FACT]** 61,933 rows (10.49%) have **more than two decimal places**. Fraud rate on those rows is **11.721%** vs 3.499% overall — a 3.35× lift.

**[INFERENCE]** Amounts with >2 dp are currency-converted (a foreign-currency purchase converted to USD leaves sub-cent residue). This is a genuine, cheap, leakage-free feature and a plausible cross-border-fraud proxy.

### Fraud rate by `ProductCD` **[FACT]**

| ProductCD | n | fraud % | median amt | `addr1` present |
|---|---|---|---|---|
| `W` | 439,670 | **2.040** | 78.50 | 99.9% |
| `C` | 68,519 | **11.687** | 31.19 | **5.0%** |
| `H` | 33,024 | 4.766 | 50.00 | 99.7% |
| `S` | 11,628 | 5.900 | 35.00 | 98.2% |
| `R` | 37,699 | 3.783 | 125.00 | 99.9% |

**[INFERENCE]** `ProductCD` is not a feature so much as a **segment marker**. `C` is a structurally different product — 5.7× the fraud rate of `W`, tiny amounts, and almost no billing address. Pooling `W` and `C` into one model and one threshold will produce a model dominated by `W` (74% of rows) while `C` supplies a third of all fraud.

**[REC]** Report all headline metrics **both pooled and per-`ProductCD`**. This is nearly free and is the kind of segment analysis that separates a research report from a leaderboard entry.

---

## 7. Table relationships

**[FACT]**

| Table | Rows | Unique `TransactionID` | Unique? | Monotonic? | ID range |
|---|---|---|---|---|---|
| `train_transaction` | 590,540 | 590,540 | Yes | Yes | 2,987,000 – 3,577,539 |
| `train_identity` | 144,233 | 144,233 | Yes | Yes | 2,987,004 – 3,577,534 |
| `test_transaction` | 506,691 | 506,691 | Yes | Yes | 3,663,549 – 4,170,239 |
| `test_identity` | 141,907 | 141,907 | Yes | Yes | 3,663,586 – 4,170,239 |
| `sample_submission` | 506,691 | 506,691 | Yes | Yes | 3,663,549 – 4,170,239 |

**[FACT]**
- One row of `*_transaction` = **one transaction**. `TransactionID` is its primary key.
- `train_identity.TransactionID ⊂ train_transaction.TransactionID` — `True`. Same for test.
- The relation is **one-to-one, optional** (zero-or-one identity row per transaction). It is *not* one-to-many.
- Identity coverage: **144,233 / 590,540 = 24.42%** (train); **141,907 / 506,691 = 28.01%** (test).
- `train ∩ test` on `TransactionID` = **0** rows. Disjoint.
- `set(sample_submission.TransactionID) == set(test_transaction.TransactionID)` — `True`.
- `TransactionID` is **perfectly contiguous** in train (all successive diffs == 1).

**[FACT] Identity missingness is not random and is strongly informative:**

| Subset | n | fraud rate |
|---|---|---|
| Has identity row | 144,233 | **7.847%** |
| No identity row | 446,307 | **2.094%** |

**[INFERENCE]** Identity presence is itself a 3.75× signal. The `id_*` block is therefore *missing-not-at-random*, and "has identity" must be an explicit binary feature. Never impute the `id_*` block with a strategy that erases the presence indicator.

**[FACT]** `corr(TransactionID, TransactionDT) = 0.99828` in train, and `TransactionID` is monotonic in time in every file.

**[INFERENCE] `TransactionID` is a chronological sequence number, i.e. a near-perfect proxy for time.** Feeding it to a model is time leakage in a thin disguise.

**[REC]** `TransactionID` is a join key and an index. Put it on a hard feature blocklist enforced in code, not in a comment.

---

## 8. Target distribution

**[FACT]**

| Class | Count | Share |
|---|---|---|
| `isFraud = 0` | 569,877 | 96.501% |
| `isFraud = 1` | **20,663** | **3.499%** |

- Imbalance ratio **27.58 : 1**
- 0 nulls; dtype int64; values exactly `{0, 1}`

**[FACT]** `isFraud` is **absent** from `test_transaction.csv`. `sample_submission.csv` contains the literal constant `0.5` for all 506,691 rows.

**[FACT]** Confirms the brief: the Kaggle test set has **no ground-truth labels** and cannot serve as a quantitative evaluation set.

---

## 9. Temporal fields and their semantics

### `TransactionDT` — investigated, not assumed

**[FACT]**

| | min | max | span | span (days) | nulls | monotonic by row |
|---|---|---|---|---|---|---|
| train | 86,400 | 15,811,131 | 15,724,731 | **182.00** | 0 | Yes |
| test | 18,403,224 | 34,214,345 | 15,811,121 | **183.00** | 0 | Yes |

- Gap train_max → test_min = 2,592,093 s = **exactly 30.00 days**
- train: 573,349 distinct values → 17,191 rows share a timestamp with another row
- Interpreted as Unix epoch, train would span **1970-01-02 → 1970-07-02**

**[INFERENCE]** `TransactionDT` is **not a Unix timestamp.** It is **seconds elapsed from an undisclosed reference instant**. Evidence: (a) a 1970 date is impossible for e-commerce card data; (b) `min == 86,400 == exactly one day`, a hallmark of a synthetic offset rather than a real clock.

**[FACT] The reference instant is aligned to local midnight — proven, not assumed.** `D9` takes exactly **24 distinct values**, evenly spaced from 0.0 to 0.9583 (= 23/24). On all 74,926 rows where `D9` is present:

```
round(D9 * 24)  ==  (TransactionDT // 3600) % 24     ->  100.0000% match, 0 mismatches
```

**[INFERENCE]** `D9` *is* hour-of-day encoded as a fraction. Two independent columns agree perfectly, so `(TransactionDT//3600)%24` recovers the **true local hour**, and the reference instant sits on a local midnight boundary.

**[FACT]** The recovered diurnal cycle is coherent and fraud-discriminative:

| Hour | n | fraud % |
|---|---|---|
| 07 | 3,704 | **10.610** <- peak fraud, volume trough |
| 08 | 2,591 | 9.301 |
| 09 | 2,479 | 8.996 |
| 19 | 42,115 | 3.474 <- volume peak |
| 13 | 20,315 | **2.289** <- min fraud |

**[INFERENCE]** Volume troughs at 07–09 and peaks at 18–21. Fraud rate is **inversely** related to legitimate volume (4.6× swing across the day), consistent with attackers operating while genuine customers sleep. Hour-of-day is a real feature, safe to derive, and the ordering is now verified rather than guessed.

**[FACT]** Day-of-week bucket `(DT//86400)%7` gives 70,223–98,502 rows per bucket and fraud rates 3.145%–3.717% — a much weaker effect than hour-of-day.

**[FACT] Derived day index:** train = days **1–182**, test = days **213–395**.

**[INFERENCE]** 182 + 30 + 183 = 395 days ≈ 13 months of coverage with a 30-day hole. **[FACT]** The `TransactionID` gap across that hole is 86,010, meaning ~86,009 transactions exist in the 30-day gap and are present in neither file — Kaggle withheld them. We do not need them; noting so we never mistake the gap for missing data to reconstruct.

### `D1`–`D15` — timedeltas

**[FACT]**

| Col | min | median | max | missing % | integral |
|---|---|---|---|---|---|
| D1 | 0 | 3 | 640 | 0.21% | Yes |
| D2 | 0 | 97 | 640 | 47.55% | Yes |
| D3 | 0 | 8 | 819 | 44.51% | Yes |
| D4 | **−122** | 26 | 869 | 28.60% | Yes |
| D8 | 0 | 37.88 | **1,707.79** | 87.31% | **No** |
| D9 | 0 | 0.67 | 0.96 | 87.31% | No (= hour/24) |
| D10 | 0 | 15 | 876 | 12.87% | Yes |
| D15 | **−83** | 52 | 879 | 15.09% | Yes |

**[INFERENCE]** `D1`–`D15` (excluding `D9`) are **days since some prior event**, in days. `D8` is non-integral → days *with a fractional part*, i.e. a timedelta with sub-day resolution; `D9` is the hour component of that same timestamp (both are 87.31% missing — the identical missingness confirms they are one pair).

**[FACT]** Maxima (640–1,707 days) far exceed the 182-day train window, and `day − D_n` reaches **−699** (D4/D10/D15) and **−1,693** (D8).

**[INFERENCE]** The referenced prior events **predate the observation window**. `D_n` carries real pre-window history — genuinely useful, and available at prediction time.

**[FACT] Forward-reference anomaly.** A tiny number of rows have negative `D_n`, i.e. they reference an event *after* the transaction:

| Col | n negative | share | min |
|---|---|---|---|
| D4 | 15 | 0.0025% | −122 |
| D15 | 15 | 0.0025% | −83 |
| D11 | 7 | 0.0012% | −53 |
| D6 | 3 | 0.0005% | −83 |
| D14 | 3 | 0.0005% | −193 |
| D12 | 2 | 0.0003% | −83 |

D1, D2, D3, D5, D7, D8, D9, D10, D13 have **no** negatives. Test has **no** negatives in any `D` column.

**[INFERENCE]** 45 rows out of 590,540. Negligible for performance, but it means "`D_n` is strictly backward-looking" is 99.992% true, not 100% true. Worth one line in the write-up; not worth special handling beyond clipping at 0.

---

## 10. Candidate node types

**[FACT] Full column inventory searched for identity-like names** (`ring|cluster|customer|user|account|merchant|seller|ip|phone|name|label|target`) across all 434 distinct columns in both train tables: **zero matches.**

**[FACT]** There is **no** customer ID, account ID, merchant ID, IP address, phone, or device serial in this dataset. Any such entity would be fabricated.

**[FACT] Measured cardinality, coverage and degree for every legitimate candidate** (train period, 590,540 transactions):

| Candidate entity | Nodes | Edges | Coverage | deg median | deg p99 | **deg max** | singleton nodes |
|---|---|---|---|---|---|---|---|
| `card1` | 13,553 | 590,540 | 100.0% | 4 | 786 | 14,932 | 3,444 |
| CardProfile `card1..6` | 13,691 | 579,498 | 98.1% | 4 | 728 | 14,112 | 3,485 |
| BillingAddr `addr1+addr2` | **437** | 524,834 | 88.9% | 2 | 25,129 | **46,324** | 171 |
| `P_emaildomain` | **59** | 496,084 | 84.0% | 305 | 154,450 | **228,355** | 0 |
| `R_emaildomain` | **60** | 137,291 | 23.2% | 113 | 39,660 | 57,147 | 0 |
| `DeviceInfo` | 1,786 | 118,666 | 20.1% | 4 | 238 | 47,722 | 440 |
| DeviceFingerprint composite | 4,880 | 71,120 | 12.0% | 2 | 242 | 2,566 | 2,044 |
| Browser `id_31` | 130 | 140,282 | 23.8% | 80 | 12,857 | 22,000 | 15 |
| OS `id_30` | 75 | 77,565 | 13.1% | 218 | 15,201 | 21,155 | 1 |
| ScreenRes `id_33` | 260 | 73,289 | 12.4% | 3 | 5,534 | 16,874 | 85 |
| NetProxy `id_19+id_20` | 2,394 | 139,212 | 23.6% | 6 | 1,080 | 5,523 | 470 |
| `ProductCD` | **5** | 590,540 | 100.0% | 37,699 | 424,823 | **439,670** | 0 |

**[FACT] Derived pseudo-account.** `card1 + addr1 + (day − D1)` — all three components readable from the current row, nothing from the future:

| Key | Groups | Singletons | median | p90 | max |
|---|---|---|---|---|---|
| `card1` | 13,553 | 3,444 (25.4%) | 4 | 46 | 14,932 |
| `card1+addr1` | 39,974 | 15,752 (39.4%) | 2 | 22 | 9,928 |
| **`card1+addr1+D1anchor`** | **217,850** | 125,222 (57.5%) | 1 | 6 | 1,414 |
| `+card2+card3+card5` | 222,452 | 128,947 (58.0%) | 1 | 6 | 1,414 |

**[INFERENCE]** `D1` = days since first activity on the card/account, so `day − D1` is a **stable per-account origin date**. Combining it with `card1+addr1` disambiguates the reuse of `card1` values across different real accounts. Adding `card2/card3/card5` changes group count by 2% — **not worth the extra complexity.**

### The decisive measurement: label homophily

**[FACT]** For each candidate entity, leave-one-out neighbour label statistics within shared-entity groups:

| Candidate entity | Nodes | Coverage | P(nb=F \| self=F) | P(nb=F \| self=L) | base | **LIFT** |
|---|---|---|---|---|---|---|
| **PseudoAccount `card1+addr1+D1anchor`** | 199,070 | 88.7% | **86.47%** | 0.33% | 2.46% | **35.12×** |
| PseudoAccount + `card2+card5` | 197,806 | 87.0% | 86.31% | 0.33% | 2.45% | 35.28× |
| `card1` | 13,553 | 100.0% | 6.30% | 3.13% | 3.50% | 1.80× |
| NetProxy `id_19+id_20` | 2,394 | 23.6% | 13.47% | 9.56% | 7.92% | 1.70× |
| CardProfile `card1..6` | 13,691 | 98.1% | 5.87% | 3.19% | 3.46% | 1.69× |
| P+R emaildomain pair | 650 | 21.4% | 12.65% | 11.55% | 8.38% | 1.51× |
| DeviceFingerprint composite | 4,880 | 12.0% | 6.56% | 3.30% | 4.47% | 1.47× |
| `P_emaildomain` | 59 | 84.0% | 4.18% | 3.99% | 3.60% | 1.16× |
| ScreenRes `id_33` | 260 | 12.4% | 5.01% | 3.92% | 4.59% | 1.09× |
| BillingAddr `addr1+addr2` | 437 | 88.9% | 2.45% | 2.35% | 2.46% | **1.00×** |
| `ProductCD` | 5 | 100.0% | 3.25% | 2.27% | 3.50% | **0.93×** |
| `DeviceInfo` | 1,786 | 20.1% | 6.45% | 6.15% | 7.25% | **0.89×** |

**[INFERENCE] This table should drive the graph schema, and it carries two uncomfortable messages.**

**First: three candidate entities carry no relational signal at all.** `DeviceInfo` (0.89×), `ProductCD` (0.93×) and `addr1+addr2` (1.00×) show fraud neighbours no more often than chance. They are also the worst degree offenders (46k–440k max degree). Including them as node types would add hub nodes that smear every embedding toward the global mean while contributing nothing — the textbook over-smoothing failure. **They belong in the feature vector, not the graph.** Note this contradicts the intuitive "device sharing implies a fraud ring" premise: at the granularity this dataset provides, `DeviceInfo` is a *model string* (`SAMSUNG SM-G892A Build/NRD90M`), not a device identity — 47,722 transactions share its top value. It is a device *type*, and there are millions of each.

**Second, and more important: the 35× pseudo-account lift is a labelling artifact, not a detection breakthrough.** An 86.47% chance that a fraud transaction's account-mate is also fraud is not a behavioural pattern any detector discovered. **[INFERENCE]** Vesta labelled at the account level: when a card is reported compromised, that transaction *and the rest of the account's activity* are flagged. So the pseudo-account grouping largely recovers the label-assignment rule itself.

This reframes the whole project, in a useful way:

- It is the honest explanation for why graph methods post spectacular numbers on IEEE-CIS. Much of the gain is label propagation through a labelling convention.
- It is **not** cheating, and it is **not** useless: in production you genuinely do know which accounts had prior confirmed fraud. That is exactly what a blocklist is.
- But it is only legitimate under a stated assumption, and it makes **H1/H2 harder to test honestly**, because a naive "graph beats XGBoost" result may be measuring nothing but this artifact.

**[FACT] Quantifying it across the proposed split** (train = day 1–121, val = 122–151, test = 152–182), using only train+val labels to build the feature:

| Test transactions with prior account history | n | fraud rate |
|---|---|---|
| account had **prior confirmed fraud** | 969 | **59.86%** |
| account had **no prior fraud** | 37,731 | **0.08%** |

**[INFERENCE]** A single strictly-backward-looking binary feature separates the two groups by ~750×. Any graph model with access to historical neighbour labels will inherit this. It must be reported as its own baseline so we can see what the *rest* of the architecture contributes.

**[REC] This becomes a required baseline, not a footnote.** Insert a **Level 3.5: "prior-fraud blocklist"** — one feature, one threshold, no learning. Every later model must beat it, and each ablation must report performance **with and without** historical-label features. Without this control, the ablation study cannot support H1 or H2, and the project's central claim collapses under review.

**[REC] Recommended node types** (justified by lift × coverage × degree sanity):

| Include | Node type | Key | Nodes | Why |
|---|---|---|---|---|
| YES | `Transaction` | `TransactionID` | 590,540 | the prediction target |
| YES | `PseudoAccount` | `card1+addr1+(day−D1)` | ~218k | lift 35.12×, coverage 88.7% — but see the artifact caveat |
| YES | `Card` | `card1` | 13,553 | lift 1.80×, coverage 100%, degree sane |
| YES | `NetProxy` | `id_19+id_20` | 2,394 | lift 1.70×, best signal in the identity block |
| YES | `DeviceFingerprint` | `DeviceInfo+DeviceType+id_30..33` | 4,880 | lift 1.47×; composite is 1.65× better than `DeviceInfo` alone |
| NO | ~~`BillingAddr`~~ | `addr1+addr2` | 437 | **lift 1.00×**, max degree 46,324 |
| NO | ~~`EmailDomain`~~ | `P/R_emaildomain` | 59/60 | lift 1.16×, max degree 228,355 |
| NO | ~~`Product`~~ | `ProductCD` | 5 | **lift 0.93×**, max degree 439,670 |
| NO | ~~`DeviceInfo` alone~~ | `DeviceInfo` | 1,786 | **lift 0.89×** — below chance |

Everything in the NO rows stays as a transaction-node **feature**. `P+R emaildomain` *pair* (1.51×) is interesting but is a 650-value categorical — capture it as a feature, not a node.

**[REC]** 5 node types, not 8. Each of the four entity types earns its place with a measured lift above 1.4× and a max degree under 6,000.

---

## 11. Candidate edge types

**[REC]** All edges are `Transaction -> Entity`, stored bidirectionally for message passing. Attributes carry the timestamp so temporal gating is enforceable at the edge level.

| # | Source | Relation | Destination | Observed or derived | Time-dependent | Edges (train) |
|---|---|---|---|---|---|---|
| 1 | Transaction | `charged_to` | PseudoAccount | **derived** (`card1+addr1+(day−D1)`) | anchor is static; edge is time-stamped | ~524k |
| 2 | Transaction | `used_card` | Card | directly observed (`card1`) | edge time-stamped | 590,540 |
| 3 | Transaction | `via_network` | NetProxy | directly observed (`id_19+id_20`) | edge time-stamped | 139,212 |
| 4 | Transaction | `from_device` | DeviceFingerprint | derived (6-field composite) | edge time-stamped | 71,120 |
| 5 | PseudoAccount | `owns_card` | Card | derived (co-occurrence) | **must be gated** — only if co-observed at t ≤ prediction time | ~218k |

Edge attributes on 1–4: `TransactionDT`, `TransactionAmt`, `Δt` since the entity's previous transaction, and the entity's prior-transaction count — all computed by backward-looking expanding window only.

**[FACT]** Edge type 5 is the only entity-to-entity edge and the only one at risk of encoding future co-occurrence. Types 1–4 are structurally incapable of leaking because each edge is anchored to exactly one transaction's own timestamp.

**[REC]** Drop edge type 5 in the first iteration. Two-hop message passing already connects accounts that share a card via `Account -> Transaction -> Card -> Transaction -> Account`. Adding the shortcut costs a leakage surface and buys a hop we already have. Add it later only if an ablation shows two-hop propagation is insufficient.

---

## 12. High-cardinality fields

**[FACT]**

| Class | Columns |
|---|---|
| **Identifier-like / blocklist** | `TransactionID` (contiguous, r=0.998 with time) |
| **High-cardinality categorical** | `card1` (13,553), `dist1` (2,651), `DeviceInfo` (1,786), `dist2` (1,751), `id_19` (522), `card2` (500), `id_21` (490), `id_20` (394), `id_25` (341), `addr1` (332), `id_33` (260) |
| **Medium** | `id_31` (130), `card5` (119), `card3` (114), `id_17` (104), `id_26` (95), `id_30` (75), `addr2` (74), `R_emaildomain` (60), `P_emaildomain` (59), `id_13` (54) |
| **Low (<=5)** | `ProductCD`, `card4`, `card6`, `M1`–`M9`, `DeviceType`, `id_12/15/16/23/27/28/29/32/34/35/36/37/38` |
| **Continuous** | `TransactionAmt`, `dist1/2`, `D1`–`D15`, `id_01`–`id_11`, most of `V*` |
| **Count-like integer** | `C1`–`C14` (all integral, verified) |

**[FACT] Value-set overlap between train and Kaggle test** — the columns whose *identities* churn:

| Column | uniq train | uniq test | overlap | **% of train values seen in test** |
|---|---|---|---|---|
| `addr1` | 332 | 292 | 183 | **55.1%** |
| `id_21` | 490 | 443 | 199 | **40.6%** |
| `id_13` | 54 | 27 | 26 | **48.1%** |
| `card1` | 13,553 | 13,244 | 9,706 | **71.6%** |
| `card5` | 119 | 102 | 83 | 69.7% |
| `DeviceInfo` | 1,786 | 2,226 | 1,213 | 67.9% |
| `card2` | 500 | 497 | 496 | **99.2%** |
| `P_emaildomain` | 59 | 60 | 59 | **100.0%** |

**[INFERENCE]** Entity *populations turn over* even where marginal distributions are stable (see §14). Only 71.6% of `card1` values recur. High-cardinality categoricals must be encoded with an explicit unseen-category path — frequency or leakage-safe target encoding fitted on train only, with a defined fallback. One-hot on `card1` is 13,553 columns and still breaks on the 28% of unseen test values.

### `C1`–`C14` **[FACT]**

All 14 are integral, 0% missing, with extreme right tails: `C1` mean 14.09 / max 4,685; `C13` mean 32.54 / max 2,918; `C3` mean 0.01 / max 26.

**[FACT] Provenance test** — within `card1` groups having >=3 transactions (8,419 groups), sorted by time:

| Col | frac constant within card1 | frac non-decreasing over time |
|---|---|---|
| C1 | 0.2317 | 0.3281 |
| C4 | 0.4530 | 0.5090 |
| C5 | 0.4225 | 0.4807 |
| C13 | 0.0634 | 0.2222 |
| C14 | 0.2037 | 0.3026 |

**[INFERENCE]** These are Vesta-supplied aggregate counts ("how many addresses/devices are associated with this card"), but **they are neither constant per card nor monotonically increasing over time**. So they are not simple running counters keyed on `card1`, and **I cannot verify from the data alone whether they were computed using future information.** They are vendor-provided, identical in kind to the `V*` block, and every baseline gets them equally — so the comparison stays fair — but the project must not claim the feature set is *provably* leakage-free.

**[REC]** State this limitation explicitly. Optionally run one ablation dropping `C1`–`C14` and `V1`–`V339` entirely, keeping only columns whose provenance we can verify. If graph gains survive that, the H1/H2 claim is far more robust.

### `V1`–`V339` **[FACT]**

339 columns collapse to **just 15 distinct missingness patterns**. Largest blocks: 46 cols at 77.91% missing; 43 cols at 0.05%; 32 cols at 0.002%; 31 cols at 76.36%; 23 cols at 12.88%.
Cardinality: min 2, median 32, max 37,367. 7 columns are binary; 92 have <=10 values; 107 have >100. 263 of 339 are integral.

**[INFERENCE]** The 15 missingness blocks are Vesta feature *families*, each computed from a different source table — which is why a family is present or absent together. This is a principled, non-arbitrary basis for dimensionality reduction: reduce **within** blocks, never across them.

**[REC]** Do not feed 339 raw `V` columns to a GNN (they would dominate a 5-node-type schema). Keep them in full for the XGBoost baselines (trees handle them fine and it keeps the baseline genuinely *strong*, which the research question requires). For graph models, reduce per block — and include the reduction as an ablation so the choice is measured, not assumed.

---

## 13. Missingness observations

**[FACT]** Distribution of column missingness:

| Bucket | `train_transaction` (394) | `train_identity` (41) |
|---|---|---|
| 0% | 95 | 3 |
| 0–5% | 17 | 14 |
| 5–25% | 70 | 5 |
| 25–50% | 38 | 7 |
| 50–75% | 6 | 3 |
| 75–90% | **166** | 0 |
| 90–99% | 2 | 9 |
| >=99% | 0 | 0 |

**[FACT]** 166 of 394 transaction columns (42%) are 75–90% missing. In `train_identity`, 9 of 41 columns exceed 90% missing (`id_21`–`id_27` at ~99.1%).

**[FACT] Train/test missingness differs sharply for some columns:**
- `D15`: 15.1% train -> **2.4%** test
- `D10`: 12.9% train -> **2.5%** test
- `D6`: 87.6% train -> **75.4%** test
- `D13`: 89.5% train -> **75.7%** test

**[INFERENCE]** Vesta's upstream data collection changed between the two periods — the `D` block got materially more complete. This is **missingness drift**, and it is invisible to any drift metric computed on observed values only. Our drift monitor must track **missingness rate per column** as a first-class signal, not just distributional shift.

**[FACT]** Bucket counts differ between train and test transaction files (e.g. 50–75% bucket: 6 train vs **115** test), consistent with the same conclusion.

**[REC]** Never use a sentinel like `−999` without a paired `is_missing` indicator. Given §7 (identity presence = 3.75× signal) and the above, missingness in this dataset is a feature, not a defect.

---

## 14. Leakage risks

Ranked by severity, each with the concrete guard.

### R1 — `TransactionID` as a feature — CRITICAL, trivially avoidable
**[FACT]** Contiguous and r = 0.998 with `TransactionDT`. **[REC]** Hard-coded blocklist enforced in the feature assembler, plus a unit test that fails if it appears in any model's input columns.

### R2 — Account-level label propagation — CRITICAL, subtle
**[FACT]** 86.47% within-pseudo-account homophily; prior-fraud accounts run 59.86% vs 0.08%.
**[INFERENCE]** Graph message passing will move labels across the split boundary if neighbours are not time-gated. Worse, it will look like model skill.
**[REC]** Three guards: (i) message passing may only traverse edges with `edge_time <= prediction_time`; (ii) any neighbour-label feature uses strictly `<` current time; (iii) the §10 blocklist baseline quantifies how much of the gain is *just* this.

### R3 — Label reporting delay is unmodelled — HIGH, a genuine dataset limitation
**[INFERENCE]** In reality a chargeback surfaces weeks to months after the transaction. At day 152 you do **not** know day-151 labels. The dataset gives no reporting-date column, so we cannot model this correctly.
**[REC]** Make the assumption explicit and test its cost: run the historical-label features under a label-availability lag `L ∈ {0, 30, 60}` days and report the degradation curve. `L = 0` is the optimistic bound every Kaggle solution implicitly assumes; `L = 30/60` is closer to production. Presenting that curve is a genuine research contribution and costs one loop.

### R4 — Full-dataset preprocessing statistics — HIGH, the classic mistake
**[REC]** Every fitted transform (imputation values, scalers, frequency maps, target encodings, PCA on `V` blocks, category vocabularies) is fitted on **train only** and applied to val/test. One `fit()` call inside the split loop is how this project dies quietly.

### R5 — Entity aggregates over the full timeline — HIGH
**[REC]** All entity statistics (degree, transaction count, mean amount, time-since-last, distinct-cards-seen) use **backward-looking expanding or rolling windows only**. Never `groupby(entity).transform('mean')` on the whole frame. Enforce with an invariant test: recomputing a feature for row *i* using only rows with `DT < DT_i` must reproduce the stored value.

### R6 — `D` column forward references — LOW, real
**[FACT]** 45 rows across D4/D6/D11/D12/D14/D15 have negative values. **[REC]** Clip at 0, log the count, mention it.

### R7 — `C`/`V` provenance unverifiable — MEDIUM, honest limitation
**[FACT]** §12 shows `C` columns are neither constant nor monotone within `card1`. **[REC]** Disclose; optionally ablate the whole vendor block.

### R8 — Threshold and hyperparameter selection on test — HIGH, procedural
**[REC]** The day 152–182 test period is opened **once**, at the end. Thresholds, K in Precision@K, drift-trigger settings, early-stopping rounds, and architecture choices are all fixed on val. Write the test-set evaluation as a separate script that loads frozen artifacts.

**[REC] The single enforcement mechanism that matters:** one function
`build_features(df, as_of_time, fitted_state) -> X`
that physically cannot see rows with `DT > as_of_time`. Every model, every ablation, every drift window calls it. Leakage then requires modifying that function rather than forgetting a detail in one of nine scripts.

---

## 15. Computational risks

**[FACT] The graph is small. This is good news measured, not hoped for.**

| Quantity | Value |
|---|---|
| Transaction nodes (train period) | 590,540 |
| Entity nodes, recommended 4 types | 13,553 + ~218,000 + 2,394 + 4,880 ≈ **238,827** |
| Total nodes | **~829,000** |
| Directed bipartite edges (4 types) | 1,325,000 |
| Bidirectional edges for message passing | **~2,650,000** |
| `edge_index` as int64 | **0.04 GB** |
| `edge_index` as int32 | **0.02 GB** |

For reference, all 12 candidate entity types would still only be 627,870 nodes / 7.08M edges / **0.11 GB**.

**[INFERENCE]** Full-graph training fits in 17.2 GB VRAM with large headroom. **Neighbour sampling is not required for scale** — only for minibatch SGD if we want it. This removes the usual GNN scaling risk entirely and lets the compute budget go to ablations instead, which is where the research value is.

**[FACT] The real blow-up is entity-to-transaction projection.** Sum of C(deg,2) over shared entities:

| Entity | Transaction–transaction pairs if projected | max hub degree |
|---|---|---|
| `ProductCD` | **100,325,490,641** | 439,670 |
| `P_emaildomain` | **33,378,464,204** | 228,355 |
| `addr1+addr2` | 5,899,439,818 | 46,324 |
| `DeviceInfo` | 1,444,857,382 | 47,722 |
| `card1` | 746,388,072 | 14,932 |

**[INFERENCE]** Projecting to a homogeneous transaction–transaction graph is **categorically infeasible** — 100 billion edges for `ProductCD` alone, ~1.5 TB as int32. Keep the graph **bipartite and heterogeneous**. This is also the direct methodological justification the brief asks for: we are not choosing heterogeneous message passing for style points, we are choosing it because the homogeneous projection is 4–5 orders of magnitude too large. Note the four excluded entity types from §10 are also the four worst offenders here — dropping them is doubly justified.

### Other measured risks

- **[FACT]** `train_transaction` alone is 1,879 MB in pandas with default dtypes. Train + test + both identity ≈ **3.6 GB**; with intermediates, an unguarded pipeline will exceed 21.6 GB free RAM. **[REC]** Downcast float64->float32 and int64->int32 on load (≈50% saving), convert to Parquet once in `data/processed/`, and never hold train and test frames simultaneously.
- **[FACT]** CSV parse of both transaction files takes minutes per run. **[REC]** Parquet + a cached feature store. Re-parsing CSVs in every experiment will dominate the project's wall-clock.
- **[INFERENCE]** 8 logical cores. XGBoost `n_jobs=8` and PyG `num_workers=0` on Windows (worker spawn on Windows is slow and often deadlocks with ROCm).
- **[INFERENCE]** 96 GB disk free is adequate but not generous; Parquet + checkpoints + SHAP caches should stay under ~20 GB. Do not write per-epoch full-graph embedding dumps.
- **[INFERENCE]** ROCm-on-Windows PyG is the top schedule risk — see §1 and the capability probe.

---

## 16. Recommended heterogeneous graph schema

**[REC]** 5 node types, 4 edge types. Every inclusion is backed by a measured number above.

```
                    +----------------------+
                    |     Transaction      |  590,540 nodes  [prediction target]
                    |  feat: Amt, log(Amt),|
                    |  hour(verified), dow,|
                    |  amt_decimals>2dp,   |
                    |  ProductCD(1-hot,5), |
                    |  P/R_emaildomain,    |
                    |  addr1/addr2,dist1/2,|
                    |  M1-M9, C1-C14,      |
                    |  D1-D15(clipped),    |
                    |  V* (block-reduced), |
                    |  id_01-id_38,        |
                    |  has_identity flag,  |
                    |  per-col is_missing  |
                    +--+-----+-----+----+--+
        charged_to |    |     |     | from_device
           ~524k e |    |     |     +---------------+
      +------------+    |     +--------+            |
      |      used_card  |  via_network |            |
      |        590,540  |      139,212 |            |  71,120
      v                 v              v            v
+---------------+ +------------+ +----------+ +------------------+
| PseudoAccount | |    Card    | | NetProxy | | DeviceFingerprint|
|   ~218,000    | |   13,553   | |   2,394  | |      4,880       |
| card1+addr1+  | |   card1    | | id_19+   | | DeviceInfo+Type+ |
|  (day - D1)   | |            | |  id_20   | | id_30/31/32/33   |
| lift 35.12x   | | lift 1.80x | |lift 1.70x| |   lift 1.47x     |
| cover 88.7%   | | cover 100% |cover 23.6% | |   cover 12.0%    |
+---------------+ +------------+ +----------+ +------------------+
```

**Entity node features** — all from backward-looking expanding windows, all `as_of_time`-gated:
`n_prior_txns`, `days_since_first_seen`, `days_since_prev_txn`, `prior_amt_mean/std/max`, `prior_distinct_<other-entity>_count`, `prior_confirmed_fraud_count` (**flagged as a historical-label feature; subject to the lag study in R3 and excluded in the no-label-features ablation**).

**Edge attributes:** `TransactionDT`, `Δt` since entity's previous transaction, `TransactionAmt`, entity's prior degree at that moment.

**[REC] Heterogeneous message passing — how, concretely.** PyG `HeteroData` with `HeteroConv`, one `SAGEConv`/`GATConv` per relation and per direction, aggregated with `sum` across relations. Node types have genuinely different feature dimensionalities (a Transaction has hundreds of features; a NetProxy has a handful of derived ones), so a per-type input projection into a shared hidden dimension is required before the first conv — that projection is the mechanical reason this cannot be silently collapsed to a homogeneous graph. Two layers only: layer 1 moves transaction evidence to entities, layer 2 returns aggregated entity context to transactions. Deeper hops mean transaction->entity->transaction->entity->transaction paths that reach unrelated accounts through `card1` hubs of degree 14,932.

**[REC]** We will **not** silently homogenise. §15 gives the quantitative reason (100 billion projected edges), and the per-type projection above gives the architectural one.

---

## 17. Dataset limitations

**[FACT] Verified absent — searched all 434 columns, zero matches:**

| The dataset does NOT contain | Consequence |
|---|---|
| **Fraud-ring / coordination labels** | H5 **cannot be validated on real data.** Discovered communities are unverifiable. |
| **Customer / person identity** | No true account entity; `PseudoAccount` is a derived proxy, and 57.5% of its groups are singletons. |
| **Account / card PAN** | `card1` is an anonymised token, not a card number; 25.4% of its values appear exactly once. |
| **Merchant / seller identity** | No merchant node. `ProductCD` has 5 values and zero homophily — it is not a merchant proxy. |
| **IP address** | Only `id_23` (`IP_PROXY:*`, 99.1% missing). An IP node type would be fabrication. |
| **Absolute calendar dates** | Reference instant undisclosed. Hour-of-day is recoverable and **verified**; real dates are not. No holiday/payday joins. |
| **Label reporting date** | R3 cannot be modelled from data — only assumed and sensitivity-tested. |
| **Feature semantics for `V*`, `C*`, `id_*`** | Anonymised. Explanations name columns, not mechanisms. SHAP will say "`V257` matters," never "velocity matters." |

**[FACT]** Ground truth for the entire Kaggle test period (506,691 rows) does not exist for us. `sample_submission.csv` is the constant 0.5.

**[INFERENCE]** Limits on what this project may claim:

1. **Fraud rings cannot be validated on IEEE-CIS.** Use "suspicious coordinated communities" / "candidate fraud rings" throughout — never "detected fraud rings." The brief already requires this; the data confirms it is not optional.
2. **[REC]** Because of #1, a **synthetic planted-pattern benchmark is mandatory, not optional** — it is the *only* way to obtain a quantitative number for H5. Plant known coordinated structures (N accounts sharing K devices in a burst window) into a copy of the real graph, then measure precision/recall of community recovery against the planted ground truth. Report real-data communities qualitatively as case studies alongside it.
3. **[FACT]** What this dataset calls a "ring" is mostly **one compromised account reused**, not multi-actor collusion: 86.47% homophily is *within* pseudo-account, whereas cross-entity lifts are 1.47–1.80×. **[FACT]** The burst signature test supports this — among 39,459 groups of >=3 identical `(card1, addr1, Amt, ProductCD)`, mean fraud rate is **2.75%**, *below* the 3.499% base rate, with a median span of 95 days. Repetition here is subscription/retry behaviour, not attack bursts. Only 497 such groups are 100% fraud. **[REC]** Say this plainly. "We looked for burst-attack signatures and the obvious one is not there" is a real finding and protects the report from an easy referee objection.
4. Explanations are structural, not semantic (anonymised features).
5. Single vendor, single 13-month window, one geography — no cross-institution generalisation claims.

---

## 18. Recommended chronological train/validation/test strategy

**[REC] Use `train_transaction.csv` only.** The Kaggle test files have no labels and are excluded from all quantitative evaluation (optional submission-format inference only, clearly labelled as unvalidated).

**[FACT]** Proposed split on the derived day index (train covers days 1–182):

| Split | Days | `TransactionDT` range | n | frauds | rate | share |
|---|---|---|---|---|---|---|
| **TRAIN** | 1–121 | 86,400 – 10,540,797 | **417,559** | 14,721 | 3.525% | 70.7% |
| **VAL** | 122–151 | 10,540,826 – 13,132,788 | **83,655** | 2,828 | 3.381% | 14.2% |
| **TEST** | 152–182 | 13,132,810 – 15,811,131 | **89,326** | 3,114 | 3.486% | 15.1% |

**[INFERENCE]** Well-balanced: ~4 months train, 1 month val, 1 month test; fraud rate stable across splits (3.38–3.53%) so no split is pathological; VAL and TEST each hold ~3,000 positives — enough for stable PR-AUC. A quantile-based 70/85 cut (days 1–120 / 121–152 / 153–182) is nearly identical; the round-day boundaries above are preferred for reproducibility and legibility.

**[FACT] Boundary behaviour measured on this split:**

| | count | share |
|---|---|---|
| TEST pseudo-accounts also in TRAIN | 14,757 | 34.9% |
| TEST pseudo-accounts also in VAL | 12,780 | 30.2% |
| **TEST pseudo-accounts cold-start** | **24,025** | **56.8%** |
| TEST transactions on accounts with prior history | 38,700 | 48.4% (fraud **1.576%**) |
| TEST transactions on cold-start accounts | 41,207 | 51.6% (fraud **2.934%**) |
| TEST `card1` values seen earlier | 5,695 / 6,334 | 89.9% |

**[INFERENCE]** This is the most operationally important number in the report. **51.6% of test transactions are on entities with no history at all, and they carry nearly double the fraud rate (2.934% vs 1.576%).** So graph and history features are structurally unavailable on exactly the harder half of the problem. A pooled metric will average a near-trivial subproblem (0.08% fraud when the account has clean history) with a genuinely hard one.

**[REC] Report every headline metric stratified into cold-start vs has-history.** A pooled PR-AUC on this split is close to meaningless, and any reviewer will ask. Note `card1` recurs at 89.9% while pseudo-accounts recur at only 43.2% — so `Card` edges are the ones carrying signal into the cold-start half, which makes the modest 1.80× `card1` lift more valuable than its size suggests.

**[REC] Drift-analysis windows:** 13 fourteen-day windows, already computed. **[FACT]** Fraud rate by window ranges **2.256% -> 4.374% (1.94×)**, with volume falling from 70,607 (window 1) to 37,693 (window 13).

**[FACT] Marginal covariate drift is mild** — PSI, train-first-half vs second-half, and train vs Kaggle test:

| Column | PSI h1 vs h2 | PSI train vs Kaggle test |
|---|---|---|
| `TransactionAmt` | 0.0102 | 0.0040 |
| `card1` | 0.0013 | 0.0014 |
| `addr1` | 0.0021 | 0.0005 |
| `C1` | 0.0081 | 0.0226 |
| `C13` | 0.0218 | 0.0007 |
| `D1` | 0.0181 | 0.0045 |
| `D15` | 0.0358 | **0.0916** |
| `dist1` | 0.0103 | 0.0110 |

All below the 0.1 "moderate shift" line except `D15` at 0.0916.

**[INFERENCE] This overturns the natural assumption behind H4, and it is the most interesting negative result so far.** There is **no strong marginal covariate drift** in this dataset. What actually changes is:

1. **Label/conditional drift** — fraud rate swings 1.94× across windows while feature marginals stay put: P(y|x) shifts, not P(x).
2. **Entity turnover** — 28.4% of `card1` and 44.9% of `addr1` values never recur; 56.8% of test accounts are cold-start. The *population* churns while the *distribution* holds.
3. **Missingness drift** — `D15` 15.1%->2.4%, `D10` 12.9%->2.5% (§13), which PSI-on-observed-values cannot see at all.

**[REC]** Consequences for the drift and adaptation design, all of which save wasted effort:
- **Do not build a PSI-based drift trigger.** Measured PSI never crosses a sensible threshold, so a PSI trigger would never fire and H4 would return a vacuous "adaptation never activated."
- **Trigger on rolling PR-AUC / Recall@K degradation on val**, plus **per-column missingness rate** and **unseen-category rate** — the three things that actually move.
- Keep PSI and KS as *reported diagnostics*, since "we measured covariate drift and it was absent" is a genuine finding worth publishing; just don't build control flow on them.
- This makes H4 a **real test with a real chance of a negative result.** If drift is mostly conditional and entity-turnover-driven, periodic scheduled retraining may match or beat triggered adaptation — which is an honest and publishable answer.

**[FACT] Alert-budget reference numbers for the TEST period** (89,326 rows, 3,114 frauds, base 3.486%):

| K | Alerts | Perfect-model Recall@K | Random-model Precision@K |
|---|---|---|---|
| 0.5% | 446 | 14.3% | 3.486% |
| 1% | 893 | 28.7% | 3.486% |
| 2% | 1,786 | 57.4% | 3.486% |
| 5% | 4,466 | 100.0% | 3.486% |

**[INFERENCE]** Recall@K is *ceilinged by K* here — at K = 1% the best achievable recall is 28.7%. Any table reporting Recall@1% must show that ceiling next to it, or the numbers look like failure when they may be near-optimal. **[REC]** Use **K = 1% and K = 2%** as the primary operating points and always print both the random floor and the perfect ceiling.

---

## 19. Recommended next implementation step

**[REC]** In order. Steps 1–2 are prerequisites; do not start step 3 before step 1 is green.

**Step 1 — ROCm/PyG capability probe (~30 min, highest priority).**
Install `scikit-learn`, `xgboost`, `torch-geometric` into `.venv` (Python 3.12). Run a toy `HeteroData` with 3 node types / 2 edge types through `HeteroConv({...: SAGEConv})` forward **and** backward on `cuda:0`. Record pass/fail and version pins in `reports/env_probe.md`. If it fails, retry CPU-only and record that as the decision — §15 shows CPU-only PyG is viable at 829k nodes / 2.65M edges, so this is a de-risked fallback, not a project-ender. **Everything downstream depends on this answer, so get it first.**

**Step 2 — Project skeleton + immutable split + Parquet conversion.**
Create `src/ configs/ models/ reports/ tests/ notebooks/`, `git init`, and a `configs/data.yaml` holding the §18 boundaries (days 1–121 / 122–151 / 152–182) as data, not code. Convert the 4 CSVs to Parquet in `data/processed/` with float32/int32 downcasting and `id-*` -> `id_*` normalisation; leave `data/raw/` untouched. Write `src/data/splits.py` returning index arrays, and a test asserting the exact row counts 417,559 / 83,655 / 89,326 and fraud counts 14,721 / 2,828 / 3,114 from this report. That test is our regression guard against silent split drift for the rest of the project.

**Step 3 — The leakage-safe feature assembler.**
Implement the single `build_features(df, as_of_time, fitted_state)` gateway from §14, with: the `TransactionID` blocklist, per-column `is_missing` indicators, the verified `hour = (DT//3600)%24`, `amt_decimals>2dp`, `D` clipping at 0, and the entity-history expanding-window aggregates. Ship it with the §14 R5 invariant test (recompute row *i* from rows with `DT < DT_i`; must match). **This is the most important file in the repository** — everything else consumes it, and it is the one place leakage can enter.

**Step 4 — Levels 1–3 baselines + the blocklist control.**
Logistic Regression, Random Forest, XGBoost, and the §10 **Level 3.5 prior-fraud blocklist**, all on the frozen split, all reporting PR-AUC / ROC-AUC / P@K / R@K at K in {1%, 2%} — pooled, **per `ProductCD`**, and **stratified cold-start vs has-history**. TEST stays sealed; VAL only.

**[REC] Do not touch graph construction until step 4 produces a strong tabular baseline.** H1 and H2 are *comparative* hypotheses. A weak baseline makes the graph look good and the research worthless — and given what §10 and §18 uncovered, a well-tuned XGBoost with `card1` frequency encoding plus one historical-fraud-count feature may already capture most of what the graph offers. Finding that out early, cheaply, is exactly what this project's stated core principle asks for.

---

## Summary of the five findings that should change the plan

1. **The 86.47% pseudo-account homophily (35× lift) is largely a labelling artifact, not detection signal.** A single prior-fraud flag separates 59.86% from 0.08% fraud. -> Add the Level 3.5 blocklist baseline; run every ablation with and without historical-label features, or H1/H2 are untestable.
2. **`DeviceInfo` (0.89×), `ProductCD` (0.93×) and `addr1+addr2` (1.00×) carry zero label homophily** while having the largest degrees (up to 439,670). -> 5 node types, not 8. The intuitive "shared device implies fraud ring" premise is not supported at this dataset's granularity.
3. **Marginal covariate drift is essentially absent (all PSI < 0.1); drift is conditional, entity-turnover and missingness-driven.** -> Do not build a PSI trigger; it would never fire. Trigger on performance + missingness + unseen-category rate.
4. **51.6% of test transactions are cold-start entities carrying ~2× the fraud rate.** -> Pooled metrics are misleading; stratify everything.
5. **The GPU is AMD/ROCm on Windows, and `torch-scatter`/`torch-sparse` have no wheel for it.** -> Probe before building. The graph is small enough (0.04 GB `edge_index`) that CPU-only is a real fallback.

**Status: Phase 1 complete. No models trained, no graph built, no raw data modified. Awaiting approval before implementation.**
