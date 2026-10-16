# Phase 4 — Heterogeneous Temporal Graph Construction & Validation

**Scope:** graph construction and validation only. No GraphSAGE/GAT training as an official experiment, no p5_scale.py, no compiled PyG extensions installed, no final model comparison, no ring discovery, no dashboard. One single-pass feasibility benchmark was run (permitted explicitly) — not a training loop.

---

## A. Final graph schema

Unchanged from Phase 1's own measured recommendation (reports/phase1_reconnaissance.md, Sections 10–11, 16) — re-verified fresh against the current data (Section D below) rather than simply cited, but not re-derived from scratch, since Phase 1 already did that analysis rigorously. **5 node types, 4 edge types.**

```
                    +----------------------+
                    |     Transaction      |  590,540 nodes, 470 features
                    +--+-----+-----+----+--+
        charged_to |    |     |     | from_device
      524,834 edges|    |     |     +---------------+
      +-------------+   |     +--------+            |
      |      used_card  |  via_network |            |  71,120 edges
      |      590,540 e  |    139,212 e |            |
      v                 v              v            v
+---------------+ +------------+ +----------+ +------------------+
| pseudo_account| |    card    | | netproxy | |device_fingerprint|
|    199,625    | |   13,553   | |   2,394  | |      4,880       |
+---------------+ +------------+ +----------+ +------------------+
```

Every edge is reversed automatically (`T.ToUndirected()`) for bidirectional message passing, giving 8 total edge-type entries in the graph object.

---

## B. Node types

| Node type | Definition | Nodes | Feature source | Feature dim |
|---|---|---|---|---|
| `transaction` | one per labeled transaction (TRAIN+VAL+TEST) | 590,540 | Phase 3's `TreeFeatureBuilder`, fit on TRAIN only, **`for_model="rf"` variant** (median-imputed, NaN-free) | 470 |
| `pseudo_account` | `card1+addr1+(day−D1)` | 199,625 | none (constant placeholder) | 1 |
| `card` | `card1` | 13,553 | none (constant placeholder) | 1 |
| `netproxy` | `id_19+id_20` | 2,394 | none (constant placeholder) | 1 |
| `device_fingerprint` | `DeviceInfo+DeviceType+id_30+id_31+id_32+id_33` | 4,880 | none (constant placeholder) | 1 |

**Why entity nodes have no engineered features:** any precomputed entity statistic (total degree, historical transaction count, etc.) would itself require the exact backward-looking discipline Phase 3's `historical.py` uses — and would risk quietly reintroducing a leakage channel if that discipline weren't perfectly reproduced. The simplest, safest choice is no static feature at all: entity representations come entirely from time-masked message passing over their neighbouring transactions.

**Why `TreeFeatureBuilder`'s `"rf"` variant, not `"xgb"`:** a real correctness issue caught during construction — XGBoost handles missing values via a learned split direction, but a GNN's per-type `nn.Linear` input projection has no such mechanism; a single NaN would propagate through the entire forward pass. Switched to the median-imputed variant before building anything; verified 0 NaN in the final transaction feature matrix.

**Deliberately excluded from transaction features:** the Phase 3 historical/entity features (`account_prior_fraud_rate`, etc.). Including them would let the transaction node pre-encode exactly the information the graph's own message passing is meant to test — this is preserved instead as a separate, explicit ablation arm for Phase 5.

---

## C. Edge/relation types

| Relation | Source → Destination | Edges (directed) | Coverage | Phase 1 lift | Edge attributes |
|---|---|---|---|---|---|
| `charged_to` | transaction → pseudo_account | 524,834 | 88.9% | 35.12× (labelling-artifact caveat — see Phase 3) | `edge_time` (day, 1–182) |
| `used_card` | transaction → card | 590,540 | 100.0% | 1.80× | `edge_time` |
| `via_network` | transaction → netproxy | 139,212 | 23.6% | 1.70× | `edge_time` |
| `from_device` | transaction → device_fingerprint | 71,120 | 12.0% | 1.47× | `edge_time` |

**Excluded, on the record (configs/graph.yaml), not silently dropped:**

| Candidate | Lift | Max degree | Why excluded |
|---|---|---|---|
| `billing_addr` (addr1+addr2) | 1.00× | 46,324 | no signal above chance; 4th-largest hub |
| `email_domain` (P/R_emaildomain) | 1.16× | 228,355 | weak lift, 2nd-largest hub |
| `product` (ProductCD) | 0.93× | 439,670 | **below** chance; largest hub in the dataset |
| `device_info_alone` (DeviceInfo) | 0.89× | 47,722 | below chance; a device *model string*, not an identity |

The `pseudo_account`↔`card` "owns_card" shortcut Phase 1 also considered was **not** built: two-hop `transaction→card→transaction→pseudo_account` message passing already reaches it, and the shortcut adds leakage surface (an entity-to-entity edge, harder to time-gate correctly) for no new reachability.

---

## D. Node/edge counts

Recomputed fresh from the current data (`scripts/estimate_graph_schema.py`), not merely cited from Phase 1 — confirmed to match within a rounding difference explained below:

| Entity | Nodes | Edges | Coverage | Phase 1 max degree | Measured max degree |
|---|---|---|---|---|---|
| pseudo_account | 199,625 | 524,834 | 88.9% (Phase 1: 88.7%) | 1,414 | 1,414 (exact match) |
| card | 13,553 | 590,540 | 100.0% | 14,932 | 14,932 (exact match) |
| netproxy | 2,394 | 139,212 | 23.6% | 5,523 | 5,523 (exact match) |
| device_fingerprint | 4,880 | 71,120 | 12.0% | 2,566 | 2,566 (exact match) |

**Totals:** 810,992 nodes (590,540 transaction + 220,452 entity) · 1,325,706 directed edges · **2,651,412 edges including reverse**.

The 0.2-point coverage difference on `pseudo_account` (88.9% here vs. 88.7% in Phase 1's homophily table) traces to a genuine, now-documented inconsistency — see Section M.

The built graph's edge counts were cross-checked against the pre-construction estimate programmatically (`check_edge_count_matches_coverage`) and matched **exactly** for every relation.

---

## E. Feature dimensions

- Transaction: **470** (Phase 3's tree-style encoding: continuous `C*`/`D*`/`V*`/`dist1-2`/numeric `id_*` median-imputed; frequency-encoded high-cardinality identifiers incl. `card1/2/3/5`, `addr1/2`, `id_30/31/33`, `DeviceInfo`; ordinal-encoded low-cardinality categoricals; 27 boolean flags; raw `hour`/`day_of_week`).
- Every entity type: **1** (constant placeholder — see Section B).

---

## F. Temporal construction methodology

**The core mechanism:** every edge stores `edge_time` = its source transaction's own day index (1–182). `apply_time_mask(graph, D)` returns a new graph containing only edges with `edge_time <= D` — arbitrary cutoff granularity, no compiled sampler needed (`pyg-lib`/`torch-sparse` remain uninstalled, per the environment constraint). Verified this produces monotonically increasing edge counts at increasing cutoffs (1,901,484 at day 121 → 2,259,934 at day 151 → 2,651,412 at day 182) and that `check_temporal_masking_mechanism` finds **zero** future edges and **zero** reachable future transactions at 5 different cutoffs (121, 151, 182, 60, 1).

**One honest precision limitation, stated plainly rather than hidden:** this phase validates the masking *mechanism* at arbitrary cutoffs, but a real Phase 5 training/evaluation run must **not** use one coarse cutoff for an entire split. Using `apply_time_mask(graph, 151)` to predict every VAL transaction at once would let an early-VAL transaction (day 122) see structural information from a later-VAL transaction (day 150) through a shared entity, within the same forward pass — a real, if narrow, violation of `information_time <= transaction_time` at the per-transaction level, even though it never crosses a split boundary. **The correct protocol, and the explicit recommendation for Phase 5 (Section N):** a rolling, per-day cutoff — evaluate each day's transactions using `apply_time_mask(graph, that_day - 1)` (or `that_day` if same-day accumulation is acceptable, matching Phase 3's convention) — not a single per-split mask. This is tractable: masking is a boolean comparison over 2.65M edges, and the feasibility benchmark (Section J) shows a single forward pass costs ~85ms once warm, so 182 rolling passes cost minutes, not hours.

**Why one shared graph object instead of three separate ones:** edges are static, observed facts ("this transaction used this card") that don't change with time — building separate TRAIN/VAL/TEST graph objects would just be pre-computing three different edge masks by hand. Keeping one object and masking on demand is simpler, and is exactly what makes the arbitrary-cutoff rolling protocol above possible without rebuilding anything.

---

## G. Leakage checks

Programmatic, not just argued — `scripts/validate_graph.py`, **8/8 passed**:

| Check | Result |
|---|---|
| No duplicate (src, dst) edges | PASS — 0 duplicates, any relation |
| Self-loops | PASS — structurally impossible (purely bipartite schema); reported explicitly rather than silently skipped |
| `edge_time` matches source transaction's own day | PASS — exact match, every forward edge |
| **Temporal masking mechanism** (5 cutoffs incl. day 121/151/182/60/1) | PASS — 0 future edges, 0 reachable future transactions at every cutoff |
| Extreme-degree hub check (threshold 20,000) | PASS — largest is `card`'s 14,932 |
| No isolated transaction nodes | PASS — 0 (every transaction has ≥1 entity edge, guaranteed by `card`'s 100% coverage) |
| Every entity node index referenced by ≥1 edge | PASS |
| Built edge counts match pre-construction estimate | PASS — exact match, every relation |

**Blocklist enforcement:** `assert_features_exclude_blocklist` run against the 470 transaction feature names before graph construction — passed; `TransactionID`/`TransactionDT`/`isFraud` confirmed absent.

**Fit-on-TRAIN-only:** the transaction feature encoder (frequency maps, median imputer, ordinal category maps) is fit exclusively on TRAIN-period rows, then applied to the full 590,540-row frame — identical discipline to Phase 2/3.

---

## H. Cold-start vs. warm-start statistics

Computed using the **exact same per-transaction, continuously-accumulating, strictly-prior methodology as Phase 3's `historical.py`** (generalized here from 2 to all 4 graph entity types), so these numbers are directly comparable to the Phase 3 report's cold-start findings — not a new, incompatible definition.

| Split | Entity | Transactions with this edge | Warm-start | Cold-start | Warm % |
|---|---|---|---|---|---|
| **VAL** | pseudo_account | 75,300 | 52,276 | 23,024 | 69.4% |
| VAL | card | 83,655 | 83,016 | 639 | 99.2% |
| VAL | netproxy | 14,186 | 14,068 | 118 | 99.2% |
| VAL | device_fingerprint | 6,084 | 5,663 | 421 | 93.1% |
| **TEST** | pseudo_account | 79,948 | 55,939 | 24,009 | 70.0% |
| TEST | card | 89,326 | 88,687 | 639 | 99.3% |
| TEST | netproxy | 17,848 | 17,740 | 108 | 99.4% |
| TEST | device_fingerprint | 8,617 | 7,894 | 723 | 91.6% |

**The pattern that matters for Phase 5:** `card` (100% coverage) and `netproxy`/`device_fingerprint` are overwhelmingly warm — almost every transaction's card, network, or device has been seen before. `pseudo_account` is the one entity type with a substantial cold-start population (~30% in both VAL and TEST), consistent with Phase 3's finding that this is also the entity type carrying most of the (largely artifactual) historical-control signal. This directly supports Phase 3's recommendation, preserved here per your instruction: **any graph-model result must be reported pooled, cold-start, and warm-start — never pooled alone.**

---

## I. Degree/hub analysis

No entity type exceeds the configured 20,000-degree hub-flag threshold. `card` is the highest-degree entity type (max 14,932, matching `card1`'s "13926" hub from Phase 1's degree table), well below the four *excluded* candidates' hub degrees (46,324 – 439,670) — direct, quantitative confirmation that excluding those four candidates avoided the over-smoothing risk Phase 1 predicted, not merely a theoretical concern.

| Entity | Median degree | Max degree | Singleton nodes (degree 1) |
|---|---|---|---|
| pseudo_account | 1 | 1,414 | majority (57.5% per Phase 1) |
| card | 4 | 14,932 | 3,444 (Phase 1) |
| netproxy | 6 | 5,523 | 470 (Phase 1) |
| device_fingerprint | 2 | 2,566 | 2,044 (Phase 1) |

---

## J. Memory and runtime measurements

**Pre-construction estimate vs. measured reality** — the estimate (using the closest available real GPU benchmark, from this session's earlier capability probe, scaled by node-count ratio) predicted ~4.23 GiB reserved for a hidden=64 SAGEConv training step; the actual feasibility benchmark on the **real graph** measured:

| Pass | Device | Nodes | Edges | Time | Peak GPU memory |
|---|---|---|---|---|---|
| TRAIN-masked, forward+backward | GPU (ROCm) | 810,992 | 1,901,484 | 1,665 ms (includes one-time kernel warmup) | **4.45 GiB reserved** / 15.98 GiB total |
| TRAIN-masked, forward+backward | CPU (8 threads) | 810,992 | 1,901,484 | 24,423 ms | — |
| VAL-masked, forward only | GPU (ROCm) | 810,992 | 2,259,934 | 85 ms | 3.24 GiB reserved |

The estimate and the measurement agree closely (4.23 GiB estimated vs. 4.45 GiB measured — the small gap is the real graph's feature/entity overhead the pure node-count scaling couldn't capture). **Full-batch training comfortably fits this GPU's 16 GiB VRAM budget**, with the CPU fallback confirmed viable (slower, not blocking) if ever needed.

**Construction time (actual, not estimated):** feature encoding 10.4s + graph assembly 1.7s = well under Phase 2/3's own pipeline steps. Total `build_graph.py` runtime: 16.8s.

**Disk footprint:** `hetero_graph.pt` is 1.1 GB (dominated by the 470-dim transaction feature matrix: 590,540 × 470 × 4 bytes ≈ 1.1 GB — the graph structure itself, per Phase 1's original estimate, is only ~32 MB).

---

## K. Graph construction artifacts

```
data/processed/graph/
├── hetero_graph.pt                          # the HeteroData object (1.1 GB)
├── graph_schema.json                        # node/edge counts, feature dims, split boundaries
├── cold_start_stats.json                    # Section H's numbers, per split
├── validation_results.json                  # Section G's 8/8 checks
├── feasibility_benchmark.json               # Section J's measured numbers
├── metadata.json                            # versions, timestamps, runtimes
├── entity_id_map_{pseudo_account,card,netproxy,device_fingerprint}.json   # key -> node index
└── transaction_feature_builder/             # fitted encoders (frequency maps, medians, ordinal maps)
data/processed/graph_size_estimate.json      # pre-construction estimate (Sections D's comparison baseline)
configs/graph.yaml                           # schema definition, as data
```

All under fixed `random_state=42`; every fitted statistic traces back to TRAIN-only rows.

---

## L. Environment/dependency status

**Unchanged from the ROCm capability-probe decision.** No new packages installed this phase. `torch_scatter`, `torch_sparse`, `pyg_lib` remain absent — full-batch training (confirmed comfortably fits, Section J) is used instead of neighbor sampling, exactly as decided earlier and required again explicitly this phase. `p5_scale.py` was not run.

---

## M. Problems encountered and solutions

1. **`train_merged.parquet` is not the feature-augmented frame.** Assumed initially it carried Phase 2's engineered columns (`hour`, `day_of_week`, `*_isna` flags) — it's actually the *pre-feature* merge output, saved before `build_base_features()` runs per split. **Fix:** load and concatenate the three `{split}_features.parquet` files instead, which also restores correct global chronological order automatically (each file is internally time-sorted; the three periods don't overlap).
2. **NaN-native tree features are unsafe for a GNN linear layer.** Initially planned to reuse `TreeFeatureBuilder`'s `"xgb"` variant (real NaN, native missing-value handling) for transaction node features — but a GNN's `nn.Linear` input projection has no equivalent mechanism, and a single NaN would propagate through the whole forward pass. **Fix:** switched to the `"rf"` variant (median-imputed, fit on TRAIN only); verified 0 NaN in the final matrix before constructing anything.
3. **A real, if minor, methodological inconsistency between Phase 3 and Phase 4.** Phase 3's `historical.py` builds its `pseudo_account`/`account` entity key with `.fillna("NA")` per component — a row missing `addr1` (11.1% of rows) still gets a key, grouped with every *other* transaction sharing the same `card1` and origin date that's *also* missing `addr1`. This is different from Phase 1's own original homophily analysis, which required all component columns non-null (`.notna().all(axis=1)`) — the convention this phase's fresh `src/graph/schema.py::entity_key` correctly follows. The two conventions produce very close but not identical `pseudo_account` populations (199,625 here vs. Phase 1's cited 199,070/217,850 depending on which Phase 1 sub-analysis, and a 0.2-point coverage difference from Phase 1's homophily table). **Resolution:** this phase's graph uses the stricter, Phase-1-consistent convention; Phase 3's approved results stand as reported (the direction and magnitude of the cold-start effect is unaffected — this is a "which transactions count as the same entity" nuance, not a leakage or timing bug, and Phase 3's own leakage self-check, which is about timing, is unaffected). Flagged here for transparency rather than silently reconciled; a future cleanup could align `historical.py` to the stricter convention if the two features are ever combined in one model.
4. Two cosmetic `PerformanceWarning`s (pandas frame fragmentation) surfaced during initial script runs and were not present in the final versions shown above — no separate fix needed here, as the same underlying pattern was already fixed in Phase 3's `tree_features.py`.

No other bugs, leakage findings, or blocking issues arose.

---

## N. Recommendation for the GraphSAGE phase

1. **Implement the per-day (or finer) rolling temporal mask for real evaluation** (Section F) — do not evaluate an entire split with one coarse per-split cutoff. The mechanism is already built and validated at arbitrary granularity; Phase 5 needs to call it in a loop, not just at the three split boundaries.
2. **Report every result pooled, cold-start, and warm-start** (Section H) — per your instruction, and because Phase 3 already showed pooled numbers alone are actively misleading on this dataset. The historical/blocklist control's numbers (pooled PR-AUC 0.6771, cold-start 0.1018) remain the reference line GraphSAGE must be compared against, especially on `pseudo_account` cold-start transactions specifically, where a graph model's real value proposition (reaching a first-seen account through a *shared* card/device/network with a known one) would have to show up if it exists at all.
3. **Use full-batch training** (Section J) — it fits with real headroom (4.45 of 15.98 GiB at hidden=64) and avoids the entire compiled-sampler problem. Reserve mini-batch sampling only if a future hidden-size/architecture choice is later shown to need it.
4. **Keep the historical features as a separate, explicit ablation arm**, not baked into transaction node features — combine them with the graph deliberately in one run and compare against graph-alone, so any performance gain can be attributed correctly (Section B).
5. **Item 3 in the "Problems" list** is worth a brief look before combining historical features with the graph in the same model — align the entity-key convention first, so `pseudo_account`'s node identity and `historical.py`'s `account` feature key refer to exactly the same population.

**Status: Phase 4 complete.** Graph constructed and validated (8/8 checks), one feasibility pass confirms it trains within this GPU's memory budget, raw data untouched throughout, no compiled graph dependencies installed, no GraphSAGE experiment run. Stopping here — awaiting your instruction before Phase 5.
