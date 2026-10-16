> Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 protocol halted at S6. Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under the original protocol.

# Tier 0 — descriptive discovery on clean real data (days 122-151), exploratory

Generated from `tier0_summary.json` by `scripts/make_phase9_exploratory_report.py`. Everything below is **descriptive**; no statistical test was run in Tier 0 and nothing here is a statistically supported finding. Label-free: no fraud labels, no model-risk scores, no R_p / R_h, no risk-based baselines. No alert threshold exists, so these are counts of eligible candidate communities (>= 3 card tokens, >= 1 infrastructure entity), not alerts.
Every community is a *candidate coordinated community; not a confirmed fraud ring*. This does not establish that IEEE-CIS contains fraud rings.

## Frozen settings used
Locked gamma per W = {'3': 0.5, '7': 0.5, '14': 0.5} (primary W = 7; W = 3, 14 pre-registered sensitivity); Louvain seed = S in {42, 123, 2024}; null tables hash-verified against the locked configuration; shared tracker started at day 122; m_min = 3; edge-weight cap 5.

## Population at the top of the table (VAL days 122-151)
83,655 transactions; 82.9% have no netproxy/device edge (structurally unobservable to infrastructure-based discovery). Fraud base rate of this population is **not computed** (requires labels; outside this authorization).

| segment (never merged) | no infrastructure edge | infrastructure edge |
|---|---|---|
| has_edge_cold | 19,187 | 3,837 |
| has_edge_warm | 49,592 | 2,684 |
| no_pseudo_account_edge | 576 | 7,779 |

## Per configuration and seed
| W | seed | community-day records | candidates/day (median; q05-q95) | cards per candidate (median / q95 / max) | records with >= 17 cards | tracks | track age (median / q95 / max) |
|---|---|---|---|---|---|---|---|
| 7 | 42 | 339 | 11.0; 7.5-18.0 | 13 / 305 / 504 | 47% | 237 | 1 / 3.0 / 30 |
| 7 | 123 | 354 | 11.0; 7.5-16.5 | 10 / 341 / 514 | 43% | 240 | 1 / 3.0 / 30 |
| 7 | 2024 | 366 | 11.0; 8.4-18.0 | 10 / 316 / 531 | 42% | 252 | 1 / 3.0 / 30 |
| 3 | 42 | 338 | 12.0; 8.4-13.0 | 23 / 133 / 166 | 62% | 288 | 1 / 1.6 / 30 |
| 3 | 123 | 334 | 11.0; 9.0-13.5 | 24 / 132 / 166 | 62% | 289 | 1 / 1.0 / 30 |
| 3 | 2024 | 334 | 11.0; 9.0-13.0 | 26 / 132 / 166 | 63% | 284 | 1 / 1.0 / 30 |
| 14 | 42 | 538 | 17.5; 8.9-24.5 | 4 / 740 / 872 | 21% | 224 | 1 / 6.8 / 30 |
| 14 | 123 | 515 | 17.0; 9.9-23.6 | 4 / 748 / 894 | 22% | 243 | 1 / 5.0 / 30 |
| 14 | 2024 | 508 | 17.0; 10.0-22.5 | 4 / 764 / 878 | 22% | 229 | 1 / 6.0 / 30 |

### Null-calibrated component percentiles of the observed candidates (descriptive; NOT a null test)
| W | seed | u_S mean (q05 / q50 / q95) | u_T mean (q05 / q50 / q95) | S+T mean score (q50 / q95) |
|---|---|---|---|---|
| 7 | 42 | 0.714 (0.30 / 0.72 / 1.00) | 0.594 (0.23 / 0.61 / 0.96) | 0.682 / 0.945 |
| 7 | 123 | 0.713 (0.26 / 0.73 / 1.00) | 0.583 (0.25 / 0.59 / 0.96) | 0.679 / 0.929 |
| 7 | 2024 | 0.717 (0.31 / 0.73 / 1.00) | 0.596 (0.25 / 0.62 / 0.96) | 0.664 / 0.920 |
| 3 | 42 | 0.616 (0.14 / 0.64 / 1.00) | 0.589 (0.20 / 0.58 / 0.97) | 0.625 / 0.946 |
| 3 | 123 | 0.636 (0.17 / 0.65 / 1.00) | 0.582 (0.20 / 0.59 / 0.96) | 0.640 / 0.939 |
| 3 | 2024 | 0.627 (0.17 / 0.64 / 1.00) | 0.583 (0.20 / 0.58 / 0.97) | 0.623 / 0.946 |
| 14 | 42 | 0.735 (0.39 / 0.83 / 1.00) | 0.617 (0.25 / 0.62 / 0.93) | 0.708 / 0.935 |
| 14 | 123 | 0.727 (0.40 / 0.83 / 1.00) | 0.614 (0.24 / 0.62 / 0.94) | 0.705 / 0.945 |
| 14 | 2024 | 0.724 (0.37 / 0.83 / 1.00) | 0.612 (0.21 / 0.62 / 0.94) | 0.696 / 0.938 |

### Hub, density and conductance diagnostics (descriptive)
| W | seed | hub share mean (q50 / q95) | density mean (q50) | conductance mean (q50) | Q recomputation max abs diff |
|---|---|---|---|---|---|
| 7 | 42 | 0.196 (0.07 / 0.78) | 0.274 (0.14) | 0.140 (0.16) | 0.0e+00 |
| 7 | 123 | 0.181 (0.00 / 0.78) | 0.293 (0.20) | 0.142 (0.16) | 0.0e+00 |
| 7 | 2024 | 0.182 (0.00 / 0.78) | 0.289 (0.20) | 0.146 (0.16) | 0.0e+00 |
| 3 | 42 | 0.245 (0.23 / 0.74) | 0.217 (0.08) | 0.110 (0.13) | 0.0e+00 |
| 3 | 123 | 0.245 (0.23 / 0.73) | 0.199 (0.08) | 0.112 (0.13) | 0.0e+00 |
| 3 | 2024 | 0.247 (0.22 / 0.74) | 0.202 (0.07) | 0.111 (0.13) | 0.0e+00 |
| 14 | 42 | 0.109 (0.00 / 0.78) | 0.441 (0.47) | 0.170 (0.20) | 0.0e+00 |
| 14 | 123 | 0.116 (0.00 / 0.78) | 0.440 (0.47) | 0.169 (0.20) | 0.0e+00 |
| 14 | 2024 | 0.116 (0.00 / 0.78) | 0.437 (0.47) | 0.167 (0.20) | 0.0e+00 |

### Lifecycle (shared tracker; tracks starting on day 122 are left-censored, tracks alive on day 151 right-censored)
| W | seed | tracks | left-censored | right-censored | observable disappearances (last day <= 148) | split flags | merge flags | consecutive-day Jaccard, same track (mean / q50) | records on tracks aged >= 2 days |
|---|---|---|---|---|---|---|---|---|---|
| 7 | 42 | 237 | 14 | 10 | 204 | 0 | 0 | 0.821 / 0.834 | 30% |
| 7 | 123 | 240 | 11 | 10 | 211 | 0 | 0 | 0.818 / 0.831 | 32% |
| 7 | 2024 | 252 | 12 | 11 | 220 | 0 | 0 | 0.816 / 0.827 | 31% |
| 3 | 42 | 288 | 12 | 11 | 255 | 0 | 0 | 0.738 / 0.675 | 15% |
| 3 | 123 | 289 | 12 | 10 | 260 | 0 | 0 | 0.719 / 0.672 | 13% |
| 3 | 2024 | 284 | 13 | 11 | 250 | 0 | 0 | 0.733 / 0.675 | 15% |
| 14 | 42 | 224 | 16 | 20 | 192 | 0 | 0 | 0.888 / 1.000 | 58% |
| 14 | 123 | 243 | 22 | 20 | 211 | 0 | 0 | 0.890 / 1.000 | 53% |
| 14 | 2024 | 229 | 18 | 15 | 199 | 0 | 0 | 0.893 / 1.000 | 55% |

Zero split/merge flags are structural, not a finding: with the frozen Jaccard >= 0.5 continuation rule and disjoint communities on each day, two children can both reach Jaccard >= 0.5 with one parent only in the degenerate exact-halves case.

### Member-transaction composition: segment x infra_observed (transaction-day incidences: a transaction is counted once per day its card is in a community's window)
| W | seed | member txn-days | cold, no infra | cold, infra | warm, no infra | warm, infra | no-edge, no infra | no-edge, infra |
|---|---|---|---|---|---|---|---|---|
| 7 | 42 | 354,033 | 17.4% | 7.5% | 52.6% | 5.1% | 1.0% | 16.4% |
| 7 | 123 | 353,685 | 17.4% | 7.5% | 52.6% | 5.1% | 1.0% | 16.5% |
| 7 | 2024 | 353,693 | 17.4% | 7.5% | 52.6% | 5.1% | 1.0% | 16.5% |
| 3 | 42 | 122,684 | 15.7% | 8.8% | 49.7% | 6.2% | 1.1% | 18.6% |
| 3 | 123 | 122,890 | 15.7% | 8.8% | 49.7% | 6.2% | 1.1% | 18.6% |
| 3 | 2024 | 122,732 | 15.7% | 8.8% | 49.7% | 6.2% | 1.1% | 18.6% |
| 14 | 42 | 809,851 | 18.6% | 6.6% | 53.4% | 4.5% | 1.0% | 15.9% |
| 14 | 123 | 810,351 | 18.6% | 6.6% | 53.5% | 4.5% | 1.0% | 15.9% |
| 14 | 2024 | 810,048 | 18.6% | 6.6% | 53.5% | 4.5% | 1.0% | 15.9% |

## Reading these numbers (descriptive observations only)
* The frozen pipeline produces roughly 11 eligible candidate communities per day at W = 7 (about 17 at W = 14); about 40-50% of the W = 7 records are communities of 17 or more card tokens (up to about 500), consistent with the design's recorded base-rate/giant-component facts (F6, F7) that infrastructure sharing is the norm. Their meaning is unknown.
* Most tracks last one day (median age 1 day at every W); a few tracks persist for the whole 30-day period (maximum age 30). Day-to-day identity is therefore unstable for most candidates.
* Observed candidates have mean null-calibrated u_S of about 0.62-0.74 and u_T of about 0.58-0.62 (above 0.5). That is a descriptive hint only: it is **not** E0, uses no permutation test, depends on the size-bucketed calibration tables (some buckets thin; nearest-bucket fallback), and says nothing about whether the structure is fraudulent.
* Member-transaction composition is not the population composition: no-edge (missing pseudo-account key) transactions are over-represented among communities' transactions relative to the VAL population, because they almost always carry an infrastructure edge. The three segments are reported separately and were never merged.
* No claim of validity, no recovery metric, no precision/recall, and no statement about H5a/H5b/H5c is made or implied.

## Limitations
Single real dataset; seeds vary only the Louvain seed (real data identical across seeds); Louvain community definitions are algorithm-dependent; tracker identity and lifecycle statistics are censored at both ends of a 30-day window; infrastructure fields are project-defined proxies with unverified semantics; 83% of VAL transactions carry no infrastructure edge.
