> Exploratory real-data analysis conducted under an explicitly authorized scope deviation after the original Phase 9 protocol halted at S6. Original G4 failed on seeds 42 and 2024 and passed on seed 123. H5a is NOT ESTABLISHED under the original protocol.

# E0 (degree-preserving null only) - descriptive per-seed results
Specification: post-Tier-0 exploratory addendum (not a preregistration). The time-shuffled null is omitted. Alerts are community-day records with A2 score >= tau_S (TRAIN-only). The null histories use the observed arrival table, so the reference distribution is descriptive and the tail proportion is a descriptive reference tail proportion (not an exact or calibrated p-value). Seeds share identical real data and differ only in Louvain seed and permutation streams; they are not independent replications and are not pooled.

A difference from this degree-preserving null is not evidence of fraud and not evidence of confirmed coordination; ordinary benign persistence (households, shared egress, repeated device use) is enough to produce one. Candidate coordinated community; not a confirmed fraud ring. This does not establish that IEEE-CIS contains fraud rings.

| seed | tau_S | TRAIN rate at tau_S (records/day) | records tied at tau_S | A observed | null A mean (SD) | null min / q05 / q50 / q95 / max | null histories with A >= observed | reference tail proportion |
|---|---|---|---|---|---|---|---|---|
| 42 | 0.858092 | 2.000 | 1 | 52 | 32.7 (5.9) | 21 / 23 / 33 / 40 / 56 | 1 of 100 | 0.0198 |
| 123 | 0.859904 | 1.967 | 1 | 41 | 31.8 (5.4) | 17 / 23 / 32 / 40 / 42 | 3 of 100 | 0.0396 |
| 2024 | 0.854286 | 2.000 | 1 | 58 | 32.5 (6.5) | 18 / 22 / 32 / 44 / 49 | 0 of 100 | 0.0099 |

With K = 100 null histories the smallest attainable tail proportion is 0.0099.

## Concentration of observed alerts (descriptive)
| seed | alerted tracks | top-1 track share of alerted community-days | top-3 share |
|---|---|---|---|
| 42 | 45 | 0.058 | 0.135 |
| 123 | 34 | 0.073 | 0.171 |
| 2024 | 47 | 0.052 | 0.138 |

## Per-day counts (observed vs mean over null histories)
| day | obs 42 / null mean 42 | obs 123 / null mean 123 | obs 2024 / null mean 2024 |
|---|---|---|---|
| 122 | 6 / 1.83 | 3 / 1.85 | 5 / 1.78 |
| 123 | 2 / 1.53 | 2 / 1.61 | 3 / 1.71 |
| 124 | 0 / 0.96 | 0 / 0.95 | 1 / 0.89 |
| 125 | 2 / 0.84 | 3 / 0.82 | 2 / 0.74 |
| 126 | 3 / 1.70 | 2 / 1.43 | 2 / 1.39 |
| 127 | 6 / 0.81 | 2 / 0.77 | 4 / 0.83 |
| 128 | 0 / 0.83 | 1 / 0.67 | 3 / 0.78 |
| 129 | 5 / 0.68 | 7 / 0.60 | 4 / 0.53 |
| 130 | 2 / 0.89 | 3 / 0.92 | 3 / 1.14 |
| 131 | 2 / 1.30 | 2 / 1.30 | 6 / 1.59 |
| 132 | 1 / 1.19 | 1 / 0.95 | 0 / 1.08 |
| 133 | 1 / 0.97 | 1 / 0.64 | 2 / 0.86 |
| 134 | 1 / 1.46 | 2 / 1.19 | 1 / 1.48 |
| 135 | 1 / 1.65 | 0 / 1.71 | 1 / 1.62 |
| 136 | 1 / 1.69 | 1 / 1.62 | 2 / 1.51 |
| 137 | 0 / 0.56 | 0 / 0.65 | 2 / 0.71 |
| 138 | 0 / 0.81 | 1 / 0.72 | 0 / 0.70 |
| 139 | 0 / 0.90 | 0 / 0.96 | 2 / 1.07 |
| 140 | 0 / 0.92 | 0 / 1.01 | 0 / 0.95 |
| 141 | 2 / 0.86 | 1 / 0.73 | 3 / 0.96 |
| 142 | 1 / 0.91 | 0 / 0.95 | 0 / 0.78 |
| 143 | 0 / 0.87 | 0 / 0.84 | 0 / 0.52 |
| 144 | 4 / 0.68 | 3 / 0.81 | 3 / 0.74 |
| 145 | 3 / 0.78 | 2 / 0.69 | 1 / 0.70 |
| 146 | 1 / 1.35 | 1 / 1.33 | 0 / 1.36 |
| 147 | 0 / 1.56 | 0 / 1.37 | 1 / 1.66 |
| 148 | 3 / 1.72 | 2 / 1.68 | 3 / 1.80 |
| 149 | 3 / 0.93 | 0 / 1.20 | 1 / 1.09 |
| 150 | 1 / 0.78 | 1 / 0.80 | 1 / 0.71 |
| 151 | 1 / 0.78 | 0 / 0.99 | 2 / 0.84 |

No threshold or formal verdict is applied. Limitations: the arrival table is held at its observed value in every null history; the 30 TRAIN windows for tau are serially dependent; the null randomizes card-infrastructure pairing only; alert counts are counts of community-days, so persistent communities count repeatedly.
