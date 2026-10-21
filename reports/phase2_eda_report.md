# Phase 2 EDA Report

Generated from `data/processed/train_merged.parquet` and `data/features/*_features.parquet` by `scripts/run_eda.py`. Source of the chronological split and column semantics: `reports/phase1_reconnaissance.md`.

## Row counts

- Merged labeled training frame: **590,540** rows, 435 columns
- With identity (`has_identity`): 144,233 (24.42%)

| split | rows | share |
|---|---|---|
| train | 417,559 | 70.71% |
| val | 83,655 | 14.17% |
| test | 89,326 | 15.13% |

## Target (isFraud)

- Overall: 569,877 legitimate / 20,663 fraud (3.4990% fraud, imbalance ratio 27.58:1)

| split | rows | fraud | fraud rate |
|---|---|---|---|
| train | 417,559 | 14,721 | 3.525% |
| val | 83,655 | 2,828 | 3.381% |
| test | 89,326 | 3,114 | 3.486% |

## Missingness

Top 20 columns by missing fraction:

| column | missing % |
|---|---|
| id_24 | 99.20% |
| id_25 | 99.13% |
| id_07 | 99.13% |
| id_08 | 99.13% |
| id_21 | 99.13% |
| id_26 | 99.13% |
| id_27 | 99.12% |
| id_23 | 99.12% |
| id_22 | 99.12% |
| dist2 | 93.63% |
| D7 | 93.41% |
| id_18 | 92.36% |
| D13 | 89.51% |
| D14 | 89.47% |
| D12 | 89.04% |
| id_04 | 88.77% |
| id_03 | 88.77% |
| D6 | 87.61% |
| id_33 | 87.59% |
| id_09 | 87.31% |

Columns by missingness bucket:

| bucket | n columns |
|---|---|
| 0% | 21 |
| 0-5% | 92 |
| 5-25% | 70 |
| 25-50% | 38 |
| 50-75% | 6 |
| 75-90% | 196 |
| >=90% | 12 |

## Temporal distribution

- Day range: 1-182 (182 days)

| split | day range | TransactionDT range |
|---|---|---|
| train | 1-121 | 86,400-10,540,797 |
| val | 122-151 | 10,540,826-13,132,788 |
| test | 152-182 | 13,132,810-15,811,131 |

14-day window fraud rate (full labeled train period):

| w | n | fraud_pct |
|---|---|---|
| 0 | 56059 | 2.713 |
| 1 | 70607 | 2.256 |
| 2 | 45458 | 3.722 |
| 3 | 40325 | 4.179 |
| 4 | 43966 | 4.374 |
| 5 | 41128 | 3.783 |
| 6 | 49820 | 3.605 |
| 7 | 42160 | 3.757 |
| 8 | 44653 | 4.26 |
| 9 | 37830 | 3.537 |
| 10 | 39992 | 3.248 |
| 11 | 40849 | 3.168 |
| 12 | 37693 | 3.926 |

## Categorical cardinalities

| column | n unique |
|---|---|
| DeviceInfo | 1,786 |
| id_33 | 260 |
| id_31 | 130 |
| id_30 | 75 |
| R_emaildomain | 60 |
| P_emaildomain | 59 |
| ProductCD | 5 |
| card4 | 4 |
| card6 | 4 |
| id_34 | 4 |
| id_23 | 3 |
| M4 | 3 |
| id_15 | 3 |
| M2 | 2 |
| M3 | 2 |
| M1 | 2 |
| id_12 | 2 |
| M9 | 2 |
| M8 | 2 |
| M7 | 2 |
| M5 | 2 |
| M6 | 2 |
| id_28 | 2 |
| id_27 | 2 |
| id_29 | 2 |
| id_16 | 2 |
| id_36 | 2 |
| id_35 | 2 |
| id_37 | 2 |
| id_38 | 2 |
| DeviceType | 2 |
| has_identity | 2 |

## Numerical distributions (representative columns)

| index | count | mean | std | min | 1% | 25% | 50% | 75% | 95% | 99% | max |
|---|---|---|---|---|---|---|---|---|---|---|---|
| TransactionAmt | 590540.0 | 135.027 | 239.163 | 0.251 | 9.244 | 43.321 | 68.769 | 125.0 | 445.0 | 1104.0 | 31937.391 |
| C1 | 590540.0 | 14.092 | 133.569 | 0.0 | 1.0 | 1.0 | 1.0 | 3.0 | 22.0 | 164.0 | 4685.0 |
| C2 | 590540.0 | 15.27 | 154.669 | 0.0 | 1.0 | 1.0 | 1.0 | 3.0 | 27.0 | 154.0 | 5691.0 |
| C13 | 590540.0 | 32.54 | 129.365 | 0.0 | 0.0 | 1.0 | 3.0 | 12.0 | 102.0 | 578.0 | 2918.0 |
| D1 | 589271.0 | 94.348 | 157.66 | 0.0 | 0.0 | 0.0 | 3.0 | 122.0 | 489.0 | 602.0 | 640.0 |
| D15 | 501427.0 | 163.745 | 202.727 | -83.0 | 0.0 | 0.0 | 52.0 | 314.0 | 566.0 | 635.0 | 879.0 |
| V1 | 311253.0 | 1.0 | 0.007 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 | 1.0 |
| V258 | 130430.0 | 1.344 | 2.016 | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 | 2.0 | 7.0 | 66.0 |

## Data quality interventions

Negative D-column values clipped to 0 (Phase 1 risk R6 — a handful of rows reference a 'days since' event that appears to be in the future):

| column | train | val | test |
|---|---|---|---|
| D11 | 5 | 2 | 0 |
| D12 | 1 | 1 | 0 |
| D14 | 3 | 0 | 0 |
| D15 | 9 | 6 | 0 |
| D4 | 8 | 7 | 0 |
| D6 | 2 | 1 | 0 |
