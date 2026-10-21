# Candidate-community investigation cards (seed 42, W = 7)

**candidate coordinated community; not a confirmed fraud ring.** This does not establish that IEEE-CIS contains fraud rings.

Descriptive, aggregate-only cards built from the existing Phase 9 Tier 0 community-day CSV. No discovery was re-run; no label, risk score or model output was used to select or rank anything. Tier 0 identity is unstable for most tracks and censored at both ends of the 30-day window (days 122-151).

Alert flag: `score_st >= tau` with tau = 0.8580917874396136 (TRAIN-calibrated, exploratory E0 lock). E0 comparisons are a descriptive reference, not exact inference and not a significance test.

Selection groups (Phase 9 design section 16): `a_top_peak_score` [26, 172, 4, 79, 92, 56, 71, 55, 43, 72]; `b_largest` [0, 133, 151, 112, 96]; `c_alerted_sample` [7, 10, 73, 206, 207]; `d_hub_dominated_high_s` [60, 91, 92, 150, 69]

## Track 0  (b_largest)

Observed days 122-132 (11 days); left-censored: True; right-censored: False; alerted days: 0.
Peak: score_st 0.8523, u_s 0.7778, u_t 0.9609, cards 504, infrastructure entities 650, hub share 0.446.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 19236, has_edge_cold / infra 8331, has_edge_warm / noinfra 63503, has_edge_warm / infra 5278, no_pseudo_account_edge / noinfra 100, no_pseudo_account_edge / infra 777.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 122 | 464 | 637 | 0.0064 | 0.396 | 0.038 | 0.770 | 0.748 | 0.759 | 9051 | no |
| 123 | 490 | 650 | 0.0062 | 0.413 | 0.030 | 0.778 | 0.748 | 0.763 | 9215 | no |
| 124 | 474 | 608 | 0.0066 | 0.407 | 0.040 | 0.770 | 0.800 | 0.785 | 9488 | no |
| 125 | 504 | 640 | 0.0062 | 0.416 | 0.034 | 0.774 | 0.772 | 0.773 | 10653 | no |
| 126 | 488 | 622 | 0.0064 | 0.406 | 0.036 | 0.770 | 0.746 | 0.758 | 9221 | no |
| 127 | 418 | 561 | 0.0072 | 0.414 | 0.063 | 0.744 | 0.772 | 0.758 | 8584 | no |
| 128 | 480 | 634 | 0.0062 | 0.371 | 0.039 | 0.756 | 0.772 | 0.764 | 9085 | no |
| 129 | 430 | 574 | 0.0069 | 0.391 | 0.044 | 0.744 | 0.739 | 0.742 | 9298 | no |
| 130 | 346 | 487 | 0.0083 | 0.339 | 0.084 | 0.726 | 0.961 | 0.843 | 6940 | no |
| 131 | 436 | 617 | 0.0066 | 0.425 | 0.042 | 0.748 | 0.957 | 0.852 | 8716 | no |
| 132 | 397 | 558 | 0.0073 | 0.446 | 0.053 | 0.730 | 0.941 | 0.835 | 6974 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 4  (a_top_peak_score)

Observed days 122-123 (2 days); left-censored: True; right-censored: False; alerted days: 1.
Peak: score_st 0.9850, u_s 0.9700, u_t 1.0000, cards 13, infrastructure entities 15, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 27, has_edge_cold / infra 29, has_edge_warm / noinfra 67, has_edge_warm / infra 9, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 122 | 13 | 15 | 0.1436 | 0.000 | 0.103 | 0.970 | 1.000 | 0.985 | 79 | yes |
| 123 | 11 | 11 | 0.1818 | 0.000 | 0.147 | 0.840 | 0.680 | 0.760 | 53 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 7  (c_alerted_sample)

Observed days 122-122 (1 days); left-censored: True; right-censored: False; alerted days: 1.
Peak: score_st 0.8750, u_s 1.0000, u_t 0.7500, cards 16, infrastructure entities 16, hub share 0.163.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 47, has_edge_cold / infra 14, has_edge_warm / noinfra 115, has_edge_warm / infra 17, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 1.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 122 | 16 | 16 | 0.1211 | 0.163 | 0.181 | 1.000 | 0.750 | 0.875 | 194 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 10  (c_alerted_sample)

Observed days 122-123 (2 days); left-censored: True; right-censored: False; alerted days: 1.
Peak: score_st 0.9455, u_s 0.9818, u_t 0.9091, cards 5, infrastructure entities 10, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 14, has_edge_cold / infra 14, has_edge_warm / noinfra 36, has_edge_warm / infra 8, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 122 | 5 | 10 | 0.2800 | 0.000 | 0.077 | 0.982 | 0.909 | 0.945 | 32 | yes |
| 123 | 5 | 10 | 0.2800 | 0.000 | 0.077 | 0.982 | 0.327 | 0.655 | 40 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 26  (a_top_peak_score)

Observed days 125-125 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.9857, u_s 1.0000, u_t 0.9714, cards 3, infrastructure entities 4, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 0, has_edge_cold / infra 4, has_edge_warm / noinfra 0, has_edge_warm / infra 4, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 125 | 3 | 4 | 0.6667 | 0.000 | 0.222 | 1.000 | 0.971 | 0.986 | 8 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 43  (a_top_peak_score)

Observed days 126-126 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.9613, u_s 0.9741, u_t 0.9486, cards 3, infrastructure entities 5, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 6, has_edge_cold / infra 1, has_edge_warm / noinfra 23, has_edge_warm / infra 5, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 126 | 3 | 5 | 0.4667 | 0.000 | 0.182 | 0.974 | 0.949 | 0.961 | 35 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 55  (a_top_peak_score)

Observed days 127-127 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.9670, u_s 0.9741, u_t 0.9600, cards 4, infrastructure entities 3, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 25, has_edge_cold / infra 3, has_edge_warm / noinfra 25, has_edge_warm / infra 2, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 127 | 4 | 3 | 0.5833 | 0.000 | 0.100 | 0.974 | 0.960 | 0.967 | 55 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 56  (a_top_peak_score)

Observed days 127-127 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.9728, u_s 0.9741, u_t 0.9714, cards 4, infrastructure entities 2, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 0, has_edge_cold / infra 5, has_edge_warm / noinfra 0, has_edge_warm / infra 1, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 127 | 4 | 2 | 0.8750 | 0.000 | 0.100 | 0.974 | 0.971 | 0.973 | 6 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 60  (d_hub_dominated_high_s)

Observed days 128-130 (3 days); left-censored: False; right-censored: False; alerted days: 0.
Peak: score_st 0.8182, u_s 1.0000, u_t 0.6364, cards 7, infrastructure entities 6, hub share 0.560.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 76, has_edge_cold / infra 16, has_edge_warm / noinfra 148, has_edge_warm / infra 24, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 3.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 128 | 7 | 5 | 0.3143 | 0.560 | 0.138 | 1.000 | 0.327 | 0.664 | 74 | no |
| 129 | 5 | 5 | 0.3600 | 0.000 | 0.160 | 1.000 | 0.364 | 0.682 | 66 | no |
| 130 | 6 | 6 | 0.3056 | 0.519 | 0.156 | 1.000 | 0.636 | 0.818 | 127 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 69  (d_hub_dominated_high_s)

Observed days 129-131 (3 days); left-censored: False; right-censored: False; alerted days: 0.
Peak: score_st 0.6428, u_s 0.9741, u_t 0.4800, cards 9, infrastructure entities 6, hub share 0.412.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 69, has_edge_cold / infra 16, has_edge_warm / noinfra 106, has_edge_warm / infra 8, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 129 | 4 | 3 | 0.5833 | 0.000 | 0.100 | 0.974 | 0.311 | 0.643 | 58 | no |
| 130 | 8 | 5 | 0.3250 | 0.000 | 0.143 | 0.941 | 0.327 | 0.634 | 71 | no |
| 131 | 9 | 6 | 0.2778 | 0.412 | 0.190 | 0.330 | 0.480 | 0.405 | 70 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 71  (a_top_peak_score)

Observed days 129-133 (5 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.9728, u_s 0.9741, u_t 0.9714, cards 4, infrastructure entities 3, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 0, has_edge_cold / infra 20, has_edge_warm / noinfra 0, has_edge_warm / infra 5, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 129 | 4 | 2 | 0.8750 | 0.000 | 0.000 | 0.974 | 0.971 | 0.973 | 5 | yes |
| 130 | 4 | 2 | 0.8750 | 0.000 | 0.000 | 0.974 | 0.311 | 0.643 | 5 | no |
| 131 | 4 | 3 | 0.5833 | 0.000 | 0.182 | 0.974 | 0.617 | 0.796 | 5 | no |
| 132 | 4 | 3 | 0.5833 | 0.000 | 0.143 | 0.974 | 0.617 | 0.796 | 5 | no |
| 133 | 4 | 3 | 0.5833 | 0.000 | 0.143 | 0.974 | 0.646 | 0.810 | 5 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 72  (a_top_peak_score)

Observed days 129-129 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.9613, u_s 0.9741, u_t 0.9486, cards 3, infrastructure entities 4, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 10, has_edge_cold / infra 4, has_edge_warm / noinfra 5, has_edge_warm / infra 2, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 129 | 3 | 4 | 0.5000 | 0.000 | 0.182 | 0.974 | 0.949 | 0.961 | 21 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 73  (c_alerted_sample)

Observed days 129-129 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.9055, u_s 0.9482, u_t 0.8629, cards 3, infrastructure entities 6, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 5, has_edge_cold / infra 3, has_edge_warm / noinfra 4, has_edge_warm / infra 1, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 129 | 3 | 6 | 0.4444 | 0.000 | 0.158 | 0.948 | 0.863 | 0.906 | 13 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 79  (a_top_peak_score)

Observed days 130-131 (2 days); left-censored: False; right-censored: False; alerted days: 2.
Peak: score_st 0.9743, u_s 1.0000, u_t 0.9486, cards 3, infrastructure entities 3, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 0, has_edge_cold / infra 8, has_edge_warm / noinfra 0, has_edge_warm / infra 8, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 130 | 3 | 3 | 0.6667 | 0.000 | 0.097 | 1.000 | 0.949 | 0.974 | 8 | yes |
| 131 | 3 | 3 | 0.6667 | 0.000 | 0.097 | 1.000 | 0.949 | 0.974 | 8 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 91  (d_hub_dominated_high_s)

Observed days 132-132 (1 days); left-censored: False; right-censored: False; alerted days: 0.
Peak: score_st 0.8182, u_s 1.0000, u_t 0.6364, cards 7, infrastructure entities 12, hub share 0.395.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 20, has_edge_cold / infra 7, has_edge_warm / noinfra 41, has_edge_warm / infra 13, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 2.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 132 | 7 | 12 | 0.2143 | 0.395 | 0.146 | 1.000 | 0.636 | 0.818 | 83 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 92  (a_top_peak_score, d_hub_dominated_high_s)

Observed days 132-132 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.9730, u_s 0.9974, u_t 0.9486, cards 3, infrastructure entities 4, hub share 0.500.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 7, has_edge_cold / infra 3, has_edge_warm / noinfra 18, has_edge_warm / infra 3, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 132 | 3 | 4 | 0.5000 | 0.500 | 0.226 | 0.997 | 0.949 | 0.973 | 31 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 96  (b_largest)

Observed days 133-133 (1 days); left-censored: False; right-censored: False; alerted days: 0.
Peak: score_st 0.7258, u_s 0.7037, u_t 0.7478, cards 303, infrastructure entities 424, hub share 0.393.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 1281, has_edge_cold / infra 478, has_edge_warm / noinfra 4053, has_edge_warm / infra 310, no_pseudo_account_edge / noinfra 6, no_pseudo_account_edge / infra 41.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 133 | 303 | 424 | 0.0082 | 0.393 | 0.124 | 0.704 | 0.748 | 0.726 | 6169 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 112  (b_largest)

Observed days 135-136 (2 days); left-censored: False; right-censored: False; alerted days: 0.
Peak: score_st 0.7937, u_s 0.7222, u_t 0.8652, cards 344, infrastructure entities 519, hub share 0.447.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 2315, has_edge_cold / infra 1146, has_edge_warm / noinfra 7960, has_edge_warm / infra 869, no_pseudo_account_edge / noinfra 9, no_pseudo_account_edge / infra 73.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 135 | 344 | 519 | 0.0075 | 0.405 | 0.072 | 0.722 | 0.772 | 0.747 | 6657 | no |
| 136 | 326 | 496 | 0.0084 | 0.447 | 0.074 | 0.722 | 0.865 | 0.794 | 5715 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 133  (b_largest)

Observed days 138-140 (3 days); left-censored: False; right-censored: False; alerted days: 0.
Peak: score_st 0.7385, u_s 0.7444, u_t 0.7500, cards 441, infrastructure entities 629, hub share 0.432.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 3523, has_edge_cold / infra 1716, has_edge_warm / noinfra 10166, has_edge_warm / infra 1232, no_pseudo_account_edge / noinfra 28, no_pseudo_account_edge / infra 123.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 138 | 300 | 406 | 0.0088 | 0.432 | 0.116 | 0.704 | 0.750 | 0.727 | 4488 | no |
| 139 | 325 | 431 | 0.0084 | 0.430 | 0.107 | 0.704 | 0.670 | 0.687 | 4937 | no |
| 140 | 441 | 629 | 0.0065 | 0.419 | 0.040 | 0.744 | 0.733 | 0.739 | 7363 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 150  (d_hub_dominated_high_s)

Observed days 141-141 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.8729, u_s 1.0000, u_t 0.7457, cards 3, infrastructure entities 6, hub share 0.364.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 1, has_edge_cold / infra 5, has_edge_warm / noinfra 1, has_edge_warm / infra 7, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 141 | 3 | 6 | 0.5000 | 0.364 | 0.154 | 1.000 | 0.746 | 0.873 | 14 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 151  (b_largest)

Observed days 142-143 (2 days); left-censored: False; right-censored: False; alerted days: 0.
Peak: score_st 0.7937, u_s 0.7259, u_t 0.8652, cards 369, infrastructure entities 529, hub share 0.416.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 2969, has_edge_cold / infra 1225, has_edge_warm / noinfra 8656, has_edge_warm / infra 842, no_pseudo_account_edge / noinfra 32, no_pseudo_account_edge / infra 106.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 142 | 361 | 529 | 0.0077 | 0.403 | 0.082 | 0.726 | 0.852 | 0.789 | 6928 | no |
| 143 | 369 | 522 | 0.0077 | 0.416 | 0.085 | 0.722 | 0.865 | 0.794 | 6902 | no |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 172  (a_top_peak_score)

Observed days 144-145 (2 days); left-censored: False; right-censored: False; alerted days: 2.
Peak: score_st 0.9857, u_s 1.0000, u_t 0.9714, cards 5, infrastructure entities 7, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 0, has_edge_cold / infra 12, has_edge_warm / noinfra 2, has_edge_warm / infra 16, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 144 | 4 | 6 | 0.3750 | 0.000 | 0.171 | 1.000 | 0.971 | 0.986 | 12 | yes |
| 145 | 5 | 7 | 0.3143 | 0.000 | 0.164 | 1.000 | 0.818 | 0.909 | 18 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 206  (c_alerted_sample)

Observed days 148-148 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.8608, u_s 0.7617, u_t 0.9600, cards 3, infrastructure entities 2, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 0, has_edge_cold / infra 0, has_edge_warm / noinfra 0, has_edge_warm / infra 0, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 8.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 148 | 3 | 2 | 0.6667 | 0.000 | 0.143 | 0.762 | 0.960 | 0.861 | 8 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## Track 207  (c_alerted_sample)

Observed days 148-148 (1 days); left-censored: False; right-censored: False; alerted days: 1.
Peak: score_st 0.9455, u_s 0.9818, u_t 0.9091, cards 8, infrastructure entities 8, hub share 0.000.
Member-transaction composition (transaction-day incidences): has_edge_cold / noinfra 9, has_edge_cold / infra 5, has_edge_warm / noinfra 5, has_edge_warm / infra 6, no_pseudo_account_edge / noinfra 0, no_pseudo_account_edge / infra 0.

| day | cards | infra | density | hub share | conductance | u_s | u_t | score_st | member txn | alerted |
|---|---|---|---|---|---|---|---|---|---|---|
| 148 | 8 | 8 | 0.2344 | 0.000 | 0.156 | 0.982 | 0.909 | 0.945 | 25 | yes |

Not available from the aggregates (not fabricated): member card identities and member infrastructure identities (Tier 0 CSVs are aggregate-only; discovery was not re-run); new-member counts and daily active members; pseudo-account counts; descriptive attributes (ProductCD, card4, card6, email domain); risk percentile distribution and learned-state signature (no risk stream was created; not authorized); any label-based column (retrospective label rate is not authorized); per-community null-matched comparison (E0 gives day-level reference counts only).
*candidate coordinated community; not a confirmed fraud ring.*

## E0 descriptive reference (day-level counts only; seed 42, W = 7)

Observed alerted community-days over days 122-151: 52. Degree-preserving null (K = 100 histories): mean 32.74, SD 5.921, min 21, median 33.0, 95th percentile 40.05, max 56; descriptive reference tail proportion 0.0198 (minimum attainable 0.0099). This is a descriptive reference distribution: it is not exact, not a significance test, not a per-community null match, and not evidence of fraud or coordination.

