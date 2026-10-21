# Class-imbalance check (comparison-A XGBoost): results

**VAL-period (days 122-151), one seed per arm, VAL used for early stopping and the best-F1 threshold. No TEST data was read. Descriptive.**

Pre-registered verdict: **WEIGHTING HURTS** (PR-AUC spread 0.0553; tolerance 0.005).

| Arm | scale_pos_weight | Best iteration | PR-AUC | ROC-AUC | PR-AUC cold-start | P@1% | R@1% | Alerts at best-F1 threshold | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | 1.00 | 998 | 0.5948 | 0.9199 | 0.4367 | 0.9211 | 0.2726 | 2026 | 0.6994 | 0.5011 | 0.5838 |
| sqrt | 5.23 | 997 | 0.5792 | 0.9142 | 0.4293 | 0.9128 | 0.2702 | 2116 | 0.6578 | 0.4922 | 0.5631 |
| full | 27.36 | 798 | 0.5395 | 0.9050 | 0.4064 | 0.8937 | 0.2645 | 2127 | 0.6173 | 0.4643 | 0.5300 |

Scope: the XGBoost comparison-A recipe only. The weighted loss of the neural graph models was not re-evaluated and focal loss was not tried. A threshold-free metric can be indifferent to weighting while alert counts and calibration change; the alert columns show that effect at the best-F1 threshold of each arm.
