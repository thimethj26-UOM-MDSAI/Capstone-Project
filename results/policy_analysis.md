## 1. Sensitivity of the policy to the precision target (validation set, n=750)

Auto-remove threshold = lowest grid threshold whose precision >= target; flag threshold = highest threshold with recall >= 0.90. Validation data only.

| Precision target | Flag >= | Remove >= | Auto-removed | Sent to review | Published | Removal precision | False removals (of NOT) | Offensive posts caught (removed + review) | Offensive posts missed |
|---|---|---|---|---|---|---|---|---|---|
| 0.85 | 0.45 | 0.80 | 36.9% | 12.9% | 50.1% | 85.9% | 39 (9.0%) | 91.2% | 8.8% |
| 0.90 (chosen) | 0.45 | 0.95 | 28.3% | 21.6% | 50.1% | 92.9% | 15 (3.5%) | 91.2% | 8.8% |
| 0.95 | 0.45 | none reaches target | 0.0% | 49.9% | 50.1% | n/a | 0 (0.0%) | 91.2% | 8.8% |

Shares of auto-removed / review / published are fractions of ALL validation posts, so the review column is the human workload.

## 2. Test-set evaluation of the saved thresholds (evaluated once; n=2500)

Thresholds from moderation/thresholds.json: flag >= 0.45, remove >= 0.95. They were chosen on validation data only.

| True label | Auto-removed | Sent to review | Published | Total |
|---|---|---|---|---|
| NOT offensive | 80 (5.4%) | 249 (16.8%) | 1156 (77.8%) | 1485 |
| Offensive | 641 (63.2%) | 244 (24.0%) | 130 (12.8%) | 1015 |

| Measure | Value |
|---|---|
| Removal precision (auto-removed posts that are truly offensive) | 88.9% |
| Wrongly auto-removed posts (NOT offensive) | 80 (5.4% of NOT posts) |
| Offensive posts auto-removed | 641 (63.2%) |
| Offensive posts sent to human review | 244 |
| Offensive posts caught in total (removed + review) | 87.2% |
| Offensive posts published (missed) | 130 (12.8%) |
| Review queue: share that is truly offensive | 49.5% |
| NOT-offensive posts sent to review | 249 (16.8% of NOT posts) |
| Human review workload (share of all posts) | 19.7% |
| Expected calibration error (ECE) | 0.089 |

Sanity check: macro-F1 at the usual 0.5 cut-off from these scores = 0.8189 (training run reported 0.8185; small differences are normal because training used GPU fp16 and this uses CPU fp32).

