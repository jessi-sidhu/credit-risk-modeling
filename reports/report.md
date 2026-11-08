# Credit Risk Modeling — Report

All numbers in this report come from `reports/artifacts/metrics.json`, written by `make train` (`python -m src.train --save`). Derived numbers (expected cost) are computed from the confusion matrices stored there.

## 1. Problem

Predict, at application time, the probability that a 36-month Lending Club loan ends in default. Logistic regression is the benchmark, the way a scorecard would be built; LightGBM is the candidate main model. The question this phase answers is how much LightGBM beats LR out of time, with a confidence interval.

## 2. Data

Source: the Kaggle Lending Club download, `data/archive/accepted_2007_to_2018q4.csv/accepted_2007_to_2018Q4.csv` (git-ignored). The rejected-applications file has a different schema and is never read. Only the 15 columns the model maps are loaded (`LC_USECOLS` in `src/data.py`).

### 2.1 Target definition

`loan_status` is mapped for finished loans only (`TARGET_MAP` in `src/data.py`):

| `loan_status` | default |
|---|---:|
| Fully Paid | 0 |
| Does not meet the credit policy. Status:Fully Paid | 0 |
| Charged Off | 1 |
| Default | 1 |
| Does not meet the credit policy. Status:Charged Off | 1 |
| Current, Late (16-30 days), Late (31-120 days), In Grace Period | dropped |

Late loans are dropped rather than counted as defaults: they don't have a final outcome yet, and some of them cure.

### 2.2 Matured cohorts and loan term

The file is a snapshot taken around 2019-03-31 (`SNAPSHOT_DATE`). For a cohort issued in 2017, the loans that have *finished* by the snapshot are mostly early payoffs and fast charge-offs; the ones still running are dropped as `Current`. Keeping those cohorts would bias the default rate. A loan is therefore kept only if `issue_d + term ≤ SNAPSHOT_DATE`.

That filter keeps 36-month loans issued up to 2016-03 but 60-month loans only up to 2014-03. An out-of-time test set from 2015 onward would contain no 60-month loans, while training would mix them in, and 60-month loans default much more often than 36-month loans. The model is therefore restricted to 36-month loans (`TERM_MONTHS = 36`).

### 2.3 Row counts

| Step | Rows |
|---|---:|
| Rows read (blank footer lines removed) | 2,260,668 |
| Finished loans (status in `TARGET_MAP`) | 1,348,099 |
| FICO, DTI, income, amount and issue date present | 1,347,721 |
| Matured by 2019-03-31 | 779,013 |
| 36-month loans | 710,794 |

Final default rate: 14.22%.

### 2.4 Out-of-time split

| Split | Issue dates | Rows | Default rate | Used for |
|---|---|---:|---:|---|
| Train | 2007-06 – 2013-12 | 175,422 | 12.63% | fitting |
| Validation | 2014-01 – 2014-12 | 162,570 | 13.73% | early stopping, model selection, threshold |
| Test | 2015-01 – 2016-03 | 372,802 | 15.18% | final report only |

The test window is larger than the training window because Lending Club's volume grew quickly. The default rate rises across the three windows; section 5 shows the effect.

## 3. Features

Every feature must be known when the credit decision is made.

| Feature | Source | Why it's allowed |
|---|---|---|
| `fico` | mean of `fico_range_low` / `fico_range_high` | Bureau score pulled at application. (`last_fico_range_*` is post-origination and is not read.) |
| `dti` | `dti` | Debt-to-income from the application |
| `annual_income` | `annual_inc` | Stated on the application |
| `emp_length` | `emp_length` | Stated on the application; parsed to 0–10 years |
| `home_ownership` | `home_ownership` | Stated on the application |
| `open_acc` | `open_acc` | Bureau data at application |
| `revol_util` | `revol_util` | Bureau data at application |
| `loan_amount` | `loan_amnt` | Requested amount |
| `term` | `term` | Requested term (always 36 after the term filter, so it carries no signal here) |
| `purpose` | `purpose` | Stated on the application |
| `grade` | `grade` | Assigned by LC at origination (see below) |
| `int_rate` | `int_rate` | Set by LC from the grade at origination (see below) |
| `loan_to_income` | `loan_amount / annual_income` | Arithmetic on application fields |
| `installment_estimate` | amortization of amount, rate, term | Arithmetic on application fields |
| `installment_to_income` | installment / monthly income | Arithmetic on application fields |
| `emp_x_homeowner` | `emp_length` × owns or mortgages home | Arithmetic on application fields (previously misnamed `credit_age_proxy`) |

**LC grade and rate.** `grade` and `int_rate` are outputs of Lending Club's own risk model. They exist at decision time, so they're not leakage, but a model using them partly re-learns LC's score. `USE_LC_GRADE` in `src/features.py` (default `True`) controls them; setting it to `False` also drops the two installment features, because they're computed from `int_rate`. The results below use `True`.

**Leakage guard.** `tests/test_features.py` fails if any of `total_pymnt`, `total_rec_prncp`, `recoveries`, `collection_recovery_fee`, `last_pymnt_d`, `last_pymnt_amnt`, `last_fico_range_high`, `last_fico_range_low`, `out_prncp` is a model feature, or is in the list of columns read from the CSV.

**Preprocessing.** Median imputation, standard scaling (LR only) and one-hot encoding (unseen categories ignored) live inside each model's sklearn `Pipeline` and are fit on the training split only.

## 4. Models

**Logistic regression** (`src/models.py`): L2, `max_iter=2000`, no class reweighting. An earlier version used `class_weight="balanced"`, which inflates predicted probabilities and would make LR look badly calibrated for reasons unrelated to ranking. Class imbalance is handled by the cost-optimal threshold instead.

**LightGBM** (`src/train.py::tune_lightgbm_early_stopping`): 30 candidates sampled over `num_leaves`, `learning_rate`, `min_child_samples`, `feature_fraction` and `reg_lambda`. Each candidate is fit on train with up to 2,000 trees and early stopping (50 rounds) on validation PR-AUC, and the candidate with the best validation PR-AUC is kept.

Selected: `num_leaves=22`, `learning_rate=0.0374`, `min_child_samples=53`, `feature_fraction=1.0`, `reg_lambda=0.0014`, 85 trees (validation PR-AUC 0.2320). Early stopping at 85 trees with a moderate learning rate means the validation set stopped rewarding extra complexity quickly.

## 5. Results (out-of-time test set)

### 5.1 Discrimination and calibration

95% CIs are percentile intervals from 1,000 bootstrap resamples of the test set. The difference is computed on the same resamples for both models (paired).

| Model | ROC-AUC [95% CI] | Gini | KS | PR-AUC [95% CI] | Brier | ECE |
|---|---|---:|---:|---|---:|---:|
| Logistic regression | 0.6803 [0.6780, 0.6824] | 0.3606 | 0.2641 | 0.2593 [0.2565, 0.2621] | 0.1233 | 0.0287 |
| LightGBM | 0.6817 [0.6794, 0.6838] | 0.3633 | 0.2661 | 0.2612 [0.2585, 0.2643] | 0.1236 | 0.0359 |

| LightGBM − LR | Estimate | 95% CI |
|---|---:|---|
| ROC-AUC | +0.0014 | [+0.0006, +0.0021] |
| PR-AUC | +0.0019 | [+0.0011, +0.0028] |

**How much does LightGBM beat LR?** By about 0.14 ROC-AUC points and 0.19 PR-AUC points. With 372,802 test loans the CIs are tight enough that the gain is clearly not zero, but it's far too small to matter in practice. On Brier the two are tied (0.1233 vs 0.1236), and LightGBM's probabilities are slightly worse calibrated (ECE 0.0359 vs 0.0287). On these application-time features, most of the signal is linear, and much of it is already summarised in LC's own `grade` / `int_rate`.

PR-AUC should be read against the test default rate of 0.1518, which is what a random ranking would score.

![Reliability diagram](figures/reliability.png)

Both curves sit above the diagonal: in every probability bin the observed default rate on 2015–16 loans is higher than predicted. That's consistent with the default rate rising from 12.6% (train) to 15.2% (test). Ranking holds up, but the probability *level* shifts over time, so the probabilities would need recalibrating on recent cohorts before being used directly (for pricing or expected loss).

### 5.2 Cost-aware threshold

Assumed costs: a missed default (FN) costs 5, a wrongly rejected good loan (FP) costs 1 (`CostMatrix`). Expected cost = 5·FN + 1·FP on the test set. The threshold that minimises expected cost is picked on the **validation** predictions and then applied to the test set unchanged.

| Model | Threshold | Flagged as default | Precision | Recall | FN | FP | Expected cost |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic regression | 0.16 | 91,081 | 0.267 | 0.430 | 32,254 | 66,747 | 228,017 |
| LightGBM | 0.15 | 105,081 | 0.259 | 0.481 | 29,368 | 77,861 | 224,701 |

At the 0.5 threshold LR flags 480 loans and LightGBM flags none, so the expected cost there (282,274 and 282,940) is essentially the cost of having no model. Against that baseline the cost-optimal threshold cuts expected cost by 19.2% (LR) and 20.6% (LightGBM). The difference *between* the two models at their operating points is 3,316 cost units (1.5%), in line with the small AUC gap.

## 6. Known limitations

- **Binary target ignores timing.** A charge-off in month 3 and one in month 35 get the same label, and loan value depends heavily on when default happens. Survival / time-to-default modeling is the next phase.
- **Accepted loans only, no reject inference.** The model is trained and evaluated on applicants LC approved. Its performance on the full applicant population is unknown.
- **LC grade used as a feature.** `grade` and `int_rate` encode LC's own risk model, so part of the measured performance belongs to LC. A run with `USE_LC_GRADE = False` would show how much the borrower fields carry on their own; it isn't reported here.
- **36-month loans only.** 60-month loans are excluded because the matured out-of-time test set contains none.
- **Default rate shifts over time.** Test cohorts default more often than training cohorts, and both models under-predict on the test set.
- **The cost matrix is a heuristic.** 5:1 is a rule of thumb, not measured loss-given-default and margin.

## 7. Notebooks

`notebooks/01`–`05` explore the same data and the same Phase 1 run (EDA, modeling, SHAP and error analysis, calibration, drift). Their findings are written in each notebook's Takeaways cell; the figures they produce are in `reports/figures/`. They only read `reports/artifacts/`, so they cannot change the numbers in this report.

## 8. Phase 2: Time-to-default

Phase 1 asks *will this loan default?* and has to throw away every loan without a final outcome: of 2.26M accepted loans it keeps 710,794. Phase 2 asks *in which quarter will it default, and what is the expected loss over its life?* That lets every loan stay in:

- **Censoring.** A loan that is still running (Current, Late, In Grace) tells us it hasn't defaulted *yet*. It contributes the quarters it was observed for and is then censored, instead of being dropped.
- **Both terms.** 60-month loans come back (650,208 of them).
- **Prepayment is a competing risk.** A loan that is paid off early can no longer default, so treating payoff as "survived" would overstate how long loans stay at risk. Each quarter has three outcomes: no event, default, prepay.

Method: a discrete-time competing-risks hazard model (Botha & Verster 2025). Each loan becomes one row per quarter at risk, and a classifier predicts {no event, default, prepay} for that quarter. From the per-quarter hazards h:

- survival `S(t) = Π_{k≤t} (1 − h_default(k) − h_prepay(k))`
- cumulative default probability `CIF_default(t) = Σ_{k≤t} S(k−1) · h_default(k)`

Phase 1's binary model is unchanged and is the comparison point. All numbers in this section come from `reports/artifacts/survival/` (`outcomes_summary.json`, `train_meta.json`, `metrics.json`, `ecl.json`), written by `make train-survival`.

### Events

| `loan_status` | Event | Duration (months since issue) |
|---|---|---|
| Charged Off, Default (+ credit-policy variant) | default | to `last_pymnt_d` + 1 (first missed payment); 1 if no payment was ever made |
| Fully Paid (+ variant), more than 2 months before term | prepay | to `last_pymnt_d` |
| Fully Paid within 2 months of term | censored | `term` |
| Current, Late, In Grace Period | censored | to `last_pymnt_d` (snapshot date if missing) |

Durations are clipped to `[1, term]`: 10,544 loans (mostly payoffs in the issue month) are clipped and kept; no loan had impossible dates.

| Term | Censored | Default | Prepay | Total |
|---|---:|---:|---:|---:|
| 36 months | 831,143 | 163,889 | 613,713 | 1,608,745 |
| 60 months | 337,647 | 105,400 | 207,161 | 650,208 |

Defaults are 19.98% of all finished loans, higher than Phase 1's 14.22% because recent cohorts have only had time to finish their early charge-offs, and 60-month loans default more. On Phase 1's own population (finished loans from matured 36-month cohorts) the outcome table gives exactly 14.22%; for matured 60-month loans it is 25.37%.

![Kaplan-Meier by grade](figures/km_by_grade.png)

Kaplan–Meier share not yet defaulted (prepayment and censoring both treated as censored here):

| Grade | 36m: S(12) | S(24) | S(36) | 60m: S(12) | S(24) | S(36) | S(60) |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | 0.9859 | 0.9572 | 0.9239 | 0.9876 | 0.9625 | 0.9310 | 0.8871 |
| B | 0.9654 | 0.9012 | 0.8406 | 0.9747 | 0.9267 | 0.8749 | 0.7972 |
| C | 0.9339 | 0.8324 | 0.7486 | 0.9522 | 0.8692 | 0.7896 | 0.6852 |
| D | 0.8968 | 0.7626 | 0.6716 | 0.9178 | 0.7947 | 0.6945 | 0.5762 |
| E–G | 0.8410 | 0.6856 | 0.5892 | 0.8541 | 0.6872 | 0.5733 | 0.4594 |

Median months to event: default 15, prepay 16 (censored loans: 20).

### Models

Same out-of-time split as Phase 1 on `issue_d`, but now with all loans: train 2007-06 – 2013-12 (230,712 loans, 2,164,708 loan-quarters), validation 2014 (235,629 loans), test 2015-01 – 2018-12 (1,792,612 loans, of which 1,021,990 censored).

- **Benchmark:** multinomial logistic regression on loan-quarter rows, with a separate quarter-of-life baseline per term (one-hot term × quarter). Validation log loss 0.30238.
- **Main:** LightGBM multiclass with `period` and `term` as features; 6-candidate random search, each early-stopped on validation log loss. Selected: 16 leaves, learning rate 0.105, 138 trees. Validation log loss 0.30147 (predicting the training base rates: 0.31570).

Numeric features are capped at their training 0.1% / 99.9% quantiles for both models: recent loans include stated incomes near zero and DTI up to 999, on which logistic regression's hazards saturated at 0 or 1.

![Predicted default curves](figures/cif_examples.png)

### Discrimination at fixed horizons (out-of-time test)

AUC of `CIF_default(h)` for "defaulted by month h", on loans whose status at h is known (defaulted by h, finished without default by h, or still observed after h). 95% CIs from 1,000 bootstrap resamples.

| Horizon | Loans | Default rate | LR AUC | LightGBM AUC | LightGBM − LR |
|---|---:|---:|---|---|---|
| 12 months | 1,405,884 | 5.94% | 0.6967 [0.6950, 0.6985] | 0.6996 [0.6977, 0.7013] | +0.0028 [+0.0023, +0.0034] |
| 24 months | 1,108,919 | 14.40% | 0.6894 [0.6879, 0.6907] | 0.6898 [0.6883, 0.6911] | +0.0004 [+0.0001, +0.0008] |
| 36 months | 947,122 | 20.00% | 0.6891 [0.6878, 0.6904] | 0.6888 [0.6875, 0.6901] | −0.0003 [−0.0007, +0.0000] |

As in Phase 1, LightGBM and the linear benchmark are essentially tied: LightGBM is ahead at 12 months, level by 36. On PR-AUC the LR is slightly ahead at every horizon (36 months: 0.3425 vs 0.3404).

PR-AUC (random ranking scores the default rate):

| Horizon | LR | LightGBM |
|---|---|---|
| 12 months | 0.1243 [0.1229, 0.1259] | 0.1232 [0.1219, 0.1246] |
| 24 months | 0.2621 [0.2603, 0.2641] | 0.2604 [0.2586, 0.2623] |
| 36 months | 0.3425 [0.3405, 0.3447] | 0.3404 [0.3383, 0.3424] |

### Phase 1 vs Phase 2, apples to apples

On the 372,802 matured 36-month test loans Phase 1 was scored on (default rate 15.18%), Phase 1's LightGBM PD vs Phase 2's LightGBM `CIF_default(36)`. Re-scoring Phase 1's saved model reproduces its published AUC exactly (0.6817).

| | Phase 1 PD | Phase 2 CIF(36) | Phase 2 − Phase 1 |
|---|---|---|---|
| ROC-AUC | 0.6817 [0.6794, 0.6838] | 0.6824 [0.6801, 0.6845] | +0.0007 [+0.0003, +0.0011] |
| PR-AUC | 0.2612 [0.2585, 0.2643] | 0.2609 [0.2580, 0.2639] | −0.0004 [−0.0010, +0.0002] |
| Brier | 0.1236 | 0.1236 | +0.0001 [+0.0000, +0.0001] |
| ECE | 0.0359 | 0.0383 | |
| Mean prediction | 0.1159 | 0.1135 | |

Training on 230,712 loans instead of Phase 1's 175,422 (both terms, unfinished loans included) ranks these loans marginally better, but does not change the picture.

### Calibration: did Phase 2 fix the under-prediction?

**No.** Mean predicted vs observed (Aalen–Johansen) cumulative default probability on the test set:

| Issue year | Horizon | Observed | LR | LightGBM | LightGBM / observed |
|---|---:|---:|---:|---:|---:|
| 2015 | 36 months | 0.1739 | 0.1433 | 0.1324 | 0.76 |
| 2016 | 36 months | 0.1636 | 0.1424 | 0.1347 | 0.82 |
| 2016 | 12 months | 0.0616 | 0.0453 | 0.0435 | 0.71 |
| 2017 | 12 months | 0.0568 | 0.0453 | 0.0445 | 0.78 |
| 2017 | 24 months | 0.0974 | 0.1036 | 0.0987 | 1.01 |
| 2018 | 12 months | 0.0256 | 0.0429 | 0.0430 | 1.68 |

For 2015–2016, where the outcomes are complete, both models still predict only 71–87% of the defaults that happened (ratios across both models and all horizons), the same gap as Phase 1. The model learns default *levels* from 2007–2013 and has no way to know that later cohorts were riskier at the same features (notebook 05: LC cut interest rates while risk rose).

The apparent over-prediction for 2017 (24 months) and 2018 (12 months) should not be read as the problem going away. A loan that stops paying is reported as Late for several months before it is charged off; for loans issued in 2017–18, many of those defaults had not been charged off by the March 2019 snapshot, so they are censored here and the observed default rate for those cohorts is understated.

By grade (36 months), the models under-predict most for the riskiest loans (E–G: observed 0.3432, LightGBM 0.2811) and are close for A–B.

![Calibration](figures/survival_calibration.png)

Recent cohorts (2016–2018), all horizons with an observable Aalen–Johansen estimate:

| Issue year | Horizon | Loans | Observed | LR | LightGBM | LR / obs. | LightGBM / obs. |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2016 | 12 | 434,344 | 0.0616 | 0.0453 | 0.0435 | 0.7345 | 0.7053 |
| 2016 | 24 | 434,344 | 0.1326 | 0.1038 | 0.0980 | 0.7829 | 0.7388 |
| 2016 | 36 | 434,344 | 0.1636 | 0.1424 | 0.1347 | 0.8704 | 0.8234 |
| 2017 | 12 | 443,065 | 0.0568 | 0.0453 | 0.0445 | 0.7973 | 0.7830 |
| 2017 | 24 | 443,065 | 0.0974 | 0.1036 | 0.0987 | 1.0640 | 1.0138 |
| 2018 | 12 | 494,110 | 0.0256 | 0.0429 | 0.0430 | 1.6764 | 1.6800 |

By grade:

| Grade | Horizon | Observed | LR | LightGBM |
|---|---:|---:|---:|---:|
| A | 36 | 0.0497 | 0.0564 | 0.0462 |
| B | 36 | 0.1082 | 0.1021 | 0.0941 |
| C | 36 | 0.1752 | 0.1570 | 0.1512 |
| D | 36 | 0.2492 | 0.2235 | 0.2199 |
| E–G | 36 | 0.3432 | 0.2937 | 0.2811 |

### Lifetime expected loss (IFRS 9-style illustration)

Not a regulatory model: no macroeconomic scenarios, no stage allocation, and LGD is a simple grade average. Per loan: `ECL = Σ_t marginal PD(t) × EAD(t) × LGD × discount(t)`, with EAD from the amortization schedule, LGD by grade from 2007–2013 charge-offs (0.87–0.89 for every grade), and discounting at the loan's own rate.

- Whole test portfolio (1,792,612 loans, $27.3bn funded): LightGBM 12-month ECL 2.99% and lifetime ECL 6.95% of funded amount (LR: 3.14% and 7.36%).
- Matured 36-month test loans (379,281 loans), undiscounted, against what was actually lost:

| Issue year | Loans | Realized loss | LR predicted | LightGBM predicted | LightGBM / realized |
|---|---:|---:|---:|---:|---:|
| 2015 | 283,171 | 7.36% | 6.04% | 5.47% | 0.74 |
| 2016 | 96,110 | 8.12% | 6.05% | 5.54% | 0.68 |
| All | 379,281 | 7.56% | 6.04% | 5.49% | 0.73 |

Expected loss inherits the PD shortfall: the model would have provisioned about three quarters of the losses that occurred.

![Predicted vs realized loss](figures/ecl_predicted_vs_realized.png)

LGD by grade (2007–2013 charge-offs, `(loan_amnt − total_rec_prncp − recoveries) / (loan_amnt − total_rec_prncp)`, clipped to [0, 1]):

| Grade | A | B | C | D | E | F | G |
|---|---:|---:|---:|---:|---:|---:|---:|
| LGD | 0.8945 | 0.8853 | 0.8781 | 0.8782 | 0.8736 | 0.8665 | 0.8683 |
| Charge-offs | 2,230 | 8,406 | 10,370 | 7,646 | 4,443 | 2,411 | 593 |

`total_rec_prncp` and `recoveries` are post-origination columns; they are read only in `src/ecl.py`, only for these training-period charge-offs and for the realized-loss comparison (a leakage test enforces this).


### In plain terms

- **Censoring.** For a loan still running at the snapshot we know it survived *up to* its last observed payment, but not what happens after. A survival model uses exactly that much: the loan counts as "at risk, no default" for each quarter it was observed, then drops out. Phase 1 had to discard these loans; here they add 1,021,990 loans to the test set and are part of training and validation too.
- **Prepayment as a competing risk.** A loan that is paid off can never default. If payoff were treated as ordinary censoring, the model would assume those loans were still exposed to default afterwards and overstate default probabilities. Modeling prepay as its own outcome lets the default curve flatten as loans pay off.
- **From hazard to default curve.** The model gives, for each quarter, the chance of defaulting or prepaying *given the loan is still active*. Multiplying the "neither" probabilities gives the chance of still being active; summing (still active at the start of the quarter) × (default hazard) gives the cumulative chance of having defaulted by any month, e.g. `CIF_default(36)`.
- **Did Phase 2 fix the under-prediction on recent cohorts?** No. Where outcomes are complete (2015–2016 issues), Phase 2 predicts 71–87% of the observed defaults, the same shortfall as Phase 1, and its expected loss covers 73% of the realized loss. The newest cohorts look over-predicted only because their defaults are not yet fully recorded. Fixing the level needs information about calendar time (macro variables, or recalibration on recent data), not a different outcome definition.

### Limitations

- **Default timing is proxied** by the last payment date (+1 month). LC doesn't give the date of the first missed payment or of the charge-off.
- **Recent cohorts are under-observed.** Loans in arrears that had not yet been charged off at the snapshot count as censored, so observed default rates for 2017–18 are too low (see calibration).
- **Prepayment is treated as a competing risk** with its own hazard, but no prepayment-specific drivers (rate incentives) are modeled.
- **No macroeconomic variables.** The models cannot react to the change in risk across cohorts, which is why the under-prediction persists.
- **Accepted loans only**, no reject inference.
- **LGD is a simple grade average** from training charge-offs, ignoring time to recovery and collection costs.
- **Training labels overlap the test period.** Loans issued up to 2013 are followed to 2019, so the models see outcomes from calendar years that also affect the test cohorts. This is also true of Phase 1.

### Future work

- Drift-aware survival models that let the hazard shift with calendar time (Peng & Lessmann 2026).
- Deep competing-risks models (Blumenstock, Lessmann & Seow 2022).
- Macroeconomic covariates (unemployment, rates) per calendar quarter.
- Deployment: a FastAPI scoring service with input-drift (PSI) and calibration monitoring.

## 9. Reproduce

```bash
make install
make test     # 123 tests on synthetic fixtures
make train    # needs data/archive/; writes metrics.json and figures/reliability.png
make notebooks
make train-survival   # Phase 2: writes reports/artifacts/survival/*.json and the Phase 2 figures
```

`RANDOM_STATE = 42` fixes the search, the model seeds and the bootstrap.

## References

- Botha, A. & Verster, T. (2025). *Approaches for modelling the term-structure of default risk under IFRS 9: a tutorial using discrete-time survival analysis.* arXiv:2507.15441.
- Peng & Lessmann (2026). Drift-aware survival models. arXiv:2601.20533.
- Blumenstock, Lessmann & Seow (2022). Deep competing-risks models for credit risk.
