# Credit Risk Modeling on Lending Club Loans

Probability-of-default model for 36-month Lending Club loans (2007–2016 originations), validated out of time. Logistic regression is the benchmark; LightGBM is the main model. Every number below comes from `reports/artifacts/metrics.json`, written by `make train`.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)
![tests](https://img.shields.io/badge/tests-81%20passing-brightgreen.svg)

## Results

Out-of-time test set: 372,802 loans issued 2015-01 to 2016-03, default rate 15.18%. Models were trained on loans issued up to 2013-12 and tuned on 2014. 95% CIs from 1,000 bootstrap resamples of the test set.

| Model | ROC-AUC [95% CI] | Gini | KS | PR-AUC [95% CI] | Brier | ECE |
|---|---|---:|---:|---|---:|---:|
| Logistic regression | 0.6803 [0.6780, 0.6824] | 0.3606 | 0.2641 | 0.2593 [0.2565, 0.2621] | 0.1233 | 0.0287 |
| LightGBM | 0.6817 [0.6794, 0.6838] | 0.3633 | 0.2661 | 0.2612 [0.2585, 0.2643] | 0.1236 | 0.0359 |

**LightGBM − LR (paired bootstrap):** ROC-AUC +0.0014 [+0.0006, +0.0021], PR-AUC +0.0019 [+0.0011, +0.0028].

The difference is statistically clear (both CIs exclude zero) but practically small. On these application-time features, the boosted model adds very little over a linear scorecard, and its probabilities are slightly less well calibrated (higher ECE). Both models under-predict default on the test cohort (see the reliability plot): the 2015–16 cohorts default more often (15.2%) than the training cohorts (12.6%).

![Reliability diagram](reports/figures/reliability.png)

**Cost-aware threshold.** A missed default is assumed to cost 5× a wrongly rejected good loan (`CostMatrix(fn_cost=5, fp_cost=1)`; expected cost = 5·FN + 1·FP). The threshold is picked on the 2014 validation set and then applied to the test set.

| Model | Threshold (from validation) | Precision | Recall | Expected cost | Expected cost at 0.5 |
|---|---:|---:|---:|---:|---:|
| Logistic regression | 0.16 | 0.267 | 0.430 | 228,017 | 282,274 |
| LightGBM | 0.15 | 0.259 | 0.481 | 224,701 | 282,940 |

At 0.5 almost no loans are flagged (LR flags 480, LightGBM none), so the 0.5 column is a "no model" baseline, not a competing operating point.

## How default is defined

From `loan_status` in the Kaggle `accepted_2007_to_2018Q4.csv` file, keeping finished loans only:

- `Fully Paid` and `Does not meet the credit policy. Status:Fully Paid` → 0
- `Charged Off`, `Default` and `Does not meet the credit policy. Status:Charged Off` → 1
- `Current`, `Late (16-30 days)`, `Late (31-120 days)` and `In Grace Period` are dropped (no final outcome yet).

Two further filters:

- **Matured cohorts only:** a loan is kept only if `issue_d + term ≤ 2019-03-31` (the snapshot date of the file). Without this, recent cohorts contain only the loans that already ended, mostly early payoffs and fast charge-offs, which biases the default rate.
- **36-month loans only:** after the maturity filter, no 60-month loan issued after 2014-03 survives, so an out-of-time test set would contain 36-month loans only. Training on both terms would mean the splits describe different products.

| Step | Rows |
|---|---:|
| Rows read | 2,260,668 |
| Finished loans | 1,348,099 |
| Key fields present | 1,347,721 |
| Matured by 2019-03-31 | 779,013 |
| 36-month loans | 710,794 |

| Split | Issue dates | Rows | Default rate |
|---|---|---:|---:|
| Train | 2007-06 – 2013-12 | 175,422 | 12.63% |
| Validation | 2014-01 – 2014-12 | 162,570 | 13.73% |
| Test | 2015-01 – 2016-03 | 372,802 | 15.18% |

## Features

All features are known when the application is decided. A test (`tests/test_features.py`) fails if any post-origination column (`total_pymnt`, `recoveries`, `last_pymnt_d`, `last_fico_range_*`, `out_prncp`, …) is a model feature or is even read from the CSV.

- **Borrower:** `fico` (mean of `fico_range_low/high` at application), `dti`, `annual_income`, `emp_length`, `home_ownership`, `open_acc`, `revol_util`
- **Loan:** `loan_amount`, `term`, `purpose`
- **LC pricing:** `grade`, `int_rate`
- **Engineered:** `loan_to_income`, `installment_estimate`, `installment_to_income`, `emp_x_homeowner` (emp_length × owns/mortgages home)

`grade` and `int_rate` are outputs of Lending Club's own risk model. They're legitimate at decision time, but using them means the model partly re-learns LC's score. `USE_LC_GRADE = False` in `src/features.py` drops them and the two installment features derived from `int_rate`.

## Quickstart

```bash
make install       # .venv + pinned deps
make test          # 81 tests (synthetic fixtures; the 1.6 GB CSV is not needed)
make smoke         # 5,000-row run on real data, no artifacts saved
make train         # full out-of-time run; writes reports/artifacts/metrics.json and the reliability plot
make train-random  # random 80/20 split, comparison only
```

`make train` needs the Kaggle Lending Club download in `data/archive/` (git-ignored). The loader picks the `accepted_*` CSV (uncompressed if present, otherwise `.csv.gz`) and never reads the rejected-loans file.

## Method

- **Split:** out of time on `issue_d` (`src/train.py`, `TRAIN_UNTIL = "2013-12"`, `TEST_FROM = "2015-01"`). The test set is only used for the final report.
- **Preprocessing:** median imputation, scaling (LR only) and one-hot encoding inside an sklearn `Pipeline`, fit on the training split only.
- **Logistic regression:** no class reweighting, so its probabilities can be compared on Brier/ECE. Class imbalance is handled by the cost-optimal threshold instead.
- **LightGBM:** 30-candidate random search, each candidate fit on train with early stopping on validation PR-AUC; the best candidate by validation PR-AUC is kept (85 trees, 22 leaves, learning rate 0.037).
- **Metrics:** ROC-AUC, Gini (= 2·AUC − 1), KS, PR-AUC, Brier, ECE (10 quantile bins), and paired bootstrap CIs (`src/evaluate.py`).

## Known limitations

- **Binary target ignores timing.** A loan that charges off in month 3 and one that charges off in month 35 are the same label. Survival / time-to-default modeling is the next phase.
- **Accepted loans only.** There's no reject inference, so the model describes applicants LC approved, not the full applicant population.
- **LC grade is a feature.** `grade` and `int_rate` come from LC's own model, so part of the measured performance is LC's. See `USE_LC_GRADE` above.
- **36-month loans only.** 60-month loans are excluded (see above); the results don't describe them.
- **Default rate shifts over time.** The test cohorts default more often than the training cohorts, and both models under-predict on the test set. Recalibrating on recent cohorts would be needed before using the probabilities directly.
- **The cost matrix is a heuristic.** 5:1 is a rough rule of thumb, not a lender's measured loss.

## Layout

- `src/data.py`: Lending Club loader, target definition, maturity and term filters, `temporal_split`
- `src/features.py`: engineered features, preprocessing, `USE_LC_GRADE`
- `src/models.py`: logistic regression and LightGBM pipelines
- `src/train.py`: out-of-time training (default) and `--random-split`
- `src/evaluate.py`: metrics, bootstrap CIs, cost-aware threshold
- `src/calibration.py`: reliability table/plot, ECE, Platt and isotonic recalibration
- `src/drift.py`, `src/monitor.py`, `src/interpret.py`, `src/errors.py`: PSI, rolling performance monitor, SHAP, error slicing (used by the notebooks)
- `src/synth.py`: small synthetic generator used **only** as a test fixture
- `reports/report.md`: full write-up
- `notebooks/` (real data; 02–05 read the `make train` artifacts and never write them; run with `make notebooks`):
  - `01_eda`: class balance, distributions by outcome, default rate by group and issue year
  - `02_modeling`: headline metrics, PR curves, cost vs threshold, cost-ratio sensitivity (thresholds picked on validation)
  - `03_interpretation`: SHAP global / beeswarm / three loans, missed defaults by purpose, grade and home ownership
  - `04_calibration`: reliability by bin and issue quarter, Platt / isotonic recalibration fit on 2014
  - `05_drift`: PSI train vs test, simulated shocks, label-delay performance monitor

## Reproducibility

`RANDOM_STATE = 42` drives the search, model seeds and bootstrap. Dependencies are pinned in `requirements.txt`.

## License

Personal portfolio project. Code can be referenced; please don't redistribute the trained artifacts.
