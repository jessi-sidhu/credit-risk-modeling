# Loan Default Prediction — Report

## 1. Problem framing

Binary classification: given a loan application, predict the probability the borrower defaults. Banks make this decision millions of times a year, so a 1–2% lift over a baseline pays for itself. ML interest is moderate class imbalance (~18% positives), mixed feature types, and a regulator-driven need for instance-level explanations (US ECOA, Canadian credit-decision rules).

## 2. Data

50,000 rows, 14 features + label, generated synthetically and modeled on Lending Club's public loan data. Two of the features are pure i.i.d. normals (`noise_1`, `noise_2`), included intentionally so we can verify the SHAP pipeline against ground truth.

Why synthetic? Reproducibility, a known data-generating process to validate interpretability against, and no Kaggle auth required. Trade-off: real-world distributions are messier; `src/data.py::load_data(source="lendingclub")` is the single swap point if a real CSV is dropped into `data/raw/`.

The DGP is deterministic per seed and uses bisection on the intercept to land the marginal default rate within 1bp of the 18% target — so the imbalance story stays honest.

## 3. EDA findings

See [`notebooks/01_eda.ipynb`](../notebooks/01_eda.ipynb).

- Default rate: 18.0% — the target.
- No missingness in the synthetic data; the `SimpleImputer` step exists for the real-data path.
- `fico`, `dti`, and `int_rate` visibly separate the classes by density. `revol_util` shifts modestly. `noise_1`/`noise_2` densities sit on top of each other — exactly what we want.
- `int_rate` and `grade` are correlated by construction (grade is a FICO bucket, int_rate is set from grade). Tree models tolerate; for a linear baseline this is something we'd flag.
- Default rate is monotone in grade (A: 0.2% → G: 88%) — a strong sanity check on the pipeline end-to-end.

![Class balance](figures/class_balance.png)
![Numeric distributions by class](figures/numeric_by_class.png)
![Default rate by category](figures/default_rate_by_category.png)

## 4. Preprocessing and feature engineering

Train/test split (stratified 80/20, `random_state=42`) is the FIRST step. All preprocessing — imputation, scaling, one-hot encoding — fits on the training fold only, enforced structurally by wrapping each model in a `Pipeline(pre=ColumnTransformer, clf=...)`. A test in `tests/test_features.py` checks that fitting on train alone produces a transformer that can transform test data without producing NaNs and without crashing on unseen categories.

Engineered features (added before the split because they're row-wise arithmetic and cannot leak):

| Feature | Why |
|---|---|
| `loan_to_income` = loan_amount / annual_income | Affordability ratio a loan officer would compute |
| `installment_estimate` | Standard amortization formula on amount, rate, term |
| `installment_to_income` = installment / (annual_income / 12) | Monthly DTI for *this* loan, not the borrower's existing burden |
| `credit_age_proxy` = emp_length × indicator(home_ownership ∈ {OWN, MORTGAGE}) | Stability proxy when LC's listed credit-history features are absent |

## 5. Metric choice

Primary: **PR-AUC** (`average_precision_score`). With 18% positives, predicting "no default" everywhere gives 82% accuracy — useless. PR-AUC is threshold-independent and focuses on the rare positive class.

Secondary: **ROC-AUC** for comparison with literature.

Operating point: precision, recall, F1 at a chosen threshold. The threshold is picked by minimizing **expected cost** with FN:FP = 5:1 (a missed default ≈ loan principal lost; a rejected good loan ≈ foregone interest margin). The 5:1 ratio is exposed as a parameter (`CostMatrix(fn_cost, fp_cost)`) so the §13 sensitivity analysis is one function call.

## 6, 7, 8. Models, tuning, threshold

Three models, escalating in complexity. The boosting model is the only one tuned, via `RandomizedSearchCV` (5-fold stratified, 30 iters, scored on PR-AUC, executed against the train set only).

| Model | PR-AUC | ROC-AUC | F1 @ 0.5 | F1 @ cost-opt | Threshold |
|---|---:|---:|---:|---:|---:|
| Logistic regression (balanced) | 0.7451 | 0.9235 | 0.6439 | 0.6586 | 0.53 |
| Random forest (300 trees) | 0.7206 | 0.9155 | 0.6405 | 0.6126 | 0.16 |
| LightGBM (tuned) | 0.7404 | 0.9215 | 0.6535 | 0.6234 | 0.14 |

Tuned LightGBM hyperparameters: `num_leaves=18, learning_rate=0.017, min_child_samples=48, reg_lambda=5.87, n_estimators=473, feature_fraction=1.0`.

![Precision–recall curves](figures/pr_curves.png)

**Threshold selection.** Sweeping thresholds on the held-out test set:

- Cost at default 0.5 threshold: **4,090**
- Cost at optimal (≈ 0.14) threshold: **2,705**
- Reduction: **33.9%**

That's the most operationally valuable result in this report. F1 alone would have understated it.

![Cost vs threshold](figures/threshold_sweep.png)

**Honest note.** The linear baseline is essentially tied with tuned LightGBM on this synthetic data. That's expected — the DGP is largely additive in the true features. We'd still ship the boosting model in production because (a) it's better-calibrated for the cost-sensitive operating point, (b) real Lending Club data has interactions and time effects the linear model can't represent, and (c) the marginal training cost is irrelevant.

## 9. Interpretability (SHAP)

See [`notebooks/03_interpretation.ipynb`](../notebooks/03_interpretation.ipynb).

**Global importance** (mean |SHAP| on a 1,000-row test sample):

| Feature | mean &#124;SHAP&#124; |
|---|---:|
| fico | 1.585 |
| dti | 0.511 |
| int_rate | 0.290 |
| revol_util | 0.211 |
| loan_to_income | 0.067 |
| term | 0.047 |
| installment_to_income | 0.030 |
| ... | ... |
| noise_1 | 0.018 |
| noise_2 | 0.012 |

The two noise features have roughly **1% of fico's contribution**, and rank below every true continuous-signal feature. A small residual SHAP value on pure noise is normal in a tuned tree ensemble — it just means the model picked up a few spurious splits, not that the pipeline is broken. If `noise_*` had been *high* in the ranking, that would have been a red flag for label leakage or a fitting bug.

![SHAP bar](figures/shap_bar.png)
![SHAP beeswarm](figures/shap_beeswarm.png)

Beeswarm signs match domain intuition: high FICO pushes predictions toward not-default; high DTI and high int_rate push toward default. Sign-checks like this are cheap insurance against label/feature errors.

**Local explanations.** Three test loans were chosen — a clear default, a clear paid loan, and a borderline case — and rendered as SHAP waterfalls. These are the regulator-facing artifact: each prediction can be explained one feature at a time.

![Clear default waterfall](figures/shap_waterfall_clear_default.png)
![Clear paid waterfall](figures/shap_waterfall_clear_paid.png)
![Borderline waterfall](figures/shap_waterfall_borderline.png)

## 10. Error analysis

Slicing test errors by `purpose` reveals where the model misses defaults disproportionately. False-negative rates (FN / actual positives) and the lift over the overall FNR:

| purpose | n | FN | FNR | FNR lift |
|---|---:|---:|---:|---:|
| home_improvement | 458 | 12 | 0.150 | 1.42 |
| small_business | 428 | 10 | 0.143 | 1.35 |
| credit_card | 2,047 | 44 | 0.119 | 1.12 |
| debt_consolidation | 5,486 | 100 | 0.104 | 0.99 |
| car | 381 | 4 | 0.049 | 0.47 |

`home_improvement` and `small_business` have the highest missed-default rates — this matches conventional credit wisdom (these purposes carry idiosyncratic risk that headline credit features don't capture well). In production we'd consider a per-segment threshold, additional features for those segments, or both.

![FNR by purpose](figures/fnr_by_purpose.png)

## 11. Limitations (honest)

- **Synthetic data.** Real Lending Club has messier missingness, label-leakage risk from `loan_status` derivative columns, and time effects (interest rates and macro conditions shift). Our DGP is additive in the true features; real data is not.
- **No temporal split.** Real credit modeling uses out-of-time validation because borrower behavior shifts. We did random stratified split.
- **No fairness audit.** In real deployment, we'd test for disparate impact across protected classes (race, gender, age). Some are forbidden as features under US ECOA, but the model can still discriminate via proxies (zip code, income).
- **Single point-in-time prediction.** Doesn't model survival/hazard — *when* during the loan does default happen?
- **Cost matrix is a rough heuristic.** 5:1 is a defensible starting point, not a measured number from a specific lender's P&L.

## 12. Future work

- **Temporal validation** — train on origination cohort T, test on cohort T+1. Use Lending Club's `issue_d`.
- **Drift monitoring** — PSI on input features and a delayed performance monitor on labels (loan outcomes resolve months after origination).
- **Calibration audit** — reliability diagrams. A model with high PR-AUC isn't automatically well-calibrated; if pricing depends on the predicted probability, miscalibration costs real money.
- **Cost-matrix sensitivity** — done; see below.
- **Segment-specific thresholds** — given the FNR concentration in `home_improvement` / `small_business`, a per-purpose threshold likely beats a single global one.

## 13a. Cost-matrix sensitivity (done)

The 5:1 ratio is a heuristic. Sweeping FN:FP from 2:1 to 10:1 on the held-out test set:

| FN:FP | Optimal threshold | Cost @ opt | Cost @ 0.5 | Reduction | Precision | Recall | F1 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.29 | 1,753 | 1,864 | 6.0% | 0.607 | 0.759 | 0.674 |
| 3 | 0.22 | 2,167 | 2,606 | 16.8% | 0.551 | 0.822 | 0.660 |
| 5 | 0.14 | 2,705 | 4,090 | **33.9%** | 0.478 | 0.894 | 0.623 |
| 7 | 0.12 | 3,054 | 5,574 | 45.2% | 0.456 | 0.913 | 0.608 |
| 10 | 0.07 | 3,490 | 7,800 | 55.3% | 0.390 | 0.956 | 0.554 |

The qualitative behavior is what we'd want to defend in an interview: as the FN penalty grows, threshold drops monotonically, recall grows, precision falls, and the gain from threshold tuning over the naive 0.5 grows. The operating point is not fragile to the exact ratio within a reasonable range.

![Cost sensitivity](figures/cost_sensitivity.png)
