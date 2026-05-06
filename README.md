# Loan Default Prediction

Binary classification on a synthetic Lending Club style dataset. Predict the probability of default, then handle the things a credit team actually deals with: cost-aware threshold selection, SHAP explanations, calibration, drift monitoring, fairness checks, and out-of-time validation.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)
![tests](https://img.shields.io/badge/tests-56%20passing-brightgreen.svg)

## Results

Tuned LightGBM hits PR-AUC 0.74 and ROC-AUC 0.92 on a held-out test set with 18% positive prevalence. Moving from the default 0.5 threshold to the cost-optimal 0.14 (under FN:FP = 5:1) cuts expected loss by 33.9%.

Out-of-time validation tells a different story. Train on cohorts through 2017-06, test on 2018, and PR-AUC drops to 0.65. That 6-7 point gap is the realistic production number; random splits hide it.

![Random vs out-of-time degradation](reports/figures/temporal_degradation.png)

## Quickstart

```bash
make install         # creates .venv, installs pinned deps
make test            # 56 tests, under 10s
make smoke           # 5k row training run, under 90s
make train           # full random-split run, ~7 min
make train-temporal  # OOT comparison, ~30s
make notebooks       # execute all 7 notebooks in order
```

`make all` chains install, test, train, train-temporal, notebooks. `make verify` wipes generated state and rebuilds from scratch. `make help` lists every target.

## What's in here

`src/` modules:
- `synth.py` synthetic data generator with optional macro and regime trends
- `data.py` `load_data`, `temporal_split`, and the Lending Club CSV swap
- `features.py` `ColumnTransformer` plus four engineered ratio features
- `models.py` LR, RF, and LightGBM Pipeline factories
- `train.py` training orchestration, with a `--temporal` mode
- `evaluate.py` PR-AUC, ROC-AUC, cost-aware threshold sweep, sensitivity
- `interpret.py` SHAP wrapper around the sklearn pipeline
- `errors.py` per-subgroup error analysis with FNR-lift
- `calibration.py` reliability tables, ECE, Platt and isotonic recalibrators
- `drift.py` Population Stability Index per feature
- `fairness.py` group metrics, parity ratios, four-fifths rule
- `monitor.py` rolling performance with configurable label-resolution lag

`notebooks/`:
- `01_eda.ipynb` distributions, correlations, default rate by category
- `02_modeling.ipynb` three-model bake-off, cost-aware threshold, FN:FP sensitivity
- `03_interpretation.ipynb` SHAP global, directional, and per-loan
- `04_calibration.ipynb` reliability diagrams plus Platt, isotonic, and CV-on-train recalibration
- `05_drift.ipynb` PSI on simulated shifts, plus the label-delay performance monitor
- `06_fairness.ipynb` per-group approval rate, FNR, FPR, four-fifths check
- `07_temporal.ipynb` random vs out-of-time split comparison

`tests/`: 56 pytest tests. Coverage includes the leakage guard, threshold selection, calibration, drift, fairness, monitor, the LC swap fixture, and `temporal_split`.

## Why these choices

**PR-AUC, not accuracy.** With 18% positives, predicting "no default" for every row scores 82% accuracy. PR-AUC focuses on the rare class.

**Cost-aware threshold, not 0.5.** Missing a default costs roughly the loan principal; rejecting a good loan costs the foregone interest margin. The 5:1 FN:FP ratio puts the optimal threshold near 0.14, far from 0.5.

**Tune the boosting model only.** Logistic regression has little to tune and RF defaults are reasonable. Spending the search budget on LightGBM, which benefits most, keeps the comparison fair.

**Pipeline-everywhere preprocessing.** Imputation, scaling, and one-hot encoding live inside an sklearn Pipeline so they refit per CV fold. A unit test (`test_features.py::test_preprocessor_fits_on_train_only`) enforces this structurally.

**Two intentional noise features in the DGP.** SHAP should rank them near zero. It does, at about 1% of FICO's mean |SHAP|. If they ranked high, that would point to leakage or a fitting bug.

**Temporal split is the production methodology.** Random splits let a model see same-window information in train and test. The synthetic data has a built-in 2018 regime change; on it, the random/OOT gap is 6-7 PR-AUC points. On real LC data the gap is usually larger.

## Real Lending Club data

Default mode is synthetic. To run on real LC data:

1. Drop an LC accepted-loans CSV into `data/raw/`.
2. Call `load_data(source="lendingclub")`. Returns the same 17-column schema as `source="synthetic"`.

The loader handles LC's quirks: `term` suffix (`" 36 months"` -> `36`), percent strings (`"12.45%"` -> `12.45`), `emp_length` variants (`"10+ years"`, `"< 1 year"`, `"n/a"`, NaN), `issue_d` parsing (`"Dec-2014"` -> `2014-12-01`), and the `loan_status` to binary `default` mapping. `tests/test_lendingclub_swap.py` covers each parsing case on a fixture, so you don't need the ~1 GB CSV to verify the loader works.

## Limitations

The synthetic DGP is largely additive in the true features, which is why the linear baseline runs essentially tied with tuned LightGBM here. Real LC data has interactions and time effects the linear model cannot represent, but on this dataset that gap doesn't show up.

The fairness audit uses a synthetic `protected_group` correlated with income. The framework ports to real protected-class data, but the disparities reported are by construction.

Default is treated as a binary outcome. For pricing or capital allocation, time-to-event (survival) modeling is the correct frame.

The temporal validation injects a constructed 2018 regime change. Real degradation patterns are messier and feature-distribution drift is usually larger than the synthetic version captures.

## Reproducibility

`RANDOM_STATE = 42` is threaded through every random operation: synthetic generation, train/test splits, model initialization, randomized search, SHAP sampling, and recalibration folds. Dependencies are pinned in `requirements.txt`. `make verify` wipes generated state and rebuilds end-to-end; the numbers in this README match a fresh run.

## Future work

- Run the pipeline on actual Lending Club CSVs. The swap path is fixture-tested but not yet exercised on real data.
- Survival or hazard modeling for time-to-default rather than the binary outcome.
- Fairness-constrained training (e.g., exponentiated gradient) with a paired before/after audit.
- A deployment wrapper (FastAPI plus MLflow) for the trained pipeline.

## License

Personal portfolio project. Code can be referenced; please don't redistribute the trained artifacts.
