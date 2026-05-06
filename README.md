# Loan Default Prediction

> Production-style credit-risk pipeline: predicts loan default probability on a Lending-Club-shaped dataset, then handles every concern a real bank would care about — cost-aware thresholding, SHAP interpretability, calibration, drift monitoring, fairness, and out-of-time validation.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-0.49-yellow.svg)
![tests](https://img.shields.io/badge/tests-56%20passing-brightgreen.svg)
![reproducible](https://img.shields.io/badge/reproducible-make%20verify-success.svg)

---

## Headline results

| Metric | Value | What it means |
|---|---:|---|
| **PR-AUC** | **0.74** | Tuned LightGBM on a held-out 10k test set with 18% positive prevalence |
| **ROC-AUC** | **0.92** | Strong ranking quality across the threshold range |
| **Expected-loss reduction** | **33.9%** | Moving from the default 0.5 threshold to the cost-optimal 0.14 under FN:FP = 5:1 |
| **OOT degradation** | **6–7 pp PR-AUC** | Honest gap between random and out-of-time splits — the kind of number a bank would ship |
| **Calibration** | **Brier 0.083, ECE 0.95%** | Probabilities trustable as inputs to expected-loss / pricing models |

![Random vs out-of-time degradation](reports/figures/temporal_degradation.png)

*Random splits overstate production performance. Temporal validation tells you what you'll actually ship.*

---

## What this project does

1. **Generates** a 50k-row synthetic loan dataset modeled on Lending Club (deterministic, calibrated to 18% default rate, with two pure-noise features included so SHAP can be verified against ground truth).
2. **Trains** three models in escalating complexity — logistic regression (balanced), random forest, tuned LightGBM — under both random and out-of-time splits.
3. **Evaluates** on PR-AUC primary / ROC-AUC secondary, then picks an operating threshold by minimizing expected business cost rather than maximizing F1.
4. **Audits** the chosen model on every dimension a regulator or production team would ask about: SHAP interpretability, per-subgroup error analysis, probability calibration, input-side drift (PSI), output-side drift (rolling performance with label lag), and disparate impact / fairness parity.
5. **Stays honest** about limits — explicitly calls out where the linear baseline ties the boosting model, where calibration recalibrators don't help, and where group fairness criteria provably can't all hold at once.

The whole pipeline is one `make all` away. Switching to real Lending Club data is one function-argument change.

---

## Tech stack

**Modeling:** scikit-learn (LR, RF, ColumnTransformer, Pipelines, RandomizedSearchCV) · LightGBM · SHAP

**Engineering:** Python 3.9 · pandas · NumPy · pytest · Jupyter · Make

**Reproducibility:** Single `RANDOM_STATE = 42` threaded through splits, models, and bootstrap; deterministic synthetic generator; pinned dependencies; `make verify` reproduces the entire pipeline from a wiped state.

---

## At-a-glance results gallery

<table>
  <tr>
    <td align="center">
      <strong>PR curves — three models</strong><br>
      <img src="reports/figures/pr_curves.png" width="380">
    </td>
    <td align="center">
      <strong>Cost vs threshold</strong><br>
      <img src="reports/figures/threshold_sweep.png" width="380">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>SHAP global importance</strong><br>
      <img src="reports/figures/shap_bar.png" width="380">
    </td>
    <td align="center">
      <strong>SHAP directional (beeswarm)</strong><br>
      <img src="reports/figures/shap_beeswarm.png" width="380">
    </td>
  </tr>
  <tr>
    <td align="center">
      <strong>PSI drift detection</strong><br>
      <img src="reports/figures/psi_scenarios.png" width="380">
    </td>
    <td align="center">
      <strong>Fairness parity by group</strong><br>
      <img src="reports/figures/fairness_metrics.png" width="380">
    </td>
  </tr>
</table>

Full discussion in [`reports/report.md`](reports/report.md).

---

## What this project demonstrates

| Capability | Where to look |
|---|---|
| **Defensible metric choice** — PR-AUC over accuracy on 18% prevalence; cost-aware threshold over the arbitrary 0.5 default | `src/evaluate.py`, notebook 02 |
| **Leakage discipline** — all preprocessing fit on the training fold via `Pipeline`; structural unit test enforces it | `src/features.py`, `tests/test_features.py` |
| **Sanity-checkable interpretability** — synthetic noise features verified to have ~1% of FICO's mean \|SHAP\| | `src/interpret.py`, notebook 03 |
| **Production hygiene** — calibration, PSI input drift, label-delay performance monitor | `src/{calibration,drift,monitor}.py`, notebooks 04, 05 |
| **Fairness awareness** — four-fifths rule + the demographic-parity / equal-opportunity / predictive-parity trilemma named explicitly | `src/fairness.py`, notebook 06 |
| **Out-of-time validation** — train ≤2017-06, test ≥2018-01 against a regime-change-injected DGP, showing 6–7 pp PR-AUC degradation vs random split | `src/{synth,train}.py`, notebook 07 |
| **Reusable swap path** — `load_data(source="lendingclub")` reads real LC CSVs and emits the same 17-column schema, validated by a fixture-based test | `src/data.py`, `tests/test_lendingclub_swap.py` |
| **Honest trade-offs** — linear baseline ties tuned LightGBM here (DGP is additive); recalibration is within noise (model is intrinsically calibrated). Discussed openly. | `reports/report.md` §6, §13b |

---

## Quickstart

Clone and run end-to-end with one command:

```bash
git clone <repo-url> credit-risk-modeling
cd credit-risk-modeling
make all              # install -> test -> train -> train-temporal -> notebooks (~10 min)
```

Or step by step:

```bash
make install          # creates .venv, installs pinned deps
make test             # runs all 56 tests, < 10s
make smoke            # quick training run, ~90s — sanity check
make train            # full random-split run (~7 min on a laptop)
make train-temporal   # OOT comparison (~30s, reuses tuned hyperparameters)
make notebooks        # execute all seven notebooks in order, in-place
make verify           # wipe generated state and reproduce from scratch
make help             # list all targets
```

---

## Repository tour

```
credit-risk-modeling/
│
├── src/                              # Importable, tested modules
│   ├── synth.py                      # Synthetic data generator (deterministic, optional macro / regime trend)
│   ├── data.py                       # load_data, temporal_split, Lending Club CSV swap
│   ├── features.py                   # ColumnTransformer + 4 engineered ratio features
│   ├── models.py                     # Pipeline factories: LR, RF, LightGBM
│   ├── train.py                      # Train + RandomizedSearchCV; --temporal mode
│   ├── evaluate.py                   # PR-AUC, ROC-AUC, cost-aware thresholds, sensitivity sweep
│   ├── interpret.py                  # SHAP wrapper for sklearn pipelines
│   ├── errors.py                     # Per-subgroup FN/FP analysis with FNR-lift
│   ├── calibration.py                # Reliability table, ECE, Platt/isotonic, CV-on-train
│   ├── drift.py                      # PSI per feature, stable/moderate/significant labels
│   ├── fairness.py                   # group_metrics, parity ratios, four-fifths rule
│   └── monitor.py                    # Rolling PR-AUC / Brier with label-resolution lag
│
├── notebooks/                        # Narrative walkthroughs (executed, outputs included)
│   ├── 01_eda.ipynb                  # Class balance, distributions, correlations
│   ├── 02_modeling.ipynb             # 3-model bake-off, cost-aware threshold, sensitivity
│   ├── 03_interpretation.ipynb       # SHAP global + local + error subgroups
│   ├── 04_calibration.ipynb          # Reliability diagrams + recalibration
│   ├── 05_drift.ipynb                # PSI scenarios + label-delay monitor
│   ├── 06_fairness.ipynb             # Per-group parity + 4/5 rule
│   └── 07_temporal.ipynb             # Random vs out-of-time validation
│
├── tests/                            # 56 tests — leakage guard, parity, calibration, drift, ...
│
├── reports/
│   ├── report.md                     # Full writeup with results tables and figures
│   ├── figures/                      # 19 PNGs (tracked in git so the report renders on a fresh clone)
│   └── artifacts/                    # Trained model + predictions (regenerated by training)
│
├── data/
│   ├── synthetic/                    # Generated CSV (gitignored)
│   └── raw/                          # Drop real Lending Club CSV here for source="lendingclub"
│
├── Makefile                          # One-command pipeline orchestration
├── requirements.txt                  # Pinned deps
└── README.md                         # This file
```

---

## Engineering decisions

A few choices worth defending:

**Synthetic data with two pure-noise features.** The data-generating process is intentionally known. The two `noise_1`/`noise_2` columns — pure i.i.d. normals — let us verify SHAP isn't hallucinating importance. Real Lending Club data swaps in via one function-argument change.

**PR-AUC as primary metric.** With 18% positive class, accuracy is meaningless (an "always-paid" model scores 82%). PR-AUC focuses on the rare class and is threshold-independent.

**Cost-aware threshold over F1.** Missing a default costs ~5× as much as rejecting a good loan (loan principal vs foregone interest margin). The cost-optimal threshold (0.14) sits well below 0.5 and reduces expected loss by 33.9%. Operating-point selection should reflect the cost matrix, not optimize a generic statistic.

**Pipeline-everywhere preprocessing.** Imputation, scaling, and one-hot encoding live inside an sklearn `Pipeline` so they're refit per CV fold. A unit test (`test_features.py::test_preprocessor_fits_on_train_only`) enforces this structurally.

**Tune the boosting model only.** Logistic regression has little to tune; RF defaults work well. Spending the tuning budget on the model that benefits most is more honest than tuning all three to the same depth and pretending the comparison is symmetric.

**Temporal split over random split.** Random splits overstate production performance because they let the model see information from the same time window in both train and test. The 6–7 pp PR-AUC gap between the two methodologies on this synthetic data is the gap a real bank would not get to hide.

**Track figures in git, not artifacts.** Figures are small and let the GitHub-rendered report look right on a fresh clone. The trained model (~1 MB) and prediction arrays are regeneratable in 7 minutes via `make train`, so they stay gitignored.

---

## Honest limitations

The project ships with [`reports/report.md` §11–13](reports/report.md) calling out limits explicitly. The big ones:

- **Synthetic data.** Real Lending Club has messier missingness, label-leakage risk from `loan_status` derivative columns, and feature-distribution drift that the synthetic DGP only partially captures. The fixture test confirms the swap path works; running it on real LC data is the natural next step.
- **No survival/hazard modeling.** Default is treated as a single binary outcome, not a time-to-event problem. For pricing or capital allocation, hazard modeling is the correct frame.
- **Fairness audit on a synthetic protected attribute.** The audit framework ports directly to real protected-class data, but the disparities reported here are by construction, not real-world.

---

## Reproducibility & determinism

- A single `RANDOM_STATE = 42` is threaded through every random operation: synthetic generation, train/test splits, model initialization, randomized search, SHAP sampling, recalibration folds.
- `make verify` wipes generated state (figures aside) and rebuilds the entire pipeline. Final numbers match this README to the last decimal.
- Dependencies are pinned in `requirements.txt` to compatible-version bounds.
- Notebook outputs are committed alongside source so reviewers can read the work without running anything.

---

## Future work

- Run on real Lending Club data (the swap path is fixture-tested but not exercised on the ~1 GB CSV).
- Survival / hazard modeling for "*when* during the loan does default happen?"
- Fairness-constrained training (e.g., AdvDebias, exponentiated gradient) with a paired before/after audit.
- A FastAPI/MLflow wrapper for the trained pipeline to demonstrate the deploy step.

---

## License

Personal portfolio project. Code may be referenced; please don't redistribute the trained artifacts.
