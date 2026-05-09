# Loan Default Prediction with Tabular Deep Learning

PyTorch FT-Transformer and MLP for credit risk on a Lending Club style dataset, benchmarked against logistic regression, random forest, and a tuned LightGBM. Plus the things a credit team actually needs around the model: cost-aware threshold selection, SHAP explanations, calibration, drift monitoring, fairness checks, and out-of-time validation.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.8-red.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6-green.svg)
![tests](https://img.shields.io/badge/tests-63%20passing-brightgreen.svg)

## Results

Five models on the same 80/20 stratified split (10k held-out test, 18% positive prevalence):

| Model | PR-AUC | ROC-AUC | Train time |
|---|---:|---:|---:|
| Logistic regression (balanced) | 0.745 | 0.924 | <1s |
| Random forest (300 trees) | 0.721 | 0.916 | 4s |
| LightGBM (tuned, 30 RandomizedSearchCV iters) | 0.740 | 0.922 | ~7 min |
| MLP (PyTorch, embeddings + 3-layer) | 0.739 | 0.921 | 16s on MPS |
| FT-Transformer (PyTorch, from scratch) | 0.737 | 0.921 | 102s on MPS |

All five within ~0.4 PR-AUC points. This matches the published finding (Grinsztajn et al. 2022) that gradient-boosted trees are the right default on small/medium tabular data; deep learning earns its place once you have multimodal features, very high-cardinality categoricals, or an order of magnitude more data.

Out-of-time validation tells a different story. Train on cohorts through 2017-06, test on 2018, and PR-AUC drops to 0.65. That 6-7 point gap is the realistic production number; random splits hide it.

![Random vs out-of-time degradation](reports/figures/temporal_degradation.png)

## Quickstart

```bash
make install         # creates .venv, installs pinned deps (incl. PyTorch)
make test            # 63 tests
make smoke           # 5k row training run, ~90s
make train           # full random-split run for LR / RF / tuned LightGBM (~7 min)
make train-temporal  # OOT comparison (~30s)
make train-dl        # MLP + FT-Transformer in PyTorch, ~3-5 min on MPS / longer on CPU
make notebooks       # execute all 8 notebooks in order
```

`make all` chains install, test, train, train-temporal, train-dl, notebooks. `make verify` wipes generated state and rebuilds end-to-end. `make help` lists every target.

## What's in here

`src/` modules:
- `synth.py` synthetic data generator with optional macro and regime trends
- `data.py` `load_data`, `temporal_split`, and the Lending Club CSV swap
- `features.py` `ColumnTransformer` plus four engineered ratio features
- `models.py` LR, RF, and LightGBM Pipeline factories
- `train.py` random-split and `--temporal` training orchestration
- `tabular_dl.py` PyTorch MLP and FT-Transformer (from-scratch implementation)
- `tabular_dl_train.py` training loop with AdamW, cosine LR, early stopping
- `evaluate.py` PR-AUC, ROC-AUC, cost-aware threshold sweep, sensitivity
- `interpret.py` SHAP wrapper around the sklearn pipeline
- `errors.py` per-subgroup error analysis with FNR-lift
- `calibration.py` reliability tables, ECE, Platt and isotonic recalibrators
- `drift.py` Population Stability Index per feature
- `fairness.py` group metrics, parity ratios, four-fifths rule
- `monitor.py` rolling performance with configurable label-resolution lag

`notebooks/`:
- `01_eda.ipynb` distributions, correlations, default rate by category
- `02_modeling.ipynb` GBM bake-off, cost-aware threshold, FN:FP sensitivity
- `03_interpretation.ipynb` SHAP global, directional, and per-loan
- `04_calibration.ipynb` reliability diagrams plus Platt, isotonic, and CV-on-train recalibration
- `05_drift.ipynb` PSI on simulated shifts, plus the label-delay performance monitor
- `06_fairness.ipynb` per-group approval rate, FNR, FPR, four-fifths check
- `07_temporal.ipynb` random vs out-of-time split comparison
- `08_tabular_dl.ipynb` MLP and FT-Transformer in PyTorch vs LightGBM, plus DL-side audits

`tests/`: 63 pytest tests. Coverage includes the leakage guard, threshold selection, calibration, drift, fairness, monitor, the LC swap fixture, `temporal_split`, and DL forward / backward / overfit-tiny-batch sanity.

## Why these choices

**PyTorch tabular DL from scratch.** FT-Transformer is implemented end-to-end (numerical tokenizer, categorical tokenizer, [CLS] prepending, pre-norm Transformer blocks, attention via `nn.MultiheadAttention`) rather than via a library wrapper. About 200 lines, easier to defend in an interview than `pytorch_tabular.models.FTTransformer(...)`.

**MLP next to FT-Transformer.** A vanilla embedding-MLP is the right floor for "did attention buy us anything." On this DGP it didn't; the MLP matches the Transformer at a fraction of the train time. That's a real finding worth showing.

**Apple Silicon MPS by default.** `tabular_dl.get_device()` prefers MPS, then CUDA, then CPU. Training the FT-Transformer on the full 50k dataset takes ~100 seconds on MPS; CPU is ~5x slower.

**`pos_weight` in BCE for class imbalance.** The 18% positives are handled at the loss level rather than via SMOTE / oversampling. Side effect: the resulting probabilities are inflated (Brier 0.125 for MLP vs 0.081 for LightGBM). Notebook 08 calls this out and recommends recalibrating before the cost-aware threshold; the existing Platt/isotonic helpers in `calibration.py` apply directly.

**PR-AUC, not accuracy.** With 18% positives, predicting "no default" for every row scores 82% accuracy. PR-AUC focuses on the rare class.

**Cost-aware threshold, not 0.5.** Missing a default costs roughly the loan principal; rejecting a good loan costs the foregone interest margin. The 5:1 FN:FP ratio puts the optimal threshold near 0.14, far from 0.5, and cuts expected loss by 33.9% on the LightGBM model.

**Tune the boosting model only.** LR has little to tune and RF defaults are reasonable. Spending the search budget on LightGBM, which benefits most, keeps the comparison fair.

**Pipeline-everywhere preprocessing.** Imputation, scaling, and one-hot encoding live inside an sklearn Pipeline so they refit per CV fold. A unit test enforces this. The PyTorch dataset uses an analogous `fit_spec(train_df)` that computes categorical vocabularies and per-feature standardization stats from train alone.

**Two intentional noise features in the DGP.** SHAP on LightGBM and permutation importance on FT-Transformer both rank them near the bottom (PR-AUC drop < 0.002 for the DL case). If they ranked high, that would point to leakage or a fitting bug.

**Temporal split is the production methodology.** Random splits let a model see same-window information in train and test. The synthetic data has a built-in 2018 regime change; on it, the random/OOT gap is 6-7 PR-AUC points. On real LC data the gap is usually larger.

## Real Lending Club data

Default mode is synthetic. To run on real LC data:

1. Drop an LC accepted-loans CSV into `data/raw/`.
2. Call `load_data(source="lendingclub")`. Returns the same 17-column schema as `source="synthetic"`.

The loader handles LC's quirks: `term` suffix, percent strings, `emp_length` variants, `issue_d` parsing, and the `loan_status` to binary `default` mapping. `tests/test_lendingclub_swap.py` covers each parsing case on a fixture, so you don't need the ~1 GB CSV to verify the loader works.

## Limitations

The synthetic DGP is largely additive in the true features, which is why LR, LightGBM, and the two PyTorch models all land within noise of each other. On real LC data with messier interactions and time effects, the gap between the model families typically opens up.

The fairness audit uses a synthetic `protected_group` correlated with income. The framework ports to real protected-class data, but the disparities reported are by construction.

Default is treated as a binary outcome. For pricing or capital allocation, time-to-event (survival) modeling is the correct frame.

The DL probabilities need recalibration for cost-sensitive deployment because of the `pos_weight` training. The recalibration helpers in `src/calibration.py` apply directly; this just isn't auto-applied in the DL training loop.

## Reproducibility

`RANDOM_STATE = 42` is threaded through every random operation: synthetic generation, train/test splits, model initialization, randomized search, SHAP sampling, recalibration folds, and PyTorch generators. Dependencies are pinned in `requirements.txt`. `make verify` wipes generated state and rebuilds end-to-end; the numbers in this README match a fresh run.

## Future work

- Add a calibration step inside the DL training loop (Platt or temperature scaling on the validation set) so the saved probabilities are usable in a cost-sensitive selector without a separate recalibration pass.
- Run the pipeline on actual Lending Club CSVs.
- Survival or hazard modeling for time-to-default rather than the binary outcome.
- Multimodal: add a `desc` field to the synthetic generator and pipe it through a small text encoder (DistilBERT or sentence-transformers) for a tabular + text bake-off.
- Fairness-constrained training (e.g., exponentiated gradient) with paired before / after audit.
- A deployment wrapper (FastAPI plus MLflow) for the trained pipeline.

## License

Personal portfolio project. Code can be referenced; please don't redistribute the trained artifacts.
