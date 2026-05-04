# Loan Default Prediction

Binary classification project: given a loan application, predict the probability of default.

Built on a synthetic dataset modeled on Lending Club, with a single swap point so the same pipeline can run on real Lending Club CSVs.

## Status

Work in progress. Roadmap:

- [x] Project scaffold
- [ ] Synthetic data generator
- [ ] EDA notebook
- [ ] Feature engineering pipeline
- [ ] Three models (LR, RF, LightGBM) with PR-AUC + cost-aware threshold
- [ ] SHAP interpretability and error analysis
- [ ] Final report

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest
```

## Layout

- `src/` — importable modules (data, features, models, evaluate, interpret)
- `notebooks/` — narrative EDA, modeling, interpretation
- `reports/` — final report and saved figures
- `tests/` — sanity tests including a leakage guard
