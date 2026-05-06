.PHONY: help install test smoke train notebooks all clean clean-all verify

PY := .venv/bin/python
JUPYTER := .venv/bin/jupyter

help:
	@echo "Targets:"
	@echo "  install     create .venv and install pinned deps"
	@echo "  test        run pytest"
	@echo "  smoke       quick training run (5k rows, 10 tuning iters, < 90s)"
	@echo "  train       full training run with 30 tuning iters (~7 min)"
	@echo "  train-temporal  random vs out-of-time split comparison (~30s)"
	@echo "  notebooks   execute all notebooks in order, in-place"
	@echo "  all         install -> test -> train -> notebooks"
	@echo "  verify      wipe generated artifacts, then run all (full reproduce)"
	@echo "  clean       remove generated figures, artifacts, and cached CSV"
	@echo "  clean-all   clean + remove .venv"

install:
	python3 -m venv .venv
	$(PY) -m pip install --quiet --upgrade pip
	$(PY) -m pip install --quiet -r requirements.txt

test:
	$(PY) -m pytest -q

smoke:
	$(PY) -m src.train --quick

train:
	$(PY) -m src.train --save

train-temporal:
	$(PY) -m src.train --temporal --save

notebooks:
	$(JUPYTER) nbconvert --to notebook --execute notebooks/01_eda.ipynb            --inplace --ExecutePreprocessor.timeout=600
	$(JUPYTER) nbconvert --to notebook --execute notebooks/02_modeling.ipynb       --inplace --ExecutePreprocessor.timeout=600
	$(JUPYTER) nbconvert --to notebook --execute notebooks/03_interpretation.ipynb --inplace --ExecutePreprocessor.timeout=600
	$(JUPYTER) nbconvert --to notebook --execute notebooks/04_calibration.ipynb    --inplace --ExecutePreprocessor.timeout=600
	$(JUPYTER) nbconvert --to notebook --execute notebooks/07_temporal.ipynb       --inplace --ExecutePreprocessor.timeout=600
	$(JUPYTER) nbconvert --to notebook --execute notebooks/05_drift.ipynb          --inplace --ExecutePreprocessor.timeout=600
	$(JUPYTER) nbconvert --to notebook --execute notebooks/06_fairness.ipynb       --inplace --ExecutePreprocessor.timeout=600

all: install test train train-temporal notebooks

clean:
	rm -rf reports/artifacts/*.joblib reports/artifacts/*.csv \
	       reports/artifacts/*.npy reports/artifacts/*.json \
	       reports/artifacts/temporal/* data/synthetic/*.csv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ipynb_checkpoints -exec rm -rf {} + 2>/dev/null || true

clean-all: clean
	rm -rf .venv

verify: clean
	$(MAKE) test
	$(MAKE) train
	$(MAKE) train-temporal
	$(MAKE) notebooks
	@echo "✔ verify: full pipeline reproduced from scratch"
