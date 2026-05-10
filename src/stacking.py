"""Stacking ensemble: a meta-learner over base-model probabilities.

Given probability predictions from K base models (here: LR, RF, LightGBM,
MLP, FT-Transformer) on a held-out set, fit a small logistic regression
meta-learner on those K columns plus the labels. The meta-learner learns
how to weight the base models against each other.

For a clean evaluation we split the held-out test set in half: half is
used as the meta-learner's training data (each base model's predictions
on those rows are out-of-sample because the base models never saw test);
the other half evaluates the stacked predictor.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _stack(probs_dict: dict[str, np.ndarray]) -> tuple[np.ndarray, list[str]]:
    names = sorted(probs_dict.keys())
    n = len(next(iter(probs_dict.values())))
    for k, v in probs_dict.items():
        if len(v) != n:
            raise ValueError(f"length mismatch: {k} has {len(v)} rows, expected {n}")
    X = np.stack([np.asarray(probs_dict[k], dtype=float) for k in names], axis=1)
    return X, names


def fit_meta(
    probs_dict: dict[str, np.ndarray],
    y: np.ndarray,
    C: float = 1.0,
    random_state: int = 42,
) -> tuple[LogisticRegression, list[str]]:
    """Fit a logistic regression meta-learner on (n, K) base-model probs.

    `C` defaults to 1.0 (sklearn default); lower values regularize more
    heavily, which is useful when the base models are strongly correlated.
    """
    X, names = _stack(probs_dict)
    y = np.asarray(y).astype(int)
    meta = LogisticRegression(C=C, max_iter=1000, random_state=random_state)
    meta.fit(X, y)
    return meta, names


def predict_meta(
    meta: LogisticRegression,
    probs_dict: dict[str, np.ndarray],
    names: list[str],
) -> np.ndarray:
    """Apply a fitted meta-learner to produce stacked probabilities. The
    `names` argument carries the column order used at fit time."""
    missing = set(names) - set(probs_dict.keys())
    if missing:
        raise KeyError(f"probs_dict missing base models: {missing}")
    X = np.stack([np.asarray(probs_dict[k], dtype=float) for k in names], axis=1)
    return meta.predict_proba(X)[:, 1]


def stack_weights(meta: LogisticRegression, names: list[str]) -> pd.Series:
    """Fitted meta-learner weights per base model, plus the intercept.
    Larger absolute value = more reliance on that base model."""
    return pd.Series(meta.coef_[0], index=names).sort_values(ascending=False)
