"""Probability calibration utilities.

Used to answer: when the model says 0.30, do 30% of those loans actually
default? Two recalibrators are provided — Platt (sigmoid, parametric)
and isotonic (non-parametric, monotonic step function).

Calibration is fit on a held-out portion of the test set so the model
itself has never seen those rows. The other half is used to evaluate
the recalibrated predictions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

_EPS = 1e-6


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def reliability_table(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    """Bin predicted probabilities (quantile-based) and report the
    empirical default rate per bin alongside the mean prediction."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    prob_pred, prob_true = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )

    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    counts = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        counts.append(int(mask.sum()))
    counts = counts[: len(prob_pred)]

    return pd.DataFrame({
        "mean_predicted": prob_pred,
        "fraction_positive": prob_true,
        "count": counts,
    })


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    """Sum of |predicted - empirical| over bins, weighted by bin size."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    n = len(y_true)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_prob >= edges[i]) & (y_prob <= edges[i + 1])
        else:
            mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if not mask.any():
            continue
        bin_pred = y_prob[mask].mean()
        bin_true = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_pred - bin_true)
    return float(ece)


def fit_platt(y_prob, y_true) -> LogisticRegression:
    """Fit a sigmoid calibrator on (logit(p), label)."""
    y_prob = np.asarray(y_prob, dtype=float)
    y_true = np.asarray(y_true).astype(int)
    lr = LogisticRegression()
    lr.fit(_logit(y_prob).reshape(-1, 1), y_true)
    return lr


def apply_platt(calibrator: LogisticRegression, y_prob) -> np.ndarray:
    y_prob = np.asarray(y_prob, dtype=float)
    return calibrator.predict_proba(_logit(y_prob).reshape(-1, 1))[:, 1]


def fit_isotonic(y_prob, y_true) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(np.asarray(y_prob, dtype=float), np.asarray(y_true).astype(int))
    return iso


def apply_isotonic(calibrator: IsotonicRegression, y_prob) -> np.ndarray:
    return calibrator.predict(np.asarray(y_prob, dtype=float))


def calibration_metrics(y_true, y_prob) -> dict:
    return {
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
    }
