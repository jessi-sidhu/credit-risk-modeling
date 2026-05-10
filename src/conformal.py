"""Conformal prediction for binary classification.

Split conformal prediction (SCP) gives distribution-free, finite-sample
coverage guarantees: with probability at least 1 - alpha, the
prediction set contains the true label. No assumptions about the
underlying model or data distribution beyond exchangeability.

For credit risk this means each loan can be assigned to one of:

    {paid}            confident the loan will be repaid
    {default}         confident the loan will default
    {paid, default}   the model isn't sure; route to manual review

The proportion of "ambiguous" sets tells you how much capacity the
model is honestly missing. Differential coverage by group tells you
whether the uncertainty is itself fair.

References:
    Vovk et al., Algorithmic Learning in a Random World (2005)
    Angelopoulos & Bates, A Gentle Introduction to Conformal
    Prediction and Distribution-Free Uncertainty Quantification (2023)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ConformalPredictor:
    """A fitted split conformal predictor for binary classification.

    `q_hat` is the (1 - alpha)-quantile of the calibration nonconformity
    scores. At test time, label y is included in the prediction set iff
    its nonconformity score is at most q_hat.
    """
    q_hat: float
    alpha: float
    n_calibration: int


def _nonconformity(y: np.ndarray, prob: np.ndarray) -> np.ndarray:
    """Score for the *true* label: 1 - p(true_label).
    Lower means the model was more confident in the right answer.
    """
    p_true = np.where(y == 1, prob, 1.0 - prob)
    return 1.0 - p_true


def fit(prob_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> ConformalPredictor:
    """Calibrate on a held-out set. Returns the threshold the predictor
    uses at test time.

    The finite-sample correction `(n+1)(1-alpha) / n` is what gives
    conformal its distribution-free guarantee; without it coverage is
    only approximate.
    """
    y_cal = np.asarray(y_cal).astype(int)
    prob_cal = np.asarray(prob_cal, dtype=float)
    scores = _nonconformity(y_cal, prob_cal)
    n = len(scores)
    # ceil((n+1)(1-alpha)) / n quantile, clipped to [0, 1]
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    q_hat = float(np.quantile(scores, q_level, method="higher"))
    return ConformalPredictor(q_hat=q_hat, alpha=alpha, n_calibration=n)


def predict_sets(prob_test: np.ndarray, predictor: ConformalPredictor) -> np.ndarray:
    """Return an (n, 2) bool array: column 0 = "paid in set", column 1
    = "default in set". Both True means the prediction set is {paid,
    default} — i.e., the model defers."""
    prob_test = np.asarray(prob_test, dtype=float)
    in_default = (1.0 - prob_test) <= predictor.q_hat   # nonconformity of label 1
    in_paid = prob_test <= predictor.q_hat              # nonconformity of label 0
    return np.stack([in_paid, in_default], axis=1)


def set_size(sets: np.ndarray) -> np.ndarray:
    """Number of labels in each prediction set (0, 1, or 2)."""
    return sets.sum(axis=1)


def coverage_metrics(y_true, sets: np.ndarray) -> dict:
    """Empirical coverage and average set size on a held-out test set."""
    y_true = np.asarray(y_true).astype(int)
    covered = sets[np.arange(len(y_true)), y_true]
    sizes = set_size(sets)
    return {
        "n": int(len(y_true)),
        "empirical_coverage": float(covered.mean()),
        "avg_set_size": float(sizes.mean()),
        "pct_singleton": float((sizes == 1).mean()),
        "pct_empty": float((sizes == 0).mean()),
        "pct_uncertain": float((sizes == 2).mean()),
    }


def coverage_by_group(y_true, sets: np.ndarray, group: Sequence) -> pd.DataFrame:
    """Per-group coverage and set-size statistics. If coverage drops
    below the target in some group, the model's uncertainty is
    discriminating across groups even though it nominally promises
    marginal coverage."""
    y_true = np.asarray(y_true).astype(int)
    g = pd.Series(np.asarray(group), name="group")
    rows = []
    for grp, idx in g.groupby(g, observed=True).groups.items():
        idx = list(idx)
        sub_sets = sets[idx]
        sub_y = y_true[idx]
        m = coverage_metrics(sub_y, sub_sets)
        rows.append({"group": grp, **m})
    return pd.DataFrame(rows).set_index("group")
