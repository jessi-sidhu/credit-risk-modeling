"""Metrics, threshold sweep, and cost-aware threshold selection.

Primary metric is PR-AUC (`average_precision_score`) — appropriate for
the imbalanced binary task. Threshold is picked by minimizing expected
business cost under a configurable FN:FP ratio rather than by maximizing
F1, because the two error types are not equally expensive.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


@dataclass(frozen=True)
class CostMatrix:
    """FN cost ≈ loan principal lost; FP cost ≈ foregone interest margin.

    Default 5:1 reflects a rough industry rule of thumb. Sensitivity
    analysis (vary fn_cost / fp_cost) is part of §13 future work.
    """
    fn_cost: float = 5.0
    fp_cost: float = 1.0


def expected_cost(y_true, y_prob, threshold: float, cost: CostMatrix = CostMatrix()) -> float:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    fn = int(((y_true == 1) & (pred == 0)).sum())
    fp = int(((y_true == 0) & (pred == 1)).sum())
    return fn * cost.fn_cost + fp * cost.fp_cost


def threshold_sweep(
    y_true,
    y_prob,
    cost: CostMatrix = CostMatrix(),
    n_thresholds: int = 91,
) -> pd.DataFrame:
    thresholds = np.linspace(0.05, 0.95, n_thresholds)
    rows = []
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    for t in thresholds:
        pred = (y_prob >= t).astype(int)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, pred, average="binary", zero_division=0
        )
        rows.append({
            "threshold": float(t),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "expected_cost": expected_cost(y_true, y_prob, t, cost),
        })
    return pd.DataFrame(rows)


def cost_optimal_threshold(y_true, y_prob, cost: CostMatrix = CostMatrix()) -> float:
    sweep = threshold_sweep(y_true, y_prob, cost)
    return float(sweep.loc[sweep["expected_cost"].idxmin(), "threshold"])


def evaluate_at_threshold(y_true, y_prob, threshold: float) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    pred = (y_prob >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    cm = confusion_matrix(y_true, pred, labels=[0, 1])
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }
