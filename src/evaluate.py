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


def cost_sensitivity(
    y_true,
    y_prob,
    ratios: tuple[float, ...] = (2.0, 3.0, 5.0, 7.0, 10.0),
    fp_cost: float = 1.0,
) -> pd.DataFrame:
    """How does the optimal threshold (and the cost it achieves) move as
    the FN:FP ratio changes? One row per ratio.

    Used for the report's §13 sensitivity bullet — the 5:1 default is a
    rough industry heuristic and we want to show the operating point
    isn't fragile to it.
    """
    rows = []
    y_true_arr = np.asarray(y_true).astype(int)
    y_prob_arr = np.asarray(y_prob)
    cost_at_default_05 = expected_cost(y_true_arr, y_prob_arr, 0.5,
                                       CostMatrix(fn_cost=fp_cost, fp_cost=fp_cost))
    for ratio in ratios:
        cost = CostMatrix(fn_cost=ratio * fp_cost, fp_cost=fp_cost)
        sweep = threshold_sweep(y_true_arr, y_prob_arr, cost)
        idx = sweep["expected_cost"].idxmin()
        thr = float(sweep.loc[idx, "threshold"])
        cost_opt = float(sweep.loc[idx, "expected_cost"])
        cost_05 = expected_cost(y_true_arr, y_prob_arr, 0.5, cost)
        rows.append({
            "fn_to_fp": float(ratio),
            "optimal_threshold": thr,
            "cost_at_optimal": cost_opt,
            "cost_at_0.5": cost_05,
            "reduction_vs_0.5": (cost_05 - cost_opt) / cost_05 if cost_05 > 0 else 0.0,
            "precision": float(sweep.loc[idx, "precision"]),
            "recall": float(sweep.loc[idx, "recall"]),
            "f1": float(sweep.loc[idx, "f1"]),
        })
    return pd.DataFrame(rows)


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
