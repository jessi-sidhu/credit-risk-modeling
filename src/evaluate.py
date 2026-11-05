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
    brier_score_loss,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .calibration import expected_calibration_error


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


# --- Headline metrics ---------------------------------------------------------

def ks_statistic(y_true, y_prob) -> float:
    """Kolmogorov-Smirnov: max gap between the score CDFs of defaulters and
    non-defaulters, i.e. max(TPR - FPR) over thresholds."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob))
    return float(np.max(tpr - fpr))


def brier_score(y_true, y_prob) -> float:
    """Mean squared error of the predicted probabilities (lower is better)."""
    return float(brier_score_loss(np.asarray(y_true).astype(int), np.asarray(y_prob)))


def headline_metrics(y_true, y_prob) -> dict:
    """ROC-AUC, Gini (= 2*AUC - 1), KS, PR-AUC, Brier, ECE."""
    auc = float(roc_auc_score(y_true, y_prob))
    return {
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks_statistic(y_true, y_prob),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob),
    }


# --- Bootstrap confidence intervals -------------------------------------------
# Resampling the test set with replacement is the same as giving each row a
# multinomial count. Sorting each model's scores once and computing AUC / AP
# from weighted cumulative counts is exact (ties handled like sklearn) and
# avoids re-sorting ~370k rows on every one of the 1,000 resamples.

def _sorted_for_bootstrap(y_true, y_prob) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_prob = np.asarray(y_prob, dtype=float)
    order = np.argsort(-y_prob, kind="mergesort")
    p_sorted = y_prob[order]
    # last index of each block of tied scores
    ends = np.r_[np.flatnonzero(np.diff(p_sorted)), len(p_sorted) - 1]
    return order, np.asarray(y_true).astype(float)[order], ends


def _weighted_auc_ap(y_sorted, w_sorted, ends) -> tuple[float, float]:
    tp = np.cumsum(w_sorted * y_sorted)[ends]
    fp = np.cumsum(w_sorted * (1.0 - y_sorted))[ends]
    tpr = tp / tp[-1]
    fpr = fp / fp[-1]
    auc = np.trapz(np.r_[0.0, tpr], np.r_[0.0, fpr])
    # blocks with zero weight in this resample add no recall; avoid 0/0
    seen = tp + fp
    precision = np.divide(tp, seen, out=np.zeros_like(tp), where=seen > 0)
    ap = np.sum(np.diff(np.r_[0.0, tpr]) * precision)
    return float(auc), float(ap)


def bootstrap_ci(
    y_true,
    probs: dict[str, np.ndarray],
    diff: tuple[str, str] | None = None,
    n_boot: int = 1_000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict:
    """Percentile bootstrap CIs for ROC-AUC and PR-AUC of each model, and
    (paired, same resamples) for the difference `diff[0] - diff[1]`.

    Returns {name: {metric: {"estimate", "ci_low", "ci_high"}}, plus
    "<a> - <b>" for the difference when `diff` is given.
    """
    y_true = np.asarray(y_true).astype(int)
    n = len(y_true)
    rng = np.random.default_rng(seed)
    prepared = {name: _sorted_for_bootstrap(y_true, p) for name, p in probs.items()}
    draws = {name: np.empty((n_boot, 2)) for name in probs}

    for b in range(n_boot):
        w = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
        for name, (order, y_s, ends) in prepared.items():
            draws[name][b] = _weighted_auc_ap(y_s, w[order], ends)

    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)

    def _summ(point: np.ndarray, samples: np.ndarray) -> dict:
        return {
            metric: {
                "estimate": float(point[i]),
                "ci_low": float(np.percentile(samples[:, i], lo)),
                "ci_high": float(np.percentile(samples[:, i], hi)),
            }
            for i, metric in enumerate(("roc_auc", "pr_auc"))
        }

    points = {
        name: np.array([roc_auc_score(y_true, p), average_precision_score(y_true, p)])
        for name, p in probs.items()
    }
    out = {name: _summ(points[name], draws[name]) for name in probs}
    if diff is not None:
        a, b = diff
        out[f"{a} - {b}"] = _summ(points[a] - points[b], draws[a] - draws[b])
    return out

