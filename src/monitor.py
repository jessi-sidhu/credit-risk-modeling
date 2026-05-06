"""Delayed-outcome performance monitor.

Real loans default 6–18 months after origination. By the time labels
arrive, the input distribution has often already moved on. This module
simulates that delay so we can plot performance over time as labels
roll in, the way a production team would actually see it.

`rolling_performance` computes per-window PR-AUC and Brier on loans
whose outcome would have resolved within the (cohort_month + lag,
cohort_month + lag + window_months] interval.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss


def rolling_performance(
    y_true,
    y_prob,
    issue_dates: Iterable,
    window_months: int = 3,
    lag_months: int = 12,
    min_positives: int = 5,
) -> pd.DataFrame:
    """Per-window PR-AUC and Brier for label-delayed monitoring.

    Args:
        y_true, y_prob: aligned arrays of length n.
        issue_dates: array-like of n datetime values (loan origination).
        window_months: width of each evaluation window.
        lag_months: months between origination and outcome resolution.
            Loans originated in month M have labels available at month
            M + lag_months — so a window centered at calendar month T
            evaluates loans originated at T - lag_months.
        min_positives: skip windows with fewer than this many positives;
            PR-AUC is unstable below it.

    Returns:
        DataFrame with one row per evaluation window: cohort start,
        cohort end, n loans, base rate, PR-AUC, Brier.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    issue = pd.to_datetime(pd.Series(issue_dates).reset_index(drop=True))

    cohort = issue.dt.to_period("M")
    months = pd.period_range(cohort.min(), cohort.max(), freq="M")

    rows = []
    for i in range(0, len(months) - window_months + 1, window_months):
        start = months[i]
        end = months[i + window_months - 1]
        mask = ((cohort >= start) & (cohort <= end)).to_numpy()
        if mask.sum() == 0 or y_true[mask].sum() < min_positives:
            continue
        evaluation_month = end + lag_months
        rows.append({
            "cohort_start": str(start),
            "cohort_end": str(end),
            "evaluation_month": str(evaluation_month),
            "n": int(mask.sum()),
            "n_positives": int(y_true[mask].sum()),
            "base_rate": float(y_true[mask].mean()),
            "pr_auc": float(average_precision_score(y_true[mask], y_prob[mask])),
            "brier": float(brier_score_loss(y_true[mask], y_prob[mask])),
        })
    return pd.DataFrame(rows)
