"""Population Stability Index — feature-level drift detection.

PSI compares two distributions (typically baseline vs production)
across the same set of bins:

    PSI = sum_i (q_i - p_i) * log(q_i / p_i)

Conventional thresholds (industry-standard for credit risk):

    PSI < 0.10   stable
    0.10–0.25    moderate shift, investigate
    PSI > 0.25   significant shift, retrain

The thresholds are heuristic; tune to your tolerance. Bin edges are
computed on the baseline so PSI is symmetric in interpretation but
asymmetric in computation (we ask: how much has the production
distribution moved away from baseline?).
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

_EPS = 1e-6


def _bin_numeric(baseline: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile bin edges from baseline. Last edge is +inf so trailing
    drift values are still binned."""
    edges = np.unique(np.quantile(baseline, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        edges = np.array([baseline.min(), baseline.max() + 1])
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _hist(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    p = counts / max(counts.sum(), 1)
    return np.where(p == 0, _EPS, p)


def psi_numeric(baseline, current, n_bins: int = 10) -> float:
    baseline = np.asarray(baseline, dtype=float)
    current = np.asarray(current, dtype=float)
    edges = _bin_numeric(baseline, n_bins)
    p = _hist(baseline, edges)
    q = _hist(current, edges)
    return float(np.sum((q - p) * np.log(q / p)))


def psi_categorical(baseline, current) -> float:
    """PSI for a single categorical column. All categories appearing in
    either input are included; missing categories get a tiny smoothing
    weight."""
    baseline = pd.Series(baseline).astype(str)
    current = pd.Series(current).astype(str)
    cats = sorted(set(baseline) | set(current))
    p = baseline.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    q = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    p = np.where(p == 0, _EPS, p)
    q = np.where(q == 0, _EPS, q)
    return float(np.sum((q - p) * np.log(q / p)))


def psi(baseline, current, n_bins: int = 10) -> float:
    """Dispatch on dtype: numeric -> psi_numeric, otherwise categorical."""
    arr = np.asarray(baseline)
    if np.issubdtype(arr.dtype, np.number):
        return psi_numeric(baseline, current, n_bins=n_bins)
    return psi_categorical(baseline, current)


def psi_report(
    baseline_df: pd.DataFrame,
    current_df: pd.DataFrame,
    columns: Iterable[str] | None = None,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Per-column PSI with a status label."""
    cols = list(columns) if columns is not None else list(baseline_df.columns)
    rows = []
    for c in cols:
        if c not in current_df.columns:
            continue
        score = psi(baseline_df[c].to_numpy(), current_df[c].to_numpy(), n_bins=n_bins)
        if score < 0.10:
            status = "stable"
        elif score < 0.25:
            status = "moderate"
        else:
            status = "significant"
        rows.append({"feature": c, "psi": score, "status": status})
    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )
