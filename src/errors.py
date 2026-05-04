"""Error-analysis helpers: per-subgroup error rates."""
from __future__ import annotations

import numpy as np
import pandas as pd


def confusion_groups(y_true, y_prob, threshold: float) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    out = np.where(
        (y_true == 1) & (pred == 1), "TP",
        np.where(
            (y_true == 0) & (pred == 0), "TN",
            np.where((y_true == 0) & (pred == 1), "FP", "FN"),
        ),
    )
    return out


def subgroup_error_rates(
    X: pd.DataFrame,
    y_true,
    y_prob,
    threshold: float,
    by: str,
) -> pd.DataFrame:
    """Per-value-of-`by`: count, FN/FP/TP/TN, false-negative rate (of
    positives), false-positive rate (of negatives), and lift over the
    overall FNR."""
    groups = confusion_groups(y_true, y_prob, threshold)
    df = pd.DataFrame({by: X[by].to_numpy(), "g": groups})
    pivot = (
        df.assign(_n=1)
        .pivot_table(index=by, columns="g", values="_n", aggfunc="sum", fill_value=0)
    )
    for k in ("TP", "FN", "FP", "TN"):
        if k not in pivot.columns:
            pivot[k] = 0
    pivot["n"] = pivot[["TP", "FN", "FP", "TN"]].sum(axis=1)
    pos = pivot["TP"] + pivot["FN"]
    neg = pivot["FP"] + pivot["TN"]
    pivot["fnr"] = np.where(pos > 0, pivot["FN"] / pos, np.nan)
    pivot["fpr"] = np.where(neg > 0, pivot["FP"] / neg, np.nan)

    overall_fnr = pivot["FN"].sum() / max(pos.sum(), 1)
    pivot["fnr_lift"] = pivot["fnr"] / overall_fnr if overall_fnr > 0 else np.nan

    return pivot[["n", "TP", "FN", "FP", "TN", "fnr", "fpr", "fnr_lift"]].sort_values("n", ascending=False)
