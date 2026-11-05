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
    # sklearn returns (prob_true, prob_pred) in that order
    prob_true, prob_pred = calibration_curve(
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


def cv_calibrated_probs(pipe, X, y, cv: int = 5, random_state: int = 42) -> np.ndarray:
    """Cross-validated train probabilities — methodologically cleaner
    inputs to a calibrator than the held-out-half-of-test approach.

    Each row of `X` gets a probability from a fold in which it was a
    held-out point, so the resulting probs are unbiased estimates of
    what the model would say on unseen data. The caller fits Platt or
    isotonic on `(cv_probs, y)` and applies to test predictions.
    """
    from sklearn.base import clone
    from sklearn.model_selection import StratifiedKFold

    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y_arr = np.asarray(y).astype(int)
    out = np.empty(len(y_arr), dtype=float)
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    for tr_idx, va_idx in skf.split(X, y_arr):
        m = clone(pipe)
        m.fit(X.iloc[tr_idx], y_arr[tr_idx])
        out[va_idx] = m.predict_proba(X.iloc[va_idx])[:, 1]
    return out


def plot_reliability(y_true, probs: dict[str, np.ndarray], path, n_bins: int = 10) -> None:
    """Reliability diagram (quantile bins) for several models on one axis,
    saved to `path`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834"]   # categorical slots 1-2, fixed order
    markers = ["o", "s"]              # secondary encoding, not color alone
    fig, ax = plt.subplots(figsize=(6.5, 6))
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")

    hi = 0.0
    for i, (name, y_prob) in enumerate(probs.items()):
        t = reliability_table(y_true, y_prob, n_bins=n_bins)
        ece = expected_calibration_error(y_true, y_prob, n_bins=n_bins)
        ax.plot(t["mean_predicted"], t["fraction_positive"], color=colors[i],
                marker=markers[i], markersize=8, lw=2,
                markeredgecolor="#fcfcfb", markeredgewidth=2,
                label=f"{name} (ECE {ece:.3f})")
        hi = max(hi, t["mean_predicted"].max(), t["fraction_positive"].max())

    lim = min(1.0, hi * 1.08)
    ax.plot([0, lim], [0, lim], color="#8a8a85", lw=1, ls="--", label="perfectly calibrated")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("Mean predicted default probability (per bin)")
    ax.set_ylabel("Observed default rate (per bin)")
    ax.set_title(f"Reliability on out-of-time test set ({n_bins} quantile bins)", loc="left")
    ax.grid(color="#e6e6e1", lw=0.8); ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=120, facecolor=fig.get_facecolor())
    plt.close(fig)

