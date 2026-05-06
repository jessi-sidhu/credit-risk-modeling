import numpy as np
import pandas as pd

from src.monitor import rolling_performance


def _series(start: str, n: int, freq: str = "D") -> pd.Series:
    return pd.Series(pd.date_range(start, periods=n, freq=freq))


def test_perfect_predictions_brier_zero():
    n = 600
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, size=n)
    p = y.astype(float)
    issue = _series("2018-01-01", n, freq="D")
    out = rolling_performance(y, p, issue, window_months=3, lag_months=6)
    assert (out["brier"] == 0).all()
    assert (out["pr_auc"] == 1.0).all()


def test_returns_dataframe_with_expected_columns():
    n = 1_000
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.2, size=n)
    p = rng.uniform(0, 1, size=n)
    issue = _series("2018-01-01", n, freq="D")
    out = rolling_performance(y, p, issue, window_months=3, lag_months=6)
    assert {"cohort_start", "cohort_end", "evaluation_month",
            "n", "n_positives", "base_rate", "pr_auc", "brier"} <= set(out.columns)


def test_skips_windows_below_min_positives():
    rng = np.random.default_rng(2)
    n = 200
    y = np.zeros(n, dtype=int)  # no positives at all
    p = rng.uniform(0, 1, size=n)
    issue = _series("2018-01-01", n)
    out = rolling_performance(y, p, issue, window_months=2, lag_months=6, min_positives=5)
    assert len(out) == 0


def test_window_count_independent_of_lag():
    """Lag shifts the evaluation_month label but not the number of
    cohort windows or their content."""
    rng = np.random.default_rng(3)
    n = 800
    y = rng.binomial(1, 0.25, size=n)
    p = rng.uniform(0, 1, size=n)
    issue = _series("2018-01-01", n, freq="D")
    a = rolling_performance(y, p, issue, window_months=3, lag_months=6)
    b = rolling_performance(y, p, issue, window_months=3, lag_months=12)
    assert len(a) == len(b)
    pd.testing.assert_series_equal(a["pr_auc"], b["pr_auc"])
    assert (a["evaluation_month"] != b["evaluation_month"]).all()


def test_random_predictions_give_brier_near_pq():
    """Brier of a constant 0.5 prediction on a Bernoulli(p) target tends
    to (0.5 - p)^2 + p(1-p). For p ≈ 0.2: ~0.25."""
    rng = np.random.default_rng(4)
    n = 5_000
    y = rng.binomial(1, 0.2, size=n)
    p = np.full(n, 0.5)
    issue = _series("2018-01-01", n, freq="D")
    out = rolling_performance(y, p, issue, window_months=3, lag_months=6)
    assert (out["brier"] - 0.25).abs().max() < 0.05
