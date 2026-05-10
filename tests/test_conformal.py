import numpy as np
import pandas as pd

from src.conformal import (
    ConformalPredictor,
    coverage_by_group,
    coverage_metrics,
    fit,
    predict_sets,
    set_size,
)


def _synthetic_calibrated(n: int, prevalence: float, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.binomial(1, prevalence, size=n)
    # well-calibrated: predicted probability ~ true conditional prob
    p_true = rng.beta(2, 8, size=n)
    p_true = p_true * (y * 0.6 + (1 - y) * 0.05)  # higher for positives
    p_obs = np.clip(p_true + rng.normal(0, 0.05, size=n), 0.01, 0.99)
    return y, p_obs


def test_fit_returns_threshold_in_unit_interval():
    y, p = _synthetic_calibrated(2_000, 0.18, seed=0)
    pred = fit(p, y, alpha=0.1)
    assert isinstance(pred, ConformalPredictor)
    assert 0.0 <= pred.q_hat <= 1.0
    assert pred.n_calibration == 2_000


def test_predict_sets_shape_and_membership():
    y, p = _synthetic_calibrated(500, 0.18, seed=1)
    pred = fit(p, y, alpha=0.1)
    sets = predict_sets(p, pred)
    assert sets.shape == (500, 2)
    assert sets.dtype == bool
    assert set(set_size(sets).tolist()) <= {0, 1, 2}


def test_marginal_coverage_close_to_target():
    """Conformal guarantees coverage >= 1 - alpha. With enough cal/test
    points the empirical coverage should be within a few percent."""
    y_cal, p_cal = _synthetic_calibrated(5_000, 0.18, seed=2)
    y_te, p_te = _synthetic_calibrated(5_000, 0.18, seed=3)
    pred = fit(p_cal, y_cal, alpha=0.1)
    sets = predict_sets(p_te, pred)
    m = coverage_metrics(y_te, sets)
    # target coverage 0.9; conformal guarantees >=0.9 in expectation,
    # so empirical coverage should be at least ~0.88 (with a few percent slack)
    assert m["empirical_coverage"] >= 0.88, f"coverage {m['empirical_coverage']:.3f} below 0.88"
    # and shouldn't be wildly over-covered either
    assert m["empirical_coverage"] <= 0.99


def test_higher_alpha_gives_smaller_sets():
    y_cal, p_cal = _synthetic_calibrated(2_000, 0.18, seed=4)
    y_te, p_te = _synthetic_calibrated(2_000, 0.18, seed=5)

    pred_strict = fit(p_cal, y_cal, alpha=0.05)  # 95% coverage target
    pred_loose = fit(p_cal, y_cal, alpha=0.30)   # 70% coverage target

    avg_strict = coverage_metrics(y_te, predict_sets(p_te, pred_strict))["avg_set_size"]
    avg_loose = coverage_metrics(y_te, predict_sets(p_te, pred_loose))["avg_set_size"]
    assert avg_loose < avg_strict, f"loose {avg_loose:.3f} should be smaller than strict {avg_strict:.3f}"


def test_perfect_classifier_gives_singleton_sets():
    """If the model is perfectly confident (prob = label), every
    prediction set should be a singleton at any reasonable alpha."""
    n = 1_000
    rng = np.random.default_rng(6)
    y = rng.binomial(1, 0.4, size=n)
    p = y.astype(float) * 0.99 + (1 - y) * 0.01  # tiny slack to avoid 0/1
    pred = fit(p, y, alpha=0.1)
    sets = predict_sets(p, pred)
    assert (set_size(sets) == 1).all()


def test_coverage_by_group_returns_dataframe():
    y_cal, p_cal = _synthetic_calibrated(2_000, 0.18, seed=7)
    pred = fit(p_cal, y_cal, alpha=0.1)
    y_te, p_te = _synthetic_calibrated(1_500, 0.18, seed=8)
    rng = np.random.default_rng(9)
    g = rng.choice(["A", "B", "C"], size=1_500)
    sets = predict_sets(p_te, pred)
    df = coverage_by_group(y_te, sets, g)
    assert sorted(df.index.tolist()) == ["A", "B", "C"]
    assert {"empirical_coverage", "avg_set_size"} <= set(df.columns)
