import numpy as np

from src.calibration import (
    apply_isotonic,
    apply_platt,
    calibration_metrics,
    expected_calibration_error,
    fit_isotonic,
    fit_platt,
    reliability_table,
)


def test_perfect_predictions_have_zero_brier_and_ece():
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1, 0, 0])
    y_prob = y_true.astype(float)
    m = calibration_metrics(y_true, y_prob)
    assert m["brier"] == 0.0
    assert m["ece"] == 0.0


def test_isotonic_reduces_brier_on_biased_probs():
    """If raw probs are systematically too high, isotonic should pull
    them back toward truth and lower the Brier score."""
    rng = np.random.default_rng(0)
    n = 5_000
    p_true = rng.beta(2, 8, size=n)  # base rate ~0.2
    y = rng.binomial(1, p_true)
    biased = np.clip(p_true + 0.15, 0, 1)  # systematically high

    iso = fit_isotonic(biased, y)
    recal = apply_isotonic(iso, biased)

    brier_raw = calibration_metrics(y, biased)["brier"]
    brier_iso = calibration_metrics(y, recal)["brier"]
    assert brier_iso < brier_raw, f"isotonic failed: {brier_iso} >= {brier_raw}"


def test_platt_reduces_brier_on_biased_probs():
    rng = np.random.default_rng(1)
    n = 5_000
    p_true = rng.beta(2, 8, size=n)
    y = rng.binomial(1, p_true)
    biased = np.clip(p_true + 0.15, 1e-3, 1 - 1e-3)

    cal = fit_platt(biased, y)
    recal = apply_platt(cal, biased)

    brier_raw = calibration_metrics(y, biased)["brier"]
    brier_platt = calibration_metrics(y, recal)["brier"]
    assert brier_platt < brier_raw, f"platt failed: {brier_platt} >= {brier_raw}"


def test_reliability_table_shape():
    rng = np.random.default_rng(2)
    n = 1_000
    y = rng.binomial(1, 0.3, size=n)
    p = rng.beta(2, 5, size=n)
    tbl = reliability_table(y, p, n_bins=8)
    assert set(tbl.columns) == {"mean_predicted", "fraction_positive", "count"}
    assert tbl["count"].sum() == n
    # bins are quantile-based; means should be roughly monotonic — allow noise
    assert tbl["mean_predicted"].iloc[0] < tbl["mean_predicted"].iloc[-1]


def test_ece_zero_for_well_calibrated_constant():
    """If every prediction equals the empirical mean, ECE should be 0."""
    rng = np.random.default_rng(3)
    n = 2_000
    y = rng.binomial(1, 0.18, size=n)
    p = np.full(n, y.mean())
    assert expected_calibration_error(y, p, n_bins=5) < 1e-9
