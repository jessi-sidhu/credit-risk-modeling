import numpy as np
import pandas as pd

from src.drift import psi, psi_categorical, psi_numeric, psi_report


def test_psi_zero_for_identical_numeric_distributions():
    rng = np.random.default_rng(0)
    a = rng.normal(0, 1, size=10_000)
    score = psi_numeric(a, a)
    assert abs(score) < 1e-9


def test_psi_zero_for_identical_categorical_distributions():
    a = ["A"] * 600 + ["B"] * 300 + ["C"] * 100
    score = psi_categorical(a, a)
    assert abs(score) < 1e-6


def test_psi_grows_with_shift():
    rng = np.random.default_rng(1)
    n = 10_000
    base = rng.normal(0, 1, size=n)
    small_shift = base + 0.2 + rng.normal(0, 0.05, size=n)
    big_shift = rng.normal(1.5, 1, size=n)

    psi_small = psi_numeric(base, small_shift)
    psi_big = psi_numeric(base, big_shift)
    assert psi_big > psi_small > 0


def test_significant_threshold_triggered_by_large_shift():
    rng = np.random.default_rng(2)
    n = 10_000
    base = rng.normal(0, 1, size=n)
    shifted = rng.normal(2.0, 1, size=n)
    assert psi_numeric(base, shifted) > 0.25


def test_psi_report_status_labels():
    rng = np.random.default_rng(3)
    base = pd.DataFrame({
        "stable_col": rng.normal(0, 1, 5_000),
        "shifted_col": rng.normal(0, 1, 5_000),
        "cat_col": rng.choice(["a", "b", "c"], size=5_000, p=[0.6, 0.3, 0.1]),
    })
    current = pd.DataFrame({
        "stable_col": rng.normal(0, 1, 5_000),
        "shifted_col": rng.normal(2.5, 1, 5_000),
        "cat_col": rng.choice(["a", "b", "c"], size=5_000, p=[0.2, 0.2, 0.6]),
    })
    rep = psi_report(base, current)
    statuses = dict(zip(rep["feature"], rep["status"]))
    assert statuses["stable_col"] == "stable"
    assert statuses["shifted_col"] == "significant"
    assert statuses["cat_col"] == "significant"


def test_psi_dispatch_handles_categorical_strings():
    base = pd.Series(["A"] * 700 + ["B"] * 300)
    current = pd.Series(["A"] * 300 + ["B"] * 700)
    assert psi(base, current) > 0.25
