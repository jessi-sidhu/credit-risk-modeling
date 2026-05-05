import numpy as np
import pandas as pd

from src.synth import generate

EXPECTED_COLS = [
    "loan_amount", "term", "int_rate", "grade", "emp_length",
    "home_ownership", "annual_income", "purpose", "dti", "fico",
    "open_acc", "revol_util", "noise_1", "noise_2", "default",
    "protected_group",
]


def test_shape_and_columns():
    df = generate(n=5_000, seed=42)
    assert df.shape == (5_000, 16)
    assert list(df.columns) == EXPECTED_COLS


def test_protected_group_does_not_directly_drive_default():
    """Group correlates with income, so we expect *some* disparity in
    default rate by group, but the dependence must be weak: a t-test
    between groups within the same income decile should be small."""
    df = generate(n=20_000, seed=42)
    # within high-income only, default rate should be roughly equal across groups
    high = df[df["annual_income"] > df["annual_income"].quantile(0.7)]
    rates = high.groupby("protected_group")["default"].mean()
    assert rates.max() - rates.min() < 0.05, rates.to_dict()


def test_default_rate_near_18_percent():
    df = generate(n=20_000, seed=42)
    rate = df["default"].mean()
    assert 0.16 <= rate <= 0.20, f"expected ~0.18 default rate, got {rate:.3f}"


def test_determinism():
    a = generate(n=2_000, seed=7)
    b = generate(n=2_000, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_noise_features_near_zero_correlation_with_label():
    df = generate(n=20_000, seed=42)
    for col in ("noise_1", "noise_2"):
        corr = float(np.corrcoef(df[col], df["default"])[0, 1])
        assert abs(corr) < 0.03, f"{col} corr with default = {corr:.3f}"


def test_signal_features_have_expected_sign():
    df = generate(n=20_000, seed=42)
    fico_corr = float(np.corrcoef(df["fico"], df["default"])[0, 1])
    dti_corr = float(np.corrcoef(df["dti"], df["default"])[0, 1])
    assert fico_corr < -0.3, f"FICO should anti-correlate; got {fico_corr:.3f}"
    assert dti_corr > 0.05, f"DTI should positively correlate; got {dti_corr:.3f}"


def test_default_rate_monotone_in_grade():
    df = generate(n=30_000, seed=42)
    by_grade = df.groupby("grade")["default"].mean()
    ordered = by_grade.reindex(["A", "B", "C", "D", "E", "F", "G"]).dropna()
    diffs = np.diff(ordered.to_numpy())
    assert (diffs >= 0).all(), f"default rate not monotone by grade: {ordered.to_dict()}"
