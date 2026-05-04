import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import load_data
from src.features import (
    CATEGORICAL,
    NUMERIC_ALL,
    engineer_features,
    make_preprocessor,
    split_xy,
)


def test_engineered_columns_added():
    df = load_data().head(50)
    out = engineer_features(df)
    for col in ("loan_to_income", "installment_estimate",
                "installment_to_income", "credit_age_proxy"):
        assert col in out.columns


def test_installment_matches_amortization_formula():
    """Hand-picked row: $10,000 @ 12% APR over 36 months ≈ $332.14/month."""
    df = pd.DataFrame({
        "loan_amount": [10_000.0],
        "int_rate": [12.0],
        "term": [36],
        "annual_income": [60_000.0],
        "emp_length": [5],
        "home_ownership": ["MORTGAGE"],
    })
    out = engineer_features(df)
    assert abs(out["installment_estimate"].iloc[0] - 332.14) < 0.5
    assert abs(out["loan_to_income"].iloc[0] - (10_000 / 60_000)) < 1e-9
    assert out["credit_age_proxy"].iloc[0] == 5.0


def test_credit_age_proxy_zero_when_renting():
    df = pd.DataFrame({
        "loan_amount": [10_000.0], "int_rate": [12.0], "term": [36],
        "annual_income": [60_000.0], "emp_length": [8],
        "home_ownership": ["RENT"],
    })
    assert engineer_features(df)["credit_age_proxy"].iloc[0] == 0.0


def test_preprocessor_fits_on_train_only():
    """Leakage guard: fit transformer on train alone, transform both;
    same column count and no NaNs in test output."""
    df = engineer_features(load_data())
    X, y = split_xy(df)
    X_tr, X_te, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=0)

    pre = make_preprocessor(scale_numeric=True)
    pre.fit(X_tr)
    Z_tr = pre.transform(X_tr)
    Z_te = pre.transform(X_te)

    assert Z_tr.shape[1] == Z_te.shape[1]
    assert not np.isnan(Z_te).any()
    names = pre.get_feature_names_out()
    assert len(names) == Z_tr.shape[1]
    for col in NUMERIC_ALL:
        assert col in names


def test_preprocessor_handles_unseen_category():
    """OneHotEncoder(handle_unknown='ignore') should not crash on unseen
    categories at transform time — important if real LC data has values
    the train fold never saw."""
    df = engineer_features(load_data().head(2000))
    X, y = split_xy(df)
    pre = make_preprocessor()
    pre.fit(X)

    new_row = X.iloc[[0]].copy()
    new_row["purpose"] = "totally_made_up"
    transformed = pre.transform(new_row)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()
