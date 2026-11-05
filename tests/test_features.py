import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import LC_USECOLS
from src.features import (
    CATEGORICAL,
    NUMERIC_ALL,
    engineer_features,
    feature_lists,
    make_preprocessor,
    split_xy,
)
from src.synth import generate


def test_engineered_columns_added():
    df = generate(n=50, seed=42)
    out = engineer_features(df)
    for col in ("loan_to_income", "installment_estimate",
                "installment_to_income", "emp_x_homeowner"):
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
    assert out["emp_x_homeowner"].iloc[0] == 5.0


def test_emp_x_homeowner_zero_when_renting():
    df = pd.DataFrame({
        "loan_amount": [10_000.0], "int_rate": [12.0], "term": [36],
        "annual_income": [60_000.0], "emp_length": [8],
        "home_ownership": ["RENT"],
    })
    assert engineer_features(df)["emp_x_homeowner"].iloc[0] == 0.0


def test_preprocessor_fits_on_train_only():
    """Leakage guard: fit transformer on train alone, transform both;
    same column count and no NaNs in test output."""
    df = engineer_features(generate(n=5_000, seed=42))
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
    df = engineer_features(generate(n=2_000, seed=42))
    X, y = split_xy(df)
    pre = make_preprocessor()
    pre.fit(X)

    new_row = X.iloc[[0]].copy()
    new_row["purpose"] = "totally_made_up"
    transformed = pre.transform(new_row)
    assert transformed.shape[0] == 1
    assert not np.isnan(transformed).any()


# Columns only known AFTER a loan is issued. Any of these in the feature
# set is target leakage.
POST_ORIGINATION = {
    "total_pymnt", "total_rec_prncp", "recoveries", "collection_recovery_fee",
    "last_pymnt_d", "last_pymnt_amnt", "last_fico_range_high",
    "last_fico_range_low", "out_prncp",
}


def test_no_post_origination_columns_in_features():
    for use_lc_grade in (True, False):
        numeric, categorical = feature_lists(use_lc_grade)
        assert not POST_ORIGINATION & set(numeric + categorical)
    assert not POST_ORIGINATION & set(NUMERIC_ALL + CATEGORICAL)


def test_no_post_origination_columns_read_from_lc():
    assert not POST_ORIGINATION & set(LC_USECOLS)


def test_use_lc_grade_false_drops_grade_and_rate_features():
    numeric, categorical = feature_lists(use_lc_grade=False)
    for col in ("grade", "int_rate", "installment_estimate", "installment_to_income"):
        assert col not in numeric + categorical
    assert "fico" in numeric and "purpose" in categorical


def test_preprocessor_respects_use_lc_grade():
    df = engineer_features(generate(n=500, seed=42))
    X, _ = split_xy(df)
    names = make_preprocessor(use_lc_grade=False).fit(X).get_feature_names_out()
    assert "int_rate" not in names
    assert not any(n.startswith("grade_") for n in names)


# --- Phase 2: outcome columns stay out of features ---------------------------

def test_outcome_columns_never_features():
    from src.survival_data import OUTCOME_COLS
    forbidden = set(OUTCOME_COLS) | POST_ORIGINATION | {"default"}
    for use_lc_grade in (True, False):
        numeric, categorical = feature_lists(use_lc_grade)
        assert not forbidden & set(numeric + categorical)


def test_last_pymnt_d_read_only_in_survival_data():
    """`last_pymnt_d` is an outcome column: only src/survival_data.py may
    reference it (for the outcome frame). `loan_status` is additionally
    allowed in src/data.py, where it defines the Phase 1 target."""
    from pathlib import Path
    allowed = {"last_pymnt_d": {"survival_data.py"},
               "loan_status": {"survival_data.py", "data.py"}}
    src = Path(__file__).resolve().parent.parent / "src"
    for col, ok in allowed.items():
        users = {p.name for p in src.glob("*.py") if col in p.read_text()}
        assert users <= ok, f"{col} referenced in {users - ok}"
