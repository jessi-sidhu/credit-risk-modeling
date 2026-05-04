"""Feature engineering and preprocessing.

Two stages, kept separate on purpose:

1. `engineer_features(df)` — pure row-wise arithmetic, no fitted state.
   Apply once after `load_data`, BEFORE the train/test split. Adding
   columns can't leak.

2. `make_preprocessor(scale_numeric)` — a `ColumnTransformer` that
   imputes, optionally scales (for the linear model), and one-hot
   encodes the categoricals. This DOES have fitted state and so MUST
   only be fit on the training fold.

Why row-wise FE outside the pipeline: easier to inspect in the notebook
and easier to reason about leakage. The downside (slightly less
"sklearny") doesn't matter at this scale.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_BASE = [
    "loan_amount", "term", "int_rate", "emp_length", "annual_income",
    "dti", "fico", "open_acc", "revol_util", "noise_1", "noise_2",
]
NUMERIC_ENGINEERED = [
    "loan_to_income", "installment_estimate",
    "installment_to_income", "credit_age_proxy",
]
NUMERIC_ALL = NUMERIC_BASE + NUMERIC_ENGINEERED
CATEGORICAL = ["grade", "home_ownership", "purpose"]


def _monthly_installment(principal: np.ndarray, apr_pct: np.ndarray, term_months: np.ndarray) -> np.ndarray:
    """Standard fixed-rate amortization formula."""
    r = (apr_pct / 100.0) / 12.0
    n = term_months
    factor = np.where(r > 0, (r * (1 + r) ** n) / ((1 + r) ** n - 1), 1.0 / n)
    return principal * factor


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ratio/derived features that mirror how a loan officer reasons."""
    out = df.copy()
    out["loan_to_income"] = out["loan_amount"] / out["annual_income"]
    out["installment_estimate"] = _monthly_installment(
        out["loan_amount"].to_numpy(dtype=float),
        out["int_rate"].to_numpy(dtype=float),
        out["term"].to_numpy(dtype=float),
    )
    out["installment_to_income"] = (
        out["installment_estimate"] / (out["annual_income"] / 12.0)
    )
    owns_home = out["home_ownership"].isin(["OWN", "MORTGAGE"]).to_numpy().astype(float)
    out["credit_age_proxy"] = out["emp_length"].to_numpy(dtype=float) * owns_home
    return out


def make_preprocessor(scale_numeric: bool = False) -> ColumnTransformer:
    if scale_numeric:
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
    else:
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])

    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        [("num", num_pipe, NUMERIC_ALL), ("cat", cat_pipe, CATEGORICAL)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df.drop(columns=["default"]), df["default"].astype(int)
