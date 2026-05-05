"""Synthetic loan dataset modeled on Lending Club.

The data-generating process is intentionally known so we can verify the
interpretability pipeline (SHAP) downstream:
- FICO and DTI are the dominant true drivers of default.
- int_rate (correlated with grade, derived from FICO) and revol_util
  contribute moderately.
- loan_to_income is a true interaction signal.
- noise_1 and noise_2 are pure i.i.d. normals; SHAP should rank them low.

Schema (14 features + 1 label):
    loan_amount, term, int_rate, grade, emp_length, home_ownership,
    annual_income, purpose, dti, fico, open_acc, revol_util,
    noise_1, noise_2, default
"""
from __future__ import annotations

import numpy as np
import pandas as pd

PURPOSES = (
    "debt_consolidation", "credit_card", "home_improvement",
    "small_business", "major_purchase", "medical", "car", "other",
)
HOME_OWNERSHIPS = ("RENT", "MORTGAGE", "OWN", "OTHER")


def _fico_to_grade(fico: np.ndarray) -> np.ndarray:
    bins = [0, 640, 660, 680, 700, 720, 740, 1000]
    labels = ["G", "F", "E", "D", "C", "B", "A"]
    return np.asarray(pd.cut(fico, bins=bins, labels=labels, right=False).astype(str))


def _calibrate_intercept(z: np.ndarray, target_rate: float) -> float:
    """Bisection: find b such that mean(sigmoid(z + b)) ≈ target_rate."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        rate = float(np.mean(1.0 / (1.0 + np.exp(-(z + mid)))))
        if rate > target_rate:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def _grade_to_int_rate(grade: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    base = {"A": 7.0, "B": 9.5, "C": 12.5, "D": 15.5,
            "E": 19.0, "F": 23.0, "G": 27.0}
    base_arr = np.array([base[g] for g in grade])
    return np.clip(base_arr + rng.normal(0, 1.0, size=len(grade)), 5.0, 30.0)


def generate(n: int = 50_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    fico = np.clip(rng.normal(700, 35, size=n), 600, 850).round().astype(int)
    grade = _fico_to_grade(fico)
    int_rate = _grade_to_int_rate(grade, rng)

    annual_income = np.clip(
        rng.lognormal(mean=11.0, sigma=0.55, size=n), 15_000, 500_000
    ).round(-2)
    loan_amount = np.clip(
        rng.lognormal(mean=9.4, sigma=0.5, size=n), 500, 40_000
    ).round(-2)

    term = rng.choice([36, 60], size=n, p=[0.7, 0.3])
    emp_length = rng.integers(0, 11, size=n)

    home_ownership = rng.choice(
        HOME_OWNERSHIPS, size=n, p=[0.45, 0.40, 0.10, 0.05]
    )
    purpose = rng.choice(
        PURPOSES, size=n, p=[0.55, 0.20, 0.05, 0.04, 0.04, 0.04, 0.04, 0.04]
    )

    dti = np.clip(rng.normal(18, 8, size=n), 0, 60).round(2)
    open_acc = np.clip(rng.poisson(10, size=n), 1, 40)
    revol_util = np.clip(rng.normal(50, 20, size=n), 0, 100).round(1)

    noise_1 = rng.normal(0, 1, size=n)
    noise_2 = rng.normal(0, 1, size=n)

    loan_to_income = loan_amount / annual_income

    z = (
        -((fico - 700) / 30) * 1.6
        + ((dti - 18) / 8) * 0.9
        + ((int_rate - 13) / 5) * 0.6
        + ((revol_util - 50) / 25) * 0.4
        + loan_to_income * 1.2
        + (term == 60).astype(float) * 0.15
    )
    # calibrate intercept so the marginal default rate ≈ target_rate
    target_rate = 0.18
    intercept = _calibrate_intercept(z, target_rate)
    p = 1.0 / (1.0 + np.exp(-(z + intercept)))
    default = (rng.random(size=n) < p).astype(int)

    # protected_group is generated AFTER everything else so adding this
    # field doesn't shift the rng stream for any earlier column. Given
    # the same seed, every other column is bit-identical to a build
    # without this attribute.  The group is correlated with income (a
    # realistic proxy effect) but does NOT influence `default` directly,
    # so any disparity surfaced in 06_fairness.ipynb comes from the
    # model relying on income/FICO — not from labels being unfair.
    log_inc = np.log(annual_income)
    z_inc = (log_inc - log_inc.mean()) / log_inc.std()
    group_score = z_inc + rng.normal(0, 0.6, size=n)
    protected_group = np.where(
        group_score > 0.5, "A",
        np.where(group_score > -0.5, "B", "C"),
    )

    return pd.DataFrame({
        "loan_amount": loan_amount,
        "term": term,
        "int_rate": int_rate.round(2),
        "grade": grade,
        "emp_length": emp_length,
        "home_ownership": home_ownership,
        "annual_income": annual_income,
        "purpose": purpose,
        "dti": dti,
        "fico": fico,
        "open_acc": open_acc,
        "revol_util": revol_util,
        "noise_1": noise_1,
        "noise_2": noise_2,
        "default": default,
        "protected_group": protected_group,
    })
