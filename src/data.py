"""Data loading layer.

`load_data` is the single swap point between synthetic and real Lending
Club data. Both sources return a DataFrame with the same columns:
    loan_amount, term, int_rate, grade, emp_length, home_ownership,
    annual_income, purpose, dti, fico, open_acc, revol_util,
    noise_1, noise_2, default
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .synth import generate

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_PATH = _PROJECT_ROOT / "data" / "synthetic" / "loans.csv"
RAW_DIR = _PROJECT_ROOT / "data" / "raw"

DEFAULT_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}


def load_data(
    source: str = "synthetic",
    n: int = 50_000,
    seed: int = 42,
    regenerate: bool = False,
) -> pd.DataFrame:
    if source == "synthetic":
        if SYNTH_PATH.exists() and not regenerate:
            return pd.read_csv(SYNTH_PATH)
        df = generate(n=n, seed=seed)
        SYNTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SYNTH_PATH, index=False)
        return df
    if source == "lendingclub":
        return _load_lendingclub()
    raise ValueError(f"Unknown source: {source!r}")


def _load_lendingclub() -> pd.DataFrame:
    candidates = sorted(RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV in {RAW_DIR}. Drop a Lending Club accepted-loans CSV there."
        )
    raw = pd.read_csv(candidates[0], low_memory=False)
    fico = (
        raw[["fico_range_low", "fico_range_high"]].mean(axis=1)
        if "fico_range_low" in raw.columns else raw["fico"]
    )
    df = pd.DataFrame({
        "loan_amount": raw["loan_amnt"],
        "term": raw["term"].astype(str).str.extract(r"(\d+)")[0].astype(int),
        "int_rate": raw["int_rate"].astype(str).str.rstrip("%").astype(float),
        "grade": raw["grade"],
        "emp_length": (
            raw["emp_length"].fillna("0").astype(str).str.extract(r"(\d+)")[0]
            .fillna("0").astype(int)
        ),
        "home_ownership": raw["home_ownership"],
        "annual_income": raw["annual_inc"],
        "purpose": raw["purpose"],
        "dti": raw["dti"],
        "fico": fico,
        "open_acc": raw["open_acc"],
        "revol_util": (
            raw["revol_util"].astype(str).str.rstrip("%").astype(float)
        ),
        "noise_1": 0.0,
        "noise_2": 0.0,
        "default": raw["loan_status"].isin(DEFAULT_STATUSES).astype(int),
    })
    return df.dropna(subset=["fico", "dti", "annual_income", "loan_amount"])
