"""Data loading layer.

`load_data` is the single swap point between synthetic and real Lending
Club data. Both sources return a DataFrame with the same columns:

    loan_amount, term, int_rate, grade, emp_length, home_ownership,
    annual_income, purpose, dti, fico, open_acc, revol_util,
    noise_1, noise_2, default, protected_group, issue_d

Real LC mapping (see `_load_lendingclub`):
    loan_amnt        -> loan_amount
    term             -> term (parsed: " 36 months" -> 36)
    int_rate         -> int_rate (parsed: "12.45%" -> 12.45)
    grade            -> grade
    emp_length       -> emp_length (parsed: "10+ years" -> 10, "< 1 year" -> 0)
    home_ownership   -> home_ownership
    annual_inc       -> annual_income
    purpose          -> purpose
    dti              -> dti
    fico_range_{lo,hi} mean -> fico
    open_acc         -> open_acc
    revol_util       -> revol_util (parsed: "45.2%" -> 45.2)
    issue_d          -> issue_d (parsed: "Dec-2014" -> 2014-12-01)
    loan_status      -> default (Charged Off / Default / Late > 30 -> 1)
    (none)           -> noise_1, noise_2 (zero-filled; for parity with synth)
    (none)           -> protected_group ("unknown"; LC has no such field)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .synth import generate

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_PATH = _PROJECT_ROOT / "data" / "synthetic" / "loans.csv"
RAW_DIR = _PROJECT_ROOT / "data" / "raw"

DEFAULT_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}


def temporal_split(
    df: pd.DataFrame,
    train_until: str = "2017-06",
    test_from: str = "2018-01",
    date_col: str = "issue_d",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split by `issue_d` into (train, validation, test) frames.

    `train_until` is inclusive; `test_from` is inclusive. The slice
    between them is the validation set (useful for tuning that needs an
    in-time hold-out without burning the test set).
    """
    if date_col not in df.columns:
        raise ValueError(f"{date_col!r} column not found in DataFrame")

    dates = pd.to_datetime(df[date_col])
    end_train = pd.Period(train_until, freq="M").to_timestamp("M")
    start_test = pd.Period(test_from, freq="M").to_timestamp()

    train = df[dates <= end_train]
    test = df[dates >= start_test]
    val = df[(dates > end_train) & (dates < start_test)]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_data(
    source: str = "synthetic",
    n: int = 50_000,
    seed: int = 42,
    regenerate: bool = False,
) -> pd.DataFrame:
    if source == "synthetic":
        if SYNTH_PATH.exists() and not regenerate:
            return pd.read_csv(SYNTH_PATH, parse_dates=["issue_d"])
        df = generate(n=n, seed=seed)
        SYNTH_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SYNTH_PATH, index=False)
        return df
    if source == "lendingclub":
        return _load_lendingclub()
    raise ValueError(f"Unknown source: {source!r}")


def _parse_lc_emp_length(series: pd.Series) -> pd.Series:
    """LC `emp_length` is messy: "10+ years", "< 1 year", "n/a", "1 year",
    "2 years", NaN. Map to clipped int years (0..10)."""
    s = series.fillna("0").astype(str).str.strip()
    s = s.where(~s.isin(["n/a", "N/A", "na"]), "0")
    s = s.where(~s.str.startswith("<"), "0")
    s = s.where(~s.str.startswith("10"), "10")
    digits = s.str.extract(r"(\d+)")[0].fillna("0").astype(int)
    return digits.clip(0, 10)


def _load_lendingclub(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    candidates = sorted(raw_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV in {raw_dir}. Drop a Lending Club accepted-loans CSV there."
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
        "emp_length": _parse_lc_emp_length(raw["emp_length"]),
        "home_ownership": raw["home_ownership"],
        "annual_income": raw["annual_inc"],
        "purpose": raw["purpose"],
        "dti": raw["dti"],
        "fico": fico,
        "open_acc": raw["open_acc"],
        "revol_util": pd.to_numeric(
            raw["revol_util"].astype(str).str.rstrip("%"),
            errors="coerce",
        ),
        "noise_1": 0.0,
        "noise_2": 0.0,
        "default": raw["loan_status"].isin(DEFAULT_STATUSES).astype(int),
        "protected_group": "unknown",
        "issue_d": pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce"),
    })
    return df.dropna(subset=["fico", "dti", "annual_income", "loan_amount", "issue_d"])
