"""Data loading layer.

`load_data` returns real Lending Club data by default. The synthetic
source (`src/synth.py`) is kept only as a small fixture generator for
tests and must not be used by any training path.

The LC loader returns:

    loan_amount, term, int_rate, grade, emp_length, home_ownership,
    annual_income, purpose, dti, fico, open_acc, revol_util,
    default, issue_d

Raw file: the Kaggle download in `data/archive/`. Only the ACCEPTED-loans
file is read (see `find_accepted_file`); the rejected-loans file has a
different schema and is never loaded.

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
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .synth import generate

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_PATH = _PROJECT_ROOT / "data" / "synthetic" / "loans.csv"
RAW_DIR = _PROJECT_ROOT / "data" / "archive"

DEFAULT_STATUSES = {"Charged Off", "Default", "Late (31-120 days)"}

# Only the raw LC columns we map, plus loan_status. The full accepted
# file has ~150 columns and is ~1.6 GB; reading a subset keeps memory down.
LC_USECOLS = [
    "loan_amnt", "term", "int_rate", "grade", "emp_length",
    "home_ownership", "annual_inc", "purpose", "dti",
    "fico_range_low", "fico_range_high", "open_acc", "revol_util",
    "issue_d", "loan_status",
]


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


DEFAULT_N = 50_000


def load_data(
    source: str = "lendingclub",
    n: int = DEFAULT_N,
    seed: int = 42,
    regenerate: bool = False,
) -> pd.DataFrame:
    if source == "synthetic":
        if SYNTH_PATH.exists() and not regenerate:
            return pd.read_csv(SYNTH_PATH, parse_dates=["issue_d"])
        df = generate(n=n, seed=seed)
        # Only cache when we generated the default-sized canonical file,
        # so callers asking for a smaller subset don't pollute the cache.
        if n == DEFAULT_N:
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


def find_accepted_file(raw_dir: Path = RAW_DIR) -> Path:
    """Locate the LC accepted-loans file under `raw_dir`.

    The Kaggle layout has folders whose names end in `.csv` (e.g.
    `accepted_2007_to_2018q4.csv/`), so only real files are considered.
    Prefer an uncompressed `.csv`; fall back to `.csv.gz`. Anything
    starting with `rejected` is never returned.
    """
    files = [p for p in Path(raw_dir).rglob("*") if p.is_file()]
    accepted = [p for p in files if p.name.lower().startswith("accepted")]
    for suffix in (".csv", ".csv.gz"):
        matches = sorted(p for p in accepted if p.name.lower().endswith(suffix))
        if matches:
            return matches[0]
    raise FileNotFoundError(
        f"No Lending Club accepted-loans file (accepted*.csv or "
        f"accepted*.csv.gz) found under {raw_dir}."
    )


def _load_lendingclub(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    path = find_accepted_file(raw_dir)
    raw = pd.read_csv(path, usecols=LC_USECOLS, low_memory=False)
    # The LC export contains summary/footer lines ("Total amount funded in
    # policy code ...") that are empty in every column we read.
    raw = raw.dropna(subset=["loan_amnt", "term"])
    fico = raw[["fico_range_low", "fico_range_high"]].mean(axis=1)
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
        "default": raw["loan_status"].isin(DEFAULT_STATUSES).astype(int),
        "issue_d": pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce"),
    })
    return df.dropna(subset=["fico", "dti", "annual_income", "loan_amount", "issue_d"])
