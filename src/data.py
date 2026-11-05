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
    loan_status      -> default (see TARGET_MAP; unfinished loans dropped)

Target definition: only FINISHED loans are kept (Fully Paid / Charged Off /
Default, including the "Does not meet the credit policy" variants), and only
from cohorts that have fully matured by SNAPSHOT_DATE, and only 36-month
loans (see TERM_MONTHS).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from .synth import generate

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTH_PATH = _PROJECT_ROOT / "data" / "synthetic" / "loans.csv"
RAW_DIR = _PROJECT_ROOT / "data" / "archive"

logger = logging.getLogger(__name__)

# Finished loans only. Anything not listed (Current, Late (16-30 days),
# Late (31-120 days), In Grace Period) has no final outcome yet and is dropped.
TARGET_MAP = {
    "Fully Paid": 0,
    "Charged Off": 1,
    "Default": 1,
    "Does not meet the credit policy. Status:Fully Paid": 0,
    "Does not meet the credit policy. Status:Charged Off": 1,
}

# Last month covered by the 2007-2018Q4 accepted file. A loan is kept only
# if issue_d + term months <= SNAPSHOT_DATE. Without this, recent cohorts
# contain only loans that already ended (mostly early payoffs and fast
# charge-offs), which biases the default rate.
SNAPSHOT_DATE = "2019-03-31"

# Only 36-month loans are modelled. With the matured-cohort filter, no
# 60-month loan issued after 2014-03 survives, so an out-of-time test set
# (2015+) would contain 36-month loans only while training mixed in
# 60-month loans that default about twice as often. Restricting to one
# product keeps train, validation and test comparable.
TERM_MONTHS = 36

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


def _is_matured(issue_d: pd.Series, term: pd.Series, snapshot_date: str) -> pd.Series:
    """True where the loan's scheduled final month is on or before the snapshot."""
    snap = pd.Timestamp(snapshot_date)
    end_month = issue_d.dt.year * 12 + (issue_d.dt.month - 1) + term
    return end_month <= snap.year * 12 + (snap.month - 1)


# Rows missing any of these are dropped (shared with the survival loader).
KEY_FIELDS = ["fico", "dti", "annual_income", "loan_amount", "issue_d"]


def parse_lc_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Map raw LC application columns to the model's feature schema.

    Only origination-time columns are used; no outcome column is read
    here. Shared by the Phase 1 loader and `src/survival_data.py`.
    """
    fico = raw[["fico_range_low", "fico_range_high"]].mean(axis=1)
    return pd.DataFrame({
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
        "issue_d": pd.to_datetime(raw["issue_d"], format="%b-%Y", errors="coerce"),
    })


def _load_lendingclub(
    raw_dir: Path = RAW_DIR,
    snapshot_date: str = SNAPSHOT_DATE,
    term_months: int | None = TERM_MONTHS,
) -> pd.DataFrame:
    path = find_accepted_file(raw_dir)
    raw = pd.read_csv(path, usecols=LC_USECOLS, low_memory=False)
    # The LC export contains summary/footer lines ("Total amount funded in
    # policy code ...") that are empty in every column we read.
    raw = raw.dropna(subset=["loan_amnt", "term"])
    counts = {"rows_read": len(raw)}
    logger.info("LC rows read: %d", len(raw))

    raw = raw[raw["loan_status"].isin(TARGET_MAP)]
    counts["finished_loans"] = len(raw)
    logger.info("after keeping finished loans only: %d", len(raw))
    feats = parse_lc_features(raw)
    df = feats.drop(columns="issue_d")
    df["default"] = raw["loan_status"].map(TARGET_MAP).astype(int)
    df["issue_d"] = feats["issue_d"]
    df = df.dropna(subset=KEY_FIELDS)
    counts["key_fields_present"] = len(df)
    logger.info("after dropping rows missing key fields: %d", len(df))

    df = df[_is_matured(df["issue_d"], df["term"], snapshot_date)]
    counts["matured"] = len(df)
    logger.info("after matured-cohort filter (snapshot %s): %d", snapshot_date, len(df))

    if term_months is not None:
        df = df[df["term"] == term_months]
        counts[f"term_{term_months}m"] = len(df)
        logger.info("after keeping %d-month loans only: %d", term_months, len(df))
    logger.info("final default rate: %.4f", df["default"].mean())

    df = df.copy()
    df.attrs["row_counts"] = {k: int(v) for k, v in counts.items()}
    df.attrs["snapshot_date"] = snapshot_date
    df.attrs["term_months"] = term_months
    df.attrs["default_rate"] = float(df["default"].mean())
    df.attrs["source_file"] = path.name
    return df
