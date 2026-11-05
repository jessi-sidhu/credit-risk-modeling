"""Round-trip a Lending-Club-shaped fixture through `_load_lendingclub`.

The real Kaggle download lives in `data/archive/` (~1.6 GB, git-ignored);
these tests don't depend on it. A 6-row fixture covers the parsing edge
cases, and small fake directory trees cover the accepted-file lookup.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import _load_lendingclub, find_accepted_file


# Rows 0-6 survive the target filters; rows 7-12 must be dropped.
_DNM = "Does not meet the credit policy. Status:"
_STATUS = [
    "Fully Paid", "Fully Paid", "Charged Off", "Default",
    _DNM + "Fully Paid", _DNM + "Charged Off", "Fully Paid",
    "Current", "Late (31-120 days)", "Late (16-30 days)", "In Grace Period",
    "Fully Paid", "Charged Off",
]
_TERM = [" 36 months", " 60 months", "36 months", " 60 months",
         " 36 months", " 60 months", " 36 months",
         " 36 months", " 36 months", " 36 months", " 36 months",
         " 60 months", " 36 months"]
_ISSUE = ["Dec-2014", "Mar-2013", "Aug-2015", "Jun-2012",
          "Jan-2010", "Nov-2009",
          "Mar-2016",   # 36m -> ends Mar-2019 == snapshot month: kept
          "Jan-2015", "Jan-2015", "Jan-2015", "Jan-2015",
          "Jan-2016",   # 60m -> ends Jan-2021: unmatured, dropped
          "Apr-2016"]   # 36m -> ends Apr-2019: unmatured, dropped
N_KEPT = 7


def _lc_fixture() -> pd.DataFrame:
    """Rows that exercise the LC quirks and target filters of `_load_lendingclub`."""
    n = len(_STATUS)
    return pd.DataFrame({
        "loan_amnt": [10_000, 20_000, 5_000, 35_000, 8_000, 15_000, 12_000] + [9_000] * (n - 7),
        "term": _TERM,
        "int_rate": ["12.45%", "8.99%", "20.50%", "15.27%", "6.89%", "24.99%", "11.00%"]
                    + ["10.00%"] * (n - 7),
        "grade": ["B", "A", "E", "C", "A", "G", "B"] + ["B"] * (n - 7),
        "emp_length": ["10+ years", "< 1 year", "5 years", "n/a",
                       "2 years", None, "3 years"] + ["1 year"] * (n - 7),
        "home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT", "MORTGAGE", "RENT", "OWN"]
                          + ["RENT"] * (n - 7),
        "annual_inc": [60_000, 120_000, 35_000, 90_000, 75_000, 45_000, 55_000]
                      + [50_000] * (n - 7),
        "purpose": ["debt_consolidation", "credit_card", "small_business",
                    "home_improvement", "car", "medical", "other"]
                   + ["credit_card"] * (n - 7),
        "dti": [18.5, 8.2, 32.1, 15.6, 10.0, 28.4, 12.0] + [15.0] * (n - 7),
        "fico_range_low":  [690, 760, 640, 715, 770, 615, 700] + [700] * (n - 7),
        "fico_range_high": [694, 764, 644, 719, 774, 619, 704] + [704] * (n - 7),
        "open_acc": [10, 8, 14, 7, 6, 12, 9] + [9] * (n - 7),
        "revol_util": ["45.2%", "20.0%", "85.5%", "55.7%", None, "92.1%", "30.0%"]
                      + ["40.0%"] * (n - 7),
        "issue_d": _ISSUE,
        "loan_status": _STATUS,
    })


def _write_fixture(tmp_path) -> "Path":
    fixture_dir = tmp_path / "lc_raw"; fixture_dir.mkdir()
    _lc_fixture().to_csv(fixture_dir / "accepted_2014_to_2018.csv", index=False)
    return fixture_dir


def test_load_lendingclub_returns_canonical_schema(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path), term_months=None)

    expected = {
        "loan_amount", "term", "int_rate", "grade", "emp_length",
        "home_ownership", "annual_income", "purpose", "dti", "fico",
        "open_acc", "revol_util", "default", "issue_d",
    }
    assert set(df.columns) == expected


def test_term_int_rate_revol_util_parsed_to_numbers(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path), term_months=None)

    assert df["term"].dtype.kind in "iu", df["term"].dtype
    assert set(df["term"].unique()) <= {36, 60}
    assert df["int_rate"].iloc[0] == pytest.approx(12.45)
    assert df["revol_util"].iloc[0] == pytest.approx(45.2)


def test_emp_length_handles_messy_strings(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path), term_months=None).reset_index(drop=True)

    # 10+ years -> 10
    assert df.loc[0, "emp_length"] == 10
    # < 1 year -> 0
    assert df.loc[1, "emp_length"] == 0
    # "5 years" -> 5
    assert df.loc[2, "emp_length"] == 5
    # n/a -> 0
    assert df.loc[3, "emp_length"] == 0


def test_loan_status_maps_to_binary_default(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path), term_months=None).reset_index(drop=True)

    # Charged Off, Default, credit-policy Charged Off -> 1; Fully Paid variants -> 0
    expected = [0, 0, 1, 1, 0, 1, 0]
    assert df["default"].tolist() == expected


def test_unfinished_loans_dropped(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path), term_months=None)
    assert len(df) == N_KEPT
    kept_dates = set(df["issue_d"].dt.strftime("%b-%Y"))
    # Current / Late / In Grace rows were all issued Jan-2015
    assert "Jan-2015" not in kept_dates


def test_unmatured_loans_dropped(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path), term_months=None)
    dates = set(df["issue_d"].dt.strftime("%b-%Y"))
    assert "Mar-2016" in dates          # ends exactly in the snapshot month
    assert "Jan-2016" not in dates      # 60m, ends 2021
    assert "Apr-2016" not in dates      # 36m, ends Apr-2019


def test_snapshot_date_controls_maturity(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path), snapshot_date="2021-12-31",
                           term_months=None)
    assert len(df) == N_KEPT + 2


def test_fico_is_average_of_range(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path), term_months=None).reset_index(drop=True)
    # row 0: low=690 high=694 -> 692
    assert df.loc[0, "fico"] == pytest.approx(692.0)


def test_issue_d_parsed_to_datetime(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path), term_months=None).reset_index(drop=True)
    assert df["issue_d"].dtype == "datetime64[ns]"
    assert df["issue_d"].iloc[0] == pd.Timestamp("2014-12-01")


def test_blank_footer_rows_dropped(tmp_path):
    """The real LC export has summary lines that are empty in every
    column we read; they must be dropped, not crash the parser."""
    fixture_dir = tmp_path / "lc_raw"; fixture_dir.mkdir()
    fx = _lc_fixture()
    path = fixture_dir / "accepted_2014_to_2018.csv"
    fx.to_csv(path, index=False)
    with open(path, "a") as f:
        f.write("," * (fx.shape[1] - 1) + "\n")
    df = _load_lendingclub(raw_dir=fixture_dir, term_months=None)
    assert len(df) == N_KEPT


def test_missing_csv_raises(tmp_path):
    empty = tmp_path / "empty"; empty.mkdir()
    with pytest.raises(FileNotFoundError):
        _load_lendingclub(raw_dir=empty)


def _kaggle_layout(root, with_csv: bool = True, with_gz: bool = True):
    """Mimic data/archive/: folders named *.csv, plus .csv.gz copies."""
    for kind in ("accepted", "rejected"):
        folder = root / f"{kind}_2007_to_2018q4.csv"
        folder.mkdir(parents=True)
        if with_csv:
            (folder / f"{kind}_2007_to_2018Q4.csv").write_text("x\n")
        if with_gz:
            (root / f"{kind}_2007_to_2018Q4.csv.gz").write_bytes(b"")


def test_find_accepted_prefers_csv_inside_folder(tmp_path):
    _kaggle_layout(tmp_path)
    found = find_accepted_file(tmp_path)
    assert found.is_file()
    assert found == tmp_path / "accepted_2007_to_2018q4.csv" / "accepted_2007_to_2018Q4.csv"


def test_find_accepted_falls_back_to_gz(tmp_path):
    _kaggle_layout(tmp_path, with_csv=False)
    assert find_accepted_file(tmp_path) == tmp_path / "accepted_2007_to_2018Q4.csv.gz"


def test_find_accepted_never_returns_rejected(tmp_path):
    folder = tmp_path / "rejected_2007_to_2018q4.csv"; folder.mkdir()
    (folder / "rejected_2007_to_2018Q4.csv").write_text("x\n")
    (tmp_path / "rejected_2007_to_2018Q4.csv.gz").write_bytes(b"")
    with pytest.raises(FileNotFoundError):
        find_accepted_file(tmp_path)


def test_find_accepted_is_case_insensitive(tmp_path):
    (tmp_path / "Accepted_LOANS.CSV").write_text("x\n")
    assert find_accepted_file(tmp_path).name == "Accepted_LOANS.CSV"


def test_default_keeps_36_month_loans_only(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path))
    assert set(df["term"]) == {36}
    # kept rows 0, 2, 4, 6 of the fixture are the 36-month ones
    assert len(df) == 4


def test_row_counts_recorded_in_attrs(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path))
    assert df.attrs["row_counts"] == {
        "rows_read": len(_STATUS), "finished_loans": 9,
        "key_fields_present": 9, "matured": N_KEPT, "term_36m": 4,
    }
    assert df.attrs["snapshot_date"] == "2019-03-31"
