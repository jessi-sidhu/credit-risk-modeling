"""Round-trip a Lending-Club-shaped fixture through `_load_lendingclub`.

The real Kaggle download lives in `data/archive/` (~1.6 GB, git-ignored);
these tests don't depend on it. A 6-row fixture covers the parsing edge
cases, and small fake directory trees cover the accepted-file lookup.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import _load_lendingclub, find_accepted_file


def _lc_fixture() -> pd.DataFrame:
    """Six rows that exercise the LC quirks `_load_lendingclub` must handle."""
    return pd.DataFrame({
        "loan_amnt": [10_000, 20_000, 5_000, 35_000, 8_000, 15_000],
        "term": [" 36 months", " 60 months", "36 months", " 60 months",
                 " 36 months", " 60 months"],
        "int_rate": ["12.45%", "8.99%", "20.50%", "15.27%", "6.89%", "24.99%"],
        "grade": ["B", "A", "E", "C", "A", "G"],
        "emp_length": ["10+ years", "< 1 year", "5 years", "n/a",
                       "2 years", None],
        "home_ownership": ["RENT", "MORTGAGE", "OWN", "RENT", "MORTGAGE", "RENT"],
        "annual_inc": [60_000, 120_000, 35_000, 90_000, 75_000, 45_000],
        "purpose": ["debt_consolidation", "credit_card", "small_business",
                    "home_improvement", "car", "medical"],
        "dti": [18.5, 8.2, 32.1, 15.6, 10.0, 28.4],
        "fico_range_low":  [690, 760, 640, 715, 770, 615],
        "fico_range_high": [694, 764, 644, 719, 774, 619],
        "open_acc": [10, 8, 14, 7, 6, 12],
        "revol_util": ["45.2%", "20.0%", "85.5%", "55.7%", None, "92.1%"],
        "issue_d": ["Dec-2014", "Mar-2015", "Aug-2016", "Jun-2017",
                    "Jan-2018", "Nov-2018"],
        "loan_status": ["Fully Paid", "Current", "Charged Off",
                        "Late (31-120 days)", "Fully Paid", "Default"],
    })


def _write_fixture(tmp_path) -> "Path":
    fixture_dir = tmp_path / "lc_raw"; fixture_dir.mkdir()
    _lc_fixture().to_csv(fixture_dir / "accepted_2014_to_2018.csv", index=False)
    return fixture_dir


def test_load_lendingclub_returns_canonical_schema(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path))

    expected = {
        "loan_amount", "term", "int_rate", "grade", "emp_length",
        "home_ownership", "annual_income", "purpose", "dti", "fico",
        "open_acc", "revol_util", "default", "issue_d",
    }
    assert set(df.columns) == expected


def test_term_int_rate_revol_util_parsed_to_numbers(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path))

    assert df["term"].dtype.kind in "iu", df["term"].dtype
    assert set(df["term"].unique()) <= {36, 60}
    assert df["int_rate"].iloc[0] == pytest.approx(12.45)
    assert df["revol_util"].iloc[0] == pytest.approx(45.2)


def test_emp_length_handles_messy_strings(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path)).reset_index(drop=True)

    # 10+ years -> 10
    assert df.loc[0, "emp_length"] == 10
    # < 1 year -> 0
    assert df.loc[1, "emp_length"] == 0
    # "5 years" -> 5
    assert df.loc[2, "emp_length"] == 5
    # n/a -> 0
    assert df.loc[3, "emp_length"] == 0


def test_loan_status_maps_to_binary_default(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path)).reset_index(drop=True)

    # Charged Off, Late (31-120 days), Default -> 1
    expected = [0, 0, 1, 1, 0, 1]
    assert df["default"].tolist() == expected


def test_fico_is_average_of_range(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path)).reset_index(drop=True)
    # row 0: low=690 high=694 -> 692
    assert df.loc[0, "fico"] == pytest.approx(692.0)


def test_issue_d_parsed_to_datetime(tmp_path):
    df = _load_lendingclub(raw_dir=_write_fixture(tmp_path)).reset_index(drop=True)
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
    df = _load_lendingclub(raw_dir=fixture_dir)
    assert len(df) == len(fx)


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
