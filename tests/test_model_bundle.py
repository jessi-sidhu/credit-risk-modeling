"""The committed model bundle loads and scores loans without the LC data."""
import numpy as np
import pandas as pd
import pytest

from src.features import engineer_features
from src.serve.bundle import BUNDLE_FILES, DEFAULT_BUNDLE, load_bundle
from src.survival import MAX_PERIODS, cumulative_incidence


def fixture_loans() -> pd.DataFrame:
    return engineer_features(pd.DataFrame({
        "loan_amount": [5_000.0, 12_000.0, 25_000.0, 35_000.0, 8_000.0],
        "term": [36, 36, 60, 60, 36],
        "int_rate": [6.5, 11.0, 15.5, 22.0, 9.0],
        "grade": ["A", "B", "C", "E", "B"],
        "emp_length": [10, 3, 0, 6, 1],
        "home_ownership": ["MORTGAGE", "RENT", "OWN", "RENT", "MORTGAGE"],
        "annual_income": [120_000.0, 55_000.0, 70_000.0, 48_000.0, 90_000.0],
        "purpose": ["credit_card", "debt_consolidation", "home_improvement", "small_business", "car"],
        "dti": [8.0, 18.0, 24.0, 33.0, 12.0],
        "fico": [780.0, 700.0, 680.0, 665.0, 720.0],
        "open_acc": [9.0, 11.0, 14.0, 7.0, np.nan],
        "revol_util": [15.0, 55.0, 70.0, 90.0, 30.0],
        "issue_d": pd.Timestamp("2016-01-01"),
    }, index=[f"fx{i}" for i in range(5)]))


@pytest.fixture(scope="module")
def bundle():
    return load_bundle()


def test_bundle_files_present_and_small():
    sizes = [(DEFAULT_BUNDLE / f).stat().st_size for f in BUNDLE_FILES]
    assert sum(sizes) < 20e6


def test_bundle_scores_without_touching_lc_data(bundle, monkeypatch):
    import src.data as data
    def no_data(*_, **__):
        raise AssertionError("the bundle must not read data/archive/")
    monkeypatch.setattr(data, "find_accepted_file", no_data)

    loans = fixture_loans()
    h = bundle.model.hazards(loans)
    assert h.shape == (5, MAX_PERIODS, 3)
    is36 = loans["term"].to_numpy() == 36
    assert not np.isnan(h[is36, :12]).any() and np.isnan(h[is36, 12:]).all()
    cif = cumulative_incidence(h)["cif_default"]
    end = cif[np.arange(5), loans["term"].to_numpy() // 3 - 1]
    assert ((end > 0) & (end < 1)).all()
    # grade A, low DTI, high FICO should be the safest of the five
    assert end.argmin() == 0


def test_bundle_metadata_is_consistent(bundle):
    assert bundle.version == "pd_survival_v1"
    assert set(bundle.lgd_by_grade["lgd"]) == set("ABCDEFG")
    assert {"default", "prepay"} <= set(bundle.level_offsets)
    fields = bundle.schema["fields"]
    assert fields["grade"]["allowed"] == list("ABCDEFG")
    assert set(bundle.reference["features"]) == set(fields)
    ref = bundle.reference["features"]["fico"]
    assert len(ref["shares"]) == len(ref["edges"])            # bins + missing bin
    assert sum(ref["shares"]) == pytest.approx(1, abs=1e-3)
