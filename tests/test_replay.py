"""Replay helpers (the replay itself needs data/archive/ and runs locally)."""
import numpy as np
import pandas as pd

from src.serve.replay import _month_add, outcome_at_12m, to_application
from src.serve.scoring import LoanApplication
from src.survival_data import EVENT_CENSORED as C, EVENT_DEFAULT as D, EVENT_PREPAY as P


def _row(duration, event, finished):
    return pd.Series({"duration": duration, "event": event, "finished": finished})


def test_outcome_known_twelve_months_after_issue():
    assert outcome_at_12m("a", _row(7, D, True)) == {"loan_id": "a", "event": "default", "duration_months": 7}
    assert outcome_at_12m("b", _row(20, D, True))["event"] == "censored"          # defaults later
    assert outcome_at_12m("b", _row(20, D, True))["duration_months"] == 12
    assert outcome_at_12m("c", _row(5, P, True))["event"] == "prepay"
    assert outcome_at_12m("d", _row(36, C, True)) == {"loan_id": "d", "event": "censored", "duration_months": 12}
    unknown = outcome_at_12m("e", _row(8, C, False))                              # open, seen for 8 months
    assert unknown["event"] == "censored" and unknown["duration_months"] == 8


def test_to_application_builds_a_valid_request():
    row = pd.Series({"loan_amount": 10_000.0, "term": 36, "int_rate": 12.5, "grade": "B", "emp_length": 3,
                     "home_ownership": "RENT", "annual_income": 50_000.0, "purpose": "car", "dti": 14.2,
                     "fico": 702.0, "open_acc": np.nan, "revol_util": 33.0, "month": "2016-04"})
    a = to_application("123", row)
    assert a["open_acc"] is None and a["issue_d"] == "2016-04" and a["loan_id"] == "123"
    LoanApplication(**a)


def test_month_add():
    assert _month_add("2016-01", -12) == "2015-01" and _month_add("2015-12", 1) == "2016-01"
