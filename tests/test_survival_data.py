"""Outcome table (duration + event) for the survival model."""
from __future__ import annotations

import pandas as pd
import pytest

from src.survival_data import (
    EVENT_CENSORED as C,
    EVENT_DEFAULT as D,
    EVENT_PREPAY as P,
    OUTCOME_COLS,
    build_outcomes,
    load_survival_data,
)

_DNM = "Does not meet the credit policy. Status:"

# (loan_status, issue_d, term, last_pymnt_d, expected event, expected duration, clipped)
CASES = [
    ("Charged Off",          "Jan-2015", 36, "Jun-2016", D, 18, False),   # 17 months + 1
    ("Default",              "Jan-2015", 36, "Mar-2015", D, 3,  False),
    (_DNM + "Charged Off",   "Jan-2015", 36, "Jan-2016", D, 13, False),
    ("Charged Off",          "Jan-2015", 36, None,       D, 1,  False),   # no payment ever
    ("Fully Paid",           "Jan-2015", 36, "Jul-2016", P, 18, False),
    (_DNM + "Fully Paid",    "Jan-2015", 36, "Jan-2016", P, 12, False),
    ("Fully Paid",           "Jan-2015", 36, "Oct-2017", P, 33, False),   # 33 < 36 - 2: prepay
    ("Fully Paid",           "Jan-2015", 36, "Nov-2017", C, 36, False),   # 34: near maturity
    ("Fully Paid",           "Jan-2015", 36, "Dec-2017", C, 36, False),
    ("Current",              "Jan-2017", 36, "Mar-2019", C, 26, False),
    ("Late (31-120 days)",   "Jan-2017", 36, "Dec-2018", C, 23, False),
    ("In Grace Period",      "Jan-2017", 36, "Feb-2019", C, 25, False),
    ("Late (16-30 days)",    "Jan-2017", 36, None,       C, 26, False),   # falls back to snapshot
    ("Fully Paid",           "Jan-2015", 36, "Jan-2015", P, 1,  True),    # paid in issue month: 0 -> 1
    ("Charged Off",          "Jan-2015", 36, "Mar-2018", D, 36, True),    # 38 + 1 -> 36
    ("Fully Paid",           "Jan-2015", 60, "Jan-2017", P, 24, False),   # 60-month prepay
]
IMPOSSIBLE = ("Fully Paid", "Jan-2015", 36, "Dec-2014")                   # paid before issue


def _raw(rows) -> pd.DataFrame:
    return pd.DataFrame({
        "loan_status": [r[0] for r in rows],
        "issue_d": pd.to_datetime([r[1] for r in rows], format="%b-%Y"),
        "term": [r[2] for r in rows],
        "last_pymnt_d": pd.to_datetime([r[3] for r in rows], format="%b-%Y"),
    })


@pytest.mark.parametrize("case", CASES, ids=[f"{c[0]}|{c[3]}" for c in CASES])
def test_build_outcomes_event_and_duration(case):
    out = build_outcomes(_raw([case]), snapshot_date="2019-03-31").iloc[0]
    assert out["event"] == case[4]
    assert out["duration"] == case[5]
    assert bool(out["clipped"]) == case[6]
    assert not out["impossible"]


def test_build_outcomes_flags_impossible_dates_and_unknown_status():
    out = build_outcomes(_raw([IMPOSSIBLE, ("Something New", "Jan-2015", 36, "Jun-2015")]))
    assert out["impossible"].all()


def _lc_csv(tmp_path, rows) -> "Path":
    n = len(rows)
    raw = pd.DataFrame({
        "id": [str(1000 + i) for i in range(n)],
        "loan_amnt": [10_000] * n,
        "term": [f" {r[2]} months" for r in rows],
        "int_rate": ["12.00%"] * n,
        "grade": ["B"] * n,
        "emp_length": ["5 years"] * n,
        "home_ownership": ["RENT"] * n,
        "annual_inc": [60_000] * n,
        "purpose": ["credit_card"] * n,
        "dti": [15.0] * n,
        "fico_range_low": [700] * n,
        "fico_range_high": [704] * n,
        "open_acc": [9] * n,
        "revol_util": ["40.0%"] * n,
        "issue_d": [r[1] for r in rows],
        "loan_status": [r[0] for r in rows],
        "last_pymnt_d": [r[3] for r in rows],
        "total_pymnt": [1.0] * n,          # post-origination: must not be read
    })
    d = tmp_path / "lc"; d.mkdir()
    path = d / "accepted_2007_to_2018Q4.csv"
    raw.to_csv(path, index=False)
    with open(path, "a") as f:                                   # LC footer line
        f.write("Total amount funded in policy code 1: 123" + "," * (raw.shape[1] - 1) + "\n")
    return d


def test_load_survival_data_end_to_end(tmp_path):
    feats, outs = load_survival_data(raw_dir=_lc_csv(tmp_path, CASES + [IMPOSSIBLE]))
    assert len(feats) == len(outs) == len(CASES)               # footer + impossible dropped
    assert feats.index.equals(outs.index)
    assert outs["event"].tolist() == [c[4] for c in CASES]
    assert outs["duration"].tolist() == [c[5] for c in CASES]
    assert outs["finished"].tolist() == [c[0] in ("Charged Off", "Default", "Fully Paid")
                                         or c[0].startswith(_DNM) for c in CASES]
    assert set(feats["term"]) == {36, 60}
    rc = feats.attrs["row_counts"]
    assert rc["impossible"] == 1 and rc["clipped"] == 2 and rc["final"] == len(CASES)


def test_feature_frame_has_no_outcome_columns(tmp_path):
    feats, outs = load_survival_data(raw_dir=_lc_csv(tmp_path, CASES))
    forbidden = set(OUTCOME_COLS) | {"default", "event", "duration", "total_pymnt"}
    assert not forbidden & set(feats.columns)
    assert {"duration", "event", "loan_status"} <= set(outs.columns)
    assert "last_pymnt_d" not in outs.columns


def test_drop_clipped_flag(tmp_path, monkeypatch):
    import src.survival_data as sd
    monkeypatch.setattr(sd, "DROP_CLIPPED", True)
    feats, _ = sd.load_survival_data(raw_dir=_lc_csv(tmp_path, CASES))
    assert len(feats) == len(CASES) - 2
