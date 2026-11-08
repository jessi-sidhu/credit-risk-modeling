"""Expected-loss building blocks."""
import numpy as np
import pandas as pd
import pytest

from src.ecl import discount_factors, ead_schedule, expected_loss, lgd_by_grade, realized_loss


def test_ead_starts_at_principal_and_amortizes_to_zero():
    ead = ead_schedule([10_000.0], [12.0], [36], n_quarters=20)[0]
    assert ead[0] == pytest.approx(10_000.0)
    assert np.all(np.diff(ead[:12]) < 0)                  # balance falls every quarter
    assert (ead[12:] == 0).all()                          # nothing after the term
    # after 33 payments of a 36-month loan, ~3 payments (~$332 each) remain
    assert ead[11] == pytest.approx(3 * 332.14, rel=0.02)


def test_ead_zero_rate_is_straight_line():
    ead = ead_schedule([3_600.0], [0.0], [36], n_quarters=12)[0]
    assert ead[:4].tolist() == pytest.approx([3600, 3300, 3000, 2700])


def test_discount_factors():
    d = discount_factors([12.0], n_quarters=2)[0]
    assert d.tolist() == pytest.approx([1.01 ** -3, 1.01 ** -6])


def test_lgd_formula_and_clipping():
    td = pd.DataFrame({
        "loan_amnt":       [10_000, 10_000, 10_000],
        "total_rec_prncp": [4_000,  4_000,  9_000],
        "recoveries":      [600,    7_000,  0],
        "grade":           ["A",    "A",    "B"],
    })
    out = lgd_by_grade(td)
    # A: (6000-600)/6000 = 0.9 and (6000-7000) clipped to 0 -> mean 0.45; B: 1000/1000 = 1
    assert out.loc["A", "lgd"] == pytest.approx(0.45)
    assert out.loc["B", "lgd"] == pytest.approx(1.0)
    assert realized_loss(td).tolist() == [5400, 0, 1000]


def test_expected_loss_hand_computation():
    mpd = np.array([[0.1, 0.2, 0.0, 0.0, 0.05]])
    ead = np.array([[100.0, 80.0, 60.0, 40.0, 20.0]])
    disc = np.ones((1, 5))
    out = expected_loss(mpd, ead, [0.5], disc)
    assert out["ecl_12m"][0] == pytest.approx(0.5 * (10 + 16))
    assert out["ecl_lifetime"][0] == pytest.approx(0.5 * (10 + 16 + 1))
    assert out["el_lifetime_undiscounted"][0] == pytest.approx(out["ecl_lifetime"][0])
