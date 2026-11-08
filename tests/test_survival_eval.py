"""Known-status sets, Aalen-Johansen calibration and Brier bootstrap."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.survival_data import EVENT_CENSORED as C, EVENT_DEFAULT as D, EVENT_PREPAY as P
from src.survival_eval import (
    _weighted_brier_draws,
    aalen_johansen_cif,
    calibration_table,
    known_status_at,
)


def _table(rows):
    # (duration, event, finished)
    return pd.DataFrame(rows, columns=["duration", "event", "finished"])


def test_known_status_at_horizon():
    t = _table([
        (5, D, True),     # defaulted by 12 -> positive
        (20, D, True),    # defaults after 12 -> negative at 12
        (8, P, True),     # prepaid before 12 -> known negative
        (8, C, False),    # open loan, last seen at 8 -> unknown at 12, excluded
        (15, C, False),   # open loan seen past 12 -> negative
        (12, D, True),    # default exactly at 12 -> positive
        (36, C, True),    # paid at term -> negative
    ])
    mask, y = known_status_at(t, 12)
    assert mask.tolist() == [True, True, True, False, True, True, True]
    assert y.tolist() == [1, 0, 0, 0, 1, 0]


def test_aalen_johansen_matches_hand_computation():
    d = np.array([1, 2, 2, 3, 3, 3, 4, 5, 5, 6])
    e = np.array([D, C, P, D, D, C, P, D, C, C])
    # t=1: 1/10 -> 0.1, S=0.9; t=2: prepay 1/9 -> S=0.8; t=3: 2/7 * 0.8 = 0.22857
    assert aalen_johansen_cif(d, e, 3) == pytest.approx(0.1 + 0.8 * 2 / 7)
    assert aalen_johansen_cif(d, e, 2) == pytest.approx(0.1)
    assert np.isnan(aalen_johansen_cif(d, e, 7))      # nobody followed to month 7


def test_aalen_johansen_without_censoring_is_empirical_share():
    d = np.array([1, 2, 3, 4, 5, 6, 7, 8])
    e = np.array([D, P, D, P, P, D, P, P])
    assert aalen_johansen_cif(d, e, 8) == pytest.approx(3 / 8)


def test_calibration_table_ratio():
    t = pd.DataFrame({
        "duration": [1, 2, 3, 4, 5, 6, 7, 8] * 2,
        "event": [D, P, D, P, P, D, P, P] * 2,
        "cif_default_8_lr": [0.75] * 16,
        "cif_default_8_lgbm": [0.375] * 16,
    })
    tbl = calibration_table(t, pd.Series(["g"] * 16), horizons=(8,))
    row = tbl.iloc[0]
    assert row["observed"] == pytest.approx(3 / 8)
    assert row["ratio_lr"] == pytest.approx(2.0)
    assert row["ratio_lgbm"] == pytest.approx(1.0)


def test_weighted_brier_draws_center_on_point_estimate():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 5_000)
    p = np.clip(0.2 + 0.3 * y + rng.normal(0, 0.1, 5_000), 0, 1)
    draws = _weighted_brier_draws(y, {"m": p}, n_boot=300, seed=0)["m"]
    point = np.mean((p - y) ** 2)
    assert np.percentile(draws, 2.5) < point < np.percentile(draws, 97.5)
