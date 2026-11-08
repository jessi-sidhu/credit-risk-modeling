"""Person-period expansion, hazard models and hazard -> CIF conversion."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import engineer_features
from src.survival import (
    MAX_PERIODS,
    HazardModel,
    cif_at,
    cumulative_incidence,
    expand_person_period,
    periods_at_risk,
)
from src.survival_data import EVENT_CENSORED as C, EVENT_DEFAULT as D, EVENT_PREPAY as P


def _loans(durations, events, terms=None, seed=0):
    n = len(durations)
    rng = np.random.default_rng(seed)
    terms = terms if terms is not None else [36] * n
    feats = pd.DataFrame({
        "loan_amount": rng.uniform(1_000, 35_000, n),
        "term": terms,
        "int_rate": rng.uniform(5, 25, n),
        "grade": rng.choice(list("ABCDEFG"), n),
        "emp_length": rng.integers(0, 11, n),
        "home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        "annual_income": rng.uniform(20_000, 200_000, n),
        "purpose": rng.choice(["credit_card", "debt_consolidation", "car"], n),
        "dti": rng.uniform(0, 40, n),
        "fico": rng.uniform(660, 850, n),
        "open_acc": rng.integers(1, 30, n).astype(float),
        "revol_util": rng.uniform(0, 100, n),
        "issue_d": pd.Timestamp("2012-01-01"),
    }, index=[f"id{i}" for i in range(n)])
    outs = pd.DataFrame({"duration": durations, "event": events}, index=feats.index)
    return engineer_features(feats), outs


def test_periods_at_risk_ceil_for_events_floor_for_censored():
    _, o = _loans([1, 3, 4, 7, 7, 2, 36], [D, D, P, D, C, C, C])
    assert periods_at_risk(o).tolist() == [1, 1, 2, 3, 2, 0, 12]


def test_expand_rows_per_loan_and_target_on_last_row_only():
    f, o = _loans([7, 13, 20, 36], [D, P, C, C], terms=[36, 60, 60, 36])
    pp = expand_person_period(f, o)
    counts = pp.groupby("loan_id", observed=True).size()
    assert counts.loc[f.index].tolist() == [3, 5, 6, 12]
    for lid, g in pp.groupby("loan_id", observed=True):
        g = g.sort_values("period")
        assert g["period"].tolist() == list(range(1, len(g) + 1))
        assert (g["y"].iloc[:-1] == 0).all()
    last = pp.sort_values("period").groupby("loan_id", observed=True).tail(1).set_index("loan_id")
    assert last.loc["id0", "y"] == D and last.loc["id1", "y"] == P


def test_censored_loans_have_no_positive_row():
    f, o = _loans([5, 9, 36, 2], [C, C, C, C])
    pp = expand_person_period(f, o)
    assert (pp["y"] == 0).all()
    assert len(pp) == 1 + 3 + 12 + 0


def test_expand_keeps_features_and_uses_compact_dtypes():
    f, o = _loans([7, 13], [D, P])
    pp = expand_person_period(f, o)
    assert pp["fico"].dtype == np.float32
    assert isinstance(pp["grade"].dtype, pd.CategoricalDtype)
    row = pp[pp["loan_id"] == "id1"].iloc[0]
    assert row["fico"] == pytest.approx(f.loc["id1", "fico"], rel=1e-6)
    assert row["grade"] == f.loc["id1", "grade"]


def test_expand_requires_aligned_index():
    f, o = _loans([7], [D])
    with pytest.raises(ValueError):
        expand_person_period(f, o.rename(index={"id0": "other"}))


def test_cumulative_incidence_matches_hand_computation():
    # two quarters, constant hazards: default 0.1, prepay 0.2
    h = np.full((1, 2, 3), np.nan)
    h[0, :, 1], h[0, :, 2] = 0.1, 0.2
    c = cumulative_incidence(h)
    assert c["survival"][0].tolist() == pytest.approx([0.7, 0.49])
    assert c["cif_default"][0].tolist() == pytest.approx([0.1, 0.1 + 0.7 * 0.1])
    assert c["cif_prepay"][0].tolist() == pytest.approx([0.2, 0.2 + 0.7 * 0.2])
    # S + CIF_default + CIF_prepay = 1 at every t
    assert (c["survival"] + c["cif_default"] + c["cif_prepay"])[0] == pytest.approx([1, 1])


def test_cif_flat_after_term():
    h = np.full((1, MAX_PERIODS, 3), np.nan)
    h[0, :12, 1], h[0, :12, 2] = 0.02, 0.05
    c = cumulative_incidence(h)["cif_default"]
    assert cif_at(c, 36) == pytest.approx(cif_at(c, 60))
    assert cif_at(c, 12) < cif_at(c, 36)


@pytest.mark.parametrize("kind", ["lr", "lgbm"])
def test_hazard_model_fits_and_predicts_curves(kind):
    rng = np.random.default_rng(1)
    n = 400
    terms = rng.choice([36, 60], n).tolist()
    ev = rng.choice([C, D, P], n, p=[0.5, 0.2, 0.3])
    dur = [int(rng.integers(1, t + 1)) for t in terms]
    f, o = _loans(dur, ev.tolist(), terms=terms, seed=1)
    pp = expand_person_period(f, o)
    params = {"n_estimators": 20} if kind == "lgbm" else {}
    m = HazardModel(kind, **params).fit(pp, eval_pp=pp if kind == "lgbm" else None)

    rows = m.predict_rows(pp)
    assert rows.shape == (len(pp), 3)
    assert np.allclose(rows.sum(axis=1), 1, atol=1e-5)

    h = m.hazards(f, chunk=150)
    assert h.shape == (n, MAX_PERIODS, 3)
    is36 = f["term"].to_numpy() == 36
    assert np.isnan(h[is36, 12:]).all() and not np.isnan(h[is36, :12]).any()
    assert not np.isnan(h[~is36]).any()
    # hazards from the loan-level path equal the row-level path
    first = pp[pp["period"] == 1].set_index("loan_id")
    first_probs = m.predict_rows(first.reset_index())
    pos = [f.index.get_loc(i) for i in first.index]
    assert np.allclose(h[pos, 0], first_probs, atol=1e-5)

    c = cumulative_incidence(h)
    assert ((c["cif_default"] >= 0) & (c["cif_default"] <= 1)).all()
    assert (np.diff(c["cif_default"], axis=1) >= -1e-7).all()


def test_extreme_and_infinite_inputs_do_not_saturate_lr_hazards():
    rng = np.random.default_rng(2)
    n = 400
    terms = rng.choice([36, 60], n).tolist()
    ev = rng.choice([C, D, P], n, p=[0.5, 0.2, 0.3])
    dur = [int(rng.integers(1, t + 1)) for t in terms]
    f, o = _loans(dur, ev.tolist(), terms=terms, seed=2)
    m = HazardModel("lr").fit(expand_person_period(f, o))
    weird = f.iloc[:3].copy()
    weird["annual_income"] = [0.0, 1.0, 1e9]
    weird = engineer_features(weird.drop(columns=["loan_to_income", "installment_estimate",
                                                  "installment_to_income", "emp_x_homeowner"]))
    weird["dti"] = [999.0, -1.0, 500.0]
    assert np.isinf(weird["loan_to_income"]).any()
    h = m.hazards(weird)
    first = h[:, 0, :]
    assert np.isfinite(first).all()
    assert (first[:, 1] < 0.9).all() and (first[:, 1] > 1e-4).all()
