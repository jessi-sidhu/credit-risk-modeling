"""Scoring core: validation, consistency with Phase 2 batch code, outputs."""
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from src.ecl import discount_factors, ead_schedule, expected_loss
from src.serve.scoring import INPUT_FIELDS, LoanApplication, get_bundle, score, score_many
from src.survival import MAX_PERIODS, apply_level_offsets, cif_at, cumulative_incidence
from tests.test_model_bundle import fixture_loans

BASE = dict(loan_amount=12_000, term=36, int_rate=13.5, grade="C", emp_length=4,
            home_ownership="RENT", annual_income=52_000, purpose="debt_consolidation",
            dti=24.5, fico=685, open_acc=11, revol_util=68)


def _apps():
    df = fixture_loans()
    return [LoanApplication(**{f: (None if pd.isna(row[f]) else row[f]) for f in INPUT_FIELDS})
            for _, row in df.iterrows()], df


@pytest.mark.parametrize("bad", [
    {"annual_income": -1},
    {"purpose": "crypto"},
    {"grade": "H"},
    {"term": 48},
    {"fico": 900},
    {"issue_d": "2016-13"},
    {"surprise_field": 1},
])
def test_invalid_applications_rejected(bad):
    with pytest.raises(ValidationError):
        LoanApplication(**{**BASE, **bad})


def test_optional_fields_may_be_missing():
    app = LoanApplication(**{k: v for k, v in BASE.items() if k not in ("open_acc", "revol_util")})
    r = score(app)
    assert 0 < r["pd_lifetime"] < 1
    assert any("not provided" in c["label"] for c in r["reason_codes"]) or r["reason_codes"]


@pytest.mark.parametrize("calibrated", [True, False])
def test_score_matches_phase2_batch_prediction(calibrated):
    apps, df = _apps()
    bundle = get_bundle()
    h = bundle.model.hazards(df)                         # Phase 2 batch path
    if calibrated:
        h = apply_level_offsets(h, [bundle.level_offsets["default"], bundle.level_offsets["prepay"]])
    curves = cumulative_incidence(h)
    lgd = df["grade"].map(bundle.lgd_by_grade["lgd"]).to_numpy()
    el = expected_loss(curves["marginal_pd_default"],
                       ead_schedule(df["loan_amount"], df["int_rate"], df["term"], MAX_PERIODS),
                       lgd, discount_factors(df["int_rate"], MAX_PERIODS))
    for i, app in enumerate(apps):
        r = score(app, calibrated=calibrated)            # one request at a time
        nq = app.term // 3
        assert r["cif_default"] == pytest.approx(curves["cif_default"][i, :nq].tolist(), abs=1e-6)
        assert r["pd_12m"] == pytest.approx(cif_at(curves["cif_default"], 12)[i], abs=1e-6)
        assert r["prepay_lifetime"] == pytest.approx(curves["cif_prepay"][i, nq - 1], abs=1e-6)
        assert r["ecl_lifetime"] == pytest.approx(el["ecl_lifetime"][i], abs=1e-6)
        assert r["ecl_12m"] == pytest.approx(el["ecl_12m"][i], abs=1e-6)


def test_batch_and_single_scoring_agree():
    apps, _ = _apps()
    batch = score_many(apps)
    for app, b in zip(apps, batch):
        s = score(app)
        assert s["pd_lifetime"] == pytest.approx(b["pd_lifetime"], abs=1e-9)
        assert s["reason_codes"] == b["reason_codes"]


def test_calibrated_is_default_and_raises_pd():
    app = LoanApplication(**BASE)
    cal, raw = score(app), score(app, calibrated=False)
    assert cal["calibrated"] is True and raw["calibrated"] is False
    assert cal["pd_lifetime"] > raw["pd_lifetime"]


def test_response_shape_reasons_and_warnings():
    r = score(LoanApplication(**{**BASE, "annual_income": 2_000_000, "term": 60}))
    assert len(r["cif_default"]) == len(r["cif_default_months"]) == 20
    assert r["cif_default_months"][-1] == 60
    assert np.all(np.diff(r["cif_default"]) >= -1e-9)
    assert r["pd_lifetime"] == pytest.approx(r["cif_default"][-1])
    assert 0 < r["ecl_12m"] < r["ecl_lifetime"] < 12_000
    assert any("annual_income" in w and "above the training range" in w for w in r["warnings"])
    assert 1 <= len(r["reason_codes"]) <= 4
    assert all(c["shap_log_odds"] > 0 and c["label"] for c in r["reason_codes"])
    assert r["model_version"] == "pd_survival_v1"
