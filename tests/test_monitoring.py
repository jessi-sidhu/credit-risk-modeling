"""Monitoring: PSI vs reference, 12-month labels, calibration alert, report."""
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.serve.app import app, state
from src.serve.bundle import load_bundle
from src.serve.monitoring import (
    calibration,
    psi_vs_reference,
    twelve_month_labels,
    write_report,
)
from tests.test_api import APP


def test_psi_vs_reference_numeric_and_categorical():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 20_000)
    edges = np.r_[-np.inf, np.quantile(base, np.linspace(0.1, 0.9, 9)), np.inf]
    ref = {"kind": "numeric", "edges": edges.tolist(), "shares": [0.1] * 10 + [0.0]}
    ref["shares"] = [max(s, 1e-6) for s in ref["shares"]]
    assert psi_vs_reference(pd.Series(rng.normal(0, 1, 20_000)), ref) < 0.01
    assert psi_vs_reference(pd.Series(rng.normal(1, 1, 20_000)), ref) > 0.25
    cat = {"kind": "categorical", "shares": {"A": 0.5, "B": 0.5}}
    assert psi_vs_reference(pd.Series(["A", "B"] * 500), cat) == pytest.approx(0, abs=1e-9)
    assert psi_vs_reference(pd.Series(["A"] * 900 + ["C"] * 100), cat) > 0.25


def test_twelve_month_labels():
    o = pd.DataFrame({"loan_id": list("abcdef"),
                      "event": ["default", "default", "prepay", "censored", "censored", "prepay"],
                      "duration_months": [5, 20, 8, 15, 6, 14]})
    lab = twelve_month_labels(o).set_index("loan_id")["default_12m"]
    assert lab.to_dict() == {"a": 1, "b": 0, "c": 0, "d": 0, "f": 0}      # e unknown: censored at 6


def _preds_and_outcomes(n, pred, rate, seed=0):
    rng = np.random.default_rng(seed)
    ids = [f"L{i}" for i in range(n)]
    preds = pd.DataFrame({"loan_id_logged": ids, "pd_12m": pred, "month": "2016-01"})
    y = rng.random(n) < rate
    outs = pd.DataFrame({"loan_id": ids, "event": np.where(y, "default", "censored"),
                         "duration_months": np.where(y, 6, 12)})
    return preds, outs


def test_calibration_alert_fires_on_under_prediction_only_with_enough_loans():
    p, o = _preds_and_outcomes(5_000, pred=0.045, rate=0.06)
    cal = calibration(p, o)
    assert cal["status"] == "alert" and cal["pooled"]["ratio"] == pytest.approx(0.75, abs=0.05)
    p, o = _preds_and_outcomes(5_000, pred=0.06, rate=0.06, seed=1)
    assert calibration(p, o)["status"] == "ok"
    p, o = _preds_and_outcomes(300, pred=0.045, rate=0.06)
    assert calibration(p, o)["status"] == "insufficient outcomes"


def test_monitoring_endpoint_and_report(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/m.db")
    with TestClient(app) as client:
        apps = [{**APP, "loan_id": f"L{i}", "issue_d": "2016-01", "dti": 10 + i % 25} for i in range(40)]
        client.post("/v1/score/batch", json={"applications": apps})
        client.post("/v1/outcomes", json={"outcomes": [
            {"loan_id": f"L{i}", "event": "default" if i % 10 == 0 else "censored",
             "duration_months": 6 if i % 10 == 0 else 12} for i in range(40)]})
        r = client.get("/v1/monitoring?month_from=2016-01&month_to=2016-01")
        assert r.status_code == 200
        body = r.json()
        assert body["n_predictions"] == 40
        feats = {d["feature"] for d in body["input_drift"]}
        assert {"dti", "int_rate", "grade"} <= feats
        assert body["calibration"]["n_loans"] == 40
        assert body["calibration"]["status"] == "insufficient outcomes"
        assert body["score_drift"][0]["month"] == "2016-01"
        assert client.get("/v1/monitoring?month_from=2016").status_code == 422
        path = write_report({k: v for k, v in body.items() if k != "request_id"}, tmp_path / "mon")
    html = path.read_text()
    assert "<img" in html and "Input drift" in html
    assert json.loads((tmp_path / "mon" / "summary.json").read_text())["n_predictions"] == 40
