"""Prediction logging and outcomes, against a temporary SQLite database."""
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

import src.serve.app as app_module
from src.serve.app import app, state
from src.serve.db import Outcome, Prediction, SCHEMA_VERSION, SchemaVersion, init_db, make_engine
from tests.test_api import APP


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/p.db")
    with TestClient(app) as c:
        yield c


def _count(model):
    with state.Session() as s:
        return s.scalar(select(func.count()).select_from(model))


def test_every_scored_request_is_logged(client):
    r = client.post("/v1/score", json={**APP, "loan_id": "L1", "issue_d": "2016-03"})
    assert r.status_code == 200
    client.post("/v1/score/batch", json={"applications": [{**APP, "loan_id": f"B{i}"} for i in range(3)]})
    assert _count(Prediction) == 4
    with state.Session() as s:
        p = s.scalars(select(Prediction).where(Prediction.loan_id == "L1")).one()
    assert p.request_id == r.json()["request_id"]
    assert p.features["issue_d"] == "2016-03" and p.features["dti"] == APP["dti"]
    assert p.pd_12m == pytest.approx(r.json()["pd_12m"]) and p.calibrated is True
    assert p.latency_ms > 0 and p.model_version == "pd_survival_v1"


def test_outcomes_endpoint(client):
    client.post("/v1/score", json={**APP, "loan_id": "L9"})
    r = client.post("/v1/outcomes", json={"outcomes": [
        {"loan_id": "L9", "event": "default", "duration_months": 7},
        {"loan_id": "never-scored", "event": "censored", "duration_months": 12}]})
    assert r.status_code == 200
    assert r.json()["stored"] == 2 and r.json()["matched_scored_loans"] == 1
    assert _count(Outcome) == 2


def test_outcomes_validation(client):
    bad = {"outcomes": [{"loan_id": "L1", "event": "late", "duration_months": 3}]}
    assert client.post("/v1/outcomes", json=bad).status_code == 422


def test_logging_failure_does_not_fail_scoring(client, monkeypatch, caplog):
    class Broken:
        def begin(self):
            raise RuntimeError("database is down")
    monkeypatch.setattr(state, "Session", Broken())
    r = client.post("/v1/score", json=APP)
    assert r.status_code == 200 and 0 < r.json()["pd_lifetime"] < 1
    assert any("failed to log" in m for m in caplog.messages)


def test_schema_version_recorded_and_checked(tmp_path):
    engine = make_engine(f"sqlite:///{tmp_path}/v.db")
    Session = init_db(engine)
    init_db(engine)                                     # idempotent
    with Session() as s:
        assert s.scalars(select(SchemaVersion.version)).all() == [SCHEMA_VERSION]
    with Session.begin() as s:
        s.add(SchemaVersion(version=SCHEMA_VERSION + 1))
    with pytest.raises(RuntimeError):
        init_db(engine)
