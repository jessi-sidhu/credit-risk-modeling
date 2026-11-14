"""API tests with FastAPI's TestClient."""
import uuid

import pytest
from fastapi.testclient import TestClient

from src.serve.app import MAX_BATCH, app

APP = dict(loan_amount=12_000, term=36, int_rate=13.5, grade="C", emp_length=4,
           home_ownership="RENT", annual_income=52_000, purpose="debt_consolidation",
           dti=24.5, fico=685, open_acc=11, revol_util=68)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # runs the startup (model load) once
        yield c


def _is_uuid(s):
    return str(uuid.UUID(s)) == s


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True
    assert body["model_version"] == "pd_survival_v1" and _is_uuid(body["request_id"])


def test_model_card(client):
    body = client.get("/v1/model").json()
    assert body["model_version"] == "pd_survival_v1"
    assert "known_under_prediction" in body and "auc_default_by_horizon" in body["test"]


def test_score_happy_path(client):
    r = client.post("/v1/score", json={**APP, "loan_id": "L1"})
    assert r.status_code == 200
    body = r.json()
    assert _is_uuid(body["request_id"]) and r.headers["X-Request-ID"] == body["request_id"]
    assert body["loan_id"] == "L1" and body["calibrated"] is True
    assert 0 < body["pd_12m"] < body["pd_lifetime"] < 1
    assert len(body["cif_default"]) == 12 and 1 <= len(body["reason_codes"]) <= 4


def test_score_raw_model_via_query(client):
    cal = client.post("/v1/score", json=APP).json()
    raw = client.post("/v1/score?calibrated=false", json=APP).json()
    assert raw["calibrated"] is False and raw["pd_lifetime"] < cal["pd_lifetime"]


def test_batch(client):
    apps = [APP, {**APP, "term": 60, "grade": "A", "int_rate": 7.0}, {**APP, "purpose": "car"}]
    r = client.post("/v1/score/batch", json={"applications": apps})
    assert r.status_code == 200
    body = r.json()
    assert len(body["results"]) == 3
    ids = [x["request_id"] for x in body["results"]]
    assert len(set(ids)) == 3 and all(_is_uuid(i) for i in ids)
    assert len(body["results"][1]["cif_default"]) == 20


def test_batch_size_limits(client):
    assert client.post("/v1/score/batch", json={"applications": []}).status_code == 422
    too_many = {"applications": [APP] * (MAX_BATCH + 1)}
    assert client.post("/v1/score/batch", json=too_many).status_code == 422


@pytest.mark.parametrize("bad,field", [
    ({"annual_income": -5}, "annual_income"),
    ({"purpose": "crypto"}, "purpose"),
    ({"term": 48}, "term"),
    ({"extra": 1}, "extra"),
])
def test_validation_errors_are_clear(client, bad, field):
    r = client.post("/v1/score", json={**APP, **bad})
    assert r.status_code == 422
    body = r.json()
    assert _is_uuid(body["request_id"]) and body["error"] == "invalid input"
    assert any(d["field"] == field for d in body["details"])


def test_missing_required_field(client):
    body = {k: v for k, v in APP.items() if k != "fico"}
    r = client.post("/v1/score", json=body)
    assert r.status_code == 422 and any(d["field"] == "fico" for d in r.json()["details"])


def test_rate_limiter_window():
    from src.serve.app import RateLimiter
    rl = RateLimiter(per_minute=3)
    assert [rl.retry_after("ip", now=t) for t in (0, 1, 2)] == [0, 0, 0]
    assert rl.retry_after("ip", now=3) == pytest.approx(57)
    assert rl.retry_after("other-ip", now=3) == 0
    assert rl.retry_after("ip", now=60.5) == 0                 # first hit expired


def test_rate_limit_returns_429(client, monkeypatch):
    import src.serve.app as app_module
    monkeypatch.setattr(app_module, "rate_limiter", app_module.RateLimiter(per_minute=2))
    codes = [client.post("/v1/score", json=APP).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/v1/score", json=APP)
    assert r.status_code == 429 and "Retry-After" in r.headers and r.json()["request_id"]
    assert client.get("/health").status_code == 200            # health is never limited


def test_landing_page_and_schema(client):
    r = client.get("/")
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
    html = r.text
    assert "<form" in html and "/v1/score" in html and "/docs" in html
    assert client.get("/favicon.ico").status_code == 204
    s = client.get("/v1/schema").json()
    assert s["fields"]["grade"]["allowed"] == list("ABCDEFG")
    assert "debt_consolidation" in s["fields"]["purpose"]["allowed"]
    assert s["request_id"]
