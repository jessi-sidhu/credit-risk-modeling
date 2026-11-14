"""FastAPI service for the Phase 2 hazard model.

    make serve        (uvicorn src.serve.app:app --reload)

POST /v1/score          one application -> default curve, PD, ECL, reasons
POST /v1/score/batch    up to MAX_BATCH applications
POST /v1/outcomes       record outcomes for previously scored loans (labels arrive later)
GET  /v1/monitoring     input drift (PSI), score drift and calibration from the logs
GET  /health            liveness and whether the model loaded
GET  /v1/model          the bundle's model card

Every scored application is logged to the database in DATABASE_URL
(default sqlite:///predictions.db); a logging failure never fails scoring.

The model bundle is loaded once at startup. Every response carries a
`request_id` (also in the X-Request-ID header).

Public deployment settings (environment variables):
    RATE_LIMIT_PER_MINUTE   requests per client IP per minute on /v1/*; 0 = off (default)
    MAX_BATCH               maximum applications per batch request (default 1000)
"""
from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .bundle import ModelBundle, load_bundle
from .db import init_db, make_engine
from .logging import log_predictions, record_outcomes
from .monitoring import monitoring_summary
from .scoring import LoanApplication, score_many

logger = logging.getLogger("src.serve")

MAX_BATCH = int(os.environ.get("MAX_BATCH", 1_000))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", 0))


class RateLimiter:
    """Sliding one-minute window per client IP (in memory, per process)."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self.hits: dict[str, deque] = defaultdict(deque)
        self.lock = threading.Lock()

    def retry_after(self, key: str, now: float | None = None) -> float:
        """0 if the request is allowed (and counts it), else seconds to wait."""
        now = time.monotonic() if now is None else now
        with self.lock:
            q = self.hits[key]
            while q and now - q[0] >= 60:
                q.popleft()
            if len(q) >= self.per_minute:
                return 60 - (now - q[0])
            q.append(now)
            return 0.0


rate_limiter = RateLimiter(RATE_LIMIT_PER_MINUTE) if RATE_LIMIT_PER_MINUTE > 0 else None


class ReasonCode(BaseModel):
    feature: str
    label: str
    shap_log_odds: float


class ScoreResult(BaseModel):
    request_id: str
    loan_id: Optional[str] = None
    model_version: str
    calibrated: bool
    pd_12m: float = Field(description="P(default within 12 months)")
    pd_lifetime: float = Field(description="P(default before the end of the term)")
    prepay_lifetime: float = Field(description="P(prepaid before the end of the term)")
    cif_default: list[float] = Field(description="cumulative default probability at the end of each quarter")
    cif_default_months: list[int]
    ecl_12m: float = Field(description="expected credit loss over the first 12 months, USD, discounted")
    ecl_lifetime: float = Field(description="expected credit loss over the term, USD, discounted")
    lgd: float
    reason_codes: list[ReasonCode]
    warnings: list[str]


class BatchRequest(BaseModel):
    applications: list[LoanApplication] = Field(min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    request_id: str
    results: list[ScoreResult]


class OutcomeIn(BaseModel):
    loan_id: str = Field(max_length=64)
    event: Literal["default", "prepay", "censored"]
    duration_months: int = Field(ge=0, le=120, description="months from issue to the event, or observed so far")
    observed_at: Optional[datetime] = None


class OutcomesRequest(BaseModel):
    outcomes: list[OutcomeIn] = Field(min_length=1, max_length=10_000)


class State:
    bundle: Optional[ModelBundle] = None
    load_error: Optional[str] = None
    Session = None
    db_error: Optional[str] = None


state = State()


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state.bundle = load_bundle()
        # warm up SHAP and the model once so the first request isn't slow
        score_many([LoanApplication(loan_amount=10_000, term=36, int_rate=12, grade="B", emp_length=5,
                                    home_ownership="RENT", annual_income=60_000, purpose="credit_card",
                                    dti=15, fico=700)], bundle=state.bundle)
        logger.info("model %s loaded", state.bundle.version)
    except Exception as exc:                       # keep serving /health so the failure is visible
        state.load_error = repr(exc)
        logger.exception("model failed to load")
    try:
        state.engine = make_engine()
        state.Session = init_db(state.engine)
    except Exception as exc:                       # scoring still works without the database
        state.Session, state.db_error = None, repr(exc)
        logger.exception("database unavailable; predictions will not be logged")
    yield
    if state.Session is not None:
        state.engine.dispose()


app = FastAPI(title="Credit risk scoring", version="1.0", lifespan=lifespan,
              description="Default curve, PD and expected loss for Lending Club-style loan applications "
                          "(Phase 2 competing-risks hazard model). Illustration only, not for credit decisions.")


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")      # behind the host's proxy
    return fwd.split(",")[0].strip() if fwd else (request.client.host if request.client else "unknown")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    request.state.started = time.perf_counter()
    if rate_limiter is not None and request.url.path.startswith("/v1/"):
        wait = rate_limiter.retry_after(_client_ip(request))
        if wait > 0:
            return JSONResponse(status_code=429, headers={"Retry-After": str(int(wait) + 1),
                                                          "X-Request-ID": request.state.request_id},
                                content={"request_id": request.state.request_id,
                                         "error": f"rate limit: {rate_limiter.per_minute} requests per minute"})
    response = await call_next(request)
    response.headers["X-Request-ID"] = request.state.request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    errors = [{"field": ".".join(str(p) for p in e["loc"] if p != "body"), "message": e["msg"]}
              for e in exc.errors()]
    return JSONResponse(status_code=422, content={"request_id": request.state.request_id,
                                                  "error": "invalid input", "details": errors})


@app.exception_handler(HTTPException)
async def http_error(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code,
                        content={"request_id": getattr(request.state, "request_id", None), "error": exc.detail})


def _bundle() -> ModelBundle:
    if state.bundle is None:
        raise HTTPException(status_code=503, detail=f"model not loaded: {state.load_error}")
    return state.bundle


def _results(request: Request, apps: list[LoanApplication], calibrated: bool) -> list[dict]:
    bundle = _bundle()
    scored = score_many(apps, calibrated=calibrated, bundle=bundle)
    ids = [request.state.request_id] if len(apps) == 1 else [str(uuid.uuid4()) for _ in apps]
    for rid, r in zip(ids, scored):
        r["request_id"] = rid
    latency_ms = (time.perf_counter() - request.state.started) * 1000 / len(apps)
    log_predictions(state.Session, [
        {"request_id": r["request_id"], "loan_id": a.loan_id, "model_version": r["model_version"],
         "features": a.model_dump(), "pd_12m": r["pd_12m"], "pd_lifetime": r["pd_lifetime"],
         "ecl_lifetime": r["ecl_lifetime"], "calibrated": calibrated, "latency_ms": latency_ms}
        for a, r in zip(apps, scored)])
    return scored


@app.post("/v1/score", response_model=ScoreResult)
def score_one(request: Request, application: LoanApplication,
              calibrated: bool = Query(True, description="apply the 2014 level offsets (recommended)")):
    return _results(request, [application], calibrated)[0]


@app.post("/v1/score/batch", response_model=BatchResponse)
def score_batch(request: Request, body: BatchRequest,
                calibrated: bool = Query(True, description="apply the 2014 level offsets (recommended)")):
    return {"request_id": request.state.request_id,
            "results": _results(request, body.applications, calibrated)}


@app.post("/v1/outcomes")
def outcomes(request: Request, body: OutcomesRequest):
    if state.Session is None:
        raise HTTPException(status_code=503, detail=f"database unavailable: {state.db_error}")
    counts = record_outcomes(state.Session, [o.model_dump() for o in body.outcomes])
    return {"request_id": request.state.request_id, **counts}


@app.get("/v1/monitoring")
def monitoring(request: Request,
               month_from: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}$", description="YYYY-MM"),
               month_to: Optional[str] = Query(None, pattern=r"^\d{4}-\d{2}$", description="YYYY-MM"),
               calibrated: Optional[bool] = Query(None, description="only raw or only calibrated predictions")):
    if state.Session is None:
        raise HTTPException(status_code=503, detail=f"database unavailable: {state.db_error}")
    return {"request_id": request.state.request_id,
            **monitoring_summary(state.Session, _bundle(), month_from, month_to, calibrated)}


@app.get("/health")
def health(request: Request):
    return {"request_id": request.state.request_id,
            "status": "ok" if state.bundle is not None else "degraded",
            "model_loaded": state.bundle is not None,
            "model_version": state.bundle.version if state.bundle else None,
            "database": state.Session is not None,
            "error": state.load_error or state.db_error}


@app.get("/v1/model")
def model_card(request: Request):
    return {"request_id": request.state.request_id, **_bundle().card}
