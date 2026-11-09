"""FastAPI service for the Phase 2 hazard model.

    make serve        (uvicorn src.serve.app:app --reload)

POST /v1/score          one application -> default curve, PD, ECL, reasons
POST /v1/score/batch    up to MAX_BATCH applications
GET  /health            liveness and whether the model loaded
GET  /v1/model          the bundle's model card

The model bundle is loaded once at startup. Every response carries a
`request_id` (also in the X-Request-ID header).
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .bundle import ModelBundle, load_bundle
from .scoring import LoanApplication, score_many

logger = logging.getLogger("src.serve")

MAX_BATCH = 1_000


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


class State:
    bundle: Optional[ModelBundle] = None
    load_error: Optional[str] = None


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
    yield


app = FastAPI(title="Credit risk scoring", version="1.0", lifespan=lifespan,
              description="Default curve, PD and expected loss for Lending Club-style loan applications "
                          "(Phase 2 competing-risks hazard model). Illustration only, not for credit decisions.")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request.state.request_id = str(uuid.uuid4())
    request.state.started = time.perf_counter()
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


@app.get("/health")
def health(request: Request):
    return {"request_id": request.state.request_id,
            "status": "ok" if state.bundle is not None else "degraded",
            "model_loaded": state.bundle is not None,
            "model_version": state.bundle.version if state.bundle else None,
            "error": state.load_error}


@app.get("/v1/model")
def model_card(request: Request):
    return {"request_id": request.state.request_id, **_bundle().card}
