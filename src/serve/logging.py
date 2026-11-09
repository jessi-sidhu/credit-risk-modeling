"""Write scored requests and arriving outcomes to the database.

Logging must never break scoring: `log_predictions` catches every error,
reports it through the standard `logging` module and returns False.
"""
from __future__ import annotations

import logging as std_logging

from sqlalchemy import select
from sqlalchemy.orm import sessionmaker

from .db import Outcome, Prediction, utcnow

logger = std_logging.getLogger("src.serve.logging")


def log_predictions(Session: sessionmaker | None, rows: list[dict]) -> bool:
    """`rows`: dicts with request_id, loan_id, model_version, features,
    pd_12m, pd_lifetime, ecl_lifetime, calibrated, latency_ms."""
    if Session is None:
        return False
    try:
        with Session.begin() as s:
            s.add_all(Prediction(received_at=utcnow(), **r) for r in rows)
        return True
    except Exception:
        logger.exception("failed to log %d prediction(s); scoring response was still returned", len(rows))
        return False


def record_outcomes(Session: sessionmaker, outcomes: list[dict]) -> dict:
    """Insert outcomes; returns how many were stored and how many match a
    previously scored loan_id."""
    ids = {o["loan_id"] for o in outcomes}
    with Session.begin() as s:
        known = set(s.scalars(select(Prediction.loan_id).where(Prediction.loan_id.in_(ids))).all())
        s.add_all(Outcome(loan_id=o["loan_id"], event=o["event"], duration_months=o["duration_months"],
                          observed_at=o.get("observed_at") or utcnow()) for o in outcomes)
    return {"stored": len(outcomes), "matched_scored_loans": sum(o["loan_id"] in known for o in outcomes)}
