"""Prediction and outcome storage (SQLAlchemy).

DATABASE_URL selects the database: default `sqlite:///predictions.db`;
Postgres works by setting e.g. `postgresql+psycopg://user:pass@host/db`.

The schema is created with `create_all` and recorded in `schema_version`,
so a fresh database is always built the same way; `init_db` refuses to run
against a database with a different schema version.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    Integer,
    String,
    create_engine,
    select,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

SCHEMA_VERSION = 1
DEFAULT_DATABASE_URL = "sqlite:///predictions.db"
EVENTS = ("default", "prepay", "censored")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    __tablename__ = "predictions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    request_id: Mapped[str] = mapped_column(String(36), unique=True, index=True)
    loan_id: Mapped[Optional[str]] = mapped_column(String(64), index=True, nullable=True)
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    model_version: Mapped[str] = mapped_column(String(32))
    features: Mapped[dict] = mapped_column(JSON)
    pd_12m: Mapped[float] = mapped_column(Float)
    pd_lifetime: Mapped[float] = mapped_column(Float)
    ecl_lifetime: Mapped[float] = mapped_column(Float)
    calibrated: Mapped[bool] = mapped_column(Boolean)
    latency_ms: Mapped[float] = mapped_column(Float)


class Outcome(Base):
    __tablename__ = "outcomes"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    loan_id: Mapped[str] = mapped_column(String(64), index=True)
    observed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    event: Mapped[str] = mapped_column(String(16))
    duration_months: Mapped[int] = mapped_column(Integer)


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


def database_url() -> str:
    return os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)


def make_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    kwargs = {"connect_args": {"check_same_thread": False}} if url.startswith("sqlite") else {"pool_pre_ping": True}
    return create_engine(url, **kwargs)


def init_db(engine: Engine) -> sessionmaker:
    Base.metadata.create_all(engine)
    Session = sessionmaker(engine, expire_on_commit=False)
    with Session.begin() as s:
        versions = s.scalars(select(SchemaVersion.version)).all()
        if not versions:
            s.add(SchemaVersion(version=SCHEMA_VERSION))
        elif max(versions) != SCHEMA_VERSION:
            raise RuntimeError(f"database schema version {max(versions)} != code version {SCHEMA_VERSION}")
    return Session
