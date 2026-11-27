"""Database schema (SQLAlchemy 2). SQLite by default, PostgreSQL in production.

    users ─┬─< devices ─┬─< trips ─< events
           │            └─< samples          (training windows, only with the user's consent)
           └─ consent_training / consent_at

Privacy by design: no image, video or audio column exists anywhere. Training
samples are 30 s of numeric face measurements (eye openness, mouth, head pose,
gaze), from which a face cannot be reconstructed.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

from sqlalchemy import JSON, Boolean, Float, ForeignKey, Integer, String, Text, create_engine, event
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship, sessionmaker


def _id() -> str:
    return uuid.uuid4().hex


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(120), default="")
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(16), default="driver")  # driver | admin
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    # opt-in, revocable: lets anonymised measurements from this user's trips be used to retrain models
    consent_training: Mapped[bool] = mapped_column(Boolean, default=False)
    consent_at: Mapped[float | None] = mapped_column(Float, nullable=True)
    devices: Mapped[list["Device"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class Device(Base):
    __tablename__ = "devices"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    kind: Mapped[str] = mapped_column(String(16), default="dashcam")  # dashcam | phone | web
    key_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)  # SHA-256 of the device key
    key_prefix: Mapped[str] = mapped_column(String(12))  # shown in the UI to tell keys apart
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    last_seen: Mapped[float | None] = mapped_column(Float, nullable=True)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    user: Mapped[User] = relationship(back_populates="devices")


class Trip(Base):
    __tablename__ = "trips"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # the in-car session id
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"), index=True)
    t_start: Mapped[float | None] = mapped_column(Float, nullable=True)
    t_end: Mapped[float | None] = mapped_column(Float, nullable=True)
    summary: Mapped[dict] = mapped_column(JSON, default=dict)
    kss: Mapped[int | None] = mapped_column(Integer, nullable=True)  # driver's own sleepiness rating (1-9)


class Event(Base):
    __tablename__ = "events"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # generated in the car: uploads are idempotent
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"))
    trip_id: Mapped[str] = mapped_column(String(64), index=True)
    t_wall: Mapped[float] = mapped_column(Float, index=True)
    t_session: Mapped[float | None] = mapped_column(Float, nullable=True)
    kind: Mapped[str] = mapped_column(String(32), index=True)
    severity: Mapped[str | None] = mapped_column(String(16), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    feedback: Mapped[str | None] = mapped_column(String(16), nullable=True)  # correct | false_alarm


class Sample(Base):
    """One training window for the temporal drowsiness model (same format as UTA-RLDD features)."""
    __tablename__ = "samples"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    device_id: Mapped[str] = mapped_column(ForeignKey("devices.id", ondelete="CASCADE"))
    trip_id: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    label: Mapped[int] = mapped_column(Integer)  # 0 alert, 1 low vigilant, 2 drowsy (UTA-RLDD scale)
    label_source: Mapped[str] = mapped_column(String(16))  # kss | false_alarm | confirmed
    data: Mapped[dict] = mapped_column(JSON)  # raw [T][9], present [T], profile [5], fps
    quality: Mapped[dict] = mapped_column(JSON, default=dict)
    accepted: Mapped[bool] = mapped_column(Boolean, default=False, index=True)
    reject_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class ModelVersion(Base):
    """Model registry: every retrained candidate, its test results and whether it was approved.
    Devices download only `approved` versions, verified by SHA-256."""
    __tablename__ = "model_versions"
    id: Mapped[str] = mapped_column(String(32), primary_key=True, default=_id)
    kind: Mapped[str] = mapped_column(String(16), default="temporal", index=True)
    version: Mapped[int] = mapped_column(Integer)
    created_at: Mapped[float] = mapped_column(Float, default=time.time)
    status: Mapped[str] = mapped_column(String(16), default="candidate", index=True)  # candidate | approved | rejected | retired
    path: Mapped[str] = mapped_column(Text)
    sha256: Mapped[str] = mapped_column(String(64))
    metrics: Mapped[dict] = mapped_column(JSON, default=dict)  # candidate vs current model on held-out real drivers
    data: Mapped[dict] = mapped_column(JSON, default=dict)  # what it was trained on
    notes: Mapped[str] = mapped_column(Text, default="")


def database_url() -> str:
    url = os.environ.get("DMS_DATABASE_URL")
    if url:
        return url
    p = Path(os.environ.get("DMS_CLOUD_DIR", Path.home() / ".dms")) / "cloud.db"
    p.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{p}"


def make_engine(url: str | None = None):
    url = url or database_url()
    if url.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool

        kw = {"connect_args": {"check_same_thread": False}}
        if url in ("sqlite://", "sqlite:///:memory:"):
            kw["poolclass"] = StaticPool  # one shared in-memory database (tests)
        eng = create_engine(url, **kw)

        @event.listens_for(eng, "connect")
        def _pragmas(conn, _):
            cur = conn.cursor()
            cur.execute("PRAGMA foreign_keys=ON")
            cur.execute("PRAGMA journal_mode=WAL")
            cur.close()
    else:
        eng = create_engine(url, pool_pre_ping=True)
    Base.metadata.create_all(eng)
    return eng


def make_sessionmaker(url: str | None = None):
    return sessionmaker(bind=make_engine(url), expire_on_commit=False)
