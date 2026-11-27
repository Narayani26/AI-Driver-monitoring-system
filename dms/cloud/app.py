"""Cloud backend API + web dashboard.

    python -m dms.cloud.app                         # http://localhost:9000 (SQLite in ~/.dms/cloud.db)
    python -m dms.cloud.app --make-admin you@example.com

Endpoints (JSON):
  POST /api/auth/register   {email, password, name}            -> {token}
  POST /api/auth/login      {email, password}                  -> {token}
  GET  /api/me  /  PATCH /api/me {consent_training}
  GET/POST /api/devices  DELETE /api/devices/{id}              -> POST returns the device key ONCE
  POST /api/ingest          (device key) {events, trips, samples}   <- the in-car Syncer
  GET  /api/trips  GET /api/trips/{id}  POST /api/trips/{id}/kss {kss}
  POST /api/events/{id}/feedback {feedback: correct|false_alarm}
  GET  /api/stats
  GET  /api/admin/dataset   (admin) training-data quality overview
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from . import auth, quality
from .db import Device, Event, Sample, Trip, User, make_sessionmaker

STATIC = Path(__file__).parent / "static"
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MAX_BODY = 8 * 1024 * 1024  # one sync batch


class Register(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=200)
    name: str = Field(default="", max_length=120)

class Login(BaseModel):
    email: str = Field(max_length=254)
    password: str = Field(max_length=200)

class MePatch(BaseModel):
    consent_training: bool | None = None
    name: str | None = Field(default=None, max_length=120)

class NewDevice(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: str = Field(default="dashcam", pattern="^(dashcam|phone|web)$")

class Kss(BaseModel):
    kss: int = Field(ge=1, le=9)

class Feedback(BaseModel):
    feedback: str = Field(pattern="^(correct|false_alarm)$")


def create_app(db_url: str | None = None) -> FastAPI:
    Session = make_sessionmaker(db_url)
    limiter = auth.LoginLimiter()
    app = FastAPI(title="DMS cloud", docs_url=None, redoc_url=None)

    @app.middleware("http")
    async def headers(request: Request, call_next):
        if int(request.headers.get("content-length") or 0) > MAX_BODY:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        resp = await call_next(request)
        resp.headers.update({
            "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY", "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'self'; style-src 'self'; script-src 'self'; frame-ancestors 'none'",
            "Cache-Control": "no-store",
        })
        return resp

    def db():
        s = Session()
        try:
            yield s
        finally:
            s.close()

    def current_user(authorization: str = Header(default=""), s=Depends(db)) -> User:
        uid = auth.read_token(authorization.removeprefix("Bearer ").strip())
        u = s.get(User, uid) if uid else None
        if u is None:
            raise HTTPException(401, "please log in")
        return u

    def current_device(authorization: str = Header(default=""), s=Depends(db)) -> Device:
        key = authorization.removeprefix("Bearer ").strip()
        d = s.scalar(select(Device).where(Device.key_hash == auth.hash_key(key))) if key else None
        if d is None or d.revoked:
            raise HTTPException(401, "unknown or revoked device key")
        return d

    def admin(u: User = Depends(current_user)) -> User:
        if u.role != "admin":
            raise HTTPException(403, "admins only")
        return u

    # ------------------------------------------------------------------ accounts


    @app.post("/api/auth/register")
    def register(body: Register, s=Depends(db)):
        email = body.email.strip().lower()
        if not EMAIL_RE.match(email):
            raise HTTPException(400, "invalid email")
        if (why := auth.password_problem(body.password)):
            raise HTTPException(400, why)
        if s.scalar(select(User).where(User.email == email)):
            raise HTTPException(409, "an account with this email exists")
        u = User(email=email, name=body.name.strip(), password_hash=auth.hash_password(body.password))
        s.add(u)
        s.commit()
        return {"token": auth.make_token(u.id)}

    @app.post("/api/auth/login")
    def login(body: Login, request: Request, s=Depends(db)):
        email, ip = body.email.strip().lower(), request.client.host if request.client else "?"
        if limiter.blocked("e:" + email, "ip:" + ip):
            raise HTTPException(429, "too many failed attempts, try again in 15 minutes")
        u = s.scalar(select(User).where(User.email == email))
        if u is None or not auth.verify_password(body.password, u.password_hash):
            limiter.fail("e:" + email, "ip:" + ip)
            raise HTTPException(401, "wrong email or password")  # same message: don't reveal which accounts exist
        limiter.ok("e:" + email)
        return {"token": auth.make_token(u.id)}

    @app.get("/api/me")
    def me(u: User = Depends(current_user)):
        return {"id": u.id, "email": u.email, "name": u.name, "role": u.role,
                "consent_training": u.consent_training, "consent_at": u.consent_at}


    @app.patch("/api/me")
    def update_me(body: MePatch, u: User = Depends(current_user), s=Depends(db)):
        u = s.merge(u)
        if body.name is not None:
            u.name = body.name.strip()
        if body.consent_training is not None and body.consent_training != u.consent_training:
            u.consent_training, u.consent_at = body.consent_training, time.time()
            if not body.consent_training:  # withdrawing consent removes the data already given
                s.query(Sample).filter(Sample.user_id == u.id).delete()
        s.commit()
        return me(u)

    @app.delete("/api/me")
    def delete_me(u: User = Depends(current_user), s=Depends(db)):
        """Right to erasure: the account and everything uploaded for it."""
        for model in (Sample, Event, Trip):
            s.query(model).filter(model.user_id == u.id).delete()
        s.delete(s.merge(u))
        s.commit()
        return {"deleted": True}

    # ------------------------------------------------------------------ devices

    @app.get("/api/devices")
    def devices(u: User = Depends(current_user), s=Depends(db)):
        rows = s.scalars(select(Device).where(Device.user_id == u.id, Device.revoked.is_(False)).order_by(Device.created_at)).all()
        return [{"id": d.id, "name": d.name, "kind": d.kind, "key_prefix": d.key_prefix, "created_at": d.created_at, "last_seen": d.last_seen}
                for d in rows]

    @app.post("/api/devices")
    def add_device(body: NewDevice, u: User = Depends(current_user), s=Depends(db)):
        key, key_hash, prefix = auth.new_device_key()
        d = Device(user_id=u.id, name=body.name.strip(), kind=body.kind, key_hash=key_hash, key_prefix=prefix)
        s.add(d)
        s.commit()
        return {"id": d.id, "name": d.name, "device_key": key,
                "note": "Copy this key now: it is shown only once. Start the car app with --sync-url and --sync-token <key>."}

    @app.delete("/api/devices/{device_id}")
    def revoke_device(device_id: str, u: User = Depends(current_user), s=Depends(db)):
        d = s.get(Device, device_id)
        if d is None or d.user_id != u.id:
            raise HTTPException(404, "no such device")
        d.revoked = True
        s.commit()
        return {"revoked": True}

    # ------------------------------------------------------------------ ingest (from the car)
    @app.post("/api/ingest")
    async def ingest(request: Request, d: Device = Depends(current_device), s=Depends(db)):
        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(400, "expected a JSON object")
        d = s.merge(d)
        user = s.get(User, d.user_id)
        n_ev = n_tr = 0
        accepted = rejected = skipped = 0
        for tr in body.get("trips") or []:
            sid = str(tr.get("session", ""))[:64]
            if not sid:
                continue
            t = s.get(Trip, sid)
            if t is not None and t.user_id != d.user_id:
                continue  # never let one account overwrite another's data
            t = t or Trip(id=sid, user_id=d.user_id, device_id=d.id)
            t.t_start, t.t_end, t.summary = tr.get("t_start"), tr.get("t_end"), tr.get("summary") or {}
            if tr.get("kss") is not None:
                t.kss = int(tr["kss"])
            s.add(t)
            n_tr += 1
        events = body.get("events") or []
        for ev in [e for e in events if e.get("kind") != "feedback"]:
            eid = str(ev.get("id", ""))[:64]
            if not eid or s.get(Event, eid) is not None:
                continue  # idempotent: the car may resend after a lost reply
            s.add(Event(id=eid, user_id=d.user_id, device_id=d.id, trip_id=str(ev.get("session", ""))[:64],
                        t_wall=float(ev.get("t_wall") or time.time()), t_session=ev.get("t_session"),
                        kind=str(ev.get("kind", ""))[:32], severity=ev.get("severity"), payload=ev.get("payload") or {}))
            n_ev += 1
        s.flush()
        for ev in [e for e in events if e.get("kind") == "feedback"]:  # the driver rated an alert in the car
            p = ev.get("payload") or {}
            if p.get("verdict") in ("correct", "false_alarm"):
                target = s.scalar(select(Event).where(Event.user_id == d.user_id, Event.trip_id == str(ev.get("session", ""))[:64],
                                                      Event.kind == str(p.get("alert", ""))[:32],
                                                      func.abs(Event.t_session - float(ev.get("t_session") or -1)) < 0.05))
                if target is not None:
                    target.feedback = p["verdict"]
        for sm in body.get("samples") or []:
            sid = str(sm.get("id", ""))[:64]
            if not sid or s.get(Sample, sid) is not None:
                continue
            if not user.consent_training:
                skipped += 1  # no consent: the window is dropped, not stored
                continue
            ok, why, q = quality.check(sm)
            s.add(Sample(id=sid, user_id=d.user_id, device_id=d.id, trip_id=str(sm.get("session", ""))[:64],
                         label=int(sm["label"]) if sm.get("label") in quality.LABELS else -1,
                         label_source=str(sm.get("label_source", ""))[:16],
                         data={k: sm.get(k) for k in ("raw", "present", "profile", "fps")} if ok else {},
                         quality=q, accepted=ok, reject_reason=why))
            accepted, rejected = accepted + ok, rejected + (not ok)
        d.last_seen = time.time()
        s.commit()
        return {"events": n_ev, "trips": n_tr, "samples": {"accepted": accepted, "rejected": rejected, "skipped_no_consent": skipped}}

    # ------------------------------------------------------------------ trips, events, feedback
    @app.get("/api/trips")
    def trips(limit: int = 50, u: User = Depends(current_user), s=Depends(db)):
        rows = s.scalars(select(Trip).where(Trip.user_id == u.id).order_by(Trip.t_end.desc()).limit(min(limit, 200))).all()
        return [{"id": t.id, "t_start": t.t_start, "t_end": t.t_end, "kss": t.kss,
                 "safety_score": t.summary.get("safety_score"), "duration_s": t.summary.get("duration_s"),
                 "alerts_total": t.summary.get("alerts_total")} for t in rows]

    @app.get("/api/trips/{trip_id}")
    def trip(trip_id: str, u: User = Depends(current_user), s=Depends(db)):
        t = s.get(Trip, trip_id)
        if t is None or t.user_id != u.id:
            raise HTTPException(404, "no such trip")
        evs = s.scalars(select(Event).where(Event.trip_id == trip_id, Event.user_id == u.id).order_by(Event.t_wall)).all()
        return {"id": t.id, "t_start": t.t_start, "t_end": t.t_end, "kss": t.kss, "summary": t.summary,
                "events": [{"id": e.id, "t_wall": e.t_wall, "t_session": e.t_session, "kind": e.kind, "severity": e.severity,
                            "detail": e.payload.get("detail"), "feedback": e.feedback} for e in evs]}


    @app.post("/api/trips/{trip_id}/kss")
    def set_kss(trip_id: str, body: Kss, u: User = Depends(current_user), s=Depends(db)):
        t = s.get(Trip, trip_id)
        if t is None or t.user_id != u.id:
            raise HTTPException(404, "no such trip")
        t.kss = body.kss
        s.commit()
        return {"kss": t.kss}


    @app.post("/api/events/{event_id}/feedback")
    def feedback(event_id: str, body: Feedback, u: User = Depends(current_user), s=Depends(db)):
        e = s.get(Event, event_id)
        if e is None or e.user_id != u.id:
            raise HTTPException(404, "no such event")
        e.feedback = body.feedback
        s.commit()
        return {"feedback": e.feedback}

    @app.get("/api/stats")
    def stats(u: User = Depends(current_user), s=Depends(db)):
        n_trips = s.scalar(select(func.count()).select_from(Trip).where(Trip.user_id == u.id))
        by_kind = dict(s.execute(select(Event.kind, func.count()).where(Event.user_id == u.id).group_by(Event.kind)).all())
        fa = dict(s.execute(select(Event.feedback, func.count()).where(Event.user_id == u.id, Event.feedback.is_not(None))
                            .group_by(Event.feedback)).all())
        n_samples = s.scalar(select(func.count()).select_from(Sample).where(Sample.user_id == u.id, Sample.accepted.is_(True)))
        return {"trips": n_trips, "alerts_by_kind": by_kind, "feedback": fa, "training_samples_given": n_samples}

    @app.get("/api/admin/dataset")
    def dataset(_: User = Depends(admin), s=Depends(db)):
        acc = dict(s.execute(select(Sample.label, func.count()).where(Sample.accepted.is_(True)).group_by(Sample.label)).all())
        rej = s.execute(select(Sample.reject_reason, func.count()).where(Sample.accepted.is_(False)).group_by(Sample.reject_reason)).all()
        drivers = s.scalar(select(func.count(func.distinct(Sample.user_id))).where(Sample.accepted.is_(True)))
        fa = s.execute(select(Event.kind, func.count()).where(Event.feedback == "false_alarm").group_by(Event.kind)).all()
        return {"accepted_by_label": {quality.LABELS[k]: v for k, v in acc.items()}, "drivers": drivers,
                "rejected_by_reason": dict(rej), "false_alarms_by_kind": dict(fa), "rules": quality.RULES}

    # ------------------------------------------------------------------ dashboard
    @app.get("/api/health")
    def health():
        return {"ok": True}

    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def make_admin(email: str, db_url: str | None = None):
    s = make_sessionmaker(db_url)()
    u = s.scalar(select(User).where(User.email == email.strip().lower()))
    if u is None:
        raise SystemExit(f"no account {email}: register it in the dashboard first")
    u.role = "admin"
    s.commit()
    print(f"{u.email} is now an admin")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--make-admin", metavar="EMAIL")
    a = ap.parse_args(argv)
    if a.make_admin:
        return make_admin(a.make_admin)
    import uvicorn

    print(f"\n  DMS cloud backend -> http://{a.host}:{a.port}\n")
    uvicorn.run(create_app(), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
