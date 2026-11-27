"""Offline-first event store with store-and-forward sync.

Works with no internet: every alert and trip summary is written to a local
SQLite file immediately. When a sync URL is configured, a background thread
uploads unsynced rows in batches whenever the network is back, retrying with
exponential backoff. Nothing is lost if the app is killed: SQLite in WAL mode
survives power loss (relevant for a device powered from a car).

Privacy by design: only event metadata is stored (time, type, severity, short
reason, numeric scores). No images, video, audio or face data ever enter the
store, so a stolen phone or SD card does not leak footage. Training samples
(30 s of numeric eye/mouth/head measurements around an alert the driver rated)
are only recorded when the driver opted in to sharing them.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY, session TEXT NOT NULL, t_wall REAL NOT NULL, t_session REAL,
    kind TEXT NOT NULL, severity TEXT, payload TEXT, synced INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS events_unsynced ON events(synced, t_wall);
CREATE TABLE IF NOT EXISTS trips (
    session TEXT PRIMARY KEY, t_start REAL, t_end REAL, summary TEXT, synced INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS samples (
    id TEXT PRIMARY KEY, session TEXT NOT NULL, t_wall REAL NOT NULL, body TEXT NOT NULL, synced INTEGER NOT NULL DEFAULT 0
);
"""


class EventStore:
    def __init__(self, path: str | Path, retention_days: float = 30.0):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.retention_s = retention_days * 86400
        self._lock = threading.Lock()
        self._db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(SCHEMA)
        try:  # owner-only permissions: other users on the device can't read the log
            self.path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def new_session() -> str:
        return uuid.uuid4().hex[:12]

    def add_event(self, session: str, kind: str, severity: str | None = None, t_session: float | None = None, **payload) -> str:
        eid = uuid.uuid4().hex
        with self._lock:
            self._db.execute("INSERT INTO events(id, session, t_wall, t_session, kind, severity, payload) VALUES (?,?,?,?,?,?,?)",
                             (eid, session, time.time(), t_session, kind, severity, json.dumps(payload, default=str)))
        return eid

    def save_trip(self, session: str, summary: dict, t_start: float | None = None):
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO trips(session, t_start, t_end, summary, synced) VALUES (?,?,?,?,0)",
                             (session, t_start, time.time(), json.dumps(summary, default=str)))

    def add_sample(self, session: str, body: dict) -> str:
        sid = uuid.uuid4().hex
        with self._lock:
            self._db.execute("INSERT INTO samples(id, session, t_wall, body) VALUES (?,?,?,?)",
                             (sid, session, time.time(), json.dumps({**body, "id": sid, "session": session})))
        return sid

    def set_trip_kss(self, session: str, kss: int):
        with self._lock:
            self._db.execute("UPDATE trips SET summary=json_set(summary, '$.kss', ?), synced=0 WHERE session=?", (kss, session))

    def unsynced_samples(self, limit: int = 20) -> list[dict]:
        with self._lock:
            rows = self._db.execute("SELECT body FROM samples WHERE synced=0 ORDER BY t_wall LIMIT ?", (limit,)).fetchall()
        return [json.loads(r[0]) for r in rows]

    def mark_samples_synced(self, ids: list[str]):
        with self._lock:
            self._db.executemany("UPDATE samples SET synced=1 WHERE id=?", [(i,) for i in ids])

    def unsynced(self, limit: int = 200) -> tuple[list[dict], list[dict]]:
        with self._lock:
            ev = self._db.execute("SELECT id, session, t_wall, t_session, kind, severity, payload FROM events WHERE synced=0 ORDER BY t_wall LIMIT ?", (limit,)).fetchall()
            tr = self._db.execute("SELECT session, t_start, t_end, summary FROM trips WHERE synced=0 LIMIT ?", (limit,)).fetchall()
        events = [{"id": r[0], "session": r[1], "t_wall": r[2], "t_session": r[3], "kind": r[4], "severity": r[5], "payload": json.loads(r[6] or "{}")} for r in ev]
        trips = [{"session": r[0], "t_start": r[1], "t_end": r[2], "summary": json.loads(r[3] or "{}")} for r in tr]
        return events, trips

    def mark_synced(self, event_ids: list[str], trip_sessions: list[str]):
        with self._lock:
            self._db.executemany("UPDATE events SET synced=1 WHERE id=?", [(i,) for i in event_ids])
            self._db.executemany("UPDATE trips SET synced=1 WHERE session=?", [(s,) for s in trip_sessions])

    def prune(self):
        """Delete synced rows older than the retention period (unsynced rows are kept until delivered)."""
        cutoff = time.time() - self.retention_s
        with self._lock:
            self._db.execute("DELETE FROM events WHERE synced=1 AND t_wall < ?", (cutoff,))
            self._db.execute("DELETE FROM trips WHERE synced=1 AND t_end < ?", (cutoff,))
            self._db.execute("DELETE FROM samples WHERE synced=1 OR t_wall < ?", (cutoff,))

    def counts(self) -> dict:
        with self._lock:
            q = lambda sql: self._db.execute(sql).fetchone()[0]
            return {"events": q("SELECT COUNT(*) FROM events"), "events_unsynced": q("SELECT COUNT(*) FROM events WHERE synced=0"),
                    "trips": q("SELECT COUNT(*) FROM trips"), "trips_unsynced": q("SELECT COUNT(*) FROM trips WHERE synced=0")}

    def close(self):
        with self._lock:
            self._db.close()


class Syncer(threading.Thread):
    """Uploads unsynced rows to `url` (HTTPS POST, JSON) when the network allows.

    The receiving server is yours (fleet dashboard, family app, ...). Requests
    carry `Authorization: Bearer <token>`; use HTTPS so data is encrypted in transit.
    """

    def __init__(self, store: EventStore, url: str, token: str | None = None, interval_s: float = 30.0, max_backoff_s: float = 900.0, timeout_s: float = 10.0):
        super().__init__(daemon=True, name="dms-sync")
        if not url.startswith("https://") and not url.startswith("http://localhost") and not url.startswith("http://127.0.0.1"):
            raise ValueError("sync URL must use https:// (plain http only allowed for localhost testing)")
        self.store, self.url, self.token = store, url, token
        self.interval_s, self.max_backoff_s, self.timeout_s = interval_s, max_backoff_s, timeout_s
        self._stop = threading.Event()
        self.last_ok: float | None = None
        self.last_error: str | None = None

    def sync_once(self) -> int:
        events, trips = self.store.unsynced()
        samples = self.store.unsynced_samples()
        if not events and not trips and not samples:
            return 0
        for tr in trips:  # the driver's own sleepiness rating travels as a trip field
            if "kss" in tr["summary"]:
                tr["kss"] = tr["summary"]["kss"]
        body = json.dumps({"events": events, "trips": trips, "samples": samples}).encode()
        req = urllib.request.Request(self.url, data=body, method="POST", headers={"Content-Type": "application/json"})
        if self.token:
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            if resp.status >= 300:
                raise urllib.error.HTTPError(self.url, resp.status, "upload rejected", resp.headers, None)
        self.store.mark_synced([e["id"] for e in events], [t["session"] for t in trips])
        self.store.mark_samples_synced([x["id"] for x in samples])
        return len(events) + len(trips) + len(samples)

    def run(self):
        backoff = self.interval_s
        while not self._stop.is_set():
            try:
                n = self.sync_once()
                self.last_ok, self.last_error = time.time(), None
                self.store.prune()
                backoff = self.interval_s if n == 0 else 1.0  # drain quickly when there is a backlog
            except Exception as e:  # offline, DNS failure, server down: keep data, retry later
                self.last_error = f"{type(e).__name__}: {e}"
                backoff = min(backoff * 2, self.max_backoff_s)
            self._stop.wait(backoff)

    def stop(self):
        self._stop.set()
