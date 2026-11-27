"""Passwords, login tokens and device keys (standard library only).

* Passwords: scrypt (memory-hard) with a per-user random salt; never stored or logged in clear.
* Login tokens: HMAC-SHA256-signed "user_id.expiry" (12 h), verified in constant time.
* Device keys: 256-bit random, shown once at registration; only their SHA-256 is stored,
  so a database leak does not let anyone impersonate a dash cam. Revocable per device.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import time
from collections import defaultdict, deque
from pathlib import Path

TOKEN_TTL_S = 12 * 3600
_SCRYPT = dict(n=2**14, r=8, p=1, dklen=32)


def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    h = hashlib.scrypt(pw.encode(), salt=salt, **_SCRYPT)
    return "scrypt$" + base64.b64encode(salt).decode() + "$" + base64.b64encode(h).decode()


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, salt, h = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=base64.b64decode(salt), **_SCRYPT)
        return hmac.compare_digest(got, base64.b64decode(h))
    except (ValueError, TypeError):
        return False


def password_problem(pw: str) -> str | None:
    if len(pw) < 10:
        return "password must have at least 10 characters"
    if pw.lower() == pw or not any(c.isdigit() for c in pw):
        return "password needs an upper-case letter and a digit"
    return None


def _secret() -> bytes:
    env = os.environ.get("DMS_CLOUD_SECRET")
    if env:
        return env.encode()
    p = Path(os.environ.get("DMS_CLOUD_DIR", Path.home() / ".dms")) / "cloud_secret"
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_bytes(secrets.token_bytes(32))
        p.chmod(0o600)
    return p.read_bytes()


def make_token(user_id: str, now: float | None = None) -> str:
    exp = int((now or time.time()) + TOKEN_TTL_S)
    msg = f"{user_id}.{exp}"
    sig = hmac.new(_secret(), msg.encode(), hashlib.sha256).hexdigest()
    return f"{msg}.{sig}"


def read_token(token: str, now: float | None = None) -> str | None:
    """user_id if the token is genuine and not expired."""
    try:
        uid, exp, sig = token.split(".")
    except ValueError:
        return None
    good = hmac.new(_secret(), f"{uid}.{exp}".encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(sig, good) or int(exp) < (now or time.time()):
        return None
    return uid


def new_device_key() -> tuple[str, str, str]:
    """(key shown once to the user, its SHA-256 for the database, short prefix for display)"""
    key = "dmsk_" + secrets.token_urlsafe(32)
    return key, hash_key(key), key[:12]


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class LoginLimiter:
    """At most `limit` failed logins per account and per client address in `window_s` (brute-force guard)."""

    def __init__(self, limit: int = 5, window_s: float = 900.0):
        self.limit, self.window_s = limit, window_s
        self._fails: dict[str, deque] = defaultdict(deque)

    def blocked(self, *keys: str) -> bool:
        now = time.time()
        for k in keys:
            q = self._fails[k]
            while q and now - q[0] > self.window_s:
                q.popleft()
            if len(q) >= self.limit:
                return True
        return False

    def fail(self, *keys: str):
        for k in keys:
            self._fails[k].append(time.time())

    def ok(self, *keys: str):
        for k in keys:
            self._fails.pop(k, None)
