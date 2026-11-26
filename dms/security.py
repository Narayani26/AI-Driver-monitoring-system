"""Security helpers for the web server and model loading.

Threat model (see docs/SECURITY.md): the camera feed of a person in their car
is highly sensitive. The server must not be usable by
  * other websites open in the same browser (cross-site WebSocket hijacking,
    DNS rebinding) - they could otherwise read camera frames or make the server
    open local files,
  * other devices on the same Wi-Fi when the server is exposed on the network,
  * tampered model files (a swapped model could silently disable alerts).
"""
from __future__ import annotations

import hashlib
import hmac
import ipaddress
import secrets
from pathlib import Path
from urllib.parse import urlparse

WEIGHTS = Path(__file__).resolve().parents[1] / "weights"
SUMS_FILE = WEIGHTS / "SHA256SUMS"
LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1", "[::1]"}
STREAM_SCHEMES = {"rtsp", "rtsps", "http", "https"}
COOKIE = "dms_token"

SECURITY_HEADERS = {
    # Only our own scripts/styles; frames are drawn from blob:/data: URLs; WebSocket to ourselves.
    "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; "
                               "media-src 'self' blob:; connect-src 'self' ws: wss:; worker-src 'self' blob:; frame-ancestors 'none'; "
                               "base-uri 'none'; form-action 'none'",
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Permissions-Policy": "camera=(self), microphone=(self), geolocation=(), payment=()",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cache-Control": "no-store",
}


def is_loopback(host: str) -> bool:
    if host in LOOPBACK_HOSTS:
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def new_token() -> str:
    return secrets.token_urlsafe(24)


def token_ok(expected: str | None, given: str | None) -> bool:
    if expected is None:
        return True
    return given is not None and hmac.compare_digest(expected.encode(), given.encode())


def host_allowed(host_header: str | None, bind_host: str, extra: set[str] = frozenset()) -> bool:
    """DNS-rebinding guard: when serving only on loopback, the Host header must be loopback too."""
    if not host_header:
        return False
    name = host_header.rsplit(":", 1)[0] if not host_header.startswith("[") else host_header.split("]")[0] + "]"
    if name in extra:
        return True
    if is_loopback(bind_host):
        return is_loopback(name)
    return True  # exposed on the network: protected by the access token instead


def origin_allowed(origin: str | None, host_header: str | None, extra: set[str] = frozenset()) -> bool:
    """Cross-site WebSocket hijacking guard: browsers always send Origin; it must be us.
    Non-browser clients (no Origin) are allowed but still need the token when one is set."""
    if origin is None:
        return True
    if origin in extra:
        return True
    o = urlparse(origin)
    return bool(o.netloc) and o.netloc == (host_header or "")


def stream_url_allowed(url: str, allow_files: bool = False) -> tuple[bool, str]:
    """Only camera streams (rtsp/http(s)) or a local camera index. Local file paths only with --allow-files,
    otherwise anyone who can reach the server could make it read and stream back any video file on disk."""
    url = url.strip()
    if url.isdigit():
        return True, ""
    p = urlparse(url)
    if p.scheme.lower() in STREAM_SCHEMES and p.netloc:
        return True, ""
    if allow_files and p.scheme in ("", "file") and Path(p.path or url).is_file():
        return True, ""
    return False, "Only rtsp://, http(s):// camera URLs or a camera number are allowed" + ("" if allow_files else " (start the server with --allow-files to play local files)")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sums(files: list[Path], out: Path = SUMS_FILE):
    out.write_text("".join(f"{sha256(f)}  {f.name}\n" for f in sorted(files)))


def verify_weights(sums_file: Path = SUMS_FILE) -> list[str]:
    """Returns a list of problems (empty = all model files match their recorded SHA-256)."""
    if not sums_file.exists():
        return [f"{sums_file} missing"]
    problems = []
    for line in sums_file.read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        f = sums_file.parent / name.strip()
        if not f.exists():
            continue  # optional model not downloaded; it simply won't be used
        if sha256(f) != digest:
            problems.append(f"{name.strip()}: checksum mismatch (file modified or corrupted)")
    return problems
