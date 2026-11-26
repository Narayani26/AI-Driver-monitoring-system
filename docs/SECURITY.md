# Security and privacy

Footage of a person inside their car is highly sensitive: it shows their face, their passengers and where they are, and it is biometric data under laws such as India's Digital Personal Data Protection Act 2023 and the EU GDPR. The design goal is that **this footage never leaves the device and is never stored**.

## 1. Principles

| Principle | How |
|---|---|
| **Process on-device, offline** | All models run locally (CPU). No cloud API is ever called with video or audio. The system works with no internet. |
| **Store nothing sensitive** | No video, image, audio or face data is written to disk. The event store keeps only metadata: time, alert type, severity, short reason ("eyes closed 1.4 s"), risk score. |
| **Minimise what syncs** | Optional sync uploads only those metadata rows and trip summaries, over HTTPS, with a bearer token. |
| **Secure by default** | The server listens on `127.0.0.1` (this device only). Exposing it on a network requires an access token and prints a warning without HTTPS. |
| **Fail safe** | Tampered models stop the server from starting; a lost camera stream is reconnected rather than silently ignored. |

## 2. Threats and the controls in the code

| Threat | Control | Where |
|---|---|---|
| **Another website open in the same browser** connects to `ws://localhost:8000` and reads your camera frames or drives the server (cross-site WebSocket hijacking) | The WebSocket `Origin` must equal the server's own host; otherwise the connection is refused (code 1008) | `dms/security.py` `origin_allowed`, `dms/web/server.py` |
| **DNS rebinding** (a malicious domain that resolves to 127.0.0.1) | When bound to loopback, the HTTP `Host` header must be a loopback name; otherwise HTTP 421 | `host_allowed`, `guard` middleware |
| **Other devices on the same Wi-Fi** open the dashboard | `--host 0.0.0.0` automatically generates an access token (or use `--token` / `DMS_TOKEN`). Without it: HTTP 401 and the WebSocket is refused. The token is moved into an `HttpOnly; SameSite=Strict` cookie and compared in constant time | `token_ok`, `guard`, `ws_endpoint` |
| **Server used to read local files** (the stream box accepted any path, so `/home/me/private.mov` could be streamed back) | Stream sources limited to `rtsp://`, `rtsps://`, `http(s)://` camera URLs or a camera number. Local files only with an explicit `--allow-files` | `stream_url_allowed` |
| **Eavesdropping on the network** | HTTPS via `--ssl-certfile/--ssl-keyfile`; the server warns when exposed without it. Sync refuses plain `http://` except to localhost | `server.main`, `Syncer` |
| **Tampered or corrupted model** (e.g. a modified model that never detects closed eyes) | SHA-256 of every model in `weights/SHA256SUMS` is verified at start-up; mismatch = refuse to start (override only with `--allow-unverified-models`). The Docker build verifies them too | `verify_weights` |
| **Clickjacking, script injection, data leaks via Referer** | Strict security headers: `Content-Security-Policy` (own scripts only, no inline scripts, `frame-ancestors 'none'`), `X-Frame-Options: DENY`, `nosniff`, `Referrer-Policy: no-referrer`, `Permissions-Policy` limiting camera/mic to this page, `Cache-Control: no-store`. All server text shown in the page is HTML-escaped | `SECURITY_HEADERS`, `app.js esc()` |
| **Resource exhaustion** (huge messages) | Frames capped at 4 MB, audio at 2 s per message, malformed JSON ignored | `handle_bytes` |
| **Stolen phone / SD card** | Nothing sensitive is stored. The event database is created owner-only (`chmod 600`); synced rows are deleted after the retention period (30 days default) | `EventStore` |
| **Process compromise on the edge box** | Docker image runs as a non-root user; the systemd unit uses `NoNewPrivileges`, `ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, write access only to `/var/lib/dms` | `Dockerfile`, `deploy/dms-edge.service` |
| **Accidental exposure of API docs** | FastAPI's `/docs`, `/redoc` and `/openapi.json` are disabled | `server.py` |

All of the above has unit or end-to-end tests in `tests/test_core.py` (`test_security_rules`, `test_web_server_security_end_to_end`, `test_event_store_offline_then_sync`).

## 3. Deployment checklist (things you must do; code can't)

- [ ] **Dash cam:** change its default Wi-Fi password and admin password; use WPA2/WPA3. Many dash cams ship with `12345678`.
- [ ] **HTTPS** whenever the dashboard is reached over a network (`mkcert` for a local CA; see `docs/DEPLOYMENT.md`).
- [ ] **Keep the token secret.** Don't paste it in chats, screenshots or slides. Rotate it by restarting with a new `--token`.
- [ ] **Full-disk encryption** on the device (FileVault on Mac, Android default encryption, LUKS on Linux).
- [ ] **Consent:** everyone being monitored (driver, passengers in view) must know about it; get written consent for any recordings used for training or evaluation.
- [ ] **Sync server** (if used): HTTPS only, authenticates the bearer token, stores only what it needs, has its own retention policy.
- [ ] **Updates:** rebuild the image or `pip install -U` regularly for security fixes in dependencies (OpenCV, FastAPI, MediaPipe).

## 4. Known limitations (be honest about these in the report)

- The access token is a single shared secret (fine for one driver/one device; a fleet product needs per-user accounts and OAuth/OIDC).
- RTSP from consumer dash cams is usually **unencrypted** on the dash cam's own Wi-Fi. Keep that network private (WPA2, strong password); prefer cameras that support RTSPS.
- Training datasets contain other people's faces under research-only licences: never ship them inside a product or repository (this repo doesn't).
- No formal penetration test has been done.
