"""Web demo / in-car dashboard: side-by-side live footage + AI analysis.

    python -m dms.web.server                                   # http://localhost:8000 (this computer only)
    python -m dms.web.server --audio --mishap                  # + cabin audio and possible-accident detection
    python -m dms.web.server --host 0.0.0.0 --ssl-certfile cert.pem --ssl-keyfile key.pem   # reachable from a phone
                                                               # (prints a URL with an access token)

Sources (chosen in the browser):
* Webcam / uploaded video: the browser sends JPEG frames over a WebSocket
  (stop-and-wait, so the server is never flooded and results stay real-time),
  plus optional 16 kHz audio chunks.
* Stream URL (RTSP/HTTP, e.g. a Wi-Fi dash cam or a phone running an IP-camera
  app): the server reads the stream itself (auto-reconnects if it drops) and
  pushes frames + results back. Same data path as the dash cam product.

Security (docs/SECURITY.md): loopback-only by default; an access token is
required whenever the server is reachable from the network; WebSocket Origin
and Host checks block other websites (cross-site WebSocket hijacking, DNS
rebinding); stream URLs are restricted to camera URLs; model files are
checksum-verified; strict security headers; nothing is recorded to disk except
alert metadata in the local event store.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import struct
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .. import security
from ..features.geometry import LEFT_EYE, RIGHT_EYE
from ..models.networks import EMOTION_CLASSES
from ..pipeline import DriverMonitor, FrameResult, MonitorConfig
from ..storage import EventStore, Syncer

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Driver Monitoring System", docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=STATIC), name="static")
CFG: dict = {"token": None, "bind_host": "127.0.0.1", "allow_files": False, "origins": set(), "audio": False, "mishap": False}
STATE: dict = {"store": None, "syncer": None, "started": time.time(), "sessions": 0}
MOUTH = (61, 81, 13, 311, 291, 402, 14, 178)
MAX_FRAME_BYTES = 4 * 1024 * 1024
MAX_AUDIO_SAMPLES = 16000 * 2  # at most 2 s per message
AUDIO_MAGIC = b"AUD0"


def _num(x, nd=4):
    if x is None:
        return None
    x = float(x)
    return None if math.isnan(x) or math.isinf(x) else round(x, nd)


def serialize(r: FrameResult, shape, proc_ms: float) -> dict:
    f, rk = r.features, r.risk
    out = {
        "type": "result",
        "t": round(r.t, 3),
        "w": shape[1],
        "h": shape[0],
        "proc_ms": round(proc_ms, 1),
        "fps": _num(r.fps, 1),
        "calibrating": _num(r.calibrating, 3),
        "face": f.face,
        "face_box": list(r.face_box) if r.face_box else None,
        "features": {k: _num(getattr(f, k)) for k in ("ear", "mar", "pitch", "yaw", "roll", "gaze_h", "gaze_v", "blink_bs", "jaw_open_bs")},
        "eye_openness": _num(r.eye_openness, 3),
        "eye_cnn_closed": _num(r.eye_cnn_closed, 3),
        "eye_closed": r.eye_closed,
        "eye_closed_s": _num(r.eye_closed_duration, 2),
        "off_road": r.off_road,
        "off_road_s": _num(r.off_road_duration, 2),
        "perclos": _num(r.perclos, 4),
        "blink_rate": _num(r.blink_rate, 1),
        "yawns": r.yawns,
        "emotion": r.emotion,
        "emotion_conf": _num(r.emotion_conf, 3),
        "emotion_probs": dict(zip(EMOTION_CLASSES, (round(float(p), 3) for p in r.emotion_probs))) if r.emotion_probs is not None else None,
        "detections": [{"label": d.label, "behaviour": d.behaviour, "conf": round(d.conf, 3), "box": list(d.box), "active": d.behaviour in r.behaviours} for d in r.detections],
        "behaviours": {k: round(v, 3) for k, v in r.behaviours.items()},
        "alertness": r.alertness,
        "audio": r.audio,
        "mishap": None if r.mishap is None else {"phase": r.mishap.phase, "countdown": round(r.mishap.countdown, 1), "evidence": r.mishap.evidence},
        "risk": None if rk is None else {"index": round(rk.index, 1), "level": rk.level, "drowsiness": round(rk.drowsiness, 3), "distraction": round(rk.distraction, 3), "affect": round(rk.affect, 3), "reasons": rk.reasons},
        "alerts": [a.as_dict() for a in r.alerts],
    }
    if r.obs is not None:
        lm = r.obs.landmarks
        out["eyes"] = [lm[list(idx), :2].round(1).tolist() for idx in (LEFT_EYE, RIGHT_EYE)]
        out["mouth"] = lm[list(MOUTH), :2].round(1).tolist()
    return out


def collect_metrics() -> list[dict]:
    """Summaries of every training run under runs/ for the 'How it works' tab."""
    rows = []
    for p in sorted((ROOT / "runs").glob("**/metrics.json")):
        try:
            m = json.loads(p.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        name = str(p.parent.relative_to(ROOT / "runs"))
        if "test" in m:
            for task, v in m["test"].items():
                split = m.get("args", {}).get("split", "subject")
                rows.append({"run": name, "task": task, "protocol": "subject-independent test split" if split == "subject" else "random split (leaky, for comparison)", "accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "n": v["n"]})
            for task, v in m.get("test_heldout", {}).items():
                rows.append({"run": name, "task": task, "protocol": "held-out dataset (never trained on)", "accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "n": v["n"]})
        elif "summary" in m:
            s = m["summary"]
            rows.append({"run": name, "task": "drowsiness (3-class)", "protocol": "subject-independent k-fold", "accuracy": s["video_accuracy"]["mean"], "macro_f1": s["window_macro_f1"]["mean"], "std": s["window_macro_f1"]["std"]})
    return rows


# --------------------------------------------------------------------------- security middleware
def _client_is_loopback(request) -> bool:
    return bool(request.client) and security.is_loopback(request.client.host)


@app.middleware("http")
async def guard(request: Request, call_next):
    if not security.host_allowed(request.headers.get("host"), CFG["bind_host"], CFG["origins"]):
        return PlainTextResponse("host not allowed", status_code=421)
    path = request.url.path
    protected = path == "/" or (path.startswith("/api/") and not (path == "/api/health" and _client_is_loopback(request)))
    given = request.query_params.get("token") or request.cookies.get(security.COOKIE)
    if protected and not security.token_ok(CFG["token"], given):
        return PlainTextResponse("access token required: open the URL printed by the server", status_code=401)
    resp = await call_next(request)
    for k, v in security.SECURITY_HEADERS.items():
        resp.headers.setdefault(k, v)
    if CFG["token"] and request.query_params.get("token") and path == "/":
        resp.set_cookie(security.COOKIE, CFG["token"], httponly=True, samesite="strict", secure=request.url.scheme == "https")
    return resp


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    public_cfg = {k: CFG.get(k) for k in ("calib_seconds", "emotion_source", "audio", "mishap")}
    return JSONResponse({"metrics": collect_metrics(), "models": CFG.get("models_info"), "config": public_cfg})


@app.get("/api/health")
def health():
    store, syncer = STATE["store"], STATE["syncer"]
    return JSONResponse({
        "status": "ok",
        "uptime_s": round(time.time() - STATE["started"], 1),
        "active_sessions": STATE["sessions"],
        "models_verified": not CFG.get("weight_problems"),
        "store": store.counts() if store else None,
        "sync": None if syncer is None else {"last_ok": syncer.last_ok, "last_error": syncer.last_error},
    })


# --------------------------------------------------------------------------- per-connection session
class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.mon: DriverMonitor | None = None
        self.stream_task: asyncio.Task | None = None
        self.stop_stream = False
        self.store: EventStore | None = STATE["store"]
        self.session_id = EventStore.new_session()
        self.t_start = time.time()

    async def send(self, obj):
        await self.ws.send_text(json.dumps(obj))

    def _log_alerts(self, r: FrameResult):
        if self.store is None:
            return
        for a in r.alerts:
            self.store.add_event(self.session_id, a.kind, a.severity, t_session=a.t, detail=a.detail,
                                 risk=None if r.risk is None else round(r.risk.index, 1))

    def save_trip(self):
        if self.store is not None and self.mon is not None and self.mon.recorder.frames:
            self.store.save_trip(self.session_id, self.mon.recorder.summary(), self.t_start)

    async def process(self, frame: np.ndarray, t: float) -> dict:
        t0 = time.perf_counter()
        r = await run_in_threadpool(self.mon.process, frame, t)
        self._log_alerts(r)
        return serialize(r, frame.shape, (time.perf_counter() - t0) * 1000)

    async def run_stream(self, url: str):
        src = int(url) if url.isdigit() else url
        live = isinstance(src, int) or "://" in str(src)
        t_start, n, retry = time.monotonic(), 0, 1.0
        await self.send({"type": "stream_started", "url": url})
        try:
            while not self.stop_stream:
                cap = await run_in_threadpool(cv2.VideoCapture, src)
                if not cap.isOpened():
                    cap.release()
                    if not live or retry > 30:
                        await self.send({"type": "error", "message": f"Cannot open stream {url!r}"})
                        return
                    await self.send({"type": "status", "message": f"camera not reachable, retrying in {retry:.0f}s"})
                    await asyncio.sleep(retry)
                    retry *= 2
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low on live RTSP streams
                fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                retry = 1.0
                try:
                    while not self.stop_stream:
                        ok, frame = await run_in_threadpool(cap.read)
                        if not ok:
                            break
                        t = time.monotonic() - t_start if live else n / fps
                        n += 1
                        h, w = frame.shape[:2]
                        if w > 960:
                            frame = cv2.resize(frame, (960, int(h * 960 / w)))
                        res = await self.process(frame, t)
                        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                        await self.ws.send_bytes(jpg.tobytes())
                        await self.send(res)
                        if not live:
                            await asyncio.sleep(max(0.0, (t_start + t) - time.monotonic()))  # play files at real speed
                finally:
                    cap.release()
                if not live:
                    break  # end of file
                if not self.stop_stream:  # dash cams drop Wi-Fi: reconnect instead of giving up
                    await self.send({"type": "status", "message": "camera stream lost, reconnecting..."})
                    await asyncio.sleep(retry)
        finally:
            await self.send({"type": "stream_ended"})

    async def handle_text(self, msg: dict):
        kind = msg.get("type")
        if kind == "reset":
            self.mon.reset(keep_profile=bool(msg.get("keep_profile")))
        elif kind == "recalibrate":
            self.save_trip()
            self.mon.reset(keep_profile=False)
            self.session_id, self.t_start = EventStore.new_session(), time.time()
        elif kind == "config":
            if "objects" in msg:
                self.mon._objects_enabled = bool(msg["objects"])
            if "calib_seconds" in msg:
                v = float(msg["calib_seconds"])
                if 1.0 <= v <= 60.0:
                    self.mon.cfg.calib_seconds = v
                    self.mon.calibrator.duration = v
        elif kind == "ack":  # driver pressed "I'm OK"
            self.mon.acknowledge(float(msg.get("t", self.mon._last_t or 0.0)))
            if self.store:
                self.store.add_event(self.session_id, "driver_ok", "info")
        elif kind == "report":
            self.save_trip()
            await self.send({"type": "report", **self.mon.recorder.full()})
        elif kind == "stream_start":
            url = str(msg.get("url", "")).strip()
            ok, why = security.stream_url_allowed(url, CFG["allow_files"])
            if not ok:
                await self.send({"type": "error", "message": why})
                return
            await self.stop()
            self.stop_stream = False
            self.mon.reset(keep_profile=False)
            self.stream_task = asyncio.create_task(self.run_stream(url))
        elif kind == "stream_stop":
            await self.stop()

    async def handle_bytes(self, data: bytes):
        if len(data) > MAX_FRAME_BYTES:
            await self.send({"type": "error", "message": "frame too large"})
            return
        if data[:4] == AUDIO_MAGIC:  # AUD0 | float64 t | float32 PCM @16 kHz
            if not CFG["audio"] or len(data) < 12:
                return
            (t,) = struct.unpack("<d", data[4:12])
            pcm = np.frombuffer(data[12:12 + 4 * MAX_AUDIO_SAMPLES], dtype="<f4")
            st = await run_in_threadpool(self.mon.push_audio, t, np.nan_to_num(pcm).clip(-1, 1))
            if st is not None:
                await self.send({"type": "audio", **st.as_dict()})
            return
        if self.stream_task is not None and not self.stream_task.done():
            return
        if len(data) < 9:
            return
        (t,) = struct.unpack("<d", data[:8])
        frame = cv2.imdecode(np.frombuffer(data[8:], np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            await self.send({"type": "error", "message": "bad frame"})
            return
        await self.send(await self.process(frame, t))

    async def stop(self):
        if self.stream_task and not self.stream_task.done():
            self.stop_stream = True
            try:
                await asyncio.wait_for(self.stream_task, 5)
            except (asyncio.TimeoutError, Exception):
                self.stream_task.cancel()
        self.stream_task = None


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    h = ws.headers
    given = ws.query_params.get("token") or ws.cookies.get(security.COOKIE)
    if not (security.host_allowed(h.get("host"), CFG["bind_host"], CFG["origins"])
            and security.origin_allowed(h.get("origin"), h.get("host"), CFG["origins"])
            and security.token_ok(CFG["token"], given)):
        await ws.close(code=1008)  # policy violation
        return
    await ws.accept()
    s = Session(ws)
    STATE["sessions"] += 1
    try:
        keys = ("face_ckpt", "cabin_ckpt", "temporal_ckpt", "yolo_weights")
        cfg = MonitorConfig(**{k: CFG.get(k) for k in keys if CFG.get(k)}, calib_seconds=CFG.get("calib_seconds", 5.0),
                            emotion_source=CFG.get("emotion_source", "pretrained"), audio=CFG["audio"], mishap=CFG["mishap"])
        s.mon = await run_in_threadpool(DriverMonitor, cfg)
        s.mon._objects_enabled = True
        orig = s.mon.object_model
        if orig is not None:  # allow the UI to switch YOLO on/off without reloading
            s.mon.object_model = lambda frame: orig(frame) if s.mon._objects_enabled else []
            s.mon.object_model.name = orig.name
        await s.send({"type": "ready", "models": {**s.mon.models_info(), "objects": orig.name if orig else None},
                      "features": {"audio": CFG["audio"], "mishap": CFG["mishap"]}})
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
            if m.get("text") is not None:
                try:
                    msg = json.loads(m["text"])
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict):
                    await s.handle_text(msg)
            elif m.get("bytes") is not None:
                await s.handle_bytes(m["bytes"])
    except WebSocketDisconnect:
        pass
    finally:
        STATE["sessions"] -= 1
        await s.stop()
        s.save_trip()
        if s.mon is not None:
            s.mon.close()


def main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1", help="127.0.0.1 = this computer only (default); 0.0.0.0 = whole network (token required)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--face-ckpt"), ap.add_argument("--cabin-ckpt"), ap.add_argument("--temporal-ckpt")
    ap.add_argument("--calib-seconds", type=float, default=5.0)
    ap.add_argument("--emotion", choices=["pretrained", "trained"], default="pretrained",
                    help="emotion model: pretrained HSEmotion (default) or the emotion head of --face-ckpt")
    ap.add_argument("--yolo", help="YOLO weights (default weights/yolo11n.pt; pass a fine-tuned best.pt)")
    ap.add_argument("--audio", action="store_true", help="enable cabin-audio events (YAMNet): snoring, shouting, sirens, crash sounds")
    ap.add_argument("--mishap", action="store_true", help="enable possible-accident detection (camera jolt / crash sound + unresponsive driver)")
    ap.add_argument("--token", help="access token (default: random when --host is not loopback; env DMS_TOKEN also works)")
    ap.add_argument("--allow-origin", action="append", default=[], help="extra allowed Origin/Host (e.g. a reverse proxy name)")
    ap.add_argument("--allow-files", action="store_true", help="let the stream box open local video files (off: only camera URLs)")
    ap.add_argument("--store", default=str(Path.home() / ".dms" / "events.db"), help="offline event store (SQLite); '' to disable")
    ap.add_argument("--sync-url", help="HTTPS endpoint to upload events/trips when online (store-and-forward)")
    ap.add_argument("--allow-unverified-models", action="store_true", help="start even if model checksums don't match")
    ap.add_argument("--ssl-certfile"), ap.add_argument("--ssl-keyfile")
    a = ap.parse_args(argv)

    problems = security.verify_weights()
    if problems and not a.allow_unverified_models:
        raise SystemExit("model integrity check failed:\n  " + "\n  ".join(problems) + "\n(re-run scripts/download_models.sh, or pass --allow-unverified-models)")
    exposed = not security.is_loopback(a.host)
    token = a.token or os.environ.get("DMS_TOKEN") or (security.new_token() if exposed else None)
    if exposed and not a.ssl_certfile:
        print("  WARNING: serving on the network without HTTPS - camera frames travel unencrypted. Use --ssl-certfile/--ssl-keyfile.")
    CFG.update(face_ckpt=a.face_ckpt, cabin_ckpt=a.cabin_ckpt, temporal_ckpt=a.temporal_ckpt, yolo_weights=a.yolo, calib_seconds=a.calib_seconds,
               emotion_source=a.emotion, audio=a.audio, mishap=a.mishap, token=token, bind_host=a.host,
               allow_files=a.allow_files, origins=set(a.allow_origin), weight_problems=problems)
    if a.store:
        STATE["store"] = EventStore(a.store)
        if a.sync_url:
            STATE["syncer"] = Syncer(STATE["store"], a.sync_url, os.environ.get("DMS_SYNC_TOKEN"))
            STATE["syncer"].start()
    probe = DriverMonitor(MonitorConfig(face_ckpt=a.face_ckpt, cabin_ckpt=a.cabin_ckpt, temporal_ckpt=a.temporal_ckpt, yolo_weights=a.yolo,
                                        emotion_source=a.emotion, audio=a.audio, mishap=a.mishap))
    CFG["models_info"] = probe.models_info()
    probe.close()
    scheme = "https" if a.ssl_certfile else "http"
    host = "localhost" if not exposed else a.host
    url = f"{scheme}://{host}:{a.port}" + (f"/?token={token}" if token else "")
    print(f"\n  Driver Monitoring System -> {url}")
    if exposed:
        print("  (from another device use this computer's IP address instead of 0.0.0.0; keep the token secret)")
    print(f"  features: audio={'on' if a.audio else 'off'}  accident-detection={'on' if a.mishap else 'off'}  "
          f"event store={a.store or 'off'}  sync={'on' if a.sync_url else 'off'}\n")
    uvicorn.run(app, host=a.host, port=a.port, ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile, log_level="warning")


if __name__ == "__main__":
    main()
