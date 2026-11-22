"""Web demo: side-by-side live footage + AI analysis dashboard.

    python -m dms.web.server                 # then open http://localhost:8000
    python -m dms.web.server --host 0.0.0.0 --ssl-certfile cert.pem --ssl-keyfile key.pem   # reachable from a phone

Sources (chosen in the browser):
* Webcam / uploaded video: the browser sends JPEG frames over a WebSocket
  (stop-and-wait, so the server is never flooded and results stay real-time).
* Stream URL (RTSP/HTTP, e.g. a Wi-Fi dash cam or a phone running an IP-camera
  app): the server reads the stream itself and pushes frames + results back.
  This is the same data path the dash cam -> phone product would use.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import struct
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from ..features.geometry import LEFT_EYE, RIGHT_EYE
from ..models.networks import EMOTION_CLASSES
from ..pipeline import DriverMonitor, FrameResult, MonitorConfig

ROOT = Path(__file__).resolve().parents[2]
STATIC = Path(__file__).parent / "static"
app = FastAPI(title="Driver Monitoring System")
app.mount("/static", StaticFiles(directory=STATIC), name="static")
CFG: dict = {}  # filled by main(): checkpoint paths etc.
MOUTH = (61, 81, 13, 311, 291, 402, 14, 178)


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
                rows.append({"run": name, "task": task, "protocol": "subject-independent test split", "accuracy": v["accuracy"], "macro_f1": v["macro_f1"], "n": v["n"]})
        elif "summary" in m:
            s = m["summary"]
            rows.append({"run": name, "task": "drowsiness (3-class)", "protocol": "subject-independent k-fold", "accuracy": s["video_accuracy"]["mean"], "macro_f1": s["window_macro_f1"]["mean"], "std": s["window_macro_f1"]["std"]})
    return rows


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/info")
def info():
    return JSONResponse({"metrics": collect_metrics(), "models": CFG.get("models_info"), "config": {k: v for k, v in CFG.items() if k != "models_info"}})


class Session:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.mon: DriverMonitor | None = None
        self.stream_task: asyncio.Task | None = None
        self.stop_stream = False

    async def send(self, obj):
        await self.ws.send_text(json.dumps(obj))

    async def process(self, frame: np.ndarray, t: float) -> dict:
        t0 = time.perf_counter()
        r = await run_in_threadpool(self.mon.process, frame, t)
        return serialize(r, frame.shape, (time.perf_counter() - t0) * 1000)

    async def run_stream(self, url: str):
        src = int(url) if url.isdigit() else url
        cap = await run_in_threadpool(cv2.VideoCapture, src)
        if not cap.isOpened():
            await self.send({"type": "error", "message": f"Cannot open stream {url!r}"})
            return
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep latency low on live RTSP streams
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        live = isinstance(src, int) or "://" in str(src)
        t_start, n = time.monotonic(), 0
        await self.send({"type": "stream_started", "url": url})
        try:
            while not self.stop_stream:
                ok, frame = await run_in_threadpool(cap.read)
                if not ok:
                    break
                if live:  # drop stale frames so we stay real-time
                    t = time.monotonic() - t_start
                else:
                    t = n / fps
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
            await self.send({"type": "stream_ended"})

    async def handle_text(self, msg: dict):
        kind = msg.get("type")
        if kind == "reset":
            self.mon.reset(keep_profile=bool(msg.get("keep_profile")))
        elif kind == "recalibrate":
            self.mon.reset(keep_profile=False)
        elif kind == "config":
            if "objects" in msg:
                self.mon._objects_enabled = bool(msg["objects"])
            if "calib_seconds" in msg:
                self.mon.cfg.calib_seconds = float(msg["calib_seconds"])
                self.mon.calibrator.duration = float(msg["calib_seconds"])
        elif kind == "report":
            await self.send({"type": "report", **self.mon.recorder.full()})
        elif kind == "stream_start":
            await self.stop()
            self.stop_stream = False
            self.mon.reset(keep_profile=False)
            self.stream_task = asyncio.create_task(self.run_stream(str(msg.get("url", "")).strip()))
        elif kind == "stream_stop":
            await self.stop()

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
    await ws.accept()
    s = Session(ws)
    try:
        cfg = MonitorConfig(**{k: CFG.get(k) for k in ("face_ckpt", "cabin_ckpt", "temporal_ckpt", "yolo_weights") if CFG.get(k)}, calib_seconds=CFG.get("calib_seconds", 5.0))
        s.mon = await run_in_threadpool(DriverMonitor, cfg)
        s.mon._objects_enabled = True
        orig = s.mon.object_model
        if orig is not None:  # allow the UI to switch YOLO on/off without reloading
            s.mon.object_model = lambda frame: orig(frame) if s.mon._objects_enabled else []
            s.mon.object_model.name = orig.name
        await s.send({"type": "ready", "models": {**s.mon.models_info(), "objects": orig.name if orig else None}})
        while True:
            m = await ws.receive()
            if m["type"] == "websocket.disconnect":
                break
            if m.get("text") is not None:
                await s.handle_text(json.loads(m["text"]))
            elif m.get("bytes") is not None and (s.stream_task is None or s.stream_task.done()):
                data = m["bytes"]
                (t,) = struct.unpack("<d", data[:8])
                frame = cv2.imdecode(np.frombuffer(data[8:], np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    await s.send({"type": "error", "message": "bad frame"})
                    continue
                await s.send(await s.process(frame, t))
    except WebSocketDisconnect:
        pass
    finally:
        await s.stop()
        if s.mon is not None:
            s.mon.close()


def main(argv=None):
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--face-ckpt"), ap.add_argument("--cabin-ckpt"), ap.add_argument("--temporal-ckpt")
    ap.add_argument("--calib-seconds", type=float, default=5.0)
    ap.add_argument("--yolo", help="YOLO weights (default weights/yolo11n.pt; pass a fine-tuned best.pt)")
    ap.add_argument("--ssl-certfile"), ap.add_argument("--ssl-keyfile")
    a = ap.parse_args(argv)
    CFG.update(face_ckpt=a.face_ckpt, cabin_ckpt=a.cabin_ckpt, temporal_ckpt=a.temporal_ckpt, yolo_weights=a.yolo, calib_seconds=a.calib_seconds)
    probe = DriverMonitor(MonitorConfig(face_ckpt=a.face_ckpt, cabin_ckpt=a.cabin_ckpt, temporal_ckpt=a.temporal_ckpt, yolo_weights=a.yolo))
    CFG["models_info"] = probe.models_info()
    probe.close()
    scheme = "https" if a.ssl_certfile else "http"
    print(f"\n  Driver Monitoring demo -> {scheme}://{'localhost' if a.host in ('127.0.0.1', '0.0.0.0') else a.host}:{a.port}\n")
    uvicorn.run(app, host=a.host, port=a.port, ssl_certfile=a.ssl_certfile, ssl_keyfile=a.ssl_keyfile, log_level="warning")


if __name__ == "__main__":
    main()
