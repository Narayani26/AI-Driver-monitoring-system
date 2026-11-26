"""Run the monitor on a webcam or video file.

    python -m dms.app.run --source 0                      # webcam, window
    python -m dms.app.run --source drive.mp4 --out annotated.mp4 --log run.jsonl --headless
    python -m dms.app.run --source drive.mp4 --out demo_side_by_side.mp4 --side-by-side --headless --report trip.json
    python -m dms.app.run --source rtsp://192.168.1.254/live      # Wi-Fi dash cam / IP camera
    python -m dms.app.run --source 0 --face-ckpt runs/face/best.pt --cabin-ckpt runs/cabin/best.pt \
                          --temporal-ckpt runs/temporal/best.pt --save-profile me.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time

from pathlib import Path

import cv2
import numpy as np

from ..pipeline import DriverMonitor, MonitorConfig, draw_overlay


def open_writer(path: str, fps: float, size) -> cv2.VideoWriter:
    """H.264 if this OpenCV build has it (plays in PowerPoint/QuickTime/browsers), else MPEG-4."""
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w
    raise RuntimeError(f"cannot open video writer for {path}")


class Speaker:
    """Non-blocking on-device text-to-speech for headless/in-car use (no browser, no internet)."""

    FEMALE_MAC = ("Samantha", "Veena", "Karen", "Moira", "Tessa", "Fiona", "Victoria")

    def __init__(self, voice: str | None = None):
        import shutil
        import subprocess

        self.cmd = next(([c] for c in ("say", "espeak-ng", "espeak", "spd-say") if shutil.which(c)), None)
        if self.cmd == ["say"]:  # macOS: pick a female voice that is installed
            installed = subprocess.run(["say", "-v", "?"], capture_output=True, text=True).stdout
            names = [line.split()[0] for line in installed.splitlines() if line.strip()]
            pick = voice if voice else next((v for v in self.FEMALE_MAC if v in names), None)
            if pick:
                self.cmd = ["say", "-v", pick, "-r", "180"]
        elif self.cmd and self.cmd[0].startswith("espeak"):
            self.cmd = self.cmd + ["-v", voice or "en-us+f3", "-s", "160"]  # +f3 = female variant
        self.proc = None
        self.sev = -1
        if self.cmd is None:
            print("no text-to-speech command found (install espeak-ng); alerts will only be printed")

    def say(self, alert):
        import subprocess

        from ..fusion.alerts import SEVERITY

        if self.cmd is None:
            return
        sev = SEVERITY[alert.severity]
        busy = self.proc is not None and self.proc.poll() is None
        if busy and sev <= self.sev:
            return  # don't interrupt a more important message
        if busy:
            self.proc.terminate()
        text = f"{alert.message} {alert.suggestion}" if sev >= 2 else alert.message
        self.proc, self.sev = subprocess.Popen(self.cmd + [text], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), sev


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--out", help="write annotated video here")
    ap.add_argument("--log", help="write per-frame JSON lines here")
    ap.add_argument("--headless", action="store_true", help="no display window")
    ap.add_argument("--face-ckpt"), ap.add_argument("--cabin-ckpt"), ap.add_argument("--temporal-ckpt")
    ap.add_argument("--emotion", choices=["pretrained", "trained"], default="pretrained")
    ap.add_argument("--profile", help="load a saved driver profile (skips calibration)")
    ap.add_argument("--save-profile", help="save the calibrated driver profile here")
    ap.add_argument("--calib-seconds", type=float, default=10.0)
    ap.add_argument("--no-alarm", action="store_true")
    ap.add_argument("--side-by-side", action="store_true", help="output video = original | analysis")
    ap.add_argument("--report", help="write the trip report JSON here at the end")
    ap.add_argument("--no-objects", action="store_true", help="disable YOLO phone/object detection")
    ap.add_argument("--yolo", help="YOLO weights (default weights/yolo11n.pt)")
    ap.add_argument("--mishap", action="store_true", help="possible-accident detection (camera jolt + unresponsive driver)")
    ap.add_argument("--speak", action="store_true", help="speak alerts with the system voice (macOS 'say', Linux 'espeak-ng'/'spd-say')")
    ap.add_argument("--voice", help="voice for --speak, e.g. Samantha / Veena / Karen on macOS (list: say -v '?'); default: a female voice")
    ap.add_argument("--sensitivity", choices=["low", "normal", "high"], default="normal", help="low = calmer, high = stricter")
    ap.add_argument("--store", help="offline event store (SQLite) for alerts + trip summary, e.g. ~/.dms/events.db")
    a = ap.parse_args(argv)

    src = int(a.source) if a.source.isdigit() else a.source
    live = isinstance(src, int) or "://" in a.source  # webcam / RTSP / HTTP camera
    cap = cv2.VideoCapture(src)
    if not cap.isOpened() and not live:
        sys.exit(f"cannot open source {a.source!r}")
    is_file = not live
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mon = DriverMonitor(MonitorConfig(face_ckpt=a.face_ckpt, cabin_ckpt=a.cabin_ckpt, temporal_ckpt=a.temporal_ckpt, profile=a.profile,
                                      calib_seconds=a.calib_seconds, objects=not a.no_objects, yolo_weights=a.yolo,
                                      emotion_source=a.emotion, mishap=a.mishap, sensitivity=a.sensitivity))
    store = session = None
    if a.store:
        from ..storage import EventStore

        store, session = EventStore(Path(a.store).expanduser()), EventStore.new_session()
    voice = Speaker(a.voice) if a.speak else None
    writer = log = None
    if a.log:
        log = open(a.log, "w")
    n, t0, saved, last_level, retry = 0, time.monotonic(), False, "OK", 1.0
    try:
        while True:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if not ok:
                if not live:
                    break
                # dash cams / Wi-Fi cameras drop out: keep reconnecting instead of exiting
                print(f"camera unavailable, reconnecting in {retry:.0f}s ...")
                cap.release()
                time.sleep(retry)
                retry = min(retry * 2, 30.0)
                cap = cv2.VideoCapture(src)
                if live:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                continue
            retry = 1.0
            # Video files use their own clock so results don't depend on processing speed.
            t = n / fps_in if is_file else time.monotonic() - t0
            r = mon.process(frame, t)
            n += 1
            vis = draw_overlay(frame, r)
            if a.side_by_side:
                vis = np.hstack([frame, vis])
            for al in r.alerts:
                print(f"[{t:7.2f}s] ALERT {al.severity.upper():8s} {al.message}  -> {al.suggestion}")
                if store:
                    store.add_event(session, al.kind, al.severity, t_session=t, detail=al.detail)
                if voice:
                    voice.say(al)
            if a.out:
                if writer is None:
                    writer = open_writer(a.out, fps_in, vis.shape[1::-1])
                writer.write(vis)
            if log:
                log.write(json.dumps(r.to_log()) + "\n")
            if a.save_profile and mon.profile and not saved:
                mon.profile.save(a.save_profile)
                saved = True
            if r.risk is not None and r.risk.level != last_level:
                if r.risk.level in ("WARNING", "CRITICAL") and not a.no_alarm:
                    print("\a", end="", flush=True)  # terminal bell; swap for a buzzer/GPIO on hardware
                print(f"[{t:7.2f}s] level {last_level} -> {r.risk.level}: {'; '.join(r.risk.reasons) or '-'}")
                last_level = r.risk.level
            if not a.headless:
                cv2.imshow("Driver Monitor", vis)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        if a.report:
            Path(a.report).write_text(json.dumps(mon.recorder.full(), indent=2))
        if store:
            store.save_trip(session, mon.recorder.summary())
            store.close()
        cap.release()
        mon.close()
        if writer:
            writer.release()
        if log:
            log.close()
        if not a.headless:
            cv2.destroyAllWindows()
    dt = time.monotonic() - t0
    print(f"processed {n} frames in {dt:.1f}s ({n / max(dt, 1e-9):.1f} FPS)")


if __name__ == "__main__":
    main()
