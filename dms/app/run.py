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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index or video path")
    ap.add_argument("--out", help="write annotated video here")
    ap.add_argument("--log", help="write per-frame JSON lines here")
    ap.add_argument("--headless", action="store_true", help="no display window")
    ap.add_argument("--face-ckpt"), ap.add_argument("--cabin-ckpt"), ap.add_argument("--temporal-ckpt")
    ap.add_argument("--profile", help="load a saved driver profile (skips calibration)")
    ap.add_argument("--save-profile", help="save the calibrated driver profile here")
    ap.add_argument("--calib-seconds", type=float, default=10.0)
    ap.add_argument("--no-alarm", action="store_true")
    ap.add_argument("--side-by-side", action="store_true", help="output video = original | analysis")
    ap.add_argument("--report", help="write the trip report JSON here at the end")
    ap.add_argument("--no-objects", action="store_true", help="disable YOLO phone/object detection")
    ap.add_argument("--yolo", help="YOLO weights (default weights/yolo11n.pt)")
    a = ap.parse_args(argv)

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"cannot open source {a.source!r}")
    is_file = not isinstance(src, int)
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0

    mon = DriverMonitor(MonitorConfig(face_ckpt=a.face_ckpt, cabin_ckpt=a.cabin_ckpt, temporal_ckpt=a.temporal_ckpt, profile=a.profile, calib_seconds=a.calib_seconds, objects=not a.no_objects, yolo_weights=a.yolo))
    writer = log = None
    if a.log:
        log = open(a.log, "w")
    n, t0, saved, last_level = 0, time.monotonic(), False, "OK"
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            # Video files use their own clock so results don't depend on processing speed.
            t = n / fps_in if is_file else time.monotonic() - t0
            r = mon.process(frame, t)
            n += 1
            vis = draw_overlay(frame, r)
            if a.side_by_side:
                vis = np.hstack([frame, vis])
            for al in r.alerts:
                print(f"[{t:7.2f}s] ALERT {al.severity.upper():8s} {al.message}  -> {al.suggestion}")
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
