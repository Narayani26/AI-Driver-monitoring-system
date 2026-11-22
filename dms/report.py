"""Session (trip) report: running statistics + downsampled timeline + alert log."""
from __future__ import annotations

import csv
import io
import time
from collections import Counter

import numpy as np

from .fusion.risk import LEVELS


class SessionRecorder:
    def __init__(self, timeline_hz: float = 2.0):
        self.period = 1.0 / timeline_hz
        self.reset()

    def reset(self):
        self.started_wall = time.time()
        self.t0 = self.t_last = None
        self._next = None
        self.timeline: list[dict] = []
        self.alerts: list[dict] = []
        self.level_time = Counter()
        self.emotion_time = Counter()
        self.behaviour_time = Counter()
        self.frames = self.face_frames = 0
        self.max_risk = 0.0
        self.risk_sum = self.risk_w = 0.0
        self.max_perclos = 0.0
        self.max_off_road = 0.0
        self.max_closed = 0.0
        self.yawns = 0
        self.blink_rates: list[float] = []

    def update(self, r, alerts, behaviours: dict | None, off_road: float, closed: float):
        t = r.t
        if self.t0 is None:
            self.t0 = self._next = t
        dt = 0.0 if self.t_last is None else max(0.0, min(t - self.t_last, 1.0))
        self.t_last = t
        self.frames += 1
        self.face_frames += int(r.features.face)
        for a in alerts:
            self.alerts.append({**a.as_dict(), "t": round(a.t - self.t0, 2)})
        if r.risk is None:
            return
        self.level_time[r.risk.level] += dt
        if r.emotion:
            self.emotion_time[r.emotion] += dt
        for b, c in (behaviours or {}).items():
            if c >= 0.5:
                self.behaviour_time[b] += dt
        self.max_risk = max(self.max_risk, r.risk.index)
        self.risk_sum += r.risk.index * dt
        self.risk_w += dt
        self.max_perclos = max(self.max_perclos, r.perclos)
        self.max_off_road = max(self.max_off_road, off_road)
        self.max_closed = max(self.max_closed, closed)
        self.yawns = max(self.yawns, r.yawns)
        if r.blink_rate > 0:
            self.blink_rates.append(r.blink_rate)
        if t >= self._next:
            self._next = t + self.period
            self.timeline.append({"t": round(t - self.t0, 2), "risk": round(r.risk.index, 1), "drowsiness": round(r.risk.drowsiness, 3),
                                  "distraction": round(r.risk.distraction, 3), "affect": round(r.risk.affect, 3), "level": r.risk.level,
                                  "perclos": round(r.perclos, 3), "emotion": r.emotion})

    def summary(self) -> dict:
        dur = 0.0 if self.t0 is None else self.t_last - self.t0
        tot = sum(self.level_time.values()) or 1.0
        by_kind = Counter(a["kind"] for a in self.alerts)
        score = 100.0 - (self.risk_sum / self.risk_w if self.risk_w else 0.0)
        return {
            "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.started_wall)),
            "duration_s": round(dur, 1),
            "frames": self.frames,
            "face_visible_pct": round(100 * self.face_frames / max(self.frames, 1), 1),
            "safety_score": round(score, 1),
            "avg_risk": round(self.risk_sum / self.risk_w, 1) if self.risk_w else 0.0,
            "max_risk": round(self.max_risk, 1),
            "time_in_level_pct": {lv: round(100 * self.level_time[lv] / tot, 1) for lv in LEVELS},
            "alerts_total": len(self.alerts),
            "alerts_by_kind": dict(by_kind),
            "max_perclos_pct": round(100 * self.max_perclos, 1),
            "longest_eye_closure_s": round(self.max_closed, 2),
            "longest_eyes_off_road_s": round(self.max_off_road, 2),
            "yawns": self.yawns,
            "mean_blink_rate": round(float(np.mean(self.blink_rates)), 1) if self.blink_rates else 0.0,
            "emotion_time_pct": {k: round(100 * v / tot, 1) for k, v in self.emotion_time.most_common()},
            "behaviour_time_s": {k: round(v, 1) for k, v in self.behaviour_time.items()},
        }

    def full(self) -> dict:
        return {"summary": self.summary(), "alerts": self.alerts, "timeline": self.timeline}

    def timeline_csv(self) -> str:
        buf = io.StringIO()
        if self.timeline:
            w = csv.DictWriter(buf, fieldnames=list(self.timeline[0]))
            w.writeheader()
            w.writerows(self.timeline)
        return buf.getvalue()
