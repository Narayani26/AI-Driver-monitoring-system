"""Possible-accident ("mishap") detection.

Evidence of an impact:
  * audio: crash / breaking / bang / skid sounds (YAMNet groups "impact", "skid")
  * video: a camera *jolt* - the whole image changes at once, far beyond the
    normal frame-to-frame change (a collision shakes the camera; normal driving
    and head movement change only part of the image)

After an impact we watch the driver for `check_s` seconds:
  * driver visible, eyes open, attentive  -> "impact" warning ("Are you OK?")
  * no face / eyes closed for >= `unresponsive_s` -> "possible_accident":
    critical alarm + `sos_countdown_s` countdown; if nobody presses "I'm OK",
    an SOS event is raised (in a product: SMS/call an emergency contact with
    GPS; here it is logged to the offline event store).

Thresholds are deliberately conservative: false SOS calls are worse than a late
"are you OK?". This is a research prototype, not a certified eCall system.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np


class JoltDetector:
    """Camera-shake detector on a tiny grayscale thumbnail.

    A jolt needs BOTH:
      * a spike in whole-frame change, far above the recent baseline, and
      * a coherent shift of the whole image (phase correlation), i.e. the camera
        itself moved.
    Each thumbnail is brightness/contrast-normalised first, so lighting changes
    (tunnels, headlights) don't count, and content that merely appears or
    disappears (a face leaving, a hand over the lens) has no coherent shift.
    """

    def __init__(self, history_s: float = 5.0, k: float = 8.0, min_diff: float = 0.35, min_shift_px: float = 2.5,
                 min_response: float = 0.08, min_texture: float = 6.0, min_history: int = 15):
        self.history_s, self.k, self.min_diff, self.min_shift_px, self.min_history = history_s, k, min_diff, min_shift_px, min_history
        self.min_response, self.min_texture = min_response, min_texture
        self._prev_std = 0.0
        self._prev: np.ndarray | None = None
        self._hist: deque[tuple[float, float]] = deque()
        self._win = cv2.createHanningWindow((64, 48), cv2.CV_32F)
        self.last_score = 0.0
        self.last_shift = 0.0

    def update(self, t: float, frame: np.ndarray) -> bool:
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame, (64, 48), interpolation=cv2.INTER_AREA).astype(np.float32)
        std = float(g.std())
        g = (g - g.mean()) / (std + 8.0)  # +8: near-uniform frames stay near zero instead of amplifying noise
        if self._prev is None:
            self._prev, self._prev_std = g, std
            return False
        diff = float(np.mean(np.abs(g - self._prev)))
        vals = np.array([d for _, d in self._hist]) if self._hist else np.array([0.0])
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) + 0.02
        self.last_score = (diff - med) / mad
        spike = len(self._hist) >= self.min_history and diff > self.min_diff and self.last_score > self.k
        self.last_shift = 0.0
        # both frames need visible structure (a covered/black lens has nothing to align)
        if spike and min(std, self._prev_std) >= self.min_texture:
            (dx, dy), response = cv2.phaseCorrelate(self._prev, g, self._win)
            if response >= self.min_response and self._shift_explains_change(self._prev, g, dx, dy):
                self.last_shift = float(np.hypot(dx, dy))
        self._prev, self._prev_std = g, std
        self._hist.append((t, diff))
        while self._hist and t - self._hist[0][0] > self.history_s:
            self._hist.popleft()
        return spike and self.last_shift >= self.min_shift_px

    @staticmethod
    def _shift_explains_change(prev: np.ndarray, cur: np.ndarray, dx: float, dy: float, ratio: float = 0.5) -> bool:
        """Camera motion means the previous frame, moved by (dx, dy), matches the
        current one. Content that appeared/vanished is not explained by a shift.
        Compared on the central region that is visible in both frames."""
        h, w = cur.shape
        mx, my = int(abs(dx)) + 2, int(abs(dy)) + 2
        if 2 * mx >= w or 2 * my >= h:
            return False
        moved = cv2.warpAffine(prev, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), borderMode=cv2.BORDER_WRAP)
        c = (slice(my, h - my), slice(mx, w - mx))
        before = float(np.mean(np.abs(cur[c] - prev[c])))
        after = float(np.mean(np.abs(cur[c] - moved[c])))
        return after <= ratio * before

    def reset(self):
        self._prev = None
        self._hist.clear()


@dataclass
class MishapState:
    phase: str = "idle"  # idle | checking | impact | possible_accident | sos | cleared
    since: float = 0.0
    evidence: str = ""
    countdown: float = 0.0  # seconds left before SOS (phase possible_accident)


class MishapDetector:
    def __init__(self, audio_thresh: float = 0.3, strong_audio: float = 0.5, check_s: float = 10.0, unresponsive_s: float = 3.0, sos_countdown_s: float = 15.0):
        """Any evidence (jolt, or crash/skid sound >= audio_thresh) starts a *silent* check.
        The spoken "impact - are you OK?" warning needs strong evidence (a jolt or sound >= strong_audio);
        weak evidence only matters if the driver then becomes unresponsive."""
        self.audio_thresh, self.strong_audio, self.check_s, self.unresponsive_s, self.sos_countdown_s = audio_thresh, strong_audio, check_s, unresponsive_s, sos_countdown_s
        self._strong = False
        self.state = MishapState()
        self._unresp_since: float | None = None
        self.events: list[tuple[float, str, str]] = []  # (t, kind, evidence) newly raised since last read

    def acknowledge(self, t: float):
        """Driver pressed "I'm OK" (or said so): cancel any pending SOS."""
        if self.state.phase in ("checking", "impact", "possible_accident"):
            self.events.append((t, "driver_ok", self.state.evidence))
        self.state = MishapState(phase="cleared", since=t)
        self._unresp_since = None

    def update(self, t: float, *, jolt: bool, audio_groups: dict[str, float] | None, face: bool, eyes_closed: bool) -> MishapState:
        a = audio_groups or {}
        evidence = []
        if jolt:
            evidence.append("camera jolt")
        if a.get("impact", 0.0) >= self.audio_thresh:
            evidence.append("crash sound")
        if a.get("skid", 0.0) >= self.audio_thresh:
            evidence.append("skid sound")
        st = self.state
        if evidence and st.phase in ("idle", "cleared"):
            self.state = st = MishapState(phase="checking", since=t, evidence=", ".join(evidence))
            self._unresp_since = None
            self._strong = jolt or max(a.get("impact", 0.0), a.get("skid", 0.0)) >= self.strong_audio
        unresponsive_now = (not face) or eyes_closed
        if st.phase in ("checking", "impact"):
            if unresponsive_now:
                self._unresp_since = t if self._unresp_since is None else self._unresp_since
                if t - self._unresp_since >= self.unresponsive_s:
                    self.state = st = MishapState(phase="possible_accident", since=t, evidence=st.evidence, countdown=self.sos_countdown_s)
                    self.events.append((t, "possible_accident", st.evidence))
            else:
                self._unresp_since = None
                if st.phase == "checking" and self._strong:
                    st.phase = "impact"
                    self.events.append((t, "impact", st.evidence))
            if st.phase in ("checking", "impact") and t - st.since > self.check_s:
                self.state = st = MishapState(phase="idle", since=t)
        elif st.phase == "possible_accident":
            st.countdown = max(0.0, self.sos_countdown_s - (t - st.since))
            if st.countdown <= 0:
                self.state = st = MishapState(phase="sos", since=t, evidence=st.evidence)
                self.events.append((t, "sos", st.evidence))
        elif st.phase == "sos" and t - st.since > 60:
            self.state = st = MishapState(phase="idle", since=t)
        elif st.phase == "cleared" and t - st.since > 5:
            self.state = st = MishapState(phase="idle", since=t)
        return st

    def pop_events(self):
        ev, self.events = self.events, []
        return ev
