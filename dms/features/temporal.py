"""Time-domain indicators built on top of per-frame features.

Everything here is driven by timestamps (seconds), not frame counts, so results
are the same at 15 FPS on a Raspberry Pi and 60 FPS on a laptop.
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from .geometry import FrameFeatures


# --------------------------------------------------------------------------- #
# Per-driver calibration
# --------------------------------------------------------------------------- #
@dataclass
class DriverProfile:
    """Baseline for one driver. Thresholds are expressed relative to it, which is
    what makes the system work across people with naturally narrow/wide eyes,
    and across camera mounting positions (neutral yaw/pitch offset)."""

    ear_open: float = 0.28
    mar_rest: float = 0.05
    yaw0: float = 0.0
    pitch0: float = 0.0
    n_samples: int = 0

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "DriverProfile":
        return cls(**json.loads(Path(path).read_text()))


class Calibrator:
    """Collects `duration` seconds of face-present frames, then emits a profile.

    Robust statistics are used so a few blinks during calibration don't matter:
    open-eye EAR is the 80th percentile, neutral head pose is the median.
    """

    def __init__(self, duration: float = 10.0, min_samples: int = 60):
        self.duration = duration
        self.min_samples = min_samples
        self._t0: float | None = None
        self._rows: list[tuple[float, float, float, float]] = []
        self.profile: DriverProfile | None = None

    @property
    def done(self) -> bool:
        return self.profile is not None

    def progress(self, t: float) -> float:
        if self._t0 is None:
            return 0.0
        return min(1.0, (t - self._t0) / self.duration)

    def update(self, t: float, f: FrameFeatures) -> DriverProfile | None:
        if self.done:
            return self.profile
        if not f.face or np.isnan(f.ear):
            return None
        if self._t0 is None:
            self._t0 = t
        self._rows.append((f.ear, f.mar, f.yaw, f.pitch))
        if t - self._t0 >= self.duration and len(self._rows) >= self.min_samples:
            a = np.array(self._rows, dtype=np.float64)
            self.profile = DriverProfile(
                ear_open=float(np.nanpercentile(a[:, 0], 80)),
                mar_rest=float(np.nanmedian(a[:, 1])),
                yaw0=float(np.nanmedian(a[:, 2])) if not np.all(np.isnan(a[:, 2])) else 0.0,
                pitch0=float(np.nanmedian(a[:, 3])) if not np.all(np.isnan(a[:, 3])) else 0.0,
                n_samples=len(a),
            )
        return self.profile


# --------------------------------------------------------------------------- #
# Eye closure: blinks, PERCLOS, microsleep
# --------------------------------------------------------------------------- #
class EyeClosureTracker:
    """Tracks eye openness relative to the driver's baseline.

    * closed  : openness < `closed_frac` (default 0.2 -> eyes >= 80 % closed, the
                P80 criterion used by PERCLOS).
    * blink   : a closed episode shorter than `microsleep_s`.
    * microsleep : closed continuously for >= `microsleep_s`.
    * PERCLOS : fraction of time closed over the last `window_s` seconds.
    """

    def __init__(self, closed_frac: float = 0.2, microsleep_s: float = 1.0, window_s: float = 60.0, min_span_s: float = 30.0):
        self.min_span_s = min_span_s  # don't trust PERCLOS from a few seconds of data
        self.closed_frac = closed_frac
        self.microsleep_s = microsleep_s
        self.window_s = window_s
        self._samples: deque[tuple[float, bool]] = deque()  # (t, closed)
        self._blinks: deque[tuple[float, float]] = deque()  # (t_end, duration)
        self._closed_since: float | None = None
        self.microsleep_events = 0
        self._in_microsleep = False

    def openness(self, ear: float, profile: DriverProfile) -> float:
        # Map EAR to [0, 1] where 1 == the driver's normal open eye. A fully
        # closed eye still has EAR ~ 0.3 * open because of eyelid thickness.
        floor = 0.3 * profile.ear_open
        return float(np.clip((ear - floor) / max(profile.ear_open - floor, 1e-6), 0.0, 1.0))

    def update(self, t: float, ear: float, profile: DriverProfile) -> None:
        if np.isnan(ear):
            return
        closed = self.openness(ear, profile) < self.closed_frac
        self._samples.append((t, closed))
        while self._samples and t - self._samples[0][0] > self.window_s:
            self._samples.popleft()
        if closed:
            if self._closed_since is None:
                self._closed_since = t
            if not self._in_microsleep and t - self._closed_since >= self.microsleep_s:
                self._in_microsleep = True
                self.microsleep_events += 1
        elif self._closed_since is not None:
            dur = t - self._closed_since
            if dur < self.microsleep_s:
                self._blinks.append((t, dur))
            self._closed_since = None
            self._in_microsleep = False
        while self._blinks and t - self._blinks[0][0] > self.window_s:
            self._blinks.popleft()

    @property
    def closed_duration(self) -> float:
        if self._closed_since is None or not self._samples:
            return 0.0
        return self._samples[-1][0] - self._closed_since

    @property
    def in_microsleep(self) -> bool:
        return self._in_microsleep

    def perclos(self) -> float:
        """Time-weighted fraction of the window with eyes closed."""
        if len(self._samples) < 2:
            return 0.0
        ts = np.array([s[0] for s in self._samples])
        cl = np.array([s[1] for s in self._samples], dtype=np.float64)
        dt = np.diff(ts)
        total = max(dt.sum(), self.min_span_s)
        return float((cl[:-1] * dt).sum() / total)

    def blink_rate(self) -> float:
        """Blinks per minute over the window (span-normalised early on)."""
        if not self._samples:
            return 0.0
        span = max(self._samples[-1][0] - self._samples[0][0], 1.0)
        return len(self._blinks) * 60.0 / span

    def mean_blink_duration(self) -> float:
        return float(np.mean([d for _, d in self._blinks])) if self._blinks else 0.0


# --------------------------------------------------------------------------- #
# Yawning
# --------------------------------------------------------------------------- #
class YawnTracker:
    """A yawn is the mouth held wide open for >= `min_s` seconds.

    `jaw_open_bs` (MediaPipe blendshape, 0..1) is preferred when available because
    it's much less sensitive to talking than raw MAR.
    """

    def __init__(self, mar_thresh: float = 0.6, jaw_thresh: float = 0.5, min_s: float = 1.5, window_s: float = 300.0):
        self.mar_thresh = mar_thresh
        self.jaw_thresh = jaw_thresh
        self.min_s = min_s
        self.window_s = window_s
        self._open_since: float | None = None
        self._counted = False
        self._yawns: deque[float] = deque()

    def update(self, t: float, mar: float, jaw_open: float) -> bool:
        if not np.isnan(jaw_open):
            is_open = jaw_open > self.jaw_thresh
        elif not np.isnan(mar):
            is_open = mar > self.mar_thresh
        else:
            return False
        new_yawn = False
        if is_open:
            if self._open_since is None:
                self._open_since, self._counted = t, False
            elif not self._counted and t - self._open_since >= self.min_s:
                self._yawns.append(t)
                self._counted = new_yawn = True
        else:
            self._open_since = None
        while self._yawns and t - self._yawns[0] > self.window_s:
            self._yawns.popleft()
        return new_yawn

    @property
    def yawning(self) -> bool:
        return self._counted and self._open_since is not None

    def count(self) -> int:
        return len(self._yawns)


# --------------------------------------------------------------------------- #
# Visual distraction (head pose / gaze)
# --------------------------------------------------------------------------- #
@dataclass
class GlanceState:
    off_road: bool = False
    off_road_duration: float = 0.0
    cumulative_off_road: float = 0.0  # within the VATS window
    long_distraction: bool = False
    vats_distraction: bool = False
    events: list[str] = field(default_factory=list)


class GlanceTracker:
    """Eyes-off-road logic modelled on Euro NCAP's two distraction types:

    * long distraction : one continuous glance away >= `long_s` (default 3 s)
    * visual attention time sharing (VATS): repeated short glances adding up to
      >= `vats_cum_s` within `vats_window_s` (default 10 s within 30 s)

    The exact protocol numbers live in Euro NCAP SD-202; they're configurable so
    you can match whichever protocol version you cite in the paper.
    """

    def __init__(
        self,
        yaw_thresh: float = 25.0,
        pitch_down_thresh: float = 20.0,
        pitch_up_thresh: float = 20.0,
        long_s: float = 3.0,
        vats_window_s: float = 30.0,
        vats_cum_s: float = 10.0,
    ):
        self.yaw_thresh = yaw_thresh
        self.pitch_down_thresh = pitch_down_thresh
        self.pitch_up_thresh = pitch_up_thresh
        self.long_s = long_s
        self.vats_window_s = vats_window_s
        self.vats_cum_s = vats_cum_s
        self._hist: deque[tuple[float, bool]] = deque()
        self._away_since: float | None = None

    def is_off_road(self, yaw: float, pitch: float, profile: DriverProfile) -> bool:
        if np.isnan(yaw) or np.isnan(pitch):
            return False
        dy = yaw - profile.yaw0
        dp = pitch - profile.pitch0
        # MediaPipe pitch is positive when looking down in image coordinates.
        return abs(dy) > self.yaw_thresh or dp > self.pitch_down_thresh or -dp > self.pitch_up_thresh

    def update(self, t: float, yaw: float, pitch: float, profile: DriverProfile, face: bool = True) -> GlanceState:
        # A missing face while we were tracking one usually means the head turned
        # far away (or the phone is in front of it) -> treat as off-road.
        off = self.is_off_road(yaw, pitch, profile) if face else True
        self._hist.append((t, off))
        while self._hist and t - self._hist[0][0] > self.vats_window_s:
            self._hist.popleft()
        if off:
            self._away_since = t if self._away_since is None else self._away_since
        else:
            self._away_since = None
        dur = 0.0 if self._away_since is None else t - self._away_since
        ts = np.array([h[0] for h in self._hist])
        offs = np.array([h[1] for h in self._hist], dtype=np.float64)
        cum = float((offs[:-1] * np.diff(ts)).sum()) if len(ts) > 1 else 0.0
        st = GlanceState(off_road=off, off_road_duration=dur, cumulative_off_road=cum)
        st.long_distraction = dur >= self.long_s
        st.vats_distraction = cum >= self.vats_cum_s and not st.long_distraction
        if st.long_distraction:
            st.events.append(f"eyes off road {dur:.1f}s")
        if st.vats_distraction:
            st.events.append(f"frequent glances away ({cum:.0f}s in {self.vats_window_s:.0f}s)")
        return st


# --------------------------------------------------------------------------- #
# Sequence buffer for the learned temporal model
# --------------------------------------------------------------------------- #
class FeatureSequence:
    """Rolling, fixed-rate resampled buffer of normalised per-frame vectors.

    The temporal model is trained on sequences sampled at `rate_hz`; at runtime
    frames arrive at an arbitrary rate, so we keep the latest vector and emit one
    sample every 1/rate_hz seconds (zero-order hold).
    """

    def __init__(self, length: int = 300, rate_hz: float = 10.0):
        self.length = length
        self.period = 1.0 / rate_hz
        self.buf: deque[np.ndarray] = deque(maxlen=length)
        self._next_t: float | None = None

    @staticmethod
    def normalise(v: np.ndarray, profile: DriverProfile) -> np.ndarray:
        """Driver-normalised features; same transform must be used in training."""
        out = v.astype(np.float32).copy()
        out[0] = v[0] / max(profile.ear_open, 1e-6)  # ear
        out[1] = v[1] - profile.mar_rest  # mar
        out[2] = (v[2] - profile.pitch0) / 30.0  # pitch
        out[3] = (v[3] - profile.yaw0) / 30.0  # yaw
        out[4] = v[4] / 30.0  # roll
        out[5] = v[5] - 0.5  # gaze_h
        out[6] = v[6] - 0.5  # gaze_v
        return np.nan_to_num(out, nan=0.0)

    def update(self, t: float, v: np.ndarray, profile: DriverProfile) -> None:
        if self._next_t is None or t - self._next_t > self.length * self.period:
            self._next_t = t  # first sample, or a long gap: don't replay stale holds
        nv = self.normalise(v, profile)
        while t >= self._next_t:
            self.buf.append(nv)
            self._next_t += self.period

    @property
    def full(self) -> bool:
        return len(self.buf) == self.length

    def array(self) -> np.ndarray:
        return np.stack(self.buf) if self.buf else np.zeros((0, len(FrameFeatures.VECTOR_FIELDS)), np.float32)
