"""Driver Risk Index: fuses drowsiness, distraction and affect into one score.

Design goals (these are what you defend in the paper):
* Explainable: every alert carries the human-readable reasons that produced it.
* Stable: exponential smoothing in *time* plus hysteresis on alert levels, so a
  single bad frame can't trigger or clear an alarm.
* Graceful degradation: every learned input is optional; with no checkpoints
  the index is computed from geometric indicators alone.
* Emotion is a *modifier*, not a primary hazard: anger/fear raise the risk of
  an existing hazard and raise a soft caution when sustained, mirroring the
  evidence that negative high-arousal states degrade driving performance.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..models.networks import ALERTNESS_CLASSES, CABIN_CLASSES, EMOTION_CLASSES

LEVELS = ("OK", "CAUTION", "WARNING", "CRITICAL")
ENTER = (0.0, 30.0, 55.0, 80.0)  # index needed to enter each level
EXIT_MARGIN = 10.0  # must drop this far below the entry threshold to leave a level


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


class _EMA:
    def __init__(self, tau: float):
        self.tau, self.v, self.t = tau, None, None

    def __call__(self, t: float, x: float) -> float:
        if self.v is None:
            self.v, self.t = x, t
            return x
        a = 1.0 - math.exp(-max(t - self.t, 0.0) / self.tau)
        self.v += a * (x - self.v)
        self.t = t
        return self.v


@dataclass
class RiskInputs:
    face: bool = True
    perclos: float = 0.0
    microsleep: bool = False
    closed_duration: float = 0.0
    mean_blink_duration: float = 0.0
    yawns_5min: int = 0
    yawning: bool = False
    off_road_duration: float = 0.0
    long_distraction: bool = False
    vats_distraction: bool = False
    alertness_probs: np.ndarray | None = None  # from temporal model
    frame_drowsy: float | None = None  # smoothed P(drowsy) from a frame-level model (NTHU-DDD style)
    cabin_probs: np.ndarray | None = None  # from CabinNet
    emotion_probs: np.ndarray | None = None  # from FaceStateNet / HSEmotion
    behaviours: dict | None = None  # from YOLO: {"phone": conf, "smoking": conf, ...}
    audio_groups: dict | None = None  # from YAMNet: {"snoring": p, "distress": p, ...}


@dataclass
class RiskOutput:
    index: float
    level: str
    drowsiness: float
    distraction: float
    affect: float
    reasons: list[str] = field(default_factory=list)


class RiskFusion:
    def __init__(self, w_drowsy: float = 1.0, w_distract: float = 1.0, affect_gain: float = 0.3, tau: float = 1.0, affect_tau: float = 5.0):
        self.w_drowsy, self.w_distract, self.affect_gain = w_drowsy, w_distract, affect_gain
        self._ema_drowsy = _EMA(tau * 3)  # drowsiness evolves slowly
        self._ema_distract = _EMA(tau * 0.5)  # distraction must react fast
        self._ema_affect = _EMA(affect_tau)
        self._ema_cabin = _EMA(tau)
        self._ema_phone = _EMA(tau * 0.7)
        self._last_microsleep: float | None = None
        self.post_microsleep_floor, self.post_microsleep_tau = 0.65, 45.0
        self.learned_cap = 0.45  # max drowsiness from learned models without measured signs
        self.cabin_cap = 0.5  # max distraction from the State Farm behaviour CNN without a detected object
        self.level_idx = 0

    # ----------------------------------------------------------------- parts
    def _drowsiness(self, r: RiskInputs, reasons: list[str]) -> float:
        s = max(
            _ramp(r.perclos, 0.08, 0.30),
            _ramp(r.mean_blink_duration, 0.25, 0.50) * 0.6,
            _ramp(r.yawns_5min, 1, 4) * 0.7,
        )
        if r.perclos >= 0.15:
            reasons.append(f"PERCLOS {r.perclos:.0%}")
        if r.yawns_5min >= 2:
            reasons.append(f"{r.yawns_5min} yawns in 5 min")
        if r.audio_groups and r.audio_groups.get("snoring", 0.0) >= 0.5:
            s = max(s, 0.9)
            reasons.append("snoring sound")
        # Learned drowsiness models (frame CNN, temporal model) are *supporting* evidence.
        # Alone they can raise at most a caution (they are trained on other people and cameras,
        # and a single model's false positive must not alarm an alert driver); together with
        # measured evidence (PERCLOS, long blinks, yawns) they can push the score higher.
        learned = []
        if r.frame_drowsy is not None:
            learned.append((r.frame_drowsy, 0.4))
            if r.frame_drowsy > 0.7:
                reasons.append(f"drowsy appearance ({r.frame_drowsy:.0%})")
        if r.alertness_probs is not None:
            p = r.alertness_probs
            names = ALERTNESS_CLASSES if len(p) == 3 else ("alert", "drowsy")
            model = float(p[1] * 0.5 + p[2]) if len(p) == 3 else float(p[1])
            learned.append((model, 0.6))
            if model > 0.6:
                reasons.append(f"temporal model: {names[int(np.argmax(p))]} ({p.max():.0%})")
        self._measured_drowsy = s >= 0.25
        if learned:
            m = sum(v * w for v, w in learned) / sum(w for _, w in learned)
            measured = s
            s = max(measured, self.learned_cap * m)  # alone: at most a caution
            if measured >= 0.25:  # measured signs present: the models may confirm them, adding at most
                s = max(s, measured + (1.0 - measured) * 0.3 * m)  # 30% of the remaining headroom
        return s

    def _distraction(self, t: float, r: RiskInputs, reasons: list[str]) -> float:
        s = _ramp(r.off_road_duration, 1.0, 3.0) * 0.8
        if r.long_distraction:
            s = 1.0
        elif r.vats_distraction:
            s = max(s, 0.8)
        self._measured_distract = s >= 0.25  # head pose / gaze (before the learned behaviour CNN)
        if r.cabin_probs is not None:
            p = r.cabin_probs
            unsafe = self._ema_cabin(t, 1.0 - float(p[CABIN_CLASSES.index("safe_driving")]))
            if unsafe > 0.5:
                k = int(np.argmax(p[1:])) + 1
                reasons.append(f"behaviour: {CABIN_CLASSES[k].replace('_', ' ')}")
            # Supporting evidence: the CNN was trained on side-mounted cameras (State Farm), so alone
            # it can raise at most a caution; a phone confirmed by the object detector goes higher.
            s = max(s, min(unsafe * (1.0 if np.argmax(p) in (1, 2, 3, 4) else 0.7), self.cabin_cap))
        if r.behaviours:
            b = r.behaviours
            phone = self._ema_phone(t, b.get("phone", 0.0))
            if max(phone, b.get("smoking", 0.0), b.get("drinking", 0.0)) >= 0.5:
                self._measured_distract = True
            if phone > 0.5:
                reasons.append(f"phone in use ({phone:.0%})")
            s = max(s, phone, 0.6 * b.get("smoking", 0.0), 0.4 * b.get("drinking", 0.0))
        return s

    def _affect(self, t: float, r: RiskInputs) -> float:
        if r.emotion_probs is None and not r.audio_groups:
            return 0.0
        neg_arousal = 0.0
        if r.emotion_probs is not None:
            p = dict(zip(EMOTION_CLASSES, r.emotion_probs))
            neg_arousal = p["anger"] + 0.7 * p["fear"] + 0.4 * p["disgust"]
        if r.audio_groups:  # shouting / screaming / crying in the cabin backs up the face
            neg_arousal = max(neg_arousal, 0.8 * r.audio_groups.get("distress", 0.0))
        return self._ema_affect(t, float(np.clip(neg_arousal, 0, 1)))

    # ----------------------------------------------------------------- main
    def update(self, t: float, r: RiskInputs) -> RiskOutput:
        reasons: list[str] = []
        d = self._ema_drowsy(t, self._drowsiness(r, reasons))
        # A microsleep is strong evidence of sleepiness: keep drowsiness elevated
        # afterwards and let it decay, instead of "forgetting" when the eyes open.
        if r.microsleep:
            self._last_microsleep = t
        if self._last_microsleep is not None and t >= self._last_microsleep:
            floor = self.post_microsleep_floor * math.exp(-(t - self._last_microsleep) / self.post_microsleep_tau)
            if floor > d:
                d = floor
                if floor > 0.3 and not r.microsleep:
                    reasons.append(f"recent microsleep ({t - self._last_microsleep:.0f}s ago)")
        x = self._ema_distract(t, self._distraction(t, r, reasons))
        a = self._affect(t, r)

        # Probabilistic OR of the two primary hazards, scaled by affect.
        base = 1.0 - (1.0 - self.w_drowsy * d) * (1.0 - self.w_distract * x)
        index = 100.0 * float(np.clip(base * (1.0 + self.affect_gain * a), 0.0, 1.0))
        snoring = bool(r.audio_groups and r.audio_groups.get("snoring", 0.0) >= 0.5)
        if not (self._measured_drowsy or self._measured_distract or snoring or r.microsleep
                or self._last_microsleep is not None and t - self._last_microsleep < 60):
            # only learned models (trained on other people / cameras) are worried: show a caution,
            # never a warning, until something measurable confirms it
            index = min(index, ENTER[2] - 1.0)
        if a > 0.6:
            reasons.append("sustained negative high-arousal affect")
            index = max(index, ENTER[1])

        # Hard overrides: these are always critical regardless of smoothing.
        if r.microsleep:
            reasons.insert(0, f"MICROSLEEP: eyes closed {r.closed_duration:.1f}s")
            index = 100.0
        if r.long_distraction:
            reasons.insert(0, f"eyes off road {r.off_road_duration:.1f}s")
            index = max(index, ENTER[3])

        self.level_idx = self._hysteresis(index)
        return RiskOutput(index=index, level=LEVELS[self.level_idx], drowsiness=d, distraction=x, affect=a, reasons=reasons)

    def _hysteresis(self, index: float) -> int:
        lvl = self.level_idx
        while lvl + 1 < len(LEVELS) and index >= ENTER[lvl + 1]:
            lvl += 1
        while lvl > 0 and index < ENTER[lvl] - EXIT_MARGIN:
            lvl -= 1
        return lvl
