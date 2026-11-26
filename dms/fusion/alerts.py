"""Turns risk state into driver-facing alerts: what to say, what to suggest,
which sound to play, and how often (cooldowns stop the system from nagging,
which is the #1 reason drivers switch DMS off).

The text lives in MESSAGES so it can be translated (e.g. Hindi / Marathi) or
tuned without touching the logic. Same module is meant to be ported 1:1 to the
mobile app.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

SEVERITY = {"info": 0, "caution": 1, "warning": 2, "critical": 3}

MESSAGES = {
    "microsleep": ("critical", "alarm", "Wake up! Your eyes were closed.", "Pull over safely as soon as possible. Do not continue driving while this tired."),
    "drowsy": ("warning", "chime", "You seem drowsy.", "Take a 15 to 20 minute break at the next safe stop. A coffee followed by a short nap works best."),
    "yawning": ("caution", "soft", "Frequent yawning detected.", "Get some fresh air and plan a break soon."),
    "eyes_off_road": ("warning", "chime", "Please look ahead at the road.", "Keep your eyes on the road. Glance at mirrors and screens for under two seconds."),
    "glances": ("caution", "soft", "You keep looking away from the road.", "Set up navigation and music before you start driving."),
    "phone": ("warning", "chime", "Please put the phone down.", "Use hands-free, or pull over safely to use your phone."),
    "smoking": ("caution", "soft", "Smoking detected.", "Keep both hands available for steering."),
    "drinking": ("info", "none", "Drinking while driving.", "Finish your drink when stopped."),
    "no_seatbelt": ("warning", "chime", "Please fasten your seatbelt.", "Seatbelts reduce the risk of death in a crash by about half."),
    "stress": ("caution", "soft", "You seem stressed.", "Take a slow, deep breath. Reduce speed and keep a safe distance."),
    "no_driver": ("caution", "soft", "Driver not visible.", "Check that the camera can see your face."),
    # audio (opt-in)
    "snoring": ("critical", "alarm", "Wake up! Snoring detected.", "You may be falling asleep. Pull over safely now."),
    "shouting": ("caution", "soft", "Raised voices detected.", "Stay calm and keep your focus on the road. The conversation can wait."),
    "siren": ("info", "chime", "Emergency vehicle nearby.", "Slow down, check your mirrors and be ready to give way."),
    "phone_ring": ("info", "soft", "Your phone is ringing.", "Let it go to voicemail, or answer hands-free."),
    "baby": ("info", "soft", "A child is crying.", "Stop somewhere safe before attending to passengers."),
    # possible accident (opt-in; needs audio and/or jolt evidence)
    "impact": ("warning", "alarm", "Sudden impact detected. Are you OK?", "If you or the car are hurt, stop safely, switch on the hazard lights and call for help."),
    "possible_accident": ("critical", "alarm", "Possible accident! Are you OK?", "Press I'm OK to cancel. Otherwise your emergency contact will be alerted."),
    "sos": ("critical", "alarm", "Emergency SOS raised.", "SOS logged. In the vehicle app this notifies your emergency contact with your location."),
}
COOLDOWN_S = {"critical": 6.0, "warning": 15.0, "caution": 45.0, "info": 120.0}


@dataclass
class Alert:
    kind: str
    severity: str
    sound: str
    message: str
    suggestion: str
    t: float
    detail: str = ""

    def as_dict(self):
        return asdict(self)


class AlertManager:
    def __init__(self, messages: dict | None = None, cooldowns: dict | None = None, drowsy_threshold: float = 0.6):
        self.messages = messages or MESSAGES
        self.cooldowns = cooldowns or COOLDOWN_S
        self.drowsy_threshold = drowsy_threshold
        self._last: dict[str, float] = {}
        self._active: dict[str, float] = {}  # kind -> since
        self._hits: dict[str, list[float]] = {}  # audio detections per kind

    def _fire(self, kind: str, t: float, detail: str = "") -> Alert | None:
        sev, sound, msg, sug = self.messages[kind]
        last = self._last.get(kind)
        if last is not None and t - last < self.cooldowns[sev] and t >= last:
            return None
        self._last[kind] = t
        return Alert(kind, sev, sound, msg, sug, t, detail)

    def _sustained(self, kind: str, cond: bool, t: float, hold: float) -> bool:
        """True once `cond` has held for `hold` seconds (debounce)."""
        if not cond:
            self._active.pop(kind, None)
            return False
        since = self._active.setdefault(kind, t)
        return t - since >= hold

    def _counted(self, key: str, hit: bool, t: float, need: int, window: float) -> bool:
        """True when `hit` occurred at >= `need` distinct times within `window` seconds.
        Callers pass the same audio state on every video frame, so a hit is only
        counted again once the audio state changes (new 0.5 s window)."""
        hist = self._hits.setdefault(key, [])
        new = hit and (not hist or t - hist[-1] >= 0.4)
        if new:
            hist.append(t)
        hist[:] = [x for x in hist if t - x <= window]
        return new and len(hist) >= need  # only fire on a fresh detection

    def reset(self):
        self._last.clear()
        self._active.clear()
        self._hits.clear()

    def update(self, t: float, *, risk, microsleep: bool, closed_duration: float, long_distraction: bool, vats: bool,
               off_road_duration: float, yawns_5min: int, face: bool, alertness: str | None, emotion_probs_affect: float,
               behaviours: dict[str, float], audio_groups: dict[str, float] | None = None,
               mishap_events: list[tuple[float, str, str]] = ()) -> list[Alert]:
        out: list[Alert] = []

        def add(a):
            if a:
                out.append(a)

        if microsleep:
            add(self._fire("microsleep", t, f"eyes closed {closed_duration:.1f}s"))
        drowsy = risk is not None and (risk.drowsiness >= self.drowsy_threshold or alertness == "drowsy")
        if self._sustained("drowsy", drowsy and not microsleep, t, 2.0):
            add(self._fire("drowsy", t, f"drowsiness {risk.drowsiness:.0%}"))
        if yawns_5min >= 3:
            add(self._fire("yawning", t, f"{yawns_5min} yawns in 5 min"))
        if long_distraction:  # also when the face is lost: usually the head turned far away
            add(self._fire("eyes_off_road", t, f"{off_road_duration:.1f}s"))
        elif vats:
            add(self._fire("glances", t))
        for b in ("phone", "smoking", "drinking", "no_seatbelt"):
            if self._sustained(b, behaviours.get(b, 0.0) >= 0.5, t, 1.0):
                add(self._fire(b, t, f"{behaviours[b]:.0%}"))
        if self._sustained("stress", emotion_probs_affect >= 0.6, t, 5.0):
            add(self._fire("stress", t))
        if self._sustained("no_driver", not face, t, 5.0):
            add(self._fire("no_driver", t))
        a = audio_groups or {}
        # Sounds are bursty (a snore, a siren's wail): count detections in a window instead of
        # requiring a continuous hold. (kind, group, detections needed, window seconds)
        for kind, group, need, window in (("snoring", "snoring", 2, 10.0), ("shouting", "distress", 2, 6.0), ("siren", "siren", 2, 10.0),
                                          ("phone_ring", "phone", 2, 8.0), ("baby", "baby", 3, 10.0)):
            if self._counted("audio_" + kind, a.get(group, 0.0) >= 0.5, t, need, window):
                add(self._fire(kind, t, f"{a.get(group, 0.0):.0%}"))
        for _, kind, evidence in mishap_events:
            if kind in self.messages:
                add(self._fire(kind, t, evidence))
        out.sort(key=lambda a: -SEVERITY[a.severity])
        return out
