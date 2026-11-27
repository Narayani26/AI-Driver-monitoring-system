"""Data-quality gate for training samples uploaded from cars.

A window only becomes training data if it really shows *the driver, driving*,
seen well enough to measure, with a trustworthy label. Every rejection is kept
with its reason, so the rules themselves can be audited and tuned.
"""
from __future__ import annotations

import math

LABELS = {0: "alert", 1: "low_vigilant", 2: "drowsy"}
LABEL_SOURCES = ("kss", "false_alarm", "confirmed")
N_FEATURES = 9  # FrameFeatures.VECTOR_FIELDS

RULES = {
    "min_seconds": 20.0,  # at least 20 s of the 30 s window
    "min_face_frac": 0.9,  # driver's face visible in >= 90% of frames
    "min_frontal_frac": 0.7,  # head close enough to the calibrated pose to measure the eyes
    "max_nan_frac": 0.1,
    "ear_open_range": (0.12, 0.5),  # plausible calibrated open-eye EAR: outside = bad calibration
    "min_speed_kmh": 5.0,  # if the device knows its speed: parked = not driving
    "min_fps": 8.0,
}


def kss_to_label(kss: int) -> int:
    """Karolinska Sleepiness Scale 1-9 -> UTA-RLDD scale. KSS >= 7 is the usual 'sleepy' threshold."""
    return 0 if kss <= 4 else 1 if kss <= 6 else 2


def check(sample: dict) -> tuple[bool, str | None, dict]:
    """(accepted, reason if rejected, measured quality numbers)"""
    q: dict = {}
    raw, present = sample.get("raw") or [], sample.get("present") or []
    meta = sample.get("meta") or {}
    fps = float(sample.get("fps") or 0)
    label, source = sample.get("label"), sample.get("label_source")

    def bad(reason):
        return False, reason, q

    if label not in LABELS:
        return bad("label must be 0 (alert), 1 (low vigilant) or 2 (drowsy)")
    if source not in LABEL_SOURCES:
        return bad(f"label_source must be one of {LABEL_SOURCES}")
    if not raw or any(len(r) != N_FEATURES for r in raw) or len(present) != len(raw):
        return bad("malformed feature window")
    q["fps"] = fps
    if fps < RULES["min_fps"]:
        return bad(f"frame rate {fps:.1f} too low (needs >= {RULES['min_fps']})")
    q["seconds"] = round(len(raw) / fps, 1)
    if q["seconds"] < RULES["min_seconds"]:
        return bad(f"window too short ({q['seconds']} s)")
    q["face_frac"] = round(sum(bool(p) for p in present) / len(present), 3)
    if q["face_frac"] < RULES["min_face_frac"]:
        return bad(f"driver's face visible only {q['face_frac']:.0%} of the time")
    vals = [v for r in raw for v in r]
    q["nan_frac"] = round(sum(1 for v in vals if v is None or (isinstance(v, float) and math.isnan(v))) / len(vals), 3)
    if q["nan_frac"] > RULES["max_nan_frac"]:
        return bad("too many missing measurements")
    prof = sample.get("profile") or []
    if len(prof) < 5 or not prof[4]:
        return bad("driver was not calibrated")
    lo, hi = RULES["ear_open_range"]
    if not lo <= float(prof[0]) <= hi:
        return bad(f"implausible calibration (open-eye EAR {float(prof[0]):.2f})")
    q["frontal_frac"] = float(meta.get("frontal_frac", 1.0))
    if q["frontal_frac"] < RULES["min_frontal_frac"]:
        return bad(f"head turned away {1 - q['frontal_frac']:.0%} of the time")
    # someone else in view is fine (the driver is tracked), but the tracked face must stay the same person
    q["max_faces"] = int(meta.get("max_faces", 1))
    if int(meta.get("driver_switches", 0)) > 0:
        return bad("tracked face changed during the window (passenger or driver swap)")
    if meta.get("speed_kmh") is not None:
        q["speed_kmh"] = float(meta["speed_kmh"])
        if q["speed_kmh"] < RULES["min_speed_kmh"]:
            return bad("vehicle not moving: not a driving sample")
    return True, None, q
