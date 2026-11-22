"""Shared labels format for your own recordings.

One CSV for all videos, one row per labelled segment (times in seconds):

    video,start_s,end_s,label
    alice_01.mov,0,40,normal
    alice_01.mov,41.0,43.5,eyes_closed
    alice_01.mov,50,54,look_away
    alice_01.mov,60,66,phone
    alice_01.mov,70,73,yawn
    alice_01.mov,80,88,anger
    alice_01.mov,90,150,drowsy

Labels: normal, eyes_closed, yawn, look_away, phone, drinking, smoking,
drowsy (optional: a longer stretch acted sleepy), or an emotion
(neutral, happy, sad, surprise, fear, disgust, anger).
`video` is the file name (no folders). The person's ID is the part of the file
name before the first "_" (alice_01.mov -> alice), so name files <person>_<n>.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ..models.networks import EMOTION_CLASSES

STATE_LABELS = {"normal", "eyes_closed", "yawn", "look_away", "phone", "drinking", "smoking", "drowsy"}
EMOTION_LABELS = set(EMOTION_CLASSES)
ALL_LABELS = STATE_LABELS | EMOTION_LABELS


@dataclass
class Segment:
    video: str
    start: float
    end: float
    label: str


def read_labels(path: str | Path) -> list[Segment]:
    df = pd.read_csv(path)
    missing = {"video", "start_s", "end_s", "label"} - set(df.columns)
    if missing:
        raise SystemExit(f"{path}: missing columns {missing}")
    segs = []
    for r in df.itertuples():
        lab = str(r.label).strip().lower()
        if lab not in ALL_LABELS:
            raise SystemExit(f"{path}: unknown label {lab!r}; use one of {sorted(ALL_LABELS)}")
        if r.end_s <= r.start_s:
            raise SystemExit(f"{path}: end_s <= start_s in row {r.Index + 2}")
        segs.append(Segment(Path(str(r.video)).name, float(r.start_s), float(r.end_s), lab))
    return segs


def subject_of(video: str, sep: str = "_") -> str:
    return Path(video).stem.split(sep)[0]
