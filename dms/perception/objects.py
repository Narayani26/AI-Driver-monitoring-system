"""YOLO object detection for in-cabin unsafe behaviour (Ultralytics, AGPL-3.0).

Out of the box we use COCO-pretrained YOLO11n, whose classes already include
`cell phone`, `bottle` and `cup`. For cigarettes / seatbelt / hands-on-wheel,
fine-tune on a driver dataset (see docs/DATASETS.md) - class names are matched
by keyword below, so a fine-tuned model drops in without code changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "yolo11n.pt"

# behaviour -> keywords that may appear in a model's class names
BEHAVIOUR_KEYWORDS = {
    "phone": ("cell phone", "phone", "mobile", "texting", "calling"),
    "drinking": ("bottle", "cup", "drink"),
    "smoking": ("cigarette", "smok"),
    "no_seatbelt": ("no_seatbelt", "no-seatbelt", "noseatbelt", "without seatbelt"),
}


@dataclass
class Detection:
    label: str
    behaviour: str
    conf: float
    box: tuple[int, int, int, int]  # x0, y0, x1, y1


def _behaviour(name: str) -> str | None:
    n = name.lower()
    # check no_seatbelt before generic matches
    for b in ("no_seatbelt", "smoking", "phone", "drinking"):
        if any(k in n for k in BEHAVIOUR_KEYWORDS[b]):
            return b
    return None


class ObjectDetector:
    def __init__(self, weights: str | Path = WEIGHTS, conf: float = 0.35, imgsz: int = 480, device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.device = conf, imgsz, device
        names = self.model.names
        self.keep = {i: _behaviour(n) for i, n in names.items() if _behaviour(n)}
        self.names = names
        self.name = f"YOLO {Path(weights).stem}"

    def __call__(self, frame: np.ndarray) -> list[Detection]:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, classes=list(self.keep), verbose=False, device=self.device)[0]
        out = []
        for b in r.boxes:
            k = int(b.cls)
            out.append(Detection(self.names[k], self.keep[k], float(b.conf), tuple(int(v) for v in b.xyxy[0].tolist())))
        return out


def contains(box, pt) -> bool:
    return box[0] <= pt[0] <= box[2] and box[1] <= pt[1] <= box[3]


def behaviour_scores(dets: list[Detection], face_box: tuple[int, int, int, int] | None, frame_shape) -> dict[str, float]:
    """Per-behaviour confidence for the driver.

    * A phone whose box contains the face centre is a *screen showing a face*
      (e.g. playing footage to the webcam in a demo), not phone use -> ignored.
    * With a face, objects far from the head/upper-body region are ignored
      (a phone lying on the passenger seat is not phone use).
    """
    scores: dict[str, float] = {}
    H, W = frame_shape[:2]
    if face_box is not None:
        fx0, fy0, fx1, fy1 = face_box
        fw, fh = fx1 - fx0, fy1 - fy0
        centre = ((fx0 + fx1) / 2, (fy0 + fy1) / 2)
        roi = (fx0 - 1.5 * fw, fy0 - 0.5 * fh, fx1 + 1.5 * fw, fy1 + 2.5 * fh)
    for d in dets:
        if face_box is not None:
            if d.behaviour == "phone" and contains(d.box, centre):
                continue
            cx, cy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
            if not contains(roi, (cx, cy)):
                continue
        scores[d.behaviour] = max(scores.get(d.behaviour, 0.0), d.conf)
    return scores
