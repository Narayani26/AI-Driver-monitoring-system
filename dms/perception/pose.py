"""Head tracking from any camera angle (side, diagonal, behind-the-shoulder).

MediaPipe's face landmarker is built for faces seen from the front: from a camera
mounted at the side (a 90-degree profile) it finds no face at all. A body-pose
model (YOLO11n-pose, COCO keypoints) still finds the nose, the visible eye, the
ear and the shoulders there, so the system keeps working:

* head nodding: how far the nose has dropped relative to the ear, in head lengths
  (independent of distance to the camera, compared with the driver's own baseline)
* eye state: a crop around the visible eye goes to the same eye-state CNN, which was
  trained on close-up single-eye images and does not need a frontal face
* presence: the driver is in the seat even when no frontal face is visible
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "yolo11n-pose.pt"
NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = range(7)
KP_CONF = 0.5


@dataclass
class HeadObservation:
    kp: np.ndarray  # (17, 3) x, y, confidence
    box: tuple[int, int, int, int]  # person box
    scale: float  # head length in pixels (nose-ear or similar)
    droop: float  # (nose_y - ear_y) / scale: grows as the head drops forward
    eye: tuple[float, float] | None  # the best visible eye

    @property
    def centre(self) -> tuple[float, float]:
        return float(self.kp[NOSE, 0]), float(self.kp[NOSE, 1])

    def eye_box(self, shape) -> tuple[int, int, int, int] | None:
        if self.eye is None:
            return None
        H, W = shape[:2]
        r = 0.3 * self.scale
        x, y = self.eye
        return int(max(x - r, 0)), int(max(y - r, 0)), int(min(x + r, W)), int(min(y + r, H))

    def face_box(self, shape) -> tuple[int, int, int, int]:
        H, W = shape[:2]
        x, y = self.centre
        r = 0.9 * self.scale
        return int(max(x - r, 0)), int(max(y - r, 0)), int(min(x + r, W)), int(min(y + r, H))


def head_from_keypoints(kp: np.ndarray, box) -> HeadObservation | None:
    ok = kp[:, 2] >= KP_CONF
    if not ok[NOSE]:
        return None
    nose = kp[NOSE, :2]
    ears = [kp[i, :2] for i in (L_EAR, R_EAR) if ok[i]]
    eyes = [(kp[i, 2], kp[i, :2]) for i in (L_EYE, R_EYE) if ok[i]]
    if ears:
        ref = np.mean(ears, axis=0)
        scale = float(np.hypot(*(nose - ref)))
    elif len(eyes) == 2:  # frontal-ish without ears
        ref = (eyes[0][1] + eyes[1][1]) / 2
        scale = 2.5 * float(np.hypot(*(eyes[0][1] - eyes[1][1])))
    else:
        return None
    if scale < 8:
        return None
    eye = tuple(float(v) for v in max(eyes, key=lambda e: e[0])[1]) if eyes else None
    return HeadObservation(kp=kp, box=tuple(int(v) for v in box), scale=scale, droop=float((nose[1] - ref[1]) / scale), eye=eye)


class HeadPoseDetector:
    def __init__(self, weights: str | Path = WEIGHTS, conf: float = 0.3, imgsz: int = 480):
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf, self.imgsz = conf, imgsz
        self.name = f"YOLO {Path(weights).stem} (head keypoints, any camera angle)"
        self._prev: tuple[float, float] | None = None

    def reset(self):
        self._prev = None

    def __call__(self, frame: np.ndarray) -> HeadObservation | None:
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, verbose=False)[0]
        if r.keypoints is None or not len(r.keypoints):
            return None
        heads = []
        for kp, b in zip(r.keypoints.data.cpu().numpy(), r.boxes.xyxy.cpu().numpy()):
            h = head_from_keypoints(kp, b)
            if h is not None:
                heads.append(h)
        if not heads:
            return None
        h = select_head(heads, self._prev, frame.shape[1])
        self._prev = h.centre
        return h


def select_head(heads: list[HeadObservation], prev, width: int) -> HeadObservation:
    """Driver = the biggest head (closest to the camera), then followed by position (passengers ignored)."""
    biggest = max(heads, key=lambda h: h.scale)
    if prev is not None:
        near = min(heads, key=lambda h: np.hypot(h.centre[0] - prev[0], h.centre[1] - prev[1]))
        if np.hypot(near.centre[0] - prev[0], near.centre[1] - prev[1]) < 0.2 * width and near.scale >= 0.6 * biggest.scale:
            return near
    return biggest
