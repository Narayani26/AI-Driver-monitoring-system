"""MediaPipe Face Landmarker (Tasks API) wrapper + crop helpers."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_MODEL = Path(__file__).resolve().parents[2] / "weights" / "face_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


@dataclass
class FaceObservation:
    landmarks: np.ndarray  # (478, 3) pixel x/y, z scaled like x
    transform: np.ndarray | None  # 4x4 facial transformation matrix
    blendshapes: dict[str, float]


class FaceLandmarker:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL, video_mode: bool = True, min_conf: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} missing. Download it with:\n  curl -L -o {model_path} {MODEL_URL}")
        self._mp = mp
        self.video_mode = video_mode
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_conf,
            min_face_presence_confidence=min_conf,
            min_tracking_confidence=min_conf,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self._last_ms = -1
        self._closed = False

    def detect(self, bgr: np.ndarray, t: float | None = None) -> FaceObservation | None:
        h, w = bgr.shape[:2]
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if self.video_mode:
            ms = int((t or 0.0) * 1000)
            ms = max(ms, self._last_ms + 1)  # MediaPipe requires strictly increasing timestamps
            self._last_ms = ms
            res = self._lm.detect_for_video(img, ms)
        else:
            res = self._lm.detect(img)
        if not res.face_landmarks:
            return None
        lm = np.array([(p.x * w, p.y * h, p.z * w) for p in res.face_landmarks[0]], dtype=np.float32)
        tf = np.array(res.facial_transformation_matrixes[0]) if res.facial_transformation_matrixes else None
        bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0]} if res.face_blendshapes else {}
        return FaceObservation(lm, tf, bs)

    def close(self):
        if not self._closed:
            self._closed = True
            self._lm.close()


def bbox_from_points(pts: np.ndarray, shape, margin: float = 0.25, square: bool = True) -> tuple[int, int, int, int]:
    x0, y0 = pts[:, 0].min(), pts[:, 1].min()
    x1, y1 = pts[:, 0].max(), pts[:, 1].max()
    cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)
    if square:
        bw = bh = max(bw, bh)
    bw, bh = bw * (1 + 2 * margin), bh * (1 + 2 * margin)
    H, W = shape[:2]
    return (int(max(cx - bw / 2, 0)), int(max(cy - bh / 2, 0)), int(min(cx + bw / 2, W)), int(min(cy + bh / 2, H)))


def crop(img: np.ndarray, box) -> np.ndarray | None:
    x0, y0, x1, y1 = box
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img[y0:y1, x0:x1]
