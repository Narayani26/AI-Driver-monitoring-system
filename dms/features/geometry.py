"""Per-frame geometric features computed from MediaPipe Face Landmarker output.

All functions take landmarks as an (N, 3) float array in *pixel* units for x/y
(z is MediaPipe's relative depth, scaled like x). Using pixel units keeps the
ratios aspect-correct on non-square frames.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

# Six-point eye contours (p1..p6 as in Soukupova & Cech, 2016) on the 478-point mesh.
# p1/p4 are the horizontal corners; (p2, p6) and (p3, p5) are vertical pairs.
LEFT_EYE = (33, 160, 158, 133, 153, 144)
RIGHT_EYE = (362, 385, 387, 263, 373, 380)
# Mouth: corners (61, 291) and three vertical inner-lip pairs.
MOUTH_CORNERS = (61, 291)
MOUTH_VERTICAL = ((81, 178), (13, 14), (311, 402))
# Iris centres (only present in the 478-point refined mesh).
LEFT_IRIS, RIGHT_IRIS = 468, 473


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a[:2] - b[:2]))


def eye_aspect_ratio(lm: np.ndarray, idx: tuple[int, ...]) -> float:
    p1, p2, p3, p4, p5, p6 = (lm[i] for i in idx)
    horiz = _dist(p1, p4)
    if horiz < 1e-6:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horiz)


def mouth_aspect_ratio(lm: np.ndarray) -> float:
    horiz = _dist(lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]])
    if horiz < 1e-6:
        return 0.0
    vert = sum(_dist(lm[a], lm[b]) for a, b in MOUTH_VERTICAL)
    return vert / (len(MOUTH_VERTICAL) * horiz)


def gaze_ratio(lm: np.ndarray) -> tuple[float, float]:
    """Horizontal/vertical iris position inside the eye, averaged over both eyes.

    0.5 means centred; returns (nan, nan) if iris landmarks are missing.
    """
    if lm.shape[0] <= RIGHT_IRIS:
        return float("nan"), float("nan")
    hs, vs = [], []
    for eye, iris in ((LEFT_EYE, LEFT_IRIS), (RIGHT_EYE, RIGHT_IRIS)):
        c1, c2 = lm[eye[0]][:2], lm[eye[3]][:2]
        top = (lm[eye[1]][:2] + lm[eye[2]][:2]) / 2
        bot = (lm[eye[4]][:2] + lm[eye[5]][:2]) / 2
        axis = c2 - c1
        n = float(np.dot(axis, axis))
        if n < 1e-6:
            continue
        hs.append(float(np.dot(lm[iris][:2] - c1, axis) / n))
        span = bot[1] - top[1]
        if abs(span) > 1e-6:
            vs.append(float((lm[iris][1] - top[1]) / span))
    h = float(np.mean(hs)) if hs else float("nan")
    v = float(np.mean(vs)) if vs else float("nan")
    return h, v


def rotation_to_euler(R: np.ndarray) -> tuple[float, float, float]:
    """Rotation matrix -> (pitch, yaw, roll) in degrees (x, y, z intrinsic)."""
    sy = np.hypot(R[0, 0], R[1, 0])
    if sy > 1e-6:
        pitch = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[1, 0], R[0, 0])
    else:  # gimbal lock
        pitch = np.arctan2(-R[1, 2], R[1, 1])
        yaw = np.arctan2(-R[2, 0], sy)
        roll = 0.0
    return tuple(float(np.degrees(a)) for a in (pitch, yaw, roll))


def head_pose_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """MediaPipe facial transformation matrix (4x4) -> (pitch, yaw, roll) degrees."""
    return rotation_to_euler(np.asarray(m, dtype=np.float64)[:3, :3])


@dataclass
class FrameFeatures:
    """Everything the temporal layer needs from one frame."""

    face: bool
    ear: float = float("nan")
    ear_left: float = float("nan")
    ear_right: float = float("nan")
    mar: float = float("nan")
    pitch: float = float("nan")
    yaw: float = float("nan")
    roll: float = float("nan")
    gaze_h: float = float("nan")
    gaze_v: float = float("nan")
    blink_bs: float = float("nan")  # mean of eyeBlinkLeft/Right blendshapes
    jaw_open_bs: float = float("nan")

    def as_dict(self) -> dict:
        return asdict(self)

    # Order matters: this is the per-frame vector fed to the temporal model.
    VECTOR_FIELDS = ("ear", "mar", "pitch", "yaw", "roll", "gaze_h", "gaze_v", "blink_bs", "jaw_open_bs")

    def vector(self) -> np.ndarray:
        return np.array([getattr(self, k) for k in self.VECTOR_FIELDS], dtype=np.float32)


def compute_features(
    lm: np.ndarray,
    transform: np.ndarray | None = None,
    blendshapes: dict[str, float] | None = None,
) -> FrameFeatures:
    el = eye_aspect_ratio(lm, LEFT_EYE)
    er = eye_aspect_ratio(lm, RIGHT_EYE)
    pitch = yaw = roll = float("nan")
    if transform is not None:
        pitch, yaw, roll = head_pose_from_matrix(transform)
    gh, gv = gaze_ratio(lm)
    bs = blendshapes or {}
    blink = np.nanmean([bs.get("eyeBlinkLeft", np.nan), bs.get("eyeBlinkRight", np.nan)]) if bs else np.nan
    return FrameFeatures(
        face=True,
        ear=(el + er) / 2,
        ear_left=el,
        ear_right=er,
        mar=mouth_aspect_ratio(lm),
        pitch=pitch,
        yaw=yaw,
        roll=roll,
        gaze_h=gh,
        gaze_v=gv,
        blink_bs=float(blink),
        jaw_open_bs=float(bs.get("jawOpen", np.nan)),
    )
