"""Notices when the camera itself was moved (re-mounted, knocked, turned to the side).

Drivers adjust phone holders and dash cams. After a move, the calibrated "normal"
position no longer matches, so every glance would look like distraction (or the face
would be lost). Re-calibrating must not be triggered by a driver who simply turns
their head, though - that is exactly what we must alert on. The two differ in the
*background*:

* driver turns / looks at a phone : the cabin behind them (seats, pillars, roof liner)
  stays where it was in the image
* camera moved                    : the whole cabin shifts, AND the driver's position or
  size in the image changes (or the view switches between frontal and side)

Both must hold for `hold_s` seconds before we re-calibrate. The driver's own body is
masked out of the background comparison; normalised cross-correlation ignores
brightness changes (tunnels, clouds), and the blurred low-resolution image is dominated
by the cabin interior rather than the moving scenery in the windows.
"""
from __future__ import annotations

import cv2
import numpy as np

SIZE = (64, 36)


def _small(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return cv2.GaussianBlur(cv2.resize(g, SIZE, interpolation=cv2.INTER_AREA), (5, 5), 0).astype(np.float32)


def _person_mask(box, shape, body=None) -> np.ndarray:
    """1 = background, 0 = the driver. `body`: the whole-person box (side view, from the pose
    model); otherwise the face box widened to the shoulders, down to the bottom of the image."""
    m = np.ones((SIZE[1], SIZE[0]), np.float32)
    if box is None and body is None:
        return m
    H, W = shape[:2]
    if body is not None:
        x0, y0, x1, y1 = body
        bx0, bx1, by0 = max(0, x0 - 0.1 * (x1 - x0)), min(W, x1 + 0.1 * (x1 - x0)), max(0, y0 - 0.1 * (y1 - y0))
    else:
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        bx0, bx1 = max(0, x0 - 1.0 * w), min(W, x1 + 1.0 * w)  # head plus shoulders
        by0 = max(0, y0 - 0.3 * h)
    sx, sy = SIZE[0] / W, SIZE[1] / H
    m[int(by0 * sy):, int(bx0 * sx): int(np.ceil(bx1 * sx))] = 0.0
    return m


def ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    w = mask > 0
    if w.sum() < 0.15 * w.size:  # the driver fills the view (close webcam): compare whole frames
        w = np.ones_like(w)
    x, y = a[w] - a[w].mean(), b[w] - b[w].mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 1e-6 else 1.0


class CameraMoveDetector:
    def __init__(self, hold_s: float = 3.0, min_ncc: float = 0.6, max_shift: float = 0.25, size_ratio: float = 1.4, period_s: float = 0.25):
        self.hold_s, self.min_ncc, self.max_shift, self.size_ratio, self.period_s = hold_s, min_ncc, max_shift, size_ratio, period_s
        self.reset()

    def reset(self):
        self._ref = None  # (small image, mask, mode, centre, size)
        self._since: float | None = None
        self._next = -1e9
        self.last_ncc = 1.0

    @property
    def armed(self) -> bool:
        return self._ref is not None

    @staticmethod
    def _geom(box, shape):
        if box is None:
            return None, None
        H, W = shape[:2]
        x0, y0, x1, y1 = box
        return ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H), (x1 - x0) / W

    def set_reference(self, frame: np.ndarray, mode: str, box, body=None):
        c, s = self._geom(box, frame.shape)
        self._ref = (_small(frame), _person_mask(box, frame.shape, body), mode, c, s)
        self._since = None

    @property
    def pending(self) -> bool:
        """A camera move is suspected right now (used to hold back distraction alerts)."""
        return self._since is not None

    def update(self, t: float, frame: np.ndarray, mode: str, box, body=None) -> bool:
        """True once, when the camera is judged to have moved."""
        if self._ref is None or t < self._next:
            return False
        self._next = t + self.period_s
        ref_img, ref_mask, ref_mode, ref_c, ref_s = self._ref
        mask = ref_mask * _person_mask(box, frame.shape, body)
        self.last_ncc = ncc(ref_img, _small(frame), mask)
        scene_changed = self.last_ncc < self.min_ncc
        c, s = self._geom(box, frame.shape)
        if mode != ref_mode:
            driver_changed = True  # frontal <-> side view, or the driver is no longer found
        elif c is None or ref_c is None:
            driver_changed = False
        else:
            shifted = np.hypot(c[0] - ref_c[0], c[1] - ref_c[1]) > self.max_shift
            rescaled = not (1 / self.size_ratio <= s / max(ref_s, 1e-6) <= self.size_ratio)
            driver_changed = shifted or rescaled
        if scene_changed and driver_changed:
            self._since = t if self._since is None else self._since
            if t - self._since >= self.hold_s:
                self._ref, self._since = None, None  # wait for the new calibration to set a new reference
                return True
        else:
            self._since = None
        return False
