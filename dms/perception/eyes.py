"""Learned eye-state classifier (the `eye` head of a trained FaceStateNet).

It complements the geometric eye aspect ratio: EAR is precise when the eye
contour is tracked well, the CNN is more robust to glasses, partial occlusion
and head turns. EyeClosureTracker fuses the two.

The eye head was trained on MRL Eye (near-infrared, grayscale), so webcam crops
are converted to grayscale before inference to match that domain.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..features.geometry import LEFT_EYE, RIGHT_EYE
from ..models.networks import EYE_CLASSES
from .face import bbox_from_points, crop

CLOSED = EYE_CLASSES.index("closed")


class EyeStateCNN:
    def __init__(self, model, img_size: int = 224, name: str = "eye-state CNN", temperature: float = 1.0):
        import torch

        self.torch, self.model, self.size, self.name, self.temperature = torch, model, img_size, name, temperature

    def __call__(self, frame: np.ndarray, landmarks: np.ndarray) -> float | None:
        """Probability that the eyes are closed (mean of both eyes), or None."""
        from ..data.transforms import to_batch

        crops = []
        for idx in (LEFT_EYE, RIGHT_EYE):
            im = crop(frame, bbox_from_points(landmarks[list(idx)], frame.shape, 0.6))
            if im is not None:
                crops.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        if not crops:
            return None
        with self.torch.inference_mode():
            p = self.torch.softmax(self.model(to_batch(crops, self.size))["eye"] / self.temperature, -1)[:, CLOSED]
        return float(p.mean())
