"""Facial emotion recognition providers.

* `TorchEmotion`  - our own FaceStateNet checkpoint (trained with dms.training.train_image).
* `HSEmotionONNX` - pretrained HSEmotion EfficientNet-B0 (AffectNet, 8 classes,
  Apache-2.0, A. Savchenko). Used as the out-of-the-box baseline so the demo
  shows emotions before you have trained anything, and as the comparison
  baseline in the paper.

Both return probabilities in our EMOTION_CLASSES order.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..models.networks import EMOTION_CLASSES

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"
HSE_PATH = WEIGHTS / "emotion_enet_b0_8.onnx"
HSE_URL = "https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx"
HSE_CLASSES = ("Anger", "Contempt", "Disgust", "Fear", "Happiness", "Neutral", "Sadness", "Surprise")
# ours <- theirs ; contempt is folded into disgust (closest basic emotion)
_MAP = {"neutral": ["Neutral"], "happy": ["Happiness"], "sad": ["Sadness"], "surprise": ["Surprise"], "fear": ["Fear"], "disgust": ["Disgust", "Contempt"], "anger": ["Anger"]}
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


class HSEmotionONNX:
    name = "HSEmotion EfficientNet-B0 (AffectNet, pretrained)"

    def __init__(self, path: str | Path = HSE_PATH, threads: int = 2):
        import onnxruntime as ort

        if not Path(path).exists():
            raise FileNotFoundError(f"{path} missing; run scripts/download_models.sh")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self._idx = [[HSE_CLASSES.index(c) for c in _MAP[e]] for e in EMOTION_CLASSES]

    def __call__(self, face_bgr: np.ndarray) -> np.ndarray:
        x = cv2.resize(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB), (224, 224)).astype(np.float32) / 255.0
        x = ((x - _MEAN) / _STD).transpose(2, 0, 1)[None]
        z = self.sess.run(None, {self.inp: x})[0][0]
        p = np.exp(z - z.max())
        p /= p.sum()
        return np.array([p[i].sum() for i in self._idx], np.float32)


class TorchEmotion:
    def __init__(self, ckpt: str | Path | None = None, model=None, ck: dict | None = None):
        import torch

        from ..models import checkpoint

        self.torch = torch
        if model is None:
            model, ck = checkpoint.load(ckpt)
        self.model = model
        self.size = ck.get("img_size", 224)
        self.temperature = float((ck.get("temperature") or {}).get("emotion", 1.0))
        self.name = f"FaceStateNet ({ck['arch'].get('backbone')}, trained)"

    def __call__(self, face_bgr: np.ndarray) -> np.ndarray:
        from ..data.transforms import to_batch

        with self.torch.inference_mode():
            return self.torch.softmax(self.model(to_batch([face_bgr], self.size))["emotion"] / self.temperature, -1)[0].numpy()


def load_emotion(ckpt: str | None = None):
    """Trained checkpoint if given, else the pretrained ONNX model if present, else None."""
    if ckpt:
        return TorchEmotion(ckpt)
    if HSE_PATH.exists():
        return HSEmotionONNX()
    return None
