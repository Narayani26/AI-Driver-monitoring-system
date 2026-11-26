"""In-cabin audio events with YAMNet (MediaPipe Audio Classifier, 521 AudioSet classes).

Raw audio is never stored: 16 kHz mono PCM chunks are classified in memory and
only event scores leave this module.

Groups (class indices from the model's own label list, resolved by name):
    snoring      -> deep drowsiness (strong evidence the driver is asleep)
    distress     -> shouting / yelling / screaming / crying  (stress, conflict)
    impact       -> smash/crash, breaking, shatter, bang, explosion  (possible accident)
    skid         -> skidding, tire squeal  (hard braking / loss of control)
    horn         -> vehicle / truck horn  (near-miss context)
    siren        -> emergency vehicles  (let them pass)
    phone        -> ringtone / telephone ringing  (distraction risk)
    baby         -> baby crying  (distraction risk)
"""
from __future__ import annotations

import zipfile
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "yamnet.tflite"
YAMNET_URL = "https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite"
SAMPLE_RATE = 16000

GROUPS: dict[str, tuple[str, ...]] = {
    "snoring": ("Snoring",),
    "distress": ("Shout", "Yell", "Screaming", "Crying, sobbing"),
    "impact": ("Smash, crash", "Breaking", "Shatter", "Bang", "Explosion"),
    "skid": ("Skidding", "Tire squeal"),
    "horn": ("Vehicle horn, car horn, honking", "Air horn, truck horn"),
    "siren": ("Siren", "Emergency vehicle", "Police car (siren)", "Ambulance (siren)", "Fire engine, fire truck (siren)", "Civil defense siren"),
    "phone": ("Ringtone", "Telephone bell ringing", "Telephone"),
    "baby": ("Baby cry, infant cry",),
}


def load_labels(model_path: str | Path = WEIGHTS) -> list[str]:
    with zipfile.ZipFile(model_path) as z:
        name = next(n for n in z.namelist() if n.endswith(".txt"))
        return z.read(name).decode().splitlines()


@dataclass
class AudioState:
    t: float = 0.0
    rms_db: float = -90.0
    groups: dict[str, float] = field(default_factory=dict)  # smoothed 0..1 per group
    top: list[tuple[str, float]] = field(default_factory=list)  # top classes of the last window

    def as_dict(self):
        return {"t": round(self.t, 2), "rms_db": round(self.rms_db, 1), "groups": {k: round(v, 3) for k, v in self.groups.items()},
                "top": [[n, round(s, 3)] for n, s in self.top]}


class GroupScorer:
    """Turns per-class scores into smoothed per-group scores. Model-free, so it is unit-testable."""

    def __init__(self, labels: list[str], groups: dict[str, tuple[str, ...]] = GROUPS, decay: float = 0.5):
        idx = {n: i for i, n in enumerate(labels)}
        self.groups = {g: [idx[n] for n in names if n in idx] for g, names in groups.items()}
        self.decay = decay  # per update; impacts are short, so keep memory brief
        self.state = {g: 0.0 for g in groups}

    def update(self, scores: np.ndarray) -> dict[str, float]:
        for g, ids in self.groups.items():
            s = float(scores[ids].max()) if ids else 0.0
            self.state[g] = max(s, self.state[g] * self.decay)
        return dict(self.state)


class AudioMonitor:
    """Feed arbitrary-length 16 kHz chunks; classifies the last ~1 s every `hop_s`."""

    def __init__(self, model_path: str | Path = WEIGHTS, hop_s: float = 0.5, window_s: float = 0.975):
        from mediapipe.tasks.python import BaseOptions, audio

        if not Path(model_path).exists():
            raise FileNotFoundError(f"{model_path} missing; run scripts/download_models.sh")
        self.labels = load_labels(model_path)
        opts = audio.AudioClassifierOptions(base_options=BaseOptions(model_asset_path=str(model_path)), max_results=-1,
                                            running_mode=audio.RunningMode.AUDIO_CLIPS)
        self._clf = audio.AudioClassifier.create_from_options(opts)
        self.scorer = GroupScorer(self.labels)
        self.window = int(window_s * SAMPLE_RATE)
        self.hop = int(hop_s * SAMPLE_RATE)
        self._buf: deque[np.ndarray] = deque()
        self._n = 0
        self._since = 0
        self.state = AudioState(groups=dict(self.scorer.state))

    def _classify(self, x: np.ndarray) -> np.ndarray:
        from mediapipe.tasks.python.components.containers import AudioData

        res = self._clf.classify(AudioData.create_from_array(x.astype(np.float32), SAMPLE_RATE))
        scores = np.zeros(len(self.labels), np.float32)
        if res:
            # AUDIO_CLIPS returns one result per ~1 s window; take the max over windows
            for r in res:
                for c in r.classifications[0].categories:
                    scores[c.index] = max(scores[c.index], c.score)
        return scores

    def push(self, t: float, pcm: np.ndarray) -> AudioState | None:
        """Returns a new AudioState when a window was classified, else None."""
        pcm = np.asarray(pcm, np.float32).ravel()
        if pcm.size == 0:
            return None
        self._buf.append(pcm)
        self._n += pcm.size
        self._since += pcm.size
        while self._n - self._buf[0].size >= self.window:
            self._n -= self._buf.popleft().size
        if self._since < self.hop or self._n < self.window // 2:
            return None
        self._since = 0
        x = np.concatenate(self._buf)[-self.window:]
        rms = float(np.sqrt(np.mean(x ** 2)) + 1e-9)
        scores = self._classify(x)
        top = np.argsort(scores)[::-1][:3]
        self.state = AudioState(t=t, rms_db=20 * np.log10(rms), groups=self.scorer.update(scores),
                                top=[(self.labels[i], float(scores[i])) for i in top if scores[i] > 0.05])
        return self.state

    def close(self):
        try:
            self._clf.close()
        except Exception:
            pass
