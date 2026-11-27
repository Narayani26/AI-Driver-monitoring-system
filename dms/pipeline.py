"""Real-time driver monitoring pipeline.

frame ─► FaceLandmarker ─► geometric features ─┬─► calibration (first N s)
                                                ├─► eye / yawn / glance trackers ──┐
                                                └─► feature sequence ─► Transformer ├─► RiskFusion ─► alerts
face crop ─► FaceStateNet (emotion)  ───────────────────────────────────────────────┤
full frame ─► CabinNet (phone, drinking, ...) ─────────────────────────────────────┘

Learned models are optional; heavy ones are throttled to a fixed rate so the
loop stays real-time on a CPU. With `async_models=True` (the live app) they run
in a background thread: every frame is answered immediately from the face
landmarks, and the learned models update their latest output as fast as the
machine allows. A slow or large model then lowers how often *it* updates,
never the video frame rate.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .data.transforms import to_batch
from .features.camera import CameraMoveDetector
from .features.geometry import LEFT_EYE, RIGHT_EYE, FrameFeatures, compute_features
from .features.temporal import (Calibrator, DriverProfile, EyeClosureTracker, FeatureSequence, GlanceTracker, HeadCalibrator,
                                HeadNodTracker, YawnTracker)
from .fusion.alerts import Alert, AlertManager
from .fusion.mishap import JoltDetector, MishapDetector, MishapState
from .fusion.risk import RiskFusion, RiskInputs, RiskOutput
from .models import checkpoint
from .models.networks import ALERTNESS_CLASSES, CABIN_CLASSES, EMOTION_CLASSES
from .perception.emotion import HSE_PATH, HSEmotionONNX, TorchEmotion
from .perception.eyes import EyeStateCNN
from .perception.face import FaceLandmarker, FaceObservation, bbox_from_points, crop
from .perception.objects import WEIGHTS as YOLO_WEIGHTS, Detection, behaviour_scores
from .perception.pose import WEIGHTS as POSE_WEIGHTS, HeadObservation
from .report import SessionRecorder


# How long a condition must last before it alerts, and how often an alert may repeat.
# "low" = calmer (fewer, later alerts), "high" = stricter (earlier alerts).
SENSITIVITY = {
    "low": dict(microsleep_s=1.5, long_s=4.0, vats_cum_s=12.0, yawn_min_s=2.0, cooldown_scale=1.5, drowsy_threshold=0.7),
    "normal": dict(microsleep_s=1.0, long_s=3.0, vats_cum_s=10.0, yawn_min_s=1.5, cooldown_scale=1.0, drowsy_threshold=0.6),
    "high": dict(microsleep_s=0.8, long_s=2.5, vats_cum_s=8.0, yawn_min_s=1.2, cooldown_scale=0.75, drowsy_threshold=0.5),
}


@dataclass
class MonitorConfig:
    landmarker: str | None = None
    face_ckpt: str | None = None
    cabin_ckpt: str | None = None
    temporal_ckpt: str | None = None
    drowsy_ckpt: str | None = None  # frame-level drowsy/alert model (e.g. trained on NTHU-DDD)
    profile: str | None = None  # load a saved DriverProfile and skip calibration
    calib_seconds: float = 10.0
    emotion: bool = True
    emotion_source: str = "pretrained"  # "pretrained" (HSEmotion/AffectNet) or "trained" (emotion head of face_ckpt)
    eye_cnn: bool = True  # use the eye head of face_ckpt together with EAR
    eye_hz: float = 8.0
    eye_max_pitch: float = 15.0  # degrees from the calibrated pose beyond which eye/mouth shape is not trusted
    eye_max_yaw: float = 25.0
    objects: bool = True  # YOLO phone / drink / smoking detection
    pose_fallback: bool = True  # side/diagonal cameras: track the head with a body-pose model when no frontal face is found
    pose_weights: str | None = None
    head_hz: float = 10.0
    yolo_weights: str | None = None
    object_hz: float = 8.0  # phone use should be caught within about a second
    face_hz: float = 5.0
    cabin_hz: float = 3.0
    temporal_hz: float = 1.0
    seq_len: int = 300  # 30 s at 10 Hz
    seq_rate_hz: float = 10.0
    threads: int = max(2, min(4, (os.cpu_count() or 2) // 2))
    async_models: bool = False  # True for live use (see module docstring); False = deterministic, for offline evaluation
    sensitivity: str = "normal"  # low | normal | high (see SENSITIVITY)
    audio: bool = False  # YAMNet cabin-audio events (needs audio pushed via push_audio)
    mishap: bool = False  # possible-accident detection (camera jolt and/or crash sounds)


@dataclass
class FrameResult:
    t: float
    calibrating: float | None  # progress 0..1 while calibrating, else None
    features: FrameFeatures
    risk: RiskOutput | None
    emotion: str | None = None
    emotion_conf: float = 0.0
    emotion_probs: np.ndarray | None = field(default=None, repr=False)
    detections: list[Detection] = field(default_factory=list)
    behaviours: dict[str, float] = field(default_factory=dict)
    alerts: list[Alert] = field(default_factory=list)
    face_box: tuple[int, int, int, int] | None = None
    off_road: bool = False
    off_road_duration: float = 0.0
    eye_closed_duration: float = 0.0
    eye_openness: float | None = None
    eye_cnn_closed: float | None = None  # eye-state CNN probability (if a face checkpoint is loaded)
    eye_closed: bool = False  # fused decision used by the trackers
    eyes_measurable: bool = True  # False while the head is tilted/turned too far to judge the eyes
    cabin: str | None = None
    cabin_conf: float = 0.0
    alertness: str | None = None
    perclos: float = 0.0
    blink_rate: float = 0.0
    yawns: int = 0
    audio: dict | None = None  # latest AudioState.as_dict()
    mishap: MishapState | None = None
    fps: float = 0.0
    mode: str = "face"  # face (frontal landmarks) | side (head keypoints, any angle) | none (no driver found)
    head: HeadObservation | None = field(default=None, repr=False)
    head_drop: float | None = None  # head lengths below the driver's normal position (side mode)
    nodding: bool = False
    nods: int = 0
    primary: str | None = None  # the view the driver was calibrated in (face | side); the other one counts as "turned away"
    camera_moved: bool = False  # the camera was moved: position is being re-learned
    obs: FaceObservation | None = field(default=None, repr=False)

    def to_log(self) -> dict:
        r = self.risk
        return {
            "t": round(self.t, 3),
            "face": self.features.face,
            "ear": _r(self.features.ear),
            "mar": _r(self.features.mar),
            "yaw": _r(self.features.yaw),
            "pitch": _r(self.features.pitch),
            "perclos": round(self.perclos, 4),
            "blink_rate": round(self.blink_rate, 2),
            "yawns": self.yawns,
            "emotion": self.emotion,
            "cabin": self.cabin,
            "behaviours": {k: round(v, 3) for k, v in self.behaviours.items()},
            "alerts": [a.kind for a in self.alerts],
            "alertness": self.alertness,
            "risk": None if r is None else round(r.index, 1),
            "level": None if r is None else r.level,
            "reasons": [] if r is None else r.reasons,
        }


def _r(x: float, n: int = 4):
    return None if x is None or np.isnan(x) else round(float(x), n)


class _Throttle:
    def __init__(self, hz: float):
        self.period, self.next = 1.0 / hz, -1e9

    def ready(self, t: float) -> bool:
        if t >= self.next:
            self.next = t + self.period
            return True
        return False


class TrainingWindow:
    """The last `seconds` of raw face measurements at `rate_hz`, kept so a window around an
    alert the driver rated can become a training sample (UTA-RLDD feature format).
    Also records what the quality gate needs: face visibility, frontal pose, faces in view and
    whether the tracked face jumped (another person taking the driver's place)."""

    def __init__(self, seconds: float = 30.0, rate_hz: float = 10.0):
        from collections import deque

        self.rate_hz, self.period = rate_hz, 1.0 / rate_hz
        self.buf = deque(maxlen=int(seconds * rate_hz))
        self._next_t: float | None = None
        self._last_c: tuple[float, float] | None = None

    def update(self, t: float, feats: FrameFeatures, frontal: bool, obs, width: int):
        switched = False
        if obs is not None:
            c = (float(obs.landmarks[:, 0].mean()), float(obs.landmarks[:, 1].mean()))
            if self._last_c is not None and np.hypot(c[0] - self._last_c[0], c[1] - self._last_c[1]) > 0.2 * width:
                switched = True
            self._last_c = c
        if self._next_t is None or t - self._next_t > 2.0:
            self._next_t = t
        while t >= self._next_t:
            self.buf.append((feats.vector(), bool(feats.face), bool(frontal), 0 if obs is None else obs.n_faces, switched))
            switched = False
            self._next_t += self.period

    def snapshot(self, profile: DriverProfile | None) -> dict | None:
        if not self.buf or profile is None:
            return None
        rows = list(self.buf)
        raw = [[None if np.isnan(v) else round(float(v), 5) for v in r[0]] for r in rows]
        return {"raw": raw, "present": [r[1] for r in rows], "fps": self.rate_hz,
                "profile": [profile.ear_open, profile.mar_rest, profile.yaw0, profile.pitch0, profile.n_samples],
                "meta": {"frontal_frac": round(float(np.mean([r[2] for r in rows])), 3), "max_faces": int(max(r[3] for r in rows)),
                         "driver_switches": int(sum(r[4] for r in rows))}}


class _ModelWorker:
    """One background thread that runs the learned models on the newest frame.
    submit() never blocks: if the worker is still busy the frame is simply not
    used for the learned models (the frame itself is still processed)."""

    def __init__(self):
        self._job = None
        self._wake = threading.Event()
        self._idle = threading.Event()
        self._idle.set()
        self.error: str | None = None
        threading.Thread(target=self._loop, daemon=True, name="dms-models").start()

    @property
    def busy(self) -> bool:
        return not self._idle.is_set()

    def submit(self, fn) -> bool:
        if self.busy:
            return False
        self._idle.clear()
        self._job = fn
        self._wake.set()
        return True

    def wait(self, timeout: float = 5.0):
        self._idle.wait(timeout)

    def _loop(self):
        while True:
            self._wake.wait()
            self._wake.clear()
            fn, self._job = self._job, None
            try:
                if fn is not None:
                    with torch.inference_mode():
                        fn()
            except Exception as e:  # noqa: BLE001 - a model error must not kill the monitor
                self.error = f"{type(e).__name__}: {e}"
            finally:
                self._idle.set()


class DriverMonitor:
    def __init__(self, cfg: MonitorConfig | None = None):
        self.cfg = cfg = cfg or MonitorConfig()
        torch.set_num_threads(cfg.threads)
        # Trained weights can produce denormal floats, which make CPU convolutions 10-100x slower.
        torch.set_flush_denormal(True)
        self.landmarker = FaceLandmarker(cfg.landmarker) if cfg.landmarker else FaceLandmarker()
        self.face_net, face_ck = checkpoint.load(cfg.face_ckpt) if cfg.face_ckpt else (None, None)
        if self.face_net is not None:
            self.face_net.eval()
        size = (face_ck or {}).get("img_size", 224)
        backbone = (face_ck or {}).get("arch", {}).get("backbone")
        eye_t = float(((face_ck or {}).get("temperature") or {}).get("eye", 1.0))
        self.eye_model = EyeStateCNN(self.face_net, size, f"FaceStateNet eye head ({backbone}, trained)", eye_t) \
            if self.face_net is not None and cfg.eye_cnn and "eye" in self.face_net.heads else None
        self.emotion_model = None
        if cfg.emotion:
            if cfg.emotion_source == "trained" and self.face_net is not None:
                self.emotion_model = TorchEmotion(model=self.face_net, ck=face_ck)
            elif HSE_PATH.exists():
                self.emotion_model = HSEmotionONNX()
            elif self.face_net is not None:
                self.emotion_model = TorchEmotion(model=self.face_net, ck=face_ck)
        self.object_model = self._load_yolo(cfg) if cfg.objects else None
        self.head_model = self._load_pose(cfg) if cfg.pose_fallback else None
        self.head_cal = HeadCalibrator(cfg.calib_seconds)
        self._head: HeadObservation | None = None
        self._head_t = self._face_seen_t = -1e9
        self._side_eye_p: float | None = None
        self.cam = CameraMoveDetector()
        self._primary: str | None = None
        self.cabin_model = self._load(cfg.cabin_ckpt)
        self.temporal_model = self._load(cfg.temporal_ckpt)
        self.drowsy_model, drowsy_ck = checkpoint.load(cfg.drowsy_ckpt) if cfg.drowsy_ckpt else (None, None)
        self._drowsy_T = float(((drowsy_ck or {}).get("temperature") or {}).get("drowsy", 1.0))
        self._drowsy_size = (drowsy_ck or {}).get("img_size", 224)
        self._drowsy_p: float | None = None
        self.calibrator = Calibrator(cfg.calib_seconds)
        if cfg.profile:
            self.calibrator.profile = DriverProfile.load(cfg.profile)
        self._build_trackers()
        self.seq = FeatureSequence(cfg.seq_len, cfg.seq_rate_hz)
        self.window = TrainingWindow(cfg.seq_len / cfg.seq_rate_hz, cfg.seq_rate_hz)
        self.fusion = RiskFusion()
        self.recorder = SessionRecorder()
        self._thr_face, self._thr_cabin, self._thr_temp = _Throttle(cfg.face_hz), _Throttle(cfg.cabin_hz), _Throttle(cfg.temporal_hz)
        self._thr_obj, self._thr_eye, self._thr_drowsy = _Throttle(cfg.object_hz), _Throttle(cfg.eye_hz), _Throttle(cfg.cabin_hz)
        self._thr_head = _Throttle(cfg.head_hz)
        self._eye_p: float | None = None
        self._emotion_p = self._cabin_p = self._alert_p = None
        self._dets: list[Detection] = []
        self._last_t, self._fps = None, 0.0
        self.audio_monitor = None  # created on first push_audio (loads YAMNet)
        self._audio = None
        self.jolt = JoltDetector()
        self.mishap = MishapDetector() if cfg.mishap else None
        self.timings: dict[str, float] = {}  # moving average of milliseconds per model call
        self._worker = _ModelWorker() if cfg.async_models else None
        self._gen = 0  # bumped by reset(): results of jobs started before a reset are dropped

    def _build_trackers(self):
        p = SENSITIVITY.get(self.cfg.sensitivity, SENSITIVITY["normal"])
        from .fusion.alerts import COOLDOWN_S

        self.eyes = EyeClosureTracker(microsleep_s=p["microsleep_s"])
        self.yawn = YawnTracker(min_s=p["yawn_min_s"])
        self.glance = GlanceTracker(long_s=p["long_s"], vats_cum_s=p["vats_cum_s"])
        self.nod = HeadNodTracker(min_s=max(p["microsleep_s"] + 0.5, 1.2))
        self.alerter = AlertManager(cooldowns={k: v * p["cooldown_scale"] for k, v in COOLDOWN_S.items()},
                                    drowsy_threshold=p["drowsy_threshold"])

    def set_sensitivity(self, level: str):
        """Switch low/normal/high on the fly; keeps the driver's calibration."""
        if level in SENSITIVITY and level != self.cfg.sensitivity:
            self.cfg.sensitivity = level
            self.reset(keep_profile=True)

    def push_audio(self, t: float, pcm: np.ndarray):
        """Feed 16 kHz mono float PCM. Returns the new AudioState when a window was classified."""
        if not self.cfg.audio:
            return None
        if self.audio_monitor is None:
            from .perception.audio import AudioMonitor

            self.audio_monitor = AudioMonitor()
        st = self.audio_monitor.push(t, pcm)
        if st is not None:
            self._audio = st
        return st

    def acknowledge(self, t: float):
        """Driver confirmed they are OK: cancels a pending accident countdown."""
        if self.mishap is not None:
            self.mishap.acknowledge(t)

    @staticmethod
    def _load_yolo(cfg):
        w = cfg.yolo_weights or YOLO_WEIGHTS
        try:
            from .perception.objects import ObjectDetector

            return ObjectDetector(w) if Path(w).exists() else None
        except ImportError:  # ultralytics not installed -> feature off
            return None

    @staticmethod
    def _load_pose(cfg):
        w = cfg.pose_weights or POSE_WEIGHTS
        try:
            from .perception.pose import HeadPoseDetector

            return HeadPoseDetector(w) if Path(w).exists() else None
        except ImportError:
            return None

    def models_info(self) -> dict:
        return {
            "landmarks": "MediaPipe Face Landmarker (478 pts, 52 blendshapes)",
            "emotion": getattr(self.emotion_model, "name", None),
            "eye": self.eye_model.name if self.eye_model else "geometric eye aspect ratio (no eye CNN loaded)",
            "objects": getattr(self.object_model, "name", None),
            "side_view": getattr(self.head_model, "name", None),
            "cabin": "CabinNet (trained)" if self.cabin_model is not None else None,
            "temporal": f"{type(self.temporal_model).__name__.replace('Drowsiness', '')} temporal model (trained)" if self.temporal_model is not None else None,
            "drowsy_frame": "frame-level drowsiness CNN (trained)" if self.drowsy_model is not None else None,
            "audio": "YAMNet (MediaPipe, 521 sound classes)" if self.cfg.audio else None,
            "mishap": "camera jolt + crash/skid sound + responsiveness check" if self.cfg.mishap else None,
        }

    def reset(self, keep_profile: bool = False):
        """New session (e.g. video restarted). Optionally keep the calibration."""
        prof = self.calibrator.profile if keep_profile else None
        self.calibrator = Calibrator(self.cfg.calib_seconds)
        self.calibrator.profile = prof
        self._build_trackers()
        self.seq = FeatureSequence(self.cfg.seq_len, self.cfg.seq_rate_hz)
        self.window = TrainingWindow(self.cfg.seq_len / self.cfg.seq_rate_hz, self.cfg.seq_rate_hz)
        self.fusion = RiskFusion()
        self.recorder.reset()
        self._emotion_p = self._cabin_p = self._alert_p = None
        self._drowsy_p = None
        self._eye_p = None
        self._dets = []
        self._last_t = None
        self._audio = None
        self._gen += 1
        self.head_cal = HeadCalibrator(self.cfg.calib_seconds)
        self._head, self._head_t, self._face_seen_t, self._side_eye_p = None, -1e9, -1e9, None
        if self.head_model is not None:
            self.head_model.reset()
        self.cam.reset()
        self._primary = None
        self.jolt.reset()
        if self.mishap is not None:
            self.mishap = MishapDetector()
        # MediaPipe VIDEO mode needs increasing timestamps; keep its clock offset.
        self._t_offset = self.landmarker._last_ms / 1000.0 + 1.0
        self.landmarker.reset_driver()

    @staticmethod
    def _load(path):
        if not path:
            return None
        if not Path(path).exists():
            raise FileNotFoundError(path)
        return checkpoint.load(path)[0]

    @property
    def profile(self) -> DriverProfile | None:
        return self.calibrator.profile

    def _recalibrate_position(self):
        """Camera moved: forget the driver's position (and open-eye shape, which depends on the
        viewing angle) but keep the trip, its report and the risk history."""
        self.calibrator = Calibrator(self.cfg.calib_seconds)
        self.head_cal = HeadCalibrator(self.cfg.calib_seconds)
        self._build_trackers()
        self.seq = FeatureSequence(self.cfg.seq_len, self.cfg.seq_rate_hz)
        self.window = TrainingWindow(self.cfg.seq_len / self.cfg.seq_rate_hz, self.cfg.seq_rate_hz)
        self._primary, self._alert_p = None, None
        self.landmarker.reset_driver()
        if self.head_model is not None:
            self.head_model.reset()

    def _frontal(self, f: FrameFeatures, prof: DriverProfile) -> bool:
        """Is the face close enough to the calibrated pose for eye/mouth shape to be trusted?"""
        if np.isnan(f.pitch) or np.isnan(f.yaw):
            return True
        return abs(f.pitch - prof.pitch0) <= self.cfg.eye_max_pitch and abs(f.yaw - prof.yaw0) <= self.cfg.eye_max_yaw

    def _timed(self, name: str, fn):
        t0 = time.perf_counter()
        out = fn()
        ms = (time.perf_counter() - t0) * 1000
        self.timings[name] = ms if name not in self.timings else 0.8 * self.timings[name] + 0.2 * ms
        return out

    def _schedule_models(self, frame: np.ndarray, obs, t: float, side: bool = False):
        """Decide which learned models are due and run them (now, or in the background)."""
        if self._worker is not None and self._worker.busy:
            return  # still working on an earlier frame: use its latest outputs
        due = []
        if side and self.head_model is not None and self._thr_head.ready(t):
            due.append("head")
        if obs is not None and self.eye_model is not None and self._thr_eye.ready(t):
            due.append("eye")
        if obs is not None and self.emotion_model is not None and self._thr_face.ready(t):
            due.append("emotion")
        if obs is not None and self.drowsy_model is not None and self._thr_drowsy.ready(t):
            due.append("drowsy")
        if self.cabin_model is not None and self._thr_cabin.ready(t):
            due.append("cabin")
        if self.object_model is not None and self._thr_obj.ready(t):
            due.append("objects")
        seq = None
        if self.temporal_model is not None and self.calibrator.done and self.seq.full and self._thr_temp.ready(t):
            due.append("temporal")
            seq = self.seq.array().copy()
        if not due:
            return
        lm = None if obs is None else obs.landmarks
        gen = self._gen

        def job():
            out = self._run_models(due, frame, lm, seq, t)
            if gen == self._gen:  # not reset meanwhile
                for k, v in out.items():
                    setattr(self, k, v)

        if self._worker is None:
            job()
        else:
            self._worker.submit(job)

    def _run_models(self, due, frame, lm, seq, t: float = 0.0) -> dict:
        out = {}
        if "head" in due:
            h = self._timed("head keypoints (side view)", lambda: self.head_model(frame))
            out["_head"], out["_head_t"] = h, t
            out["_side_eye_p"] = None
            if h is not None and self.eye_model is not None and (b := h.eye_box(frame.shape)) is not None:
                out["_side_eye_p"] = self._timed("eye CNN", lambda: self.eye_model.on_crop(crop(frame, b)))
        if "eye" in due:
            out["_eye_p"] = self._timed("eye CNN", lambda: self.eye_model(frame, lm))
        if "emotion" in due:
            face = crop(frame, bbox_from_points(lm, frame.shape, 0.15))
            if face is not None:
                p = self._timed("emotion", lambda: self.emotion_model(face))
                # light temporal smoothing: single-frame FER is noisy
                out["_emotion_p"] = p if self._emotion_p is None else 0.6 * self._emotion_p + 0.4 * p
        if "drowsy" in due:
            p = self._timed("drowsy CNN", lambda: float(torch.softmax(self.drowsy_model(to_batch([frame], self._drowsy_size)) / self._drowsy_T, -1)[0, 1]))
            out["_drowsy_p"] = p if self._drowsy_p is None else 0.85 * self._drowsy_p + 0.15 * p  # ~2 s smoothing at 3 Hz
        if "cabin" in due:
            out["_cabin_p"] = self._timed("cabin CNN", lambda: torch.softmax(self.cabin_model(to_batch([frame])), -1)[0].numpy())
        if "objects" in due:
            out["_dets"] = self._timed("objects (YOLO)", lambda: self.object_model(frame))
        if "temporal" in due:
            x = torch.from_numpy(seq).unsqueeze(0)
            out["_alert_p"] = self._timed("temporal model", lambda: torch.softmax(self.temporal_model(x), -1)[0].numpy())
        return out

    @torch.inference_mode()
    def process(self, frame: np.ndarray, t: float | None = None) -> FrameResult:
        t = time.monotonic() if t is None else t
        if self._last_t is not None and t > self._last_t:
            self._fps = 0.9 * self._fps + 0.1 / (t - self._last_t)
        self._last_t = t

        obs = self._timed("face landmarks", lambda: self.landmarker.detect(frame, t + getattr(self, "_t_offset", 0.0)))
        feats = compute_features(obs.landmarks, obs.transform, obs.blendshapes) if obs else FrameFeatures(face=False)
        face_box = bbox_from_points(obs.landmarks, frame.shape, 0.1) if obs else None
        jolt = self.jolt.update(t, frame) if self.mishap is not None else False
        audio_groups = self._audio.groups if (self._audio is not None and t - self._audio.t < 2.0) else None

        # --- learned per-frame models (throttled; in the background when async) --
        if obs is not None:
            self._face_seen_t = t
        if obs is None:
            self._eye_p = self._emotion_p = None
        # no frontal face for a moment: side/diagonal camera (or head turned far) -> track the head by keypoints
        side = obs is None and self.head_model is not None and t - self._face_seen_t > 0.7
        self._schedule_models(frame, obs, t, side)
        head = self._head if side and self._head is not None and t - self._head_t < 1.0 else None
        mode = "face" if obs is not None else "side" if head is not None else "none"
        if face_box is None and head is not None:
            face_box = head.face_box(frame.shape)
        behaviours = behaviour_scores(self._dets, face_box, frame.shape)

        # --- camera moved since calibration? then re-learn the position (trip and report are kept) ---
        if self.cam.update(t, frame, mode, face_box, head.box if head is not None else None):
            self._recalibrate_position()
            moved = self.alerter._fire("camera_moved", t, f"background match {self.cam.last_ncc:.2f}")
            res = FrameResult(t=t, calibrating=0.0, features=feats, risk=None, obs=obs, fps=self._fps, detections=self._dets,
                              behaviours=behaviours, face_box=face_box, mode=mode, head=head, camera_moved=True,
                              alerts=[moved] if moved else [])
            self.recorder.update(res, res.alerts, behaviours, 0.0, 0.0)
            return res
        # the view the driver was calibrated in is "looking at the road"; seeing them in the other view means turned away
        primary = self._primary
        if primary == "face" and mode == "side":
            mode, head = "none", None  # face lost, only the side of the head visible: turned away
        driver_seen = (mode == primary) if primary else mode != "none"

        res = FrameResult(t=t, calibrating=None, features=feats, risk=None, obs=obs, fps=self._fps,
                          detections=self._dets, behaviours=behaviours, face_box=face_box, mode=mode, head=head,
                          audio=self._audio.as_dict() if self._audio is not None else None)
        if self._emotion_p is not None:
            k = int(np.argmax(self._emotion_p))
            res.emotion, res.emotion_conf, res.emotion_probs = EMOTION_CLASSES[k], float(self._emotion_p[k]), self._emotion_p
        if self._cabin_p is not None:
            k = int(np.argmax(self._cabin_p))
            res.cabin, res.cabin_conf = CABIN_CLASSES[k], float(self._cabin_p[k])

        # --- calibration: learns the driver's normal position from whatever angle the camera has.
        # Whichever view (frontal face or side of the head) finishes first becomes the primary view.
        if self._primary is None:
            if obs is not None:
                self.calibrator.update(t, feats)
            if head is not None:
                self.head_cal.update(t, head.droop)
            if self.calibrator.done or self.head_cal.done:
                self._primary = "face" if self.calibrator.done else "side"
                if self._primary == "face":
                    self.head_cal = HeadCalibrator(self.cfg.calib_seconds)
                else:
                    self.calibrator = Calibrator(self.cfg.calib_seconds)
            else:
                res.calibrating = (self.calibrator.progress(t) if mode == "face" else self.head_cal.progress(t) if mode == "side" else 0.0)
                return res  # still learning (or still looking for the driver): no risk shown, nothing is "safe" yet
        primary = self._primary
        res.primary = primary
        if not self.cam.armed and mode == primary:
            self.cam.set_reference(frame, mode, face_box, head.box if head is not None else None)
        if primary == "side" and mode == "face":
            obs, head = None, None  # side-mounted camera sees a frontal face: the driver turned towards the side
            feats = FrameFeatures(face=False)
        prof = self.profile or DriverProfile()

        # --- temporal indicators ----------------------------------------------
        frontal = feats.face and self._frontal(feats, prof)
        if frontal:
            self.eyes.update(t, feats.ear, prof, self._eye_p)
            self.yawn.update(t, feats.mar, feats.jaw_open_bs)
        elif head is not None and self._side_eye_p is not None:
            self.eyes.update(t, float("nan"), prof, self._side_eye_p)  # visible eye from the side
            self.yawn.pause()
        else:
            self.eyes.pause()
            self.yawn.pause()
        if head is not None and primary == "side":
            res.head_drop = head.droop - self.head_cal.droop0
            self.nod.update(t, res.head_drop)
        else:
            self.nod.pause()
        res.nodding, res.nods = self.nod.nodding, self.nod.count()
        if feats.face:
            self.seq.update(t, feats.vector(), prof)
        self.window.update(t, feats, frontal, obs, frame.shape[1])
        # side view: no gaze angles, but the driver is there (not "looking away")
        g = self.glance.update(t, feats.yaw, feats.pitch, prof, face=driver_seen)

        if self._alert_p is not None:
            names = ALERTNESS_CLASSES if len(self._alert_p) == 3 else ("alert", "drowsy")
            res.alertness = names[int(np.argmax(self._alert_p))]

        eyes_judged = frontal or (head is not None and self._side_eye_p is not None)
        res.perclos, res.blink_rate, res.yawns = self.eyes.perclos(), self.eyes.blink_rate(), self.yawn.count()
        res.risk = self.fusion.update(
            t,
            RiskInputs(
                face=driver_seen,
                perclos=res.perclos,
                microsleep=self.eyes.in_microsleep,
                closed_duration=self.eyes.closed_duration,
                mean_blink_duration=self.eyes.mean_blink_duration(),
                yawns_5min=res.yawns,
                yawning=self.yawn.yawning,
                off_road_duration=g.off_road_duration,
                long_distraction=g.long_distraction,
                vats_distraction=g.vats_distraction,
                alertness_probs=self._alert_p,
                frame_drowsy=self._drowsy_p,
                cabin_probs=self._cabin_p,
                emotion_probs=self._emotion_p,
                behaviours=behaviours,
                audio_groups=audio_groups,
                head_nod=res.nodding,
                nods_5min=res.nods,
                head_drop=res.head_drop,
            ),
        )
        res.off_road, res.off_road_duration = g.off_road, g.off_road_duration
        res.eye_cnn_closed = self._eye_p if obs is not None else self._side_eye_p
        res.eye_closed = bool(eyes_judged and self.eyes.last_closed)
        res.eyes_measurable = bool(eyes_judged)
        res.eye_closed_duration = self.eyes.closed_duration
        if feats.face and not np.isnan(feats.ear):
            res.eye_openness = self.eyes.openness(feats.ear, prof)
        mishap_events = []
        if self.mishap is not None:
            res.mishap = self.mishap.update(t, jolt=jolt, audio_groups=audio_groups, face=driver_seen,
                                           eyes_closed=bool(res.eye_closed or res.nodding))
            mishap_events = self.mishap.pop_events()
        res.alerts = self.alerter.update(
            t, risk=res.risk, microsleep=self.eyes.in_microsleep, closed_duration=self.eyes.closed_duration,
            # while a camera move is suspected, "look at the road" waits (the view, not the driver, changed)
            long_distraction=g.long_distraction and not self.cam.pending, vats=g.vats_distraction and not self.cam.pending,
            off_road_duration=g.off_road_duration, yawns_5min=res.yawns, face=driver_seen or self.cam.pending, alertness=res.alertness, emotion_probs_affect=res.risk.affect,
            behaviours=behaviours, audio_groups=audio_groups, mishap_events=mishap_events, head_nod=res.nodding,
        )
        self.recorder.update(res, res.alerts, behaviours, g.off_road_duration, self.eyes.closed_duration)
        return res

    def close(self):
        self.landmarker.close()
        if self.audio_monitor is not None:
            self.audio_monitor.close()


# --------------------------------------------------------------------------- #
# Visualisation
# --------------------------------------------------------------------------- #
LEVEL_COLORS = {"OK": (80, 200, 80), "CAUTION": (0, 200, 255), "WARNING": (0, 140, 255), "CRITICAL": (0, 0, 255)}


def draw_overlay(frame: np.ndarray, r: FrameResult) -> np.ndarray:
    out = frame.copy()
    H, W = out.shape[:2]
    if r.obs is not None:
        for idx in (LEFT_EYE, RIGHT_EYE):
            pts = r.obs.landmarks[list(idx), :2].astype(np.int32)
            cv2.polylines(out, [pts], True, (255, 255, 0), 1, cv2.LINE_AA)
        x0, y0, x1, y1 = bbox_from_points(r.obs.landmarks, out.shape, 0.1)
        cv2.rectangle(out, (x0, y0), (x1, y1), (200, 200, 200), 1)
    for d in r.detections:
        c = (0, 0, 255) if d.behaviour in r.behaviours else (160, 160, 160)
        cv2.rectangle(out, d.box[:2], d.box[2:], c, 2)
        cv2.putText(out, f"{d.label} {d.conf:.0%}", (d.box[0], max(d.box[1] - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1, cv2.LINE_AA)

    def put(text, y, color=(255, 255, 255), scale=0.55):
        # dark box behind the text (an outline stroke renders misaligned with OpenCV 5's font engine)
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        roi = out[max(y - th - 4, 0): y + base + 2, 6: min(14 + tw, W)]
        roi[:] = (roi * 0.35).astype(np.uint8)
        cv2.putText(out, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)

    if r.calibrating is not None:
        put(f"Calibrating - look at the road ({r.calibrating:.0%})", 30, (0, 255, 255), 0.7)
        cv2.rectangle(out, (10, 40), (10 + int((W - 20) * r.calibrating), 50), (0, 255, 255), -1)
        return out

    f, rk = r.features, r.risk
    color = LEVEL_COLORS[rk.level]
    cv2.rectangle(out, (0, 0), (W, 6), color, -1)
    put(f"{rk.level}  risk {rk.index:5.1f}", 32, color, 0.8)
    y = 58
    lines = [
        f"EAR {f.ear:.3f}  MAR {f.mar:.3f}" if f.face else "NO FACE",
        f"yaw {f.yaw:+.0f}  pitch {f.pitch:+.0f}" if f.face else "",
        f"PERCLOS {r.perclos:.0%}  blinks/min {r.blink_rate:.0f}  yawns {r.yawns}",
        f"drowsy {rk.drowsiness:.2f}  distract {rk.distraction:.2f}  affect {rk.affect:.2f}",
    ]
    if r.alertness:
        lines.append(f"alertness: {r.alertness}")
    if r.emotion:
        lines.append(f"emotion: {r.emotion} ({r.emotion_conf:.0%})")
    if r.cabin:
        lines.append(f"behaviour: {r.cabin.replace('_', ' ')} ({r.cabin_conf:.0%})")
    lines.append(f"{r.fps:.0f} FPS")
    for line in filter(None, lines):
        put(line, y)
        y += 22
    for i, reason in enumerate(rk.reasons[:3]):
        put("! " + reason, H - 15 - 22 * i, color, 0.6)
    return out
