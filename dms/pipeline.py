"""Real-time driver monitoring pipeline.

frame ─► FaceLandmarker ─► geometric features ─┬─► calibration (first N s)
                                                ├─► eye / yawn / glance trackers ──┐
                                                └─► feature sequence ─► Transformer ├─► RiskFusion ─► alerts
face crop ─► FaceStateNet (emotion)  ───────────────────────────────────────────────┤
full frame ─► CabinNet (phone, drinking, ...) ─────────────────────────────────────┘

Learned models are optional; heavy ones are throttled to a fixed rate so the
loop stays real-time on a CPU.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from .data.transforms import to_batch
from .features.geometry import LEFT_EYE, RIGHT_EYE, FrameFeatures, compute_features
from .features.temporal import Calibrator, DriverProfile, EyeClosureTracker, FeatureSequence, GlanceTracker, YawnTracker
from .fusion.alerts import Alert, AlertManager
from .fusion.mishap import JoltDetector, MishapDetector, MishapState
from .fusion.risk import RiskFusion, RiskInputs, RiskOutput
from .models import checkpoint
from .models.networks import ALERTNESS_CLASSES, CABIN_CLASSES, EMOTION_CLASSES
from .perception.emotion import HSE_PATH, HSEmotionONNX, TorchEmotion
from .perception.eyes import EyeStateCNN
from .perception.face import FaceLandmarker, FaceObservation, bbox_from_points, crop
from .perception.objects import WEIGHTS as YOLO_WEIGHTS, Detection, behaviour_scores
from .report import SessionRecorder


@dataclass
class MonitorConfig:
    landmarker: str | None = None
    face_ckpt: str | None = None
    cabin_ckpt: str | None = None
    temporal_ckpt: str | None = None
    profile: str | None = None  # load a saved DriverProfile and skip calibration
    calib_seconds: float = 10.0
    emotion: bool = True
    emotion_source: str = "pretrained"  # "pretrained" (HSEmotion/AffectNet) or "trained" (emotion head of face_ckpt)
    eye_cnn: bool = True  # use the eye head of face_ckpt together with EAR
    eye_hz: float = 15.0
    objects: bool = True  # YOLO phone / drink / smoking detection
    yolo_weights: str | None = None
    object_hz: float = 4.0
    face_hz: float = 5.0
    cabin_hz: float = 3.0
    temporal_hz: float = 1.0
    seq_len: int = 300  # 30 s at 10 Hz
    seq_rate_hz: float = 10.0
    threads: int = 2
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
    cabin: str | None = None
    cabin_conf: float = 0.0
    alertness: str | None = None
    perclos: float = 0.0
    blink_rate: float = 0.0
    yawns: int = 0
    audio: dict | None = None  # latest AudioState.as_dict()
    mishap: MishapState | None = None
    fps: float = 0.0
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


class DriverMonitor:
    def __init__(self, cfg: MonitorConfig | None = None):
        self.cfg = cfg = cfg or MonitorConfig()
        torch.set_num_threads(cfg.threads)
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
        self.cabin_model = self._load(cfg.cabin_ckpt)
        self.temporal_model = self._load(cfg.temporal_ckpt)
        self.calibrator = Calibrator(cfg.calib_seconds)
        if cfg.profile:
            self.calibrator.profile = DriverProfile.load(cfg.profile)
        self.eyes = EyeClosureTracker()
        self.yawn = YawnTracker()
        self.glance = GlanceTracker()
        self.seq = FeatureSequence(cfg.seq_len, cfg.seq_rate_hz)
        self.fusion = RiskFusion()
        self.alerter = AlertManager()
        self.recorder = SessionRecorder()
        self._thr_face, self._thr_cabin, self._thr_temp = _Throttle(cfg.face_hz), _Throttle(cfg.cabin_hz), _Throttle(cfg.temporal_hz)
        self._thr_obj, self._thr_eye = _Throttle(cfg.object_hz), _Throttle(cfg.eye_hz)
        self._eye_p: float | None = None
        self._emotion_p = self._cabin_p = self._alert_p = None
        self._dets: list[Detection] = []
        self._last_t, self._fps = None, 0.0
        self.audio_monitor = None  # created on first push_audio (loads YAMNet)
        self._audio = None
        self.jolt = JoltDetector()
        self.mishap = MishapDetector() if cfg.mishap else None

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

    def models_info(self) -> dict:
        return {
            "landmarks": "MediaPipe Face Landmarker (478 pts, 52 blendshapes)",
            "emotion": getattr(self.emotion_model, "name", None),
            "eye": self.eye_model.name if self.eye_model else "geometric eye aspect ratio (no eye CNN loaded)",
            "objects": getattr(self.object_model, "name", None),
            "cabin": "CabinNet (trained)" if self.cabin_model is not None else None,
            "temporal": "Conv-Transformer (trained)" if self.temporal_model is not None else None,
            "audio": "YAMNet (MediaPipe, 521 sound classes)" if self.cfg.audio else None,
            "mishap": "camera jolt + crash/skid sound + responsiveness check" if self.cfg.mishap else None,
        }

    def reset(self, keep_profile: bool = False):
        """New session (e.g. video restarted). Optionally keep the calibration."""
        prof = self.calibrator.profile if keep_profile else None
        self.calibrator = Calibrator(self.cfg.calib_seconds)
        self.calibrator.profile = prof
        self.eyes, self.yawn, self.glance = EyeClosureTracker(), YawnTracker(), GlanceTracker()
        self.seq = FeatureSequence(self.cfg.seq_len, self.cfg.seq_rate_hz)
        self.fusion, self.alerter = RiskFusion(), AlertManager()
        self.recorder.reset()
        self._emotion_p = self._cabin_p = self._alert_p = None
        self._eye_p = None
        self._dets = []
        self._last_t = None
        self._audio = None
        self.jolt.reset()
        if self.mishap is not None:
            self.mishap = MishapDetector()
        # MediaPipe VIDEO mode needs increasing timestamps; keep its clock offset.
        self._t_offset = self.landmarker._last_ms / 1000.0 + 1.0

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

    @torch.inference_mode()
    def process(self, frame: np.ndarray, t: float | None = None) -> FrameResult:
        t = time.monotonic() if t is None else t
        if self._last_t is not None and t > self._last_t:
            self._fps = 0.9 * self._fps + 0.1 / (t - self._last_t)
        self._last_t = t

        obs = self.landmarker.detect(frame, t + getattr(self, "_t_offset", 0.0))
        feats = compute_features(obs.landmarks, obs.transform, obs.blendshapes) if obs else FrameFeatures(face=False)
        face_box = bbox_from_points(obs.landmarks, frame.shape, 0.1) if obs else None
        jolt = self.jolt.update(t, frame) if self.mishap is not None else False
        audio_groups = self._audio.groups if (self._audio is not None and t - self._audio.t < 2.0) else None

        # --- learned per-frame models (throttled) -----------------------------
        if obs is None:
            self._eye_p = None
        elif self.eye_model is not None and self._thr_eye.ready(t):
            self._eye_p = self.eye_model(frame, obs.landmarks)
        if obs is None:
            self._emotion_p = None
        elif self.emotion_model is not None and self._thr_face.ready(t):
            face = crop(frame, bbox_from_points(obs.landmarks, frame.shape, 0.15))
            if face is not None:
                p = self.emotion_model(face)
                # light temporal smoothing: single-frame FER is noisy
                self._emotion_p = p if self._emotion_p is None else 0.6 * self._emotion_p + 0.4 * p
        if self.cabin_model is not None and self._thr_cabin.ready(t):
            self._cabin_p = torch.softmax(self.cabin_model(to_batch([frame])), -1)[0].numpy()
        if self.object_model is not None and self._thr_obj.ready(t):
            self._dets = self.object_model(frame)
        behaviours = behaviour_scores(self._dets, face_box, frame.shape)

        res = FrameResult(t=t, calibrating=None, features=feats, risk=None, obs=obs, fps=self._fps,
                          detections=self._dets, behaviours=behaviours, face_box=face_box,
                          audio=self._audio.as_dict() if self._audio is not None else None)
        if self._emotion_p is not None:
            k = int(np.argmax(self._emotion_p))
            res.emotion, res.emotion_conf, res.emotion_probs = EMOTION_CLASSES[k], float(self._emotion_p[k]), self._emotion_p
        if self._cabin_p is not None:
            k = int(np.argmax(self._cabin_p))
            res.cabin, res.cabin_conf = CABIN_CLASSES[k], float(self._cabin_p[k])

        # --- calibration -------------------------------------------------------
        if not self.calibrator.done:
            self.calibrator.update(t, feats)
            if not self.calibrator.done:
                res.calibrating = self.calibrator.progress(t)
                return res
        prof = self.profile

        # --- temporal indicators ----------------------------------------------
        if feats.face:
            self.eyes.update(t, feats.ear, prof, self._eye_p)
            self.yawn.update(t, feats.mar, feats.jaw_open_bs)
            self.seq.update(t, feats.vector(), prof)
        g = self.glance.update(t, feats.yaw, feats.pitch, prof, face=feats.face)

        if self.temporal_model is not None and self.seq.full and self._thr_temp.ready(t):
            x = torch.from_numpy(self.seq.array()).unsqueeze(0)
            self._alert_p = torch.softmax(self.temporal_model(x), -1)[0].numpy()
        if self._alert_p is not None:
            res.alertness = ALERTNESS_CLASSES[int(np.argmax(self._alert_p))]

        res.perclos, res.blink_rate, res.yawns = self.eyes.perclos(), self.eyes.blink_rate(), self.yawn.count()
        res.risk = self.fusion.update(
            t,
            RiskInputs(
                face=feats.face,
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
                cabin_probs=self._cabin_p,
                emotion_probs=self._emotion_p,
                behaviours=behaviours,
                audio_groups=audio_groups,
            ),
        )
        res.off_road, res.off_road_duration = g.off_road, g.off_road_duration
        res.eye_cnn_closed, res.eye_closed = self._eye_p, bool(feats.face and self.eyes.last_closed)
        res.eye_closed_duration = self.eyes.closed_duration
        if feats.face and not np.isnan(feats.ear):
            res.eye_openness = self.eyes.openness(feats.ear, prof)
        mishap_events = []
        if self.mishap is not None:
            res.mishap = self.mishap.update(t, jolt=jolt, audio_groups=audio_groups, face=feats.face,
                                           eyes_closed=bool(feats.face and self.eyes.last_closed))
            mishap_events = self.mishap.pop_events()
        res.alerts = self.alerter.update(
            t, risk=res.risk, microsleep=self.eyes.in_microsleep, closed_duration=self.eyes.closed_duration,
            long_distraction=g.long_distraction, vats=g.vats_distraction, off_road_duration=g.off_road_duration,
            yawns_5min=res.yawns, face=feats.face, alertness=res.alertness, emotion_probs_affect=res.risk.affect,
            behaviours=behaviours, audio_groups=audio_groups, mishap_events=mishap_events,
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
