"""Video -> per-frame feature sequences for the temporal drowsiness model.

UTA-RLDD layout (one folder per subject, three self-labelled videos each):
    <root>/Fold1_part1/01/0.mov   alert          (KSS-like level 0)
    <root>/Fold1_part1/01/5.mov   low vigilant   (level 5)
    <root>/Fold1_part1/01/10.mov  drowsy         (level 10)
Any layout works as long as the file stem is the level and the parent folder is
the subject ID.

    python -m dms.data.sequences /data/uta-rldd features/rldd --fps 10

Per-subject calibration: each subject's DriverProfile is computed from their
*alert* video and stored with every sequence of that subject, so the model
learns from driver-normalised features exactly as it sees them at runtime.
Calibrating on the drowsy video itself would normalise away the very signal
we are trying to detect.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

from ..features.geometry import FrameFeatures, compute_features
from ..features.temporal import DriverProfile, FeatureSequence

VIDEO_EXT = {".mov", ".mp4", ".m4v", ".avi", ".mkv"}
RLDD_LABELS = {"0": 0, "5": 1, "10": 2}


def extract_video(path: Path, landmarker_factory, fps: float = 10.0, max_seconds: float | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Returns (features[T, F] raw at `fps`, face_present[T], face_rate)."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(int(round(src_fps / fps)), 1)
    lmk = landmarker_factory()
    feats, present, i = [], [], 0
    nan_vec = FrameFeatures(face=False).vector()
    try:
        while True:
            ok = cap.grab()
            if not ok:
                break
            if i % step == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    break
                t = i / src_fps
                if max_seconds and t > max_seconds:
                    break
                obs = lmk.detect(frame, t)
                if obs is None:
                    feats.append(nan_vec)
                    present.append(False)
                else:
                    feats.append(compute_features(obs.landmarks, obs.transform, obs.blendshapes).vector())
                    present.append(True)
            i += 1
    finally:
        cap.release()
        lmk.close()
    F = np.stack(feats) if feats else np.zeros((0, len(nan_vec)), np.float32)
    P = np.array(present, dtype=bool)
    return F, P, float(P.mean()) if len(P) else 0.0


def profile_from_features(F: np.ndarray) -> DriverProfile:
    ok = ~np.isnan(F[:, 0])
    if ok.sum() < 10:
        return DriverProfile()
    G = F[ok]
    return DriverProfile(
        ear_open=float(np.nanpercentile(G[:, 0], 80)),
        mar_rest=float(np.nanmedian(G[:, 1])),
        pitch0=float(np.nanmedian(G[:, 2])),
        yaw0=float(np.nanmedian(G[:, 3])),
        n_samples=int(ok.sum()),
    )


def normalise_sequence(F: np.ndarray, prof: DriverProfile) -> np.ndarray:
    return np.stack([FeatureSequence.normalise(v, prof) for v in F]) if len(F) else F


def main(argv=None):
    from ..perception.face import FaceLandmarker

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-seconds", type=float, default=None)
    ap.add_argument("--dataset", default="rldd")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    vids = sorted(p for p in a.root.rglob("*") if p.suffix.lower() in VIDEO_EXT and p.stem in RLDD_LABELS)
    by_subject = defaultdict(dict)
    for v in vids:
        by_subject[v.parent.name][RLDD_LABELS[v.stem]] = v
    print(f"{len(vids)} videos, {len(by_subject)} subjects")

    for subj, videos in sorted(by_subject.items()):
        raw = {}
        for label, v in sorted(videos.items()):
            dst = a.out / f"{a.dataset}_{subj}_{label}.npz"
            if dst.exists():
                raw[label] = np.load(dst)["raw"]
                continue
            F, P, rate = extract_video(v, lambda: FaceLandmarker(video_mode=True), a.fps, a.max_seconds)
            raw[label] = F
            np.savez_compressed(dst, raw=F, present=P, label=label, subject=f"{a.dataset}_{subj}", fps=a.fps, source=str(v))
            print(f"  {subj} label={label}: {len(F)} steps, face {rate:.0%}")
        # alert video defines the baseline; fall back to all videos if it's missing
        base = raw.get(0, np.concatenate(list(raw.values())) if raw else np.zeros((0, 9)))
        prof = profile_from_features(base)
        for label in raw:
            dst = a.out / f"{a.dataset}_{subj}_{label}.npz"
            d = dict(np.load(dst, allow_pickle=True))
            d["profile"] = np.array(list(asdict(prof).values()), dtype=np.float64)
            np.savez_compressed(dst, **d)


if __name__ == "__main__":
    main()
