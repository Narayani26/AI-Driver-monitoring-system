"""Video -> per-frame feature sequences for the temporal drowsiness model.

UTA-RLDD layout (one folder per subject, three self-labelled videos each):
    <root>/Fold1_part1/01/0.mov   alert          (KSS-like level 0)
    <root>/Fold1_part1/01/5.mov   low vigilant   (level 5)
    <root>/Fold1_part1/01/10.mov  drowsy         (level 10)
Also accepted: label folders (alert/ low_vigilant/ drowsy/ with one video per
subject) and <subject>_<label>.mp4 names; see find_videos().

    python -m dms.data.sequences /data/uta-rldd features/rldd --fps 10 --workers 4 --max-seconds 360

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


LABEL_WORDS = {"alert": 0, "awake": 0, "low_vigilant": 1, "lowvigilant": 1, "low vigilant": 1, "low-vigilant": 1,
               "drowsy": 2, "sleepy": 2}


def find_videos(root: Path) -> list[tuple[str, int, Path]]:
    """(subject, label, path) for UTA-RLDD-style data. Accepts
       <subject>/0.mov|5.mov|10.mov            (official layout, any extension/case)
       <label word>/<subject>.mp4               (e.g. alert/01.mp4, drowsy/01.mp4)
       <subject>_<0|5|10>.mp4 or <subject>-<label word>.mp4 in one folder"""
    out = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in VIDEO_EXT or not p.is_file():
            continue
        stem, parent = p.stem.lower(), p.parent.name.lower()
        if stem in RLDD_LABELS:
            out.append((p.parent.name, RLDD_LABELS[stem], p))
        elif parent in LABEL_WORDS:
            out.append((p.stem, LABEL_WORDS[parent], p))
        else:  # <subject><sep><label>, longest label first so "low_vigilant" wins over "vigilant"
            labels = sorted({**RLDD_LABELS, **LABEL_WORDS}.items(), key=lambda kv: -len(kv[0]))
            for word, lab in labels:
                for sep in ("_", "-", " "):
                    if stem.endswith(sep + word) and len(stem) > len(word) + 1:
                        out.append((p.stem[: -len(word) - 1], lab, p))
                        break
                else:
                    continue
                break
    return out


def _extract_job(job):
    """Runs in a worker process (one MediaPipe instance per process)."""
    from ..perception.face import FaceLandmarker

    subj, label, path, dst, fps, max_seconds, dataset = job
    F, P, rate = extract_video(path, lambda: FaceLandmarker(video_mode=True), fps, max_seconds)
    np.savez_compressed(dst, raw=F, present=P, label=label, subject=f"{dataset}_{subj}", fps=fps, source=str(path))
    return subj, label, len(F), rate


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--fps", type=float, default=10.0)
    ap.add_argument("--max-seconds", type=float, default=None, help="use only the first N seconds of each video (faster)")
    ap.add_argument("--workers", type=int, default=1, help="videos processed in parallel (one per CPU core)")
    ap.add_argument("--dataset", default="rldd")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)

    vids = find_videos(a.root)
    by_subject = defaultdict(dict)
    for subj, label, v in vids:
        by_subject[subj][label] = v
    print(f"{len(vids)} videos, {len(by_subject)} subjects")
    if not vids:
        raise SystemExit("no videos found: expected <subject>/0.mov,5.mov,10.mov (or label folders); "
                         "run `python -m dms.data.probe <folder>` and check the layout")
    jobs = []
    for subj, videos in by_subject.items():
        for label, v in videos.items():
            dst = a.out / f"{a.dataset}_{subj}_{label}.npz"
            if not dst.exists():  # resumable: already-extracted videos are skipped
                jobs.append((subj, label, v, dst, a.fps, a.max_seconds, a.dataset))
    print(f"{len(jobs)} videos to process with {a.workers} worker(s)")
    if a.workers > 1 and jobs:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(a.workers) as pool:
            for i, (subj, label, n, rate) in enumerate(pool.imap_unordered(_extract_job, jobs), 1):
                print(f"  [{i}/{len(jobs)}] {subj} label={label}: {n} steps, face {rate:.0%}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            subj, label, n, rate = _extract_job(job)
            print(f"  [{i}/{len(jobs)}] {subj} label={label}: {n} steps, face {rate:.0%}", flush=True)

    # per-subject calibration from the alert video (fallback: all of that subject's videos)
    for subj, videos in sorted(by_subject.items()):
        raw = {lab: np.load(a.out / f"{a.dataset}_{subj}_{lab}.npz")["raw"] for lab in videos}
        base = raw.get(0, np.concatenate(list(raw.values())) if raw else np.zeros((0, 9)))
        prof = profile_from_features(base)
        for label in raw:
            dst = a.out / f"{a.dataset}_{subj}_{label}.npz"
            d = dict(np.load(dst, allow_pickle=True))
            d["profile"] = np.array(list(asdict(prof).values()), dtype=np.float64)
            np.savez_compressed(dst, **d)
    print(f"done: {len(list(a.out.glob('*.npz')))} sequences in {a.out}")


if __name__ == "__main__":
    main()
