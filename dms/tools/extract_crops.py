"""Turn your own labelled recordings into training crops + a manifest.

    python -m dms.tools.extract_crops --videos recordings/ --labels recordings/labels.csv --out data/own

Writes
    data/own/eye/{open,closed}/*.jpg       eye crops   -> eye head
    data/own/face/{yawn,no_yawn}/*.jpg     face crops  -> yawn head
    data/own/face/<emotion>/*.jpg          face crops  -> emotion head
    data/own/manifest.csv                  ready for dms.training.train_image (subject column filled)

"normal" segments provide the negatives (open eyes, no yawn, neutral face).
Frames within 0.3 s of a segment boundary are skipped (transitions are ambiguous).
Then fine-tune, e.g.:
    python -m dms.training.train_image --kind face --manifest data/own/manifest.csv \
        --init-ckpt runs/face/best.pt --freeze-epochs 2 --epochs 8 --lr 1e-4 --out runs/face_own
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import pandas as pd

from ..data.manifests import TASK_COLS
from ..features.geometry import LEFT_EYE, RIGHT_EYE
from ..models.networks import EMOTION_CLASSES
from ..perception.face import FaceLandmarker, bbox_from_points, crop
from .labels import EMOTION_LABELS, read_labels, subject_of

MARGIN_S = 0.3


def targets_for(label: str) -> list[tuple[str, str, int]]:
    """label -> [(crop kind, folder, (task, class))]"""
    if label == "normal":
        return [("eye", "open", ("eye", 1)), ("face", "no_yawn", ("yawn", 0)), ("face", "neutral", ("emotion", EMOTION_CLASSES.index("neutral")))]
    if label == "eyes_closed":
        return [("eye", "closed", ("eye", 0))]
    if label == "yawn":
        return [("face", "yawn", ("yawn", 1))]
    if label in EMOTION_LABELS:
        return [("face", label, ("emotion", EMOTION_CLASSES.index(label)))]
    return []  # look_away / phone / ... are evaluated, not cropped


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=Path, required=True, help="folder containing the recordings")
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=5.0, help="crops per second of labelled video")
    a = ap.parse_args(argv)

    segs = read_labels(a.labels)
    by_video = defaultdict(list)
    for s in segs:
        by_video[s.video].append(s)
    rows, counts = [], Counter()
    for video, vsegs in sorted(by_video.items()):
        path = next((p for p in a.videos.rglob(video)), None)
        if path is None:
            print(f"!! {video} not found under {a.videos}, skipped")
            continue
        subj = f"own_{subject_of(video)}"
        cap = cv2.VideoCapture(str(path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        step = max(int(round(fps / a.fps)), 1)
        lmk = FaceLandmarker(video_mode=True)
        i = 0
        try:
            while cap.grab():
                if i % step:
                    i += 1
                    continue
                ok, frame = cap.retrieve()
                t = i / fps
                i += 1
                if not ok:
                    break
                seg = next((s for s in vsegs if s.start + MARGIN_S <= t <= s.end - MARGIN_S), None)
                if seg is None:
                    continue
                tg = targets_for(seg.label)
                if not tg:
                    continue
                obs = lmk.detect(frame, t)
                if obs is None:
                    continue
                for kind, folder, (task, cls) in tg:
                    boxes = [bbox_from_points(obs.landmarks[list(e)], frame.shape, 0.6) for e in (LEFT_EYE, RIGHT_EYE)] if kind == "eye" \
                        else [bbox_from_points(obs.landmarks, frame.shape, 0.15)]
                    for k, box in enumerate(boxes):
                        im = crop(frame, box)
                        if im is None:
                            continue
                        dst = a.out / kind / folder / f"{subj}__{Path(video).stem}_{i:06d}_{k}.jpg"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(dst), im)
                        r = {"path": str(dst.resolve()), "subject": subj, "source": "own", "split": ""}
                        r.update({c: -1 for c in TASK_COLS})
                        r[task] = cls
                        rows.append(r)
                        counts[f"{kind}/{folder}"] += 1
        finally:
            cap.release()
            lmk.close()
        print(f"{video}: done ({subj})")
    if not rows:
        raise SystemExit("no crops written - check labels and video names")
    a.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out / "manifest.csv", index=False)
    print(f"{len(rows)} crops, {len({r['subject'] for r in rows})} people -> {a.out / 'manifest.csv'}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
