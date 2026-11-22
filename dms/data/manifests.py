"""Convert public datasets into one manifest format.

Manifest CSV columns:
    path, subject, source, split, eye, yawn, emotion, cabin
Task columns hold a class index, or -1 when that image has no label for the
task. `subject` is what makes subject-independent splits possible; `split` is
optional (blank = assign automatically, by subject).

Usage:
    python -m dms.data.manifests mrl       /data/mrlEyes_2018_01      manifests/mrl.csv
    python -m dms.data.manifests statefarm /data/state-farm           manifests/statefarm.csv
    python -m dms.data.manifests fer       /data/fer2013              manifests/fer.csv
    python -m dms.data.manifests affectnet /data/affectnet            manifests/affectnet.csv
    python -m dms.data.manifests folders   /data/yawn_eye_dataset     manifests/yawn.csv --map yawn=yawn:1 no_yawn=yawn:0 Closed=eye:0 Open=eye:1
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

from ..models.networks import EMOTION_CLASSES

TASK_COLS = ("eye", "yawn", "emotion", "cabin")
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

EMOTION_ALIASES = {
    "neutral": "neutral", "happy": "happy", "happiness": "happy", "sad": "sad", "sadness": "sad",
    "surprise": "surprise", "surprised": "surprise", "fear": "fear", "fearful": "fear",
    "disgust": "disgust", "disgusted": "disgust", "anger": "anger", "angry": "anger",
}
# AffectNet numeric folders: 0 neutral 1 happy 2 sad 3 surprise 4 fear 5 disgust 6 anger 7 contempt
AFFECTNET_IDS = {str(i): c for i, c in enumerate(EMOTION_CLASSES)}


def _row(path, subject, source, split="", **labels):
    r = {"path": str(path), "subject": str(subject), "source": source, "split": split}
    r.update({c: int(labels.get(c, -1)) for c in TASK_COLS})
    return r


def _images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def mrl_eye(root: Path) -> pd.DataFrame:
    """MRL Eye Dataset. Filename: s0001_00001_<gender>_<glasses>_<eyestate>_<reflect>_<light>_<sensor>.png"""
    rows = []
    for p in _images(root):
        parts = p.stem.split("_")
        if len(parts) < 5 or not parts[0].startswith("s"):
            continue
        rows.append(_row(p, f"mrl_{parts[0]}", "mrl", eye=int(parts[4])))  # 0 closed, 1 open
    return pd.DataFrame(rows)


def statefarm(root: Path) -> pd.DataFrame:
    """State Farm Distracted Driver Detection (Kaggle). Uses driver_imgs_list.csv for driver IDs."""
    lst = pd.read_csv(root / "driver_imgs_list.csv")
    img_dir = root / "imgs" / "train"
    rows = [_row(img_dir / r.classname / r.img, f"sf_{r.subject}", "statefarm", cabin=int(r.classname[1:])) for r in lst.itertuples()]
    return pd.DataFrame(rows)


def emotion_folders(root: Path, source: str, id_map: dict | None = None) -> pd.DataFrame:
    """ImageFolder-style emotion data (FER2013 / RAF-DB / AffectNet / KMU-FED exports).

    Folder names are mapped via aliases (or `id_map` for numeric folders); unknown
    classes such as 'contempt' are skipped. 'train'/'test'/'val' parent folders
    are kept as the split. These sets have no subject IDs, so each image is its
    own group -- note this limitation in the paper.
    """
    rows = []
    for p in _images(root):
        name = p.parent.name.lower()
        cls = (id_map or {}).get(name) or EMOTION_ALIASES.get(name)
        if cls is None:
            continue
        split = next((s for s in ("train", "val", "test") if s in {q.name.lower() for q in p.parents}), "")
        split = {"val": "val", "test": "test", "train": "train"}.get(split, "")
        rows.append(_row(p, f"{source}_{p.stem}", source, split, emotion=EMOTION_CLASSES.index(cls)))
    return pd.DataFrame(rows)


def folders(root: Path, mapping: dict[str, tuple[str, int]], source: str, subject_regex: str | None = None) -> pd.DataFrame:
    """Generic: folder name -> (task, label). Optional regex extracts subject from filename."""
    rows, rx = [], re.compile(subject_regex) if subject_regex else None
    for p in _images(root):
        if p.parent.name not in mapping:
            continue
        task, label = mapping[p.parent.name]
        m = rx.search(p.name) if rx else None
        subj = f"{source}_{m.group(1)}" if m else f"{source}_{p.stem}"
        rows.append(_row(p, subj, source, **{task: label}))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["mrl", "statefarm", "fer", "affectnet", "rafdb", "kmufed", "folders"])
    ap.add_argument("root", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--map", nargs="*", default=[], help="folders: NAME=task:label ...")
    ap.add_argument("--subject-regex", help="folders: regex with one group capturing subject id")
    ap.add_argument("--source", default=None)
    a = ap.parse_args(argv)
    if a.kind == "mrl":
        df = mrl_eye(a.root)
    elif a.kind == "statefarm":
        df = statefarm(a.root)
    elif a.kind == "affectnet":
        df = emotion_folders(a.root, "affectnet", AFFECTNET_IDS)
    elif a.kind in ("fer", "rafdb", "kmufed"):
        df = emotion_folders(a.root, a.kind)
    else:
        mapping = {}
        for m in a.map:
            name, tl = m.split("=")
            task, label = tl.split(":")
            assert task in TASK_COLS, task
            mapping[name] = (task, int(label))
        df = folders(a.root, mapping, a.source or a.root.name, a.subject_regex)
    if df.empty:
        sys.exit("no images found - check the dataset path/layout")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} images, {df.subject.nunique()} subjects -> {a.out}")
    for c in TASK_COLS:
        vc = df[c][df[c] >= 0].value_counts().sort_index()
        if len(vc):
            print(f"  {c}: {dict(vc)}")


if __name__ == "__main__":
    main()
