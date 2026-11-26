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

TASK_COLS = ("eye", "yawn", "emotion", "cabin", "drowsy")
BOX_COLS = ("bx", "by", "bw", "bh")  # optional face box (pixels) when images are not face-cropped
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

EMOTION_ALIASES = {
    "neutral": "neutral", "happy": "happy", "happiness": "happy", "sad": "sad", "sadness": "sad",
    "surprise": "surprise", "surprised": "surprise", "fear": "fear", "fearful": "fear",
    "disgust": "disgust", "disgusted": "disgust", "anger": "anger", "angry": "anger",
}
# AffectNet numeric folders: 0 neutral 1 happy 2 sad 3 surprise 4 fear 5 disgust 6 anger 7 contempt
AFFECTNET_IDS = {str(i): c for i, c in enumerate(EMOTION_CLASSES)}
# RAF-DB basic-emotion label numbers (1-7), as used by the official release and its Kaggle copies
RAFDB_IDS = {"1": "surprise", "2": "fear", "3": "disgust", "4": "happy", "5": "sad", "6": "anger", "7": "neutral"}
# Yawn/eye folder names seen in public datasets (case varies between copies)
YAWN_EYE_MAP = {"yawn": ("yawn", 1), "no_yawn": ("yawn", 0), "no yawn": ("yawn", 0), "noyawn": ("yawn", 0),
                "closed": ("eye", 0), "close": ("eye", 0), "closed_eyes": ("eye", 0), "open": ("eye", 1), "open_eyes": ("eye", 1)}


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


def _find(root: Path, name: str) -> Path:
    """First file/dir called `name` under root (Kaggle mirrors nest folders differently)."""
    hit = next(iter(sorted(root.rglob(name), key=lambda p: len(p.parts))), None)
    if hit is None:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hit


def statefarm(root: Path) -> pd.DataFrame:
    """State Farm Distracted Driver Detection (Kaggle). Uses driver_imgs_list.csv for driver IDs."""
    lst = pd.read_csv(_find(root, "driver_imgs_list.csv"))
    img_dir = next(d.parent for d in sorted(root.rglob("c0"), key=lambda p: len(p.parts)) if d.is_dir() and d.parent.name == "train")
    rows = [_row(img_dir / r.classname / r.img, f"sf_{r.subject}", "statefarm", cabin=int(r.classname[1:])) for r in lst.itertuples()]
    return pd.DataFrame(rows)


def _image_id(source: str, p: Path, root: Path) -> str:
    """Unique per-image group ID for datasets without person IDs. Uses the path
    inside the dataset, not just the file name: FER2013 has files with the same
    name in train/ and test/, which would otherwise look like one person in two
    splits and trip the leakage check."""
    return f"{source}_{p.relative_to(root).with_suffix('').as_posix()}"


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
        inner = {q.lower() for q in p.relative_to(root).parts[:-1]}  # only folders inside the dataset
        split = next((s for s in ("train", "val", "test") if s in inner), "")
        split = {"val": "val", "test": "test", "train": "train"}.get(split, "")
        rows.append(_row(p, _image_id(source, p, root), source, split, emotion=EMOTION_CLASSES.index(cls)))
    return pd.DataFrame(rows)


_PATH_COLS = ("subdirectory_filepath", "image", "image_path", "img", "path", "file", "filename", "pth", "img_path", "image_name")
_LABEL_COLS = ("expression", "label", "emotion", "exp", "class", "hard_label", "category")
_BOX = {"face_x": "bx", "face_y": "by", "face_width": "bw", "face_height": "bh"}


def _affectnet_label(v) -> int | None:
    """AffectNet numbering (0 neutral ... 6 anger, 7 contempt, 8-10 none/uncertain/non-face) or a name."""
    s = str(v).strip().lower()
    if s.lstrip("-").isdigit():
        i = int(s)
        return i if 0 <= i < len(EMOTION_CLASSES) else None
    cls = EMOTION_ALIASES.get(s)
    return EMOTION_CLASSES.index(cls) if cls else None


def affectnet(root: Path, source: str = "affectnet") -> pd.DataFrame:
    """AffectNet / AffectNet+ in any of its usual layouts:
      1. emotion-named or numbered folders          -> emotion_folders
      2. official: <split>/images/<id>.jpg + <split>/annotations/<id>_exp.npy
      3. CSV files with an image-path column and an expression column (official
         training.csv / validation.csv), optional face box (face_x, face_y,
         face_width, face_height) and optional soft-label columns (one
         probability per emotion, AffectNet+): the most likely emotion is used.
    'val'/'validation' folders or CSVs become the test split (AffectNet's
    official evaluation set); there are no person IDs, so each image is its own group."""
    import numpy as np

    df = emotion_folders(root, source, AFFECTNET_IDS)
    if not df.empty and len(df) > 50:
        return df
    rows = []
    for ann in sorted(p for p in root.rglob("annotations") if p.is_dir()):  # layout 2
        img_dir = ann.parent / "images"
        if not img_dir.is_dir():
            continue
        split = "test" if any(k in ann.parent.name.lower() for k in ("val", "test")) else ""
        for img in _images(img_dir):
            f = ann / f"{img.stem}_exp.npy"
            if not f.exists():
                continue
            lab = _affectnet_label(np.load(f, allow_pickle=True).item() if f.suffix == ".npy" else f.read_text())
            if lab is not None:
                rows.append(_row(img, _image_id(source, img, root), source, split, emotion=lab))
    if rows:
        return pd.DataFrame(rows)
    by_name = {p.name: p for p in _images(root)}  # layout 3
    for csv_path in sorted(root.rglob("*.csv")):
        try:
            t = pd.read_csv(csv_path, low_memory=False)
        except Exception:  # noqa: BLE001 - skip unreadable side files
            continue
        cols = {c.lower().strip(): c for c in t.columns}
        pcol = next((cols[c] for c in _PATH_COLS if c in cols), None)
        lcol = next((cols[c] for c in _LABEL_COLS if c in cols), None)
        soft = [cols[e] for e in EMOTION_CLASSES if e in cols] or [cols[k] for k, v in EMOTION_ALIASES.items() if k in cols]
        if pcol is None or (lcol is None and len(soft) < 7):
            continue
        split = "test" if any(k in csv_path.stem.lower() for k in ("val", "test")) else ""
        for r in t.itertuples(index=False):
            rec = r._asdict() if hasattr(r, "_asdict") else dict(zip(t.columns, r))
            rel = str(rec[pcol])
            p = next((c for c in (csv_path.parent / rel, root / rel) if c.is_file()), None) or by_name.get(Path(rel).name)
            if p is None:
                continue
            if lcol is not None:
                lab = _affectnet_label(rec[lcol])
            else:
                probs = [float(rec[c]) for c in soft[:7]]
                lab = int(np.argmax(probs))
            if lab is None:
                continue
            row = _row(p, _image_id(source, p, root), source, split, emotion=lab)
            for k, v in _BOX.items():
                if k in cols:
                    row[v] = float(rec[cols[k]])
            rows.append(row)
    return pd.DataFrame(rows)


NTHU_DROWSY = {"drowsy": 1, "sleepy": 1, "notdrowsy": 0, "not_drowsy": 0, "not drowsy": 0, "nondrowsy": 0,
               "non_drowsy": 0, "non drowsy": 0, "alert": 0, "nonsleepy": 0}
SUBJECT_IN_NAME = re.compile(r"^(\d{1,3})[_\-]")


def nthu_frames(root: Path, source: str = "nthu") -> tuple[pd.DataFrame, float]:
    """NTHU-DDD frames in drowsy / notdrowsy folders. Person ID from a leading number in the
    file name (e.g. 001_glasses_sleepyCombination_12_drowsy.jpg) or from a numbered parent
    folder. Returns (manifest, fraction of images with a person ID)."""
    rows, with_id = [], 0
    for p in _images(root):
        lab = NTHU_DROWSY.get(p.parent.name.lower())
        if lab is None:
            continue
        m = SUBJECT_IN_NAME.match(p.name)
        # only folders INSIDE the dataset count (a path like /private/var/folders/25/... must not become person "25")
        inner = p.relative_to(root).parts[:-1]
        sid = m.group(1) if m else next((q for q in reversed(inner) if q.isdigit() and len(q) <= 3), None)
        if sid is not None:
            with_id += 1
            subj = f"{source}_{int(sid):03d}"
        else:
            subj = _image_id(source, p, root)
        rows.append(_row(p, subj, source, drowsy=lab))
    df = pd.DataFrame(rows)
    frac = with_id / len(rows) if rows else 0.0
    if frac and df[df.subject.str.match(rf"^{source}_\d{{3}}$")].subject.nunique() < 3:
        frac = 0.0  # fewer than 3 distinct people: not real person IDs, can't split by person
    return df, frac


def folders(root: Path, mapping: dict[str, tuple[str, int]], source: str, subject_regex: str | None = None,
            case_insensitive: bool = False) -> pd.DataFrame:
    """Generic: folder name -> (task, label). Optional regex extracts subject from filename."""
    rows, rx = [], re.compile(subject_regex) if subject_regex else None
    if case_insensitive:
        mapping = {k.lower(): v for k, v in mapping.items()}
    for p in _images(root):
        name = p.parent.name.lower() if case_insensitive else p.parent.name
        if name not in mapping:
            continue
        task, label = mapping[name]
        m = rx.search(p.name) if rx else None
        subj = f"{source}_{m.group(1)}" if m else _image_id(source, p, root)
        rows.append(_row(p, subj, source, **{task: label}))
    return pd.DataFrame(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["mrl", "statefarm", "fer", "affectnet", "rafdb", "kmufed", "yawneye", "nthu", "folders"])
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
        df = affectnet(a.root)
    elif a.kind == "nthu":
        df, frac = nthu_frames(a.root)
        print(f"person IDs found for {frac:.0%} of images" + ("" if frac > 0.9 else "  <- WARNING: splits will leak people"))
    elif a.kind == "rafdb":
        df = emotion_folders(a.root, "rafdb", RAFDB_IDS)
    elif a.kind == "yawneye":
        df = folders(a.root, YAWN_EYE_MAP, a.source or "yawn", a.subject_regex, case_insensitive=True)
    elif a.kind in ("fer", "kmufed"):
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
