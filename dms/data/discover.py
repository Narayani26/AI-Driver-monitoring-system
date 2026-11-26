"""Find every supported dataset under a folder (e.g. /kaggle/input or data/) by its
*contents*, not its name, and build the manifests.

Kaggle copies of the same dataset use different names and folder layouts, so
we look for tell-tale files instead:

    driver_imgs_list.csv                         -> State Farm (cabin behaviour)
    s0001_00001_*.png                            -> MRL Eye (eye state, 37 people)
    folders yawn / no_yawn (+ Open / Closed)     -> Yawn-Eye (yawn, eye state)
    folders named after emotions (angry, happy…) -> FER2013 / AffectNet / any emotion set
    folders 1..7 (RAF-DB label numbers)          -> RAF-DB
    folders 0..7 (AffectNet label numbers)       -> AffectNet
    images/<id>.jpg + annotations/<id>_exp.npy   -> official AffectNet / AffectNet+ layout
    a CSV with an image-file column + emotion    -> any emotion dataset labelled by a table
    'kmu' in the dataset name                    -> KMU-FED: in-car emotions, TEST ONLY (never trained on)
    folders drowsy / notdrowsy (NTHU-DDD style)  -> frame-level drowsiness, ONLY if person IDs are in the file
                                                    names (>= 90%); otherwise skipped (train/test would share people)

    python -m dms.data.discover /kaggle/input manifests        # prints what was found
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from . import manifests as M

SKIP_DIRS = {"competitions", "datasets", "notebooks", "models", "__MACOSX"}
EMOTION_NAMES = set(M.EMOTION_ALIASES)


@dataclass
class Found:
    kind: str  # statefarm | mrl | yawneye | emotion | rafdb | kmufed | skipped
    root: Path
    source: str
    note: str = ""


def dataset_roots(base: Path) -> list[Path]:
    """Kaggle mounts inputs as /kaggle/input/{datasets/<owner>,competitions}/<slug> (new) or /kaggle/input/<slug> (old)."""
    roots = []
    for p in sorted(base.iterdir()):
        if not p.is_dir():
            continue
        if p.name == "datasets":
            roots += [d for owner in sorted(p.iterdir()) if owner.is_dir() for d in sorted(owner.iterdir()) if d.is_dir()]
        elif p.name == "competitions":
            roots += [d for d in sorted(p.iterdir()) if d.is_dir()]
        elif p.name not in SKIP_DIRS:
            roots.append(p)
    return roots


def _dir_names(root: Path, limit: int = 20000) -> set[str]:
    names = set()
    for i, d in enumerate(root.rglob("*")):
        if i > limit:
            break
        if d.is_dir():
            names.add(d.name.lower())
    return names


def classify(root: Path) -> Found | None:
    slug = root.name.lower()
    source = "".join(c if c.isalnum() else "_" for c in slug).strip("_")[:30]
    if next(root.rglob("driver_imgs_list.csv"), None):
        return Found("statefarm", root, "statefarm")
    if next(root.rglob("s0001_*.png"), None):
        return Found("mrl", root, "mrl")
    names = _dir_names(root)
    if "kmu" in slug:
        return Found("kmufed", root, "kmufed", "test only")
    if "drowsy" in names and ({"notdrowsy", "not_drowsy", "not drowsy", "nondrowsy", "non_drowsy", "non drowsy"} & names):
        # NTHU-DDD-style frames; used only if person IDs can be recovered from file names (checked in build())
        return Found("drowsyframes", root, "nthu" if "nthu" in slug else "drowsy_" + source)
    if "yawn" in names and ({"no_yawn", "no yawn", "noyawn"} & names):
        return Found("yawneye", root, "yawn_" + source)
    if len(EMOTION_NAMES & names) >= 4:
        src = "fer" if "fer" in slug else "affectnet" if "affect" in slug else "emo_" + source
        return Found("emotion", root, src)
    if {str(i) for i in range(1, 8)} <= names and "0" not in names:
        return Found("rafdb", root, "rafdb")
    if {str(i) for i in range(0, 7)} <= names:  # AffectNet numbering 0=neutral ... 6=anger (7=contempt dropped)
        return Found("affectnet_ids", root, "affectnet")
    if next(root.rglob("*_exp.npy"), None):  # official AffectNet / AffectNet+ layout
        return Found("affectnet_npy", root, "affectnet")
    if next(root.rglob("*.csv"), None) and next(iter(M._images(root)), None):
        src = "affectnet" if "affect" in slug else "emo_" + source
        return Found("emotion_table", root, src, "labels read from a CSV table")
    return None


LABEL_COLS = ("expression", "human-label", "human_label", "hard_label", "label", "labels", "emotion", "class", "category")
PATH_HINTS = ("path", "file", "image", "img", "name", "subdirectory")
AFFECTNET_NUM = {i: c for i, c in enumerate(M.EMOTION_CLASSES)}  # 0 neutral … 6 anger; 7 contempt / 8-10 none, uncertain, no-face dropped


def _split_of(p: Path) -> str:
    parts = {q.name.lower() for q in p.parents}
    for key, split in (("train", "train"), ("training", "train"), ("val", "val"), ("valid", "val"), ("validation", "val"), ("test", "test")):
        if any(x == key or x.startswith(key + "_") or x.endswith("_" + key) for x in parts):
            return split
    return ""


def _label_to_idx(v, numeric_map: dict) -> int | None:
    if isinstance(v, str) and not v.strip().lstrip("-").isdigit():
        name = M.EMOTION_ALIASES.get(v.strip().lower())
        return M.EMOTION_CLASSES.index(name) if name else None
    try:
        name = numeric_map.get(int(float(v)))
    except (TypeError, ValueError):
        return None
    return M.EMOTION_CLASSES.index(name) if name else None


def npy_annotations(root: Path) -> pd.DataFrame:
    """Official AffectNet layout: <split>/images/<id>.jpg + <split>/annotations/<id>_exp.npy (label number)."""
    import numpy as np

    rows = []
    for exp in root.rglob("*_exp.npy"):
        stem = exp.name[: -len("_exp.npy")]
        img_dir = exp.parent.parent / "images"
        img = next((img_dir / f"{stem}{ext}" for ext in (".jpg", ".jpeg", ".png") if (img_dir / f"{stem}{ext}").exists()), None)
        if img is None:
            continue
        idx = _label_to_idx(np.load(exp).item(), AFFECTNET_NUM)
        if idx is not None:
            rows.append(M._row(img, M._image_id("affectnet", img, root), "affectnet", _split_of(img), emotion=idx))
    return pd.DataFrame(rows)


def table_annotations(root: Path, source: str) -> pd.DataFrame:
    """A CSV listing image files and an emotion label (names or AffectNet numbers)."""
    images = {}
    for p in M._images(root):
        images.setdefault(p.name, p)
    rows = []
    for csv in sorted(root.rglob("*.csv")):
        if csv.stat().st_size > 300_000_000:
            continue
        try:
            df = pd.read_csv(csv, low_memory=False)
        except Exception:
            continue
        cols = {c.lower(): c for c in df.columns}
        lab = next((cols[c] for c in LABEL_COLS if c in cols), None)
        path = next((c for c in df.columns if any(h in c.lower() for h in PATH_HINTS) and not pd.api.types.is_numeric_dtype(df[c])), None)
        if lab is None or path is None:
            continue
        num_map = AFFECTNET_NUM if ("affect" in source or (pd.to_numeric(df[lab], errors="coerce") == 0).any()) else \
            {int(k): v for k, v in M.RAFDB_IDS.items()}
        for fname, lv in zip(df[path].astype(str), df[lab]):
            img = images.get(Path(fname).name)
            idx = _label_to_idx(lv, num_map)
            if img is not None and idx is not None:
                rows.append(M._row(img, M._image_id(source, img, root), source, _split_of(img), emotion=idx))
        if rows:
            break
    return pd.DataFrame(rows)


def tree(root: Path, depth: int = 3, per_dir: int = 6) -> str:
    """Short folder listing, to paste when a dataset isn't recognised."""
    lines = []

    def walk(d: Path, level: int):
        entries = sorted(d.iterdir())
        for e in entries[:per_dir]:
            lines.append("    " * level + e.name + ("/" if e.is_dir() else ""))
            if e.is_dir() and level + 1 < depth:
                walk(e, level + 1)
        if len(entries) > per_dir:
            lines.append("    " * level + f"... ({len(entries)} entries)")

    walk(root, 0)
    return "\n".join(lines)


def build(found: Found) -> pd.DataFrame:
    if found.kind == "statefarm":
        return M.statefarm(found.root)
    if found.kind == "mrl":
        return M.mrl_eye(found.root)
    if found.kind == "yawneye":
        return M.folders(found.root, M.YAWN_EYE_MAP, found.source, case_insensitive=True)
    if found.kind == "emotion":
        return M.emotion_folders(found.root, found.source)
    if found.kind == "affectnet_ids":
        return M.emotion_folders(found.root, "affectnet", M.AFFECTNET_IDS)
    if found.kind == "rafdb":
        return M.emotion_folders(found.root, "rafdb", M.RAFDB_IDS)
    if found.kind == "affectnet_npy":
        return npy_annotations(found.root)
    if found.kind == "emotion_table":
        if found.source.startswith("affectnet"):  # AffectNet CSVs: also face boxes + AffectNet+ soft labels
            df = M.affectnet(found.root, found.source)
            if not df.empty:
                return df
        return table_annotations(found.root, found.source)
    if found.kind == "drowsyframes":
        df, frac = M.nthu_frames(found.root, found.source)
        found.note = f"person IDs for {frac:.0%} of frames"
        return df if frac >= 0.9 else df.iloc[0:0]
    if found.kind == "kmufed":
        df = M.emotion_folders(found.root, "kmufed")
        return df.assign(split="test")
    return pd.DataFrame()


def discover(base: str | Path, out_dir: str | Path) -> dict:
    """Returns {"face": [...], "face_test": [...], "cabin": [...], "found": [Found, ...]} with manifest paths."""
    base, out_dir = Path(base), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"face": [], "face_test": [], "cabin": [], "drowsy": [], "found": []}
    used_sources: set[str] = set()
    for root in dataset_roots(base):
        if root.resolve() == out_dir.resolve():  # our own output folder, not a dataset
            continue
        f = classify(root)
        if f is None:
            print(f"  ?  {root}  (not recognised, ignored). Its folders - paste this to get it supported:\n" + tree(root))
            continue
        res["found"].append(f)
        if f.kind == "skipped":
            print(f"  -  {root}  -> skipped: {f.note}")
            continue
        if f.source in used_sources:  # two copies of the same dataset: use the first only
            print(f"  -  {root}  -> duplicate of {f.source}, ignored")
            continue
        try:
            df = build(f)
        except Exception as e:  # a broken/unexpected copy must not stop the whole run
            print(f"  !  {root}  -> {f.kind}: failed ({type(e).__name__}: {e})")
            continue
        if df.empty:
            if f.kind == "drowsyframes":
                print(f"  !  {root}  -> skipped ({f.note}): without person IDs, train and test would share people "
                      "and the score would be inflated. Its folders:\n" + tree(root))
            else:
                print(f"  !  {root}  -> {f.kind}: no labelled images found. Its folders:\n" + tree(root))
            continue
        if f.source.startswith("affectnet") and "split" in df:
            # AffectNet has no public test set; its official validation set is what papers report on.
            # Use it as our test set; our validation split is carved from the training images.
            df["split"] = df["split"].replace({"val": "test"})
        used_sources.add(f.source)
        path = out_dir / f"{f.source}.csv"
        df.to_csv(path, index=False)
        tasks = {c: int((df[c] >= 0).sum()) for c in M.TASK_COLS if (df[c] >= 0).any()}
        print(f"  ✓  {root}  -> {f.kind:9s} {len(df):7d} images, {df.subject.nunique():6d} groups, labels {tasks}"
              + (f"  [{f.note}]" if f.note else ""))
        key = {"statefarm": "cabin", "kmufed": "face_test", "drowsyframes": "drowsy"}.get(f.kind, "face")
        res[key].append(str(path))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("base", type=Path, help="folder containing the datasets (e.g. /kaggle/input or data)")
    ap.add_argument("out", type=Path, help="where to write manifests")
    a = ap.parse_args(argv)
    if not a.base.is_dir():
        sys.exit(f"{a.base} is not a folder")
    r = discover(a.base, a.out)
    print("\nface training :", r["face"], "\nface test only:", r["face_test"], "\ncabin         :", r["cabin"], "\ndrowsy frames :", r["drowsy"])


if __name__ == "__main__":
    main()
