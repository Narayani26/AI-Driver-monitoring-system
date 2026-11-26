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
    'kmu' in the dataset name                    -> KMU-FED: in-car emotions, TEST ONLY (never trained on)
    folders 'Drowsy' / 'Non Drowsy'              -> skipped: frames cut from videos with no person IDs
                                                    (train/test would share people -> inflated scores)

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
    if {"drowsy", "non drowsy"} <= names or {"drowsy", "non_drowsy"} <= names:
        return Found("skipped", root, source, "no person IDs -> leaky splits; not used for training")
    if "yawn" in names and ({"no_yawn", "no yawn", "noyawn"} & names):
        return Found("yawneye", root, "yawn_" + source)
    if len(EMOTION_NAMES & names) >= 4:
        src = "fer" if "fer" in slug else "affectnet" if "affect" in slug else "emo_" + source
        return Found("emotion", root, src)
    if {str(i) for i in range(1, 8)} <= names and "0" not in names:
        return Found("rafdb", root, "rafdb")
    if {str(i) for i in range(0, 7)} <= names:  # AffectNet numbering 0=neutral ... 6=anger (7=contempt dropped)
        return Found("affectnet_ids", root, "affectnet")
    return None


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
    if found.kind == "kmufed":
        df = M.emotion_folders(found.root, "kmufed")
        return df.assign(split="test")
    return pd.DataFrame()


def discover(base: str | Path, out_dir: str | Path) -> dict:
    """Returns {"face": [...], "face_test": [...], "cabin": [...], "found": [Found, ...]} with manifest paths."""
    base, out_dir = Path(base), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = {"face": [], "face_test": [], "cabin": [], "found": []}
    used_sources: set[str] = set()
    for root in dataset_roots(base):
        f = classify(root)
        if f is None:
            print(f"  ?  {root}  (not recognised, ignored)")
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
            print(f"  !  {root}  -> {f.kind}: no images found")
            continue
        used_sources.add(f.source)
        path = out_dir / f"{f.source}.csv"
        df.to_csv(path, index=False)
        tasks = {c: int((df[c] >= 0).sum()) for c in M.TASK_COLS if (df[c] >= 0).any()}
        print(f"  ✓  {root}  -> {f.kind:9s} {len(df):7d} images, {df.subject.nunique():6d} groups, labels {tasks}"
              + (f"  [{f.note}]" if f.note else ""))
        key = "cabin" if f.kind == "statefarm" else "face_test" if f.kind == "kmufed" else "face"
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
    print("\nface training :", r["face"], "\nface test only:", r["face_test"], "\ncabin         :", r["cabin"])


if __name__ == "__main__":
    main()
