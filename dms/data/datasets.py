"""Datasets + leakage-free splitting."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Dataset, WeightedRandomSampler

from .manifests import TASK_COLS
from .transforms import bgr_to_rgb3


def load_manifests(paths: list[str]) -> pd.DataFrame:
    df = pd.concat([pd.read_csv(p, dtype={"subject": str, "split": str}) for p in paths], ignore_index=True)
    df["split"] = df["split"].fillna("")
    for c in TASK_COLS:  # manifests written before a task existed
        df[c] = df[c].fillna(-1).astype(int) if c in df else -1
    return df


def subject_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Fill blank `split` values so that no subject appears in two splits.

    Rows that already have a split (e.g. FER2013's official test set) keep it.
    Splitting is done per `source` so each dataset is represented in every split.
    """
    df = df.copy()
    for _, g in df[df.split == ""].groupby("source"):
        subjects = g.subject.values
        idx = np.arange(len(g))
        if len(np.unique(subjects)) < 3:
            raise ValueError(f"source {g.source.iloc[0]!r} has <3 subjects; cannot split by subject")
        gss = GroupShuffleSplit(1, test_size=test_frac, random_state=seed)
        trval, te = next(gss.split(idx, groups=subjects))
        gss2 = GroupShuffleSplit(1, test_size=val_frac / (1 - test_frac), random_state=seed)
        tr, va = next(gss2.split(trval, groups=subjects[trval]))
        lab = np.empty(len(g), dtype=object)
        lab[trval[tr]], lab[trval[va]], lab[te] = "train", "val", "test"
        df.loc[g.index, "split"] = lab
    # Sources with an official train/test but no val: carve val out of train by subject.
    for src, g in df.groupby("source"):
        if (g.split == "val").any() or not (g.split == "train").any():
            continue
        tr = g[g.split == "train"]
        gss = GroupShuffleSplit(1, test_size=val_frac, random_state=seed)
        _, va = next(gss.split(np.arange(len(tr)), groups=tr.subject.values))
        df.loc[tr.index[va], "split"] = "val"
    assert_no_leakage(df)
    return df


def random_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> pd.DataFrame:
    """Image-level random split that IGNORES subjects. Only for the leakage study
    (paper experiment E3): the same driver appears in train and test, which is
    how many published papers get ~99%. Never use it for reported results."""
    df = df.copy()
    rng = np.random.default_rng(seed)
    blank = df.index[df.split == ""]
    u = rng.random(len(blank))
    df.loc[blank, "split"] = np.where(u < test_frac, "test", np.where(u < test_frac + val_frac, "val", "train"))
    return df


def assert_no_leakage(df: pd.DataFrame) -> None:
    per = df.groupby("subject").split.nunique()
    bad = per[per > 1]
    if len(bad):
        raise AssertionError(f"{len(bad)} subjects appear in multiple splits, e.g. {list(bad.index[:5])}")


class ManifestDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform, tasks: tuple[str, ...] = TASK_COLS):
        self.df = df.reset_index(drop=True)
        self.tf = transform
        self.tasks = tasks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = cv2.imread(r.path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(r.path)
        if "bw" in r and r.bw == r.bw and r.bw > 0:  # face box given (AffectNet full images): crop with 15% margin
            m = 0.15 * max(r.bw, r.bh)
            x0, y0 = int(max(r.bx - m, 0)), int(max(r.by - m, 0))
            x1, y1 = int(min(r.bx + r.bw + m, img.shape[1])), int(min(r.by + r.bh + m, img.shape[0]))
            if x1 - x0 > 8 and y1 - y0 > 8:
                img = img[y0:y1, x0:x1]
        x = self.tf(bgr_to_rgb3(img))
        return x, {t: torch.tensor(int(r[t])) for t in self.tasks}


def balanced_sampler(df: pd.DataFrame, tasks: tuple[str, ...]) -> WeightedRandomSampler:
    """Each (source, task, class) bucket gets equal total weight, so a 80k-image
    eye dataset doesn't drown a 1k-image driver-emotion dataset."""
    keys = []
    for _, r in df.iterrows():
        k = next(((r.source, t, int(r[t])) for t in tasks if r[t] >= 0), (r.source, "none", -1))
        keys.append(k)
    keys = pd.Series(keys)
    w = 1.0 / keys.map(keys.value_counts()).values
    return WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double), num_samples=len(df), replacement=True)
