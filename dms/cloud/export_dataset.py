"""Export accepted real-world training windows for retraining the temporal model.

    python -m dms.cloud.export_dataset features/fleet
    python -m dms.training.train_temporal features/fleet --model hybrid --binary --folds 5
    # or: test the current model on real drivers it never saw
    python -m dms.training.train_temporal features/rldd --test-dir features/fleet/test ...

Writes the same .npz format as the UTA-RLDD extractor (raw, present, label, subject,
fps, profile), so the existing training code needs no change. Drivers are split by
person (a hash of the account id: no names or emails leave the database): the same
driver is never in both train and test. Weak labels (from alerts nobody disputed)
are kept out of the test set so test numbers stay trustworthy.
"""
from __future__ import annotations

import argparse
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
from sqlalchemy import select

from .db import Sample, make_sessionmaker


def pseudonym(user_id: str) -> str:
    return "drv_" + hashlib.sha256(("dms-export:" + user_id).encode()).hexdigest()[:10]


def split_of(subject: str, test_frac: float) -> str:
    h = int(hashlib.sha256(subject.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_frac else "train"


def export(out: Path, db_url: str | None = None, test_frac: float = 0.2) -> Counter:
    s = make_sessionmaker(db_url)()
    counts: Counter = Counter()
    for sm in s.scalars(select(Sample).where(Sample.accepted.is_(True))):
        subj = pseudonym(sm.user_id)
        part = split_of(subj, test_frac)
        if part == "test" and sm.label_source == "confirmed":
            continue  # weak label: training only
        d = sm.data
        raw = np.array([[np.nan if v is None else v for v in r] for r in d["raw"]], dtype=np.float32)
        dst = out / part / f"{subj}_{sm.id[:12]}_{sm.label}.npz"
        dst.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dst, raw=raw, present=np.array(d["present"], dtype=bool), label=sm.label, subject=subj,
                            fps=float(d["fps"]), profile=np.array(d["profile"], dtype=np.float64), source=f"fleet:{sm.label_source}")
        counts[(part, sm.label)] += 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", type=Path)
    ap.add_argument("--test-frac", type=float, default=0.2, help="share of *drivers* held out for testing")
    a = ap.parse_args(argv)
    c = export(a.out, test_frac=a.test_frac)
    for (part, label), n in sorted(c.items()):
        print(f"  {part:5s} label {label}: {n} windows")
    print(f"exported {sum(c.values())} windows to {a.out}/train and {a.out}/test")


if __name__ == "__main__":
    main()
