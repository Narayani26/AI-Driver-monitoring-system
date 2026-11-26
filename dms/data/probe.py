"""Print the real structure of every dataset under a folder, so unknown layouts
can be seen at a glance (folders, file types, counts, example names, label files).

    python -m dms.data.probe /kaggle/input
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from .discover import dataset_roots

LABEL_EXT = {".csv", ".json", ".txt", ".npy", ".xlsx", ".mat"}


def probe(root: Path, max_files: int = 300000) -> str:
    exts, dirs, examples, label_files = Counter(), Counter(), {}, []
    n = 0
    for p in root.rglob("*"):
        n += 1
        if n > max_files:
            break
        rel = p.relative_to(root)
        if p.is_dir():
            if len(rel.parts) <= 4:
                dirs[str(rel)] += 0
            continue
        ext = p.suffix.lower()
        exts[ext] += 1
        if len(rel.parts) > 1:
            dirs[str(rel.parent)] += 1
        examples.setdefault(ext, [])
        if len(examples[ext]) < 4:
            examples[ext].append(str(rel))
        if ext in LABEL_EXT and len(label_files) < 8 and not p.name.endswith("_lnd.npy"):
            label_files.append(p)
    out = [f"== {root}  ({sum(exts.values())} files{'+' if n > max_files else ''})",
           "   file types: " + ", ".join(f"{e or '(none)'} x{c}" for e, c in exts.most_common(8))]
    top = sorted(dirs.items(), key=lambda kv: (kv[0].count("/"), kv[0]))[:25]
    out.append("   folders (files directly inside):")
    out += [f"     {d}/  ({c})" for d, c in top]
    for e, ex in examples.items():
        out.append(f"   e.g. {e}: " + " | ".join(ex))
    for f in label_files[:4]:
        if f.suffix.lower() == ".csv":
            try:
                with open(f, newline="", errors="replace") as fh:
                    rows = [next(csv.reader(fh)) for _ in range(3)]
                out.append(f"   {f.relative_to(root)} -> header {rows[0][:12]} | first row {rows[1][:12]}")
            except Exception as e:  # noqa: BLE001 - probing must never crash
                out.append(f"   {f.relative_to(root)} (unreadable: {e})")
        elif f.suffix.lower() == ".json":
            try:
                out.append(f"   {f.relative_to(root)} -> {f.read_text(errors='replace')[:300]!r}")
            except OSError as e:
                out.append(f"   {f.relative_to(root)} (unreadable: {e})")
        elif f.suffix.lower() == ".npy":
            try:
                import numpy as np

                out.append(f"   {f.relative_to(root)} -> {np.load(f, allow_pickle=True)!r}"[:200])
            except Exception as e:  # noqa: BLE001
                out.append(f"   {f.relative_to(root)} (unreadable: {e})")
        else:
            out.append(f"   label-like file: {f.relative_to(root)}")
    return "\n".join(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("base", type=Path)
    a = ap.parse_args(argv)
    for r in dataset_roots(a.base):
        print(probe(r) + "\n")


if __name__ == "__main__":
    main()
