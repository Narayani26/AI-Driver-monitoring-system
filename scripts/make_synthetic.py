"""Generate tiny synthetic datasets so the whole training stack can be smoke-tested
without downloading anything. NOT for reporting results.

    python scripts/make_synthetic.py /tmp/synth
    python -m dms.training.train_image --kind face --manifest /tmp/synth/face.csv --no-pretrained --epochs 1 --max-steps 3 --workers 0 --img-size 96 --out /tmp/synth/runs/face
    python -m dms.training.train_temporal /tmp/synth/seq --folds 3 --epochs 3 --steps-per-epoch 30 --out /tmp/synth/runs/temporal
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dms.features.temporal import DriverProfile
from dataclasses import asdict

root = Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic")
rng = np.random.default_rng(0)

# ---- images: 3 sources x 8 subjects, each image labelled for one task -------
rows = []
(root / "img").mkdir(parents=True, exist_ok=True)
for src, task, ncls in (("eyes", "eye", 2), ("yawns", "yawn", 2), ("faces", "emotion", 7)):
    for s in range(8):
        for k in range(6):
            y = int(rng.integers(ncls))
            img = np.full((64, 64, 3), 40 + 25 * y, np.uint8) + rng.integers(0, 20, (64, 64, 3), dtype=np.uint8)
            p = root / "img" / f"{src}_{s}_{k}.png"
            cv2.imwrite(str(p), img)
            r = {"path": str(p), "subject": f"{src}_{s}", "source": src, "split": "", "eye": -1, "yawn": -1, "emotion": -1, "cabin": -1}
            r[task] = y
            rows.append(r)
pd.DataFrame(rows).to_csv(root / "face.csv", index=False)

# ---- sequences: alert / low-vigilant / drowsy with increasing eye closure ----
(root / "seq").mkdir(exist_ok=True)
fps, secs = 10, 240
for s in range(12):
    ear_open = rng.uniform(0.24, 0.34)
    for label, (blink_p, blink_len, droop) in enumerate(((0.03, 2, 0.0), (0.05, 3, 0.08), (0.08, 6, 0.18))):
        T = fps * secs
        ear = np.full(T, ear_open) * (1 - droop) + rng.normal(0, 0.01, T)
        t = 0
        while t < T:
            if rng.random() < blink_p:
                L = int(rng.integers(1, blink_len + 1))
                ear[t : t + L] = 0.3 * ear_open
                t += L
            t += 1
        F = np.zeros((T, 9), np.float32)
        F[:, 0] = ear
        F[:, 1] = 0.05 + rng.normal(0, 0.01, T)
        F[:, 2] = rng.normal(5 * label, 3, T)  # head nods down more when drowsy
        F[:, 3] = rng.normal(0, 4, T)
        F[:, 5:7] = 0.5
        prof = DriverProfile(ear_open=ear_open, mar_rest=0.05)
        np.savez_compressed(root / "seq" / f"syn_{s:02d}_{label}.npz", raw=F, present=np.ones(T, bool), label=label, subject=f"syn_{s:02d}", fps=fps, profile=np.array(list(asdict(prof).values())))
print("wrote", root)
