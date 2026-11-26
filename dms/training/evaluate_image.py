"""Evaluate a trained image model on extra labelled data it never saw (e.g. AffectNet+ validation,
KMU-FED, your own crops) - no retraining needed.

    python -m dms.training.evaluate_image runs/face/best.pt manifests/affectnet_plus.csv
    python -m dms.training.evaluate_image runs/drowsy/best.pt manifests/nthu.csv --name nthu_all
    python -m dms.training.evaluate_image hsemotion manifests/affectnet_plus.csv   # the pretrained demo model, for comparison

Writes <checkpoint folder>/eval_<name>.json (shown in the web app's results table) and a confusion matrix PNG.
Uses the checkpoint's calibration (temperature) and its test-time augmentation setting.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ..data.datasets import ManifestDataset, load_manifests
from ..data.transforms import eval_transform
from ..models import checkpoint
from .metrics import plot_confusion
from .train_image import CLASSES, SINGLE_TASK, collect_logits, metrics_from_logits


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpt", type=Path, help="checkpoint, or 'hsemotion' for the pretrained emotion model used by the demo")
    ap.add_argument("manifest", nargs="+")
    ap.add_argument("--name", help="label for the result file (default: manifest file name)")
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args(argv)

    if str(a.ckpt) == "hsemotion":
        return eval_hsemotion(a)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, ck = checkpoint.load(a.ckpt)
    model.to(device)
    kind = ck["kind"]
    tasks = tuple(model.tasks) if kind == "face" else (SINGLE_TASK[kind],)
    df = load_manifests(a.manifest)
    present = tuple(t for t in tasks if (df[t] >= 0).any())
    if not present:
        raise SystemExit(f"the manifest has no labels for this model's tasks {tasks}")
    df = df[(df[list(present)] >= 0).any(axis=1)]
    dl = DataLoader(ManifestDataset(df, eval_transform(ck.get("img_size", 224)), present), batch_size=a.bs, num_workers=a.workers)
    z = collect_logits(model, dl, present, device, kind, bool(ck.get("tta")))
    res = metrics_from_logits(z, ck.get("temperature") or {})
    name = a.name or Path(a.manifest[0]).stem
    out = a.ckpt.parent
    for t, m in res.items():
        plot_confusion(m["confusion"], CLASSES[t], out / f"confusion_{t}_{name}.png", f"{t} on {name} (never trained on)")
        print(f"{name} {t}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} balAcc={m['balanced_accuracy']:.4f} ECE={m['ece']:.4f} n={m['n']}")
    (out / f"eval_{name}.json").write_text(json.dumps({"checkpoint": str(a.ckpt), "data": a.manifest, "test_heldout": res}, indent=2))
    print(f"saved {out / f'eval_{name}.json'}")


def eval_hsemotion(a):
    """Pretrained HSEmotion (AffectNet-trained) on the same data: the baseline to beat."""
    import cv2
    import numpy as np

    from ..models.networks import EMOTION_CLASSES
    from ..perception.emotion import HSEmotionONNX
    from .metrics import classification_metrics

    df = load_manifests(a.manifest)
    df = df[df.emotion >= 0]
    if df.empty:
        raise SystemExit("no emotion labels in the manifest")
    model, probs, ys = HSEmotionONNX(), [], []
    for r in df.itertuples():
        img = cv2.imread(r.path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        probs.append(model(img))
        ys.append(int(r.emotion))
    m = classification_metrics(np.array(ys), np.stack(probs), EMOTION_CLASSES)
    name = a.name or Path(a.manifest[0]).stem
    out = Path(__file__).resolve().parents[2] / "runs" / "hsemotion_pretrained"
    out.mkdir(parents=True, exist_ok=True)
    plot_confusion(m["confusion"], EMOTION_CLASSES, out / f"confusion_emotion_{name}.png", f"pretrained HSEmotion on {name}")
    (out / f"eval_{name}.json").write_text(json.dumps({"checkpoint": "HSEmotion EfficientNet-B0 (pretrained)", "data": a.manifest,
                                                        "test_heldout": {"emotion": m}}, indent=2))
    print(f"{name} emotion (pretrained HSEmotion): acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} n={m['n']}")
    print(f"saved {out / f'eval_{name}.json'}")


if __name__ == "__main__":
    main()
