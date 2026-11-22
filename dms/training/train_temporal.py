"""Train/evaluate the temporal drowsiness model with subject-independent CV.

    python -m dms.training.train_temporal features/rldd --model transformer --folds 5 --out runs/temporal
    python -m dms.training.train_temporal features/rldd --model gru         --folds 5 --out runs/temporal_gru
    python -m dms.training.train_temporal features/rldd --model rf          --folds 5 --out runs/temporal_rf

    # cross-dataset: train on everything in A, test on B
    python -m dms.training.train_temporal features/rldd --test-dir features/nthu --out runs/xdata

Reports window-level and video-level (majority vote) metrics per fold and
mean ± std, which is what reviewers will ask for.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import GroupKFold

from ..features.temporal import DriverProfile
from ..models import checkpoint
from ..models.networks import ALERTNESS_CLASSES, build_temporal, count_params
from .metrics import classification_metrics, plot_confusion
from ..data.sequences import normalise_sequence


def load_dir(d: Path) -> list[dict]:
    vids = []
    for p in sorted(Path(d).glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        prof = DriverProfile(*z["profile"][:4], int(z["profile"][4])) if "profile" in z else DriverProfile()
        vids.append({"x": normalise_sequence(z["raw"], prof).astype(np.float32), "y": int(z["label"]), "subject": str(z["subject"]), "name": p.stem})
    if not vids:
        raise SystemExit(f"no .npz sequences in {d}")
    return vids


def windows(x: np.ndarray, length: int, stride: int) -> np.ndarray:
    if len(x) < length:
        x = np.concatenate([x, np.zeros((length - len(x), x.shape[1]), x.dtype)])
    idx = range(0, len(x) - length + 1, stride)
    return np.stack([x[i : i + length] for i in idx])


def window_stats(W: np.ndarray) -> np.ndarray:
    """Hand-crafted baseline features per window (classic DMS literature)."""
    ear = W[..., 0]
    # openness < 0.2 (see EyeClosureTracker) == ear/ear_open < 0.3 + 0.2 * 0.7
    closed = (ear < 0.3 + 0.2 * 0.7).astype(np.float32)
    feats = [ear.mean(1), ear.std(1), ear.min(1), closed.mean(1), np.abs(np.diff(closed, axis=1)).sum(1)]
    for j in range(1, W.shape[2]):
        feats += [W[..., j].mean(1), W[..., j].std(1)]
    return np.stack(feats, 1)


def augment(x: torch.Tensor) -> torch.Tensor:
    x = x + 0.02 * torch.randn_like(x)
    x = x * (1 + 0.05 * torch.randn(x.shape[0], 1, x.shape[2]))  # per-feature gain (camera/person variation)
    mask = torch.rand(x.shape[0], x.shape[1], 1) < 0.05  # dropped frames
    return x.masked_fill(mask, 0.0)


def train_nn(train, a, device):
    model = build_temporal(a.model, in_dim=train[0]["x"].shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    rng = np.random.default_rng(a.seed)
    steps = a.epochs * a.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * s / steps)))
    model.train()
    for step in range(steps):
        # sample random windows uniformly over videos -> class-balanced (3 videos/subject)
        xb, yb = [], []
        for v in rng.choice(len(train), a.bs):
            x = train[v]["x"]
            s = rng.integers(0, max(len(x) - a.seq_len, 0) + 1)
            w = x[s : s + a.seq_len]
            if len(w) < a.seq_len:
                w = np.concatenate([w, np.zeros((a.seq_len - len(w), x.shape[1]), np.float32)])
            xb.append(w)
            yb.append(train[v]["y"])
        xb = augment(torch.from_numpy(np.stack(xb))).to(device)
        loss = ce(model(xb), torch.tensor(yb, device=device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if (step + 1) % a.steps_per_epoch == 0:
            print(f"    epoch {(step + 1) // a.steps_per_epoch}/{a.epochs} loss {float(loss):.4f}")
    return model.eval()


@torch.no_grad()
def predict_nn(model, vids, a, device):
    out = []
    for v in vids:
        W = torch.from_numpy(windows(v["x"], a.seq_len, a.eval_stride))
        P = torch.cat([torch.softmax(model(W[i : i + 256].to(device)), -1).cpu() for i in range(0, len(W), 256)]).numpy()
        out.append(P)
    return out


def evaluate(vids, probs) -> dict:
    yw = np.concatenate([np.full(len(p), v["y"]) for v, p in zip(vids, probs)])
    pw = np.concatenate(probs)
    yv = np.array([v["y"] for v in vids])
    pv = np.stack([p.mean(0) for p in probs])  # video-level: mean of window probs
    return {"window": classification_metrics(yw, pw, ALERTNESS_CLASSES), "video": classification_metrics(yv, pv, ALERTNESS_CLASSES)}


def run_fold(train, test, a, device):
    if a.model == "rf":
        from sklearn.ensemble import RandomForestClassifier

        Xtr = np.concatenate([window_stats(windows(v["x"], a.seq_len, a.eval_stride)) for v in train])
        ytr = np.concatenate([np.full(len(windows(v["x"], a.seq_len, a.eval_stride)), v["y"]) for v in train])
        rf = RandomForestClassifier(400, min_samples_leaf=5, n_jobs=-1, random_state=a.seed, class_weight="balanced").fit(Xtr, ytr)
        return None, [rf.predict_proba(window_stats(windows(v["x"], a.seq_len, a.eval_stride))) for v in test]
    model = train_nn(train, a, device)
    return model, predict_nn(model, test, a, device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("--test-dir", type=Path, help="cross-dataset test set (disables CV)")
    ap.add_argument("--model", choices=["transformer", "gru", "rf"], default="transformer")
    ap.add_argument("--out", default="runs/temporal")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seq-len", type=int, default=300, help="steps (300 = 30 s at 10 Hz)")
    ap.add_argument("--eval-stride", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fit-all", action="store_true", help="after CV, retrain on all data and save as deploy.pt")
    a = ap.parse_args(argv)
    torch.manual_seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    vids = load_dir(a.data)
    subjects = np.array([v["subject"] for v in vids])
    print(f"{len(vids)} videos, {len(set(subjects))} subjects, model={a.model}")

    results = []
    if a.test_dir:
        test = load_dir(a.test_dir)
        model, probs = run_fold(vids, test, a, device)
        results.append(evaluate(test, probs))
        folds = [("cross", model)]
    else:
        folds = []
        gkf = GroupKFold(n_splits=a.folds)
        for k, (tr, te) in enumerate(gkf.split(vids, groups=subjects)):
            assert not set(subjects[tr]) & set(subjects[te])
            print(f"fold {k}: train {len(tr)} videos / test {len(te)} videos ({len(set(subjects[te]))} unseen subjects)")
            model, probs = run_fold([vids[i] for i in tr], [vids[i] for i in te], a, device)
            r = evaluate([vids[i] for i in te], probs)
            results.append(r)
            folds.append((k, model))
            print(f"  window macroF1 {r['window']['macro_f1']:.4f}  video acc {r['video']['accuracy']:.4f}")

    summary = {}
    for level in ("window", "video"):
        for m in ("accuracy", "balanced_accuracy", "macro_f1", "kappa", "ece"):
            vals = [r[level][m] for r in results]
            summary[f"{level}_{m}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    cm = np.sum([r["window"]["confusion"] for r in results], axis=0)
    plot_confusion(cm, ALERTNESS_CLASSES, out / "confusion_window.png", f"{a.model} - subject-independent")
    if folds[0][1] is not None:  # save the first fold's model (or retrain on all data for deployment)
        checkpoint.save(out / "best.pt", folds[0][1], "temporal", {"kind": a.model, "in_dim": vids[0]["x"].shape[1]}, seq_len=a.seq_len)
        summary["params"] = count_params(folds[0][1])
    if a.fit_all and a.model != "rf":
        print("retraining on all subjects for deployment")
        final = train_nn(vids, a, device)
        checkpoint.save(out / "deploy.pt", final, "temporal", {"kind": a.model, "in_dim": vids[0]["x"].shape[1]}, seq_len=a.seq_len)
    (out / "metrics.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(a).items()}, "summary": summary, "folds": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
