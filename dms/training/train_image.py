"""Train FaceStateNet (multi-task, partial labels) or CabinNet.

    # multi-task face model on eye + yawn + emotion data
    python -m dms.training.train_image --kind face \
        --manifest manifests/mrl.csv manifests/yawn.csv manifests/affectnet.csv --out runs/face

    # cabin behaviour model, driver-independent split
    python -m dms.training.train_image --kind cabin --manifest manifests/statefarm.csv \
        --backbone efficientnet_b0 --out runs/cabin
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..data.datasets import ManifestDataset, balanced_sampler, load_manifests, random_split, subject_split
from ..data.transforms import eval_transform, train_transform
from ..models import checkpoint
from ..models.networks import CABIN_CLASSES, EMOTION_CLASSES, EYE_CLASSES, YAWN_CLASSES, CabinNet, FaceStateNet
from .metrics import classification_metrics, plot_confusion

CLASSES = {"eye": EYE_CLASSES, "yawn": YAWN_CLASSES, "emotion": EMOTION_CLASSES, "cabin": CABIN_CLASSES}


def get_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["face", "cabin"], required=True)
    ap.add_argument("--manifest", nargs="+", required=True)
    ap.add_argument("--test-manifest", nargs="*", default=[], help="datasets used ONLY for testing (e.g. KMU-FED for in-car emotion)")
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--backbone", default=None, help="any timm model; default mobilenetv3_large_100 (face) / efficientnet_b0 (cabin)")
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=1.0, help="warmup epochs")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--no-balance", action="store_true")
    ap.add_argument("--split", choices=["subject", "random"], default="subject",
                    help="'random' ignores who is in the image - ONLY for the leakage study (E3)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=0, help="debug: stop each epoch after N steps")
    ap.add_argument("--init-ckpt", help="fine-tune: start from this checkpoint (e.g. a model trained on public data)")
    ap.add_argument("--freeze-epochs", type=int, default=0, help="fine-tune: train only the heads for the first N epochs")
    return ap.parse_args(argv)


@torch.no_grad()
def evaluate(model, loader, tasks, device, kind):
    model.eval()
    ys = {t: [] for t in tasks}
    ps = {t: [] for t in tasks}
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        out = model(x)
        out = out if kind == "face" else {"cabin": out}
        for t in tasks:
            m = y[t] >= 0
            if m.any():
                ps[t].append(torch.softmax(out[t].float(), -1)[m.to(device)].cpu().numpy())
                ys[t].append(y[t][m].numpy())
    return {t: classification_metrics(np.concatenate(ys[t]), np.concatenate(ps[t]), CLASSES[t]) for t in tasks if ys[t]}


def main(argv=None):
    a = get_args(argv)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    tasks = ("eye", "yawn", "emotion") if a.kind == "face" else ("cabin",)

    df = load_manifests(a.manifest)
    df = df[(df[list(tasks)] >= 0).any(axis=1)]
    if a.split == "random":
        print("WARNING: random image split - the same people are in train and test (leakage study only)")
        df = random_split(df, a.val_frac, a.test_frac, a.seed)
    else:
        df = subject_split(df, a.val_frac, a.test_frac, a.seed)
    if a.test_manifest:  # held-out datasets: never trained on, always test
        held = load_manifests(a.test_manifest)
        held = held[(held[list(tasks)] >= 0).any(axis=1)].assign(split="heldout")
        df = pd.concat([df, held], ignore_index=True)
    df.to_csv(out / "split.csv", index=False)
    present = tuple(t for t in tasks if (df[t] >= 0).any())
    print({s: int((df.split == s).sum()) for s in ("train", "val", "test", "heldout")}, "tasks:", present)

    # Horizontal flips would swap left/right-hand cabin classes (texting_right <-> texting_left).
    ttf = train_transform(a.img_size, flip=a.kind == "face")
    etf = eval_transform(a.img_size)
    tr_df = df[df.split == "train"]
    tr = ManifestDataset(tr_df, ttf, tasks)
    va = ManifestDataset(df[df.split == "val"], etf, tasks)
    te = ManifestDataset(df[df.split == "test"], etf, tasks)
    sampler = None if a.no_balance else balanced_sampler(tr_df, present)
    kw = dict(batch_size=a.bs, num_workers=a.workers, pin_memory=device == "cuda", persistent_workers=a.workers > 0)
    tr_dl = DataLoader(tr, sampler=sampler, shuffle=sampler is None, drop_last=True, **kw)
    va_dl, te_dl = DataLoader(va, **kw), DataLoader(te, **kw)
    held_df = df[df.split == "heldout"]
    held_dl = DataLoader(ManifestDataset(held_df, etf, tasks), **kw) if len(held_df) else None

    if a.init_ckpt and not a.backbone:  # fine-tuning: reuse the checkpoint's architecture
        a.backbone = torch.load(a.init_ckpt, map_location="cpu", weights_only=False)["arch"]["backbone"]
    backbone = a.backbone or ("mobilenetv3_large_100" if a.kind == "face" else "efficientnet_b0")
    pretrained = not a.no_pretrained and not a.init_ckpt  # no ImageNet download when weights come from a checkpoint
    if a.kind == "face":
        arch = {"backbone": backbone}
        model = FaceStateNet(backbone, pretrained=pretrained)
    else:
        arch = {"backbone": backbone}
        model = CabinNet(backbone, pretrained=pretrained)
    if a.init_ckpt:
        init, ck = checkpoint.load(a.init_ckpt)
        if ck["kind"] != a.kind:
            raise SystemExit(f"--init-ckpt is a {ck['kind']} model, expected {a.kind}")
        if ck["arch"].get("backbone") != backbone:
            raise SystemExit(f"--init-ckpt backbone {ck['arch'].get('backbone')} != --backbone {backbone}")
        model.load_state_dict(init.state_dict())
        print(f"fine-tuning from {a.init_ckpt}")
    model.to(device)
    backbone_params = list((model.backbone if a.kind == "face" else model.net).parameters())

    def set_backbone_trainable(flag: bool):
        for p_ in backbone_params:
            p_.requires_grad_(flag)
        if a.kind == "cabin":  # the classifier layer lives inside model.net; keep it trainable
            for p_ in model.net.get_classifier().parameters():
                p_.requires_grad_(True)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    steps_per_epoch = a.max_steps or len(tr_dl)
    total, warm = a.epochs * steps_per_epoch, int(a.warmup * steps_per_epoch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / max(warm, 1) if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(total - warm, 1))))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(enabled=use_amp)
    ce = torch.nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)

    best, history = -1.0, []
    for ep in range(a.epochs):
        set_backbone_trainable(ep >= a.freeze_epochs)
        model.train()
        t0, run_loss, n = time.time(), 0.0, 0
        for i, (x, y) in enumerate(tr_dl):
            if a.max_steps and i >= a.max_steps:
                break
            x = x.to(device, non_blocking=True)
            y = {k: v.to(device) for k, v in y.items()}
            with torch.autocast(device, enabled=use_amp):
                o = model(x)
                loss = model.loss(o, y, a.label_smoothing)[0] if a.kind == "face" else ce(o, y["cabin"])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += float(loss.detach())
            n += 1
        val = evaluate(model, va_dl, present, device, a.kind)
        score = float(np.mean([m["macro_f1"] for m in val.values()]))
        rec = {"epoch": ep, "loss": run_loss / max(n, 1), "val_macro_f1": score, **{f"val_{t}_f1": m["macro_f1"] for t, m in val.items()}, "time_s": time.time() - t0}
        history.append(rec)
        print(json.dumps(rec))
        if score > best:
            best = score
            checkpoint.save(out / "best.pt", model, a.kind, arch, epoch=ep, val=val, img_size=a.img_size)
        checkpoint.save(out / "last.pt", model, a.kind, arch, epoch=ep)

    model, _ = checkpoint.load(out / "best.pt")
    model.to(device)
    test = evaluate(model, te_dl, present, device, a.kind)
    for t, m in test.items():
        plot_confusion(m["confusion"], CLASSES[t], out / f"confusion_{t}.png", f"{t} (test, {'subject-independent' if a.split == 'subject' else 'RANDOM split'})")
        print(f"TEST {t}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} balAcc={m['balanced_accuracy']:.4f} ECE={m['ece']:.4f}")
    result = {"args": vars(a), "history": history, "test": test}
    if held_dl is not None:
        held_present = tuple(t for t in present if (held_df[t] >= 0).any())
        result["test_heldout"] = heldout = evaluate(model, held_dl, held_present, device, a.kind)
        for t, m in heldout.items():
            plot_confusion(m["confusion"], CLASSES[t], out / f"confusion_{t}_heldout.png", f"{t} (held-out dataset, never trained on)")
            print(f"HELD-OUT {t}: acc={m['accuracy']:.4f} macroF1={m['macro_f1']:.4f} (n={m['n']}, sources: {sorted(held_df.source.unique())})")
    (out / "metrics.json").write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
