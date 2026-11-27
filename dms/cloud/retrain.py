"""Continuous-improvement pipeline: real drives -> better drowsiness model -> devices.

    python -m dms.cloud.retrain --base features/rldd --current runs_temporal2/hybrid_binary/deploy.pt

Steps (each is logged and the outcome is stored in the `model_versions` table):
 1. Export consented, quality-checked windows from the database, split BY DRIVER into train/test
    (no person in both), see dms.cloud.export_dataset.
 2. Enough data? (`--min-drivers` real drivers to train with, `--min-test-drivers` to test on, both classes present.)
    If not, stop: a model judged on two people is not evidence.
 3. Train a candidate on public data (UTA-RLDD) + the real-drive training windows.
 4. Test the candidate AND the model currently in use on the held-out real drivers.
 5. Approve only if the candidate is at least as good (macro-F1) AND does not raise the false-alarm rate
    (alert windows predicted drowsy) by more than `--max-fa-increase`, AND clears an absolute floor.
    Otherwise it is stored as `rejected`, with its numbers, for the audit trail.
 6. Approved checkpoints are copied to the model store with their SHA-256; devices fetch them with
    `python -m dms.cloud.fetch_model` and verify the hash before use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sqlalchemy import func, select

from .db import ModelVersion, make_sessionmaker
from .export_dataset import export


def model_dir() -> Path:
    d = Path(os.environ.get("DMS_MODEL_DIR", Path(os.environ.get("DMS_CLOUD_DIR", Path.home() / ".dms")) / "models"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def evaluate_checkpoint(ckpt: Path, test_dir: Path) -> dict:
    """Window/video metrics + false-alarm rate of a binary temporal checkpoint on a feature folder."""
    from ..models import checkpoint
    from ..training.train_temporal import BINARY_CLASSES, evaluate, load_dir, predict_nn

    model, ck = checkpoint.load(ckpt)
    vids = load_dir(test_dir, binary=True)
    probs = predict_nn(model, vids, SimpleNamespace(seq_len=ck.get("seq_len", 300), eval_stride=50), "cpu")
    if probs and probs[0].shape[1] == 3:  # a 3-class model: alert vs (low-vigilant + drowsy)
        probs = [np.stack([p[:, 0], p[:, 1] + p[:, 2]], 1) for p in probs]
    r = evaluate(vids, probs, BINARY_CLASSES)
    cm = np.array(r["window"]["confusion"], dtype=float)
    fa = float(cm[0, 1] / cm[0].sum()) if cm[0].sum() else 0.0
    return {"window_macro_f1": r["window"]["macro_f1"], "window_accuracy": r["window"]["accuracy"],
            "video_accuracy": r["video"]["accuracy"], "false_alarm_rate": fa, "n_windows": int(cm.sum())}


def drivers(folder: Path) -> tuple[int, set]:
    files = list(folder.glob("*.npz")) if folder.exists() else []
    subj = {f.name.split("_")[0] + "_" + f.name.split("_")[1] for f in files}
    labels = {int(f.stem.rsplit("_", 1)[1]) for f in files}
    return len(subj), labels


def decide(cand: dict, cur: dict | None, min_f1: float, max_fa_increase: float) -> tuple[bool, str]:
    if cand["window_macro_f1"] < min_f1:
        return False, f"macro-F1 {cand['window_macro_f1']:.3f} below the floor {min_f1}"
    if cur is None:
        return True, "first model trained with real-drive data"
    if cand["window_macro_f1"] + 1e-9 < cur["window_macro_f1"]:
        return False, f"worse than the current model ({cand['window_macro_f1']:.3f} < {cur['window_macro_f1']:.3f})"
    if cand["false_alarm_rate"] > cur["false_alarm_rate"] + max_fa_increase:
        return False, f"more false alarms ({cand['false_alarm_rate']:.1%} vs {cur['false_alarm_rate']:.1%})"
    return True, f"macro-F1 {cur['window_macro_f1']:.3f} -> {cand['window_macro_f1']:.3f}, false alarms {cur['false_alarm_rate']:.1%} -> {cand['false_alarm_rate']:.1%}"


def run(a) -> dict:
    from ..training import train_temporal

    Session = make_sessionmaker(a.db_url)
    work = Path(a.work or tempfile.mkdtemp(prefix="dms_retrain_"))
    fleet = work / "fleet"
    if fleet.exists():
        shutil.rmtree(fleet)
    counts = export(fleet, a.db_url, test_frac=a.test_frac)
    n_tr, lab_tr = drivers(fleet / "train")
    n_te, lab_te = drivers(fleet / "test")
    data = {"fleet_windows": {f"{p}_{l}": n for (p, l), n in counts.items()}, "fleet_train_drivers": n_tr, "fleet_test_drivers": n_te,
            "base": str(a.base) if a.base else None}
    print(f"real-drive data: {n_tr} training drivers, {n_te} test drivers, windows {dict(counts)}")
    if n_tr < a.min_drivers or n_te < a.min_test_drivers or not {0, 2} <= lab_te:
        msg = (f"not enough real-drive data yet (need >= {a.min_drivers} training drivers, >= {a.min_test_drivers} test drivers "
               f"and both alert and drowsy windows in the test set); nothing trained")
        print(msg)
        return {"status": "skipped", "reason": msg, "data": data}

    # 3. candidate = public dataset + real-drive training windows
    train_dir = work / "train_all"
    if train_dir.exists():
        shutil.rmtree(train_dir)
    train_dir.mkdir(parents=True)
    for src in ([Path(a.base)] if a.base else []) + [fleet / "train"]:
        for f in src.glob("*.npz"):
            (train_dir / f"{src.name}_{f.name}").symlink_to(f.resolve())
    out = work / "candidate"
    train_temporal.main([str(train_dir), "--test-dir", str(fleet / "test"), "--model", a.model, "--binary", "--drop", "0.3",
                         "--epochs", str(a.epochs), "--steps-per-epoch", str(a.steps_per_epoch), "--out", str(out)])
    cand = evaluate_checkpoint(out / "best.pt", fleet / "test")
    current = a.current
    if current is None:  # default: the model the registry currently hands out to devices
        with Session() as s0:
            m = s0.scalar(select(ModelVersion).where(ModelVersion.kind == "temporal", ModelVersion.status == "approved")
                          .order_by(ModelVersion.version.desc()))
            current = Path(m.path) if m is not None and Path(m.path).is_file() else None
    cur = evaluate_checkpoint(Path(current), fleet / "test") if current else None
    ok, why = decide(cand, cur, a.min_f1, a.max_fa_increase)
    print(f"candidate {json.dumps(cand)}\ncurrent   {json.dumps(cur)}\n-> {'APPROVED' if ok else 'REJECTED'}: {why}")

    s = Session()
    version = (s.scalar(select(func.max(ModelVersion.version)).where(ModelVersion.kind == "temporal")) or 0) + 1
    dst = model_dir() / f"temporal_v{version}.pt"
    shutil.copy2(out / "best.pt", dst)
    if ok:  # the previously approved model is retired (kept for rollback)
        for mv in s.scalars(select(ModelVersion).where(ModelVersion.kind == "temporal", ModelVersion.status == "approved")):
            mv.status = "retired"
    mv = ModelVersion(kind="temporal", version=version, status="approved" if ok else "rejected", path=str(dst), sha256=sha256(dst),
                      metrics={"candidate": cand, "current": cur, "decision": why, "test": "held-out real drivers"}, data=data,
                      notes=f"trained {time.strftime('%Y-%m-%d %H:%M')} model={a.model}")
    s.add(mv)
    s.commit()
    print(f"registered temporal v{version} ({mv.status}) -> {dst}")
    return {"status": mv.status, "version": version, "reason": why, "metrics": mv.metrics, "data": data}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, help="public training features to keep (e.g. features/rldd)")
    ap.add_argument("--current", type=Path, help="checkpoint in use now (default: the latest approved model in the registry)")
    ap.add_argument("--model", default="hybrid", choices=["hybrid", "transformer", "gru"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--steps-per-epoch", type=int, default=100)
    ap.add_argument("--test-frac", type=float, default=0.3, help="share of real drivers held out for testing")
    ap.add_argument("--min-drivers", type=int, default=5)
    ap.add_argument("--min-test-drivers", type=int, default=2)
    ap.add_argument("--min-f1", type=float, default=0.6)
    ap.add_argument("--max-fa-increase", type=float, default=0.02)
    ap.add_argument("--work", help="working folder (default: a temporary one)")
    ap.add_argument("--db-url", default=None, help="database URL (default: DMS_DATABASE_URL or ~/.dms/cloud.db)")
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
