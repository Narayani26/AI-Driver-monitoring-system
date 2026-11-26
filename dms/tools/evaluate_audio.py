"""Measure the cabin-audio alerts on labelled sound datasets (ESC-50, UrbanSound8K, your own clips).

YAMNet is already trained on ~2M AudioSet clips, so the useful work is to
*measure* and *tune* our alert rules: for each alert group, how many real
events are caught (recall) and how often other everyday sounds trigger it
(false-alarm rate), at several score thresholds.

    # ESC-50 (github.com/karolpiczak/ESC-50): meta/esc50.csv + audio/
    python -m dms.tools.evaluate_audio --csv ESC-50/meta/esc50.csv --audio ESC-50/audio --out runs/audio_eval.json
    # UrbanSound8K (Kaggle chrisfilo/urbansound8k): metadata/UrbanSound8K.csv + audio/fold*/
    python -m dms.tools.evaluate_audio --csv UrbanSound8K/metadata/UrbanSound8K.csv --audio UrbanSound8K/audio --label-col class \
        --file-col slice_file_name --out runs/audio_eval_us8k.json

A clip counts as "alerted" with the same rule the live system uses: the group's
score reaches the threshold in >= 2 analysis windows (0.5 s apart). Impact
(crash) uses 1 window because crashes are single events.
"""
from __future__ import annotations

import argparse
import csv
import json
import wave
from collections import defaultdict
from pathlib import Path

import numpy as np

from ..perception.audio import SAMPLE_RATE, AudioMonitor

# dataset category -> our alert group. Anything not listed is a "negative" (should NOT alert).
CATEGORY_TO_GROUP = {
    # ESC-50
    "siren": "siren", "snoring": "snoring", "glass_breaking": "impact", "crying_baby": "baby", "car_horn": "horn",
    # UrbanSound8K
    "gun_shot": "impact",
}
GROUPS = ("snoring", "distress", "impact", "skid", "horn", "siren", "phone", "baby")
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7)
WINDOWS_NEEDED = {g: (1 if g in ("impact", "skid") else 2) for g in GROUPS}


def load_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        sr, ch, width = w.getframerate(), w.getnchannels(), w.getsampwidth()
        raw = w.readframes(w.getnframes())
    dtype = {1: np.uint8, 2: np.int16, 4: np.int32}[width]
    x = np.frombuffer(raw, dtype).astype(np.float32)
    x = (x - 128) / 128 if width == 1 else x / float(np.iinfo(dtype).max)
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    if sr != SAMPLE_RATE:
        x = np.interp(np.arange(0, len(x), sr / SAMPLE_RATE), np.arange(len(x)), x).astype(np.float32)
    return x


def clip_scores(monitor: AudioMonitor, x: np.ndarray) -> dict[str, list[float]]:
    """Per-window raw group scores (no decay) for one clip."""
    monitor.scorer.decay = 0.0
    monitor.scorer.state = {k: 0.0 for k in monitor.scorer.state}
    monitor._buf.clear()
    monitor._n = monitor._since = 0
    per = defaultdict(list)
    hop = 4000
    for i in range(0, len(x), hop):
        st = monitor.push(i / SAMPLE_RATE, x[i:i + hop])
        if st is not None:
            for g, v in st.groups.items():
                per[g].append(v)
    return per


def alerted(scores: list[float], thr: float, need: int) -> bool:
    return sum(s >= thr for s in scores) >= need


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--audio", type=Path, required=True)
    ap.add_argument("--file-col", default="filename")
    ap.add_argument("--label-col", default="category")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    files = {p.name: p for p in a.audio.rglob("*.wav")}
    rows = [r for r in csv.DictReader(open(a.csv)) if r[a.file_col] in files]
    if not rows:
        raise SystemExit("no audio files from the CSV were found under --audio")
    monitor = AudioMonitor()
    results = []
    for i, r in enumerate(rows):
        per = clip_scores(monitor, load_wav(files[r[a.file_col]]))
        results.append((r[a.label_col], per))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(rows)} clips")
    monitor.close()

    report = {"clips": len(results), "groups": {}}
    print(f"\n{len(results)} clips.  recall = events caught; FA rate = other sounds that wrongly trigger the alert\n")
    print(f"{'group':9s} {'events':>6s} " + " ".join(f"{'thr ' + str(t):>15s}" for t in THRESHOLDS))
    for g in GROUPS:
        pos = [per[g] for cat, per in results if CATEGORY_TO_GROUP.get(cat) == g]
        neg = [(cat, per[g]) for cat, per in results if CATEGORY_TO_GROUP.get(cat) != g]
        row, cells = {}, []
        for t in THRESHOLDS:
            rec = float(np.mean([alerted(s, t, WINDOWS_NEEDED[g]) for s in pos])) if pos else None
            fa_cats = [c for c, s in neg if alerted(s, t, WINDOWS_NEEDED[g])]
            fa = len(fa_cats) / max(len(neg), 1)
            row[str(t)] = {"recall": rec, "false_alarm_rate": fa, "false_alarms_by_category": {c: fa_cats.count(c) for c in sorted(set(fa_cats))}}
            cells.append(f"{'R ' + (f'{rec:.2f}' if rec is not None else ' -  ')} FA {fa:.3f}".rjust(15))
        report["groups"][g] = {"events": len(pos), "by_threshold": row}
        print(f"{g:9s} {len(pos):6d} " + " ".join(cells))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2))
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
