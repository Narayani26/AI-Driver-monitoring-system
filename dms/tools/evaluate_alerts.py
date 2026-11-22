"""System-level evaluation on your own labelled recordings (paper experiment E7).

1) run the system on each recording and log every frame:
    python -m dms.app.run --source recordings/alice_01.mov --log logs/alice_01.jsonl --headless --calib-seconds 10
2) score the alerts against your labels:
    python -m dms.tools.evaluate_alerts --labels recordings/labels.csv --logs logs/ --out runs/eval_own.json

For each hazard type: how many labelled events were detected (recall), how long
it took (detection delay), and how many alerts fired outside any matching
event (false alarms, also per hour of driving).

Event -> expected alert:
    eyes_closed (>= 1 s)  -> microsleep
    look_away   (>= 3 s)  -> eyes_off_road
    phone       (>= 1.5 s)-> phone
    smoking / drinking    -> smoking / drinking
    anger / fear (>= 6 s) -> stress
    yawn                  -> yawn counter increases (per-yawn detection)
    drowsy (optional)     -> drowsy
Follow-on alerts are not false alarms: "drowsy"/"yawning" within 2 min after an
eyes_closed or yawn event, "glances"/"no_driver" around a look_away.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .labels import read_labels

RULES = {  # label -> (alert kind, minimum segment length for the alert to be expected)
    "eyes_closed": ("microsleep", 1.0),
    "look_away": ("eyes_off_road", 3.0),
    "phone": ("phone", 1.5),
    "smoking": ("smoking", 1.5),
    "drinking": ("drinking", 1.5),
    "anger": ("stress", 6.0),
    "fear": ("stress", 6.0),
    "yawn": ("yawn", 1.5),
}


FOLLOW_ON_S = 120.0  # a drowsy/yawning alert within 2 min of an eye-closure or yawn is not a false alarm


def load_log(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def alert_times(log: list[dict]) -> dict[str, list[float]]:
    """alert kind -> times. The yawn counter is turned into one 'yawn' event per increment."""
    out = defaultdict(list)
    prev_y = 0
    for r in log:
        for k in r.get("alerts", []):
            out[k].append(r["t"])
        y = r.get("yawns") or 0
        if y > prev_y:
            out["yawn"].append(r["t"])
        prev_y = y
    return out


def score(segs, alerts: dict[str, list[float]], tolerance: float):
    """Returns per-kind dicts of events, detected, delays, false_alarms."""
    res = defaultdict(lambda: {"events": 0, "detected": 0, "delays": [], "false_alarms": 0})
    windows = defaultdict(list)
    for s in segs:
        if s.label not in RULES:
            continue
        kind, min_len = RULES[s.label]
        windows[kind].append((s.start, s.end + tolerance))
        if s.end - s.start < min_len:
            continue  # too short: an alert is allowed but not required
        res[kind]["events"] += 1
        hits = [t for t in alerts.get(kind, []) if s.start <= t <= s.end + tolerance]
        if hits:
            res[kind]["detected"] += 1
            res[kind]["delays"].append(min(hits) - s.start)
    # Follow-on alerts that are correct but not tied to one labelled event:
    for s in segs:
        if s.label in ("eyes_closed", "yawn"):
            windows["drowsy"].append((s.start, s.end + FOLLOW_ON_S))
            windows["yawning"].append((s.start, s.end + FOLLOW_ON_S))
        if s.label == "drowsy":
            windows["drowsy"].append((s.start, s.end + tolerance))
            res["drowsy"]["events"] += 1
            hits = [t for t in alerts.get("drowsy", []) if s.start <= t <= s.end + tolerance]
            if hits:
                res["drowsy"]["detected"] += 1
                res["drowsy"]["delays"].append(min(hits) - s.start)
        if s.label == "look_away":
            windows["glances"].append((s.start, s.end + 30.0))
            windows["no_driver"].append((s.start, s.end + tolerance))
    for kind, ts in alerts.items():
        for t in ts:
            if not any(a <= t <= b for a, b in windows.get(kind, [])):
                res[kind]["false_alarms"] += 1
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--logs", type=Path, required=True, help="folder of <video stem>.jsonl logs")
    ap.add_argument("--tolerance", type=float, default=3.0, help="seconds after an event an alert still counts")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    segs = read_labels(a.labels)
    by_video = defaultdict(list)
    for s in segs:
        by_video[s.video].append(s)
    total = defaultdict(lambda: {"events": 0, "detected": 0, "delays": [], "false_alarms": 0})
    hours = 0.0
    for video, vsegs in sorted(by_video.items()):
        log_path = a.logs / (Path(video).stem + ".jsonl")
        if not log_path.exists():
            print(f"!! no log for {video} ({log_path}), skipped")
            continue
        log = load_log(log_path)
        hours += (log[-1]["t"] - log[0]["t"]) / 3600 if len(log) > 1 else 0
        for kind, r in score(vsegs, alert_times(log), a.tolerance).items():
            for k in ("events", "detected", "false_alarms"):
                total[kind][k] += r[k]
            total[kind]["delays"] += r["delays"]

    report = {"hours": round(hours, 4), "tolerance_s": a.tolerance, "per_kind": {}}
    print(f"\n{'alert':15s} {'events':>6s} {'found':>6s} {'recall':>7s} {'delay s':>8s} {'false':>6s} {'FA/h':>6s}")
    for kind, r in sorted(total.items()):
        rec = r["detected"] / r["events"] if r["events"] else float("nan")
        delay = float(np.mean(r["delays"])) if r["delays"] else float("nan")
        fah = r["false_alarms"] / hours if hours else float("nan")
        report["per_kind"][kind] = {"events": r["events"], "detected": r["detected"], "recall": rec, "mean_delay_s": delay, "false_alarms": r["false_alarms"], "false_alarms_per_hour": fah}
        print(f"{kind:15s} {r['events']:6d} {r['detected']:6d} {rec:7.2f} {delay:8.2f} {r['false_alarms']:6d} {fah:6.1f}")
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(report, indent=2, default=lambda x: None))
        print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
