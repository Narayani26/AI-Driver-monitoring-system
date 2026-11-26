# AI Driver Monitoring System

Real-time, camera-based detection of **drowsiness**, **distraction / unsafe behaviour** and **affect (emotion)**. The three are fused into one explainable **Driver Risk Index** aligned with the Euro NCAP 2026 driver-engagement definitions.

Research background, datasets, novelty claims and the experiment plan: **[docs/RESEARCH.md](docs/RESEARCH.md)**.

```
camera ─► MediaPipe Face Landmarker (478 pts, 52 blendshapes, head pose)
            ├─► EAR / MAR / head pose / gaze ─► per-driver calibration
            ├─► blinks · PERCLOS · microsleep · yawns · long & repeated glances away
            ├─► 30 s feature sequence ─► Conv-Transformer ─► alert / low-vigilant / drowsy
            ├─► face crop ─► multi-task CNN ─► emotion
frame ──────┴─► cabin CNN ─► phone / texting / drinking / reaching back …
                        ▼
              Driver Risk Index 0–100 → OK / CAUTION / WARNING / CRITICAL + reasons
```

## Quick start: live demo (web app)

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # macOS: Python 3.11-3.12 recommended
pip install -r requirements.txt
bash scripts/download_models.sh                              # MediaPipe + HSEmotion + YOLO11n (~25 MB)
python -m dms.web.server                                     # open http://localhost:8000
```

The browser shows **live footage on the left and the AI detection on the right**, plus a risk gauge, live indicators, emotion bars, a real-time timeline, and an alert feed with suggestions (spoken aloud, with a buzzer). Sources:

* **Webcam**, including holding a phone that plays footage up to the webcam: a phone *showing* a face is recognised as a screen, not phone use.
* **Video file**: upload a clip; detection follows the video's own clock, and the trip report opens when it ends.
* **Dash cam / IP stream**: RTSP/HTTP URL of a Wi-Fi dash cam or a phone running an IP-camera app.

It works offline. Pretrained models are used until you train your own; add checkpoints with `--face-ckpt`, `--temporal-ckpt`, `--cabin-ckpt`, `--yolo`. Trained runs in `runs/` appear automatically in the **How it works** tab.

**Optional features** (off by default):
* `--audio`: cabin audio (YAMNet): snoring, shouting, sirens, phone ringing, crash/skid sounds; analysed in memory, never recorded.
* `--mishap`: possible-accident detection: camera jolt or crash sound + unresponsive driver → "Are you OK?" countdown → SOS event.
* `--sensitivity low|normal|high`: how quickly alerts fire (also switchable in the page); a female alert voice is chosen by default and can be changed in the page.
* `--face-ckpt runs/face/best.pt`: use your trained eye-state network (fused with eye geometry) and optionally `--emotion trained`.

**Security and offline:** loopback-only by default, access token when exposed (`--host 0.0.0.0`), Origin/Host checks, model checksums, no footage stored, offline event store with store-and-forward sync (`--sync-url`). See [SECURITY.md](docs/SECURITY.md) and [PRODUCTION.md](docs/PRODUCTION.md).

**Command-line / demo videos for slides**
```bash
python -m dms.app.run --source 0                                                    # webcam window
python -m dms.app.run --source clip.mov --out clip_demo.mp4 --side-by-side --headless --report trip.json
python -m dms.app.run --source rtsp://192.168.1.254/live                            # dash cam
```

> Linux servers/containers: MediaPipe needs `libegl1 libgles2` (`apt-get install -y libegl1 libgles2`).

**Start here → [Complete guide: VS Code setup, testing, datasets, Kaggle, training, fine-tuning](docs/GUIDE.md)**

**Docs:** [Production & deployment](docs/PRODUCTION.md) · [Security](docs/SECURITY.md) · [Research study](docs/RESEARCH.md) · [Datasets & download plan](docs/DATASETS.md) · [Mac vs Kaggle, dash cam & phone deployment](docs/DEPLOYMENT.md) · [Demo script](docs/DEMO.md)

## Training

### 1. Build manifests (one CSV format for every dataset, with subject IDs)
```bash
python -m dms.data.manifests mrl       data/mrlEyes_2018_01   manifests/mrl.csv
python -m dms.data.manifests affectnet data/affectnet         manifests/affectnet.csv
python -m dms.data.manifests kmufed    data/kmu-fed           manifests/kmufed.csv
python -m dms.data.manifests statefarm data/state-farm        manifests/statefarm.csv
python -m dms.data.manifests folders   data/yawn_eye_dataset  manifests/yawn.csv --map yawn=yawn:1 no_yawn=yawn:0
```

### 2. Image models (subject-independent split is automatic and leakage-checked)
```bash
python -m dms.training.train_image --kind face  --manifest manifests/mrl.csv manifests/yawn.csv manifests/affectnet.csv --out runs/face
python -m dms.training.train_image --kind cabin --manifest manifests/statefarm.csv --out runs/cabin
```

### 3. Temporal drowsiness model (UTA-RLDD)
```bash
python -m dms.data.sequences data/uta-rldd features/rldd --fps 10        # ~hours on CPU; run once
python -m dms.training.train_temporal features/rldd --model transformer --folds 5 --fit-all --out runs/temporal
python -m dms.training.train_temporal features/rldd --model gru --folds 5 --out runs/temporal_gru     # baseline
python -m dms.training.train_temporal features/rldd --model rf  --folds 5 --out runs/temporal_rf      # baseline
python -m dms.training.train_temporal features/rldd --test-dir features/nthu --out runs/cross         # cross-dataset
```

### 4. Export & benchmark
```bash
python -m dms.training.export runs/temporal/deploy.pt --int8
python -m dms.training.export runs/face/best.pt
```

Every run writes `metrics.json` (accuracy, macro-F1, balanced accuracy, κ, ECE, per-class, confusion) and confusion-matrix PNGs ready for the paper.

### Smoke test without any data
```bash
PYTHONPATH=. python scripts/make_synthetic.py /tmp/synth
python -m dms.training.train_temporal /tmp/synth/seq --folds 3 --epochs 3 --steps-per-epoch 30 --out /tmp/synth/runs/t
pytest
```

## Layout

| Path | Contents |
|---|---|
| `dms/perception/face.py` | MediaPipe Face Landmarker wrapper, crops |
| `dms/features/geometry.py` | EAR, MAR, head pose from transformation matrix, iris gaze |
| `dms/features/temporal.py` | calibration, blinks/PERCLOS/microsleep, yawns, Euro NCAP-style glance logic, sequence buffer |
| `dms/models/networks.py` | FaceStateNet (multi-task, partial labels), CabinNet, Conv-Transformer, GRU |
| `dms/fusion/risk.py` | Driver Risk Index with smoothing, hysteresis, reasons |
| `dms/pipeline.py` | real-time orchestrator + overlay |
| `dms/perception/emotion.py`, `objects.py` | emotion (trained or pretrained HSEmotion), YOLO phone/objects |
| `dms/fusion/alerts.py` | spoken alerts + suggestions, debounce, cooldowns |
| `dms/fusion/mishap.py` | possible-accident detection (jolt detector + responsiveness check + SOS countdown) |
| `dms/perception/audio.py`, `eyes.py` | YAMNet cabin audio; trained eye-state CNN |
| `dms/storage.py`, `dms/security.py` | offline event store + sync; security checks |
| `Dockerfile`, `deploy/` | edge deployment (Docker, systemd) |
| `dms/report.py` | trip report (summary, timeline, alerts) |
| `dms/web/` | FastAPI server + side-by-side dashboard (no external JS) |
| `dms/data/` | manifests, leakage-free splits, augmentations, video → feature sequences |
| `dms/training/` | image & temporal trainers, metrics, ONNX export/benchmark |
| `dms/tools/` | your own recordings: `extract_crops` (labelled video → training crops), `evaluate_alerts` (recall, delay, false alarms/hour) |
| `notebooks/kaggle_train.ipynb` | GPU training on Kaggle (generated by `scripts/build_notebooks.py`) |
| `scripts/` | `download_models.sh`, `get_datasets.sh` (Kaggle CLI), `make_synthetic.py` |
| `.vscode/` | interpreter, test discovery, run/debug configurations, tasks |
| `.github/workflows/tests.yml` | CI: unit tests + training smoke test on every push |
| `tests/` | unit tests on synthetic signals with known answers |

## Status

| Component | State |
|---|---|
| Web demo: webcam / video upload / IP stream, side-by-side, voice + buzzer alerts with suggestions, trip report | ✅ verified in headless Chromium against a scripted test clip (alerts fire at the expected times) |
| Real-time pipeline: landmarks, calibration, PERCLOS/blinks/microsleep/yawns, Euro NCAP-style distraction, YOLO phone, emotion, risk fusion | ✅ ~30 FPS on a 4-core CPU with emotion + YOLO |
| Unit/integration tests (33) + CI on GitHub Actions | ✅ passing |
| Training / evaluation / export code, fine-tuning (`--init-ckpt`), leakage study (`--split random`), held-out test sets, Kaggle notebook | ✅ verified on synthetic data and a mock Kaggle folder layout |
| Our own trained weights & real-dataset results | ⏳ need datasets (docs/DATASETS.md) and Kaggle/Mac training (docs/DEPLOYMENT.md) |
| Trained models (Kaggle): eye 97.9%, emotion 69.2%, behaviour 92.1% on unseen drivers; calibrated | ✅ |
| Cabin audio, accident detection, offline store, security hardening | ✅ tested (unit, integration, real ESC-50 sounds, browser) |
| Dockerfile / systemd unit | ✅ Docker image built + smoke-tested in CI (GitHub Actions); systemd unit written |
| Android app (dash cam → phone, locked screen) | 📐 designed (docs/PRODUCTION.md), not built |
