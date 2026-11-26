# Complete guide: setup, testing, datasets, Kaggle, training, fine-tuning

Follow the sections in order the first time. Every command is run from the project folder in a VS Code terminal, unless it says Kaggle.

| Part | Where it runs |
|---|---|
| Live demo, tests, temporal model, video feature extraction, YOLO quick fine-tunes | **MacBook M5** (VS Code) |
| Face model (eye/yawn/emotion), cabin model, leakage study, big YOLO fine-tunes | **Kaggle** (free GPU) |

---

## 1. VS Code setup on the MacBook

### 1.1 Install the tools (once)
1. **Homebrew** (macOS package manager): paste the install command from https://brew.sh into Terminal.
2. **Git + Python 3.12**:
   ```bash
   brew install git python@3.12
   ```
   Use 3.12, not the newest Python: MediaPipe and ONNX Runtime publish Mac wheels for it first.
3. **VS Code**: https://code.visualstudio.com → drag to Applications → open it → `Cmd+Shift+P` → "Shell Command: Install 'code' command in PATH".

### 1.2 Get the code
```bash
cd ~/Documents
git clone https://github.com/Narayani26/AI-Driver-monitoring-system.git
cd AI-Driver-monitoring-system
git checkout feature/dms-implementation   # only until the pull request is merged into main
code .
```
VS Code asks to install the **recommended extensions** (Python, Pylance, Debugpy, Jupyter, Ruff). Click **Install All**.

### 1.3 Create the environment
`Cmd+Shift+P` → **Tasks: Run Task** → **setup: create venv + install**. Or in the terminal (`` Ctrl+` ``):
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
bash scripts/download_models.sh          # MediaPipe face model, HSEmotion, YOLO11n (~25 MB)
```
Then `Cmd+Shift+P` → **Python: Select Interpreter** → pick `./.venv/bin/python`. The settings in `.vscode/` already point there, so new terminals activate it automatically.

### 1.4 macOS camera permission
* Browser demo: Chrome/Safari asks for the camera the first time. Click **Allow**.
* Command-line webcam (`dms.app.run --source 0`): System Settings → **Privacy & Security → Camera** → enable **Visual Studio Code** (and/or Terminal), then restart VS Code.

### 1.5 What's in VS Code for you
* **Run and Debug** panel (`Cmd+Shift+D`), ready-made configurations:
  * *Demo: web app*: starts the demo at http://localhost:8000
  * *Demo: web app with trained models*
  * *Demo: webcam window (CLI)*
  * *Demo: side-by-side video from a clip*: asks for a video path
  * *Train: temporal model*
  * *Smoke test: train on synthetic data*
  * *Debug: current file*
  Breakpoints work in all of them.
* **Testing** panel (flask icon): every unit test, runnable one by one.
* **Tasks** (`Cmd+Shift+P` → Run Task): setup, download models, run tests, make synthetic data, start web demo.

---

## 2. Testing everything

### 2.1 Unit tests (1 minute)
```bash
python -m pytest -q
```
Expect `20 passed`. They check, using synthetic signals with known answers:
* eye/mouth ratios and head-pose maths
* blink counting, PERCLOS, microsleep detection
* yawn duration rule, the Euro NCAP 3 s and 10-in-30 s distraction rules
* calibration
* risk levels and hysteresis, the post-microsleep decay
* alert debounce and cooldowns
* "phone showing a face" being ignored
* the models' output shapes and masked multi-task loss
* checkpoint save/load
* that the subject split never leaks a person while the random split does
* the alert scoring tool

GitHub runs the same tests (plus a mini training run) on every push: see the **Actions** tab of the repository.

### 2.2 Training smoke test (2 minutes, no datasets needed)
Run and Debug → **Smoke test: train on synthetic data**, or:
```bash
PYTHONPATH=. python scripts/make_synthetic.py /tmp/dms_synth
python -m dms.training.train_temporal /tmp/dms_synth/seq --folds 3 --epochs 3 --steps-per-epoch 30 --out /tmp/dms_synth/runs/t
python -m dms.training.train_image --kind face --manifest /tmp/dms_synth/face.csv --no-pretrained \
  --backbone mobilenetv3_small_100 --epochs 1 --bs 8 --max-steps 3 --workers 0 --img-size 96 --out /tmp/dms_synth/runs/face
```
This only checks that the training code runs. The scores are meaningless.

### 2.3 Live demo checklist (5 minutes)
```bash
python -m dms.web.server        # open http://localhost:8000 in Chrome
```
Press **Start** (Webcam), sit normally and look at the screen for the 5 s calibration. Then check each row:

| Do this | Expected on the right-hand side |
|---|---|
| Sit normally for 20 s | Level **OK**, eyes "open", blinks/min ≈ 10–20, emotion mostly neutral |
| Close your eyes for 2 s | Eye outlines turn red, "eyes closed 1.x s", **CRITICAL**, alarm + voice "Wake up! …" |
| …then open them | Risk stays elevated (WARNING), reason "recent microsleep", soon "You seem drowsy …" |
| Yawn widely, holding ~2 s, three times within 5 min | Yawn counter 1→3, then "Frequent yawning detected" |
| Turn your head to the side for 4 s | Attention "away 3.x s", "Please look ahead at the road" at 3 s |
| Short glances away, repeatedly, for 30 s | "You keep looking away from the road" |
| Hold your phone to your ear / text at chest height for 2 s | Red **cell phone** box, "Please put the phone down" |
| Hold up a phone *playing a video of a face* | Dashed grey box "(ignored)": no phone alert |
| Angry face for ~6 s | Emotion bars shift to anger, "You seem stressed" |
| Press **Stop** | Trip report: safety score, alerts by type, time in each level, risk curve; Download JSON/CSV work |

Also check **Video file** (upload any clip; the report opens at the end) and the **How it works** tab.

If alerts are too sensitive or not sensitive enough for your face or lighting, press **Recalibrate** while looking straight ahead. Thresholds live in `dms/features/temporal.py` (`EyeClosureTracker`, `YawnTracker`, `GlanceTracker`).

### 2.4 Test the dash-cam path (IP stream)
1. On an Android phone, install an IP-camera app (e.g. "IP Webcam"), start the server, and note the URL it shows (e.g. `http://192.168.1.23:8080/video`). The phone and the Mac must be on the same Wi-Fi.
2. In the demo choose **Dash cam / IP stream**, paste the URL, press **Start**.
3. A real dash cam with RTSP works the same way (`rtsp://<camera-ip>/...`; the path depends on the model).

### 2.5 Test on your own recordings, with numbers for the paper
1. Record clips with QuickTime (File → New Movie Recording) on the demo laptop. Name them `<person>_<n>.mov`, e.g. `riya_01.mov`. Act out a script (§6.1).
2. Write `recordings/labels.csv` (format in §6.1).
3. Run the system on each clip and score it:
   ```bash
   mkdir -p logs
   for f in recordings/*.mov; do
     python -m dms.app.run --source "$f" --log "logs/$(basename "${f%.*}").jsonl" --headless --no-alarm --calib-seconds 10
   done
   python -m dms.tools.evaluate_alerts --labels recordings/labels.csv --logs logs --out runs/eval_own.json
   ```
   The output table gives, per alert type: events, detected, **recall**, **mean detection delay**, **false alarms** and **false alarms per hour**. That is paper experiment E7.

---

## 3. Getting the datasets

### 3.1 Kaggle account + API token (needed for the Mac script and for Kaggle notebooks)
1. Create an account at https://www.kaggle.com and **verify your phone number** (Settings → Phone verification). Without it you get no GPU and no internet in notebooks.
2. Settings → **API → Create New Token** downloads `kaggle.json`.
3. On the Mac:
   ```bash
   pip install kaggle six
   mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
   ```
4. Open https://www.kaggle.com/c/state-farm-distracted-driver-detection → **Join Competition / I understand and accept** (once).

### 3.2 Download
```bash
bash scripts/get_datasets.sh              # State Farm + FER2013 + MRL Eye into data/
```
Then the manual ones (links and details in `docs/DATASETS.md`):

| Dataset | Put it in | Notes |
|---|---|---|
| Yawn-Eye (Kaggle: search "yawn eye dataset") | `data/yawn_eye/` | `kaggle datasets download -d <owner>/<slug> -p data/yawn_eye --unzip` |
| UTA-RLDD | `data/uta-rldd/` | 111 GB in 5 folds; see §5.3 for processing one fold at a time |
| KMU-FED | `data/kmu-fed/` | small; test set for in-car emotion |
| AffectNet / RAF-DB / NTHU-DDD / 100-Driver | `data/<name>/` | request forms; send them now |

### 3.3 Build manifests
A manifest is one CSV per dataset with columns `path, subject, source, split, eye, yawn, emotion, cabin`. The `subject` column is what lets training keep every person entirely in train *or* test.
```bash
python -m dms.data.manifests statefarm data/state-farm manifests/statefarm.csv
python -m dms.data.manifests fer       data/fer2013    manifests/fer.csv
python -m dms.data.manifests mrl       data/mrl        manifests/mrl.csv
python -m dms.data.manifests folders   data/yawn_eye   manifests/yawn.csv --source yawn \
       --map yawn=yawn:1 no_yawn=yawn:0 Closed=eye:0 Open=eye:1
python -m dms.data.manifests kmufed    data/kmu-fed    manifests/kmufed.csv
python -m dms.data.manifests affectnet data/affectnet  manifests/affectnet.csv
```
Each command prints the number of images, the number of people, and the class counts. Sanity-check them against the dataset's description.

> FER2013, AffectNet and the Yawn-Eye set have **no person IDs**, so their splits are per-image. Say this in the paper; it's why KMU-FED (people known) is the honest emotion test.

---

## 4. Kaggle setup (GPU training)

1. https://www.kaggle.com → **Create → New Notebook**.
2. **File → Import Notebook → GitHub**, then paste the repo URL and pick `notebooks/kaggle_train.ipynb`. Or upload the file from your Mac (File → Import Notebook → Upload).
3. Right panel → **Session options**: Accelerator **GPU T4 x2**; Internet **On**; Persistence: *Files only*.
4. Right panel → **Add Input** → attach:
   * Competition `state-farm-distracted-driver-detection`
   * Dataset `msambare/fer2013`
   * An MRL eye dataset (search "MRL eye", e.g. `imadeddinedjerarda/mrl-eye-dataset`)
   * Optional: a Yawn-Eye dataset
5. **Private datasets** (AffectNet, KMU-FED, 100-Driver, your own crops): zip the folder on your Mac → kaggle.com → **Datasets → New Dataset** → upload the zip → visibility **Private** → then *Add Input → Your Datasets*. You can also upload from the terminal: `kaggle datasets init -p folder`, edit `dataset-metadata.json`, then `kaggle datasets create -p folder`.
6. Optional **Secrets** (Add-ons → Secrets): `ROBOFLOW_API_KEY` for YOLO data; `GITHUB_TOKEN` if the repo is private.
7. First cell: set `BRANCH` (`main` after merging, otherwise `feature/dms-implementation`).
8. **Run All** to watch it live. For long runs, use **Save Version → Save & Run All (Commit)**: it runs in the background for up to 12 h, even with your laptop closed.
9. When done: **Output** panel → download `dms_runs.zip` → unzip into the project folder on the Mac (creates `runs/face`, `runs/cabin`, …).

Weekly GPU quota is about 30 hours. A face model takes ~1–2 h, the cabin model ~1 h ×2 (both splits), and YOLO ~1 h, so everything fits in one week with room for reruns.

---

## 5. Training each model

Every run writes into `runs/<name>/`:
* `best.pt` (best checkpoint)
* `metrics.json` (accuracy, macro-F1, balanced accuracy, κ, calibration ECE, per-class scores, confusion matrix)
* `confusion_*.png` (paper-ready figures)
* `split.csv` (exactly who was in train/val/test, for reproducibility)

The web app's *How it works* tab reads `runs/` automatically.

### 5.1 Face model: eye state + yawn + emotion (Kaggle, notebook section 3)
```bash
python -m dms.training.train_image --kind face \
  --manifest manifests/mrl.csv manifests/yawn.csv manifests/fer.csv \
  --backbone mobilenetv3_large_100 --epochs 15 --bs 128 --lr 5e-4 --out runs/face
```
* **On the Mac instead:** same command with `--bs 64 --workers 4` (uses the Apple GPU via MPS automatically). Plug in the charger.
* **Good results look like:** eye macro-F1 ≥ 0.95; FER2013 emotion accuracy 0.65–0.72 (state of the art ≈ 0.75); AffectNet-7 ≈ 0.60–0.65.
* **Driver-domain test (E2):** add `--test-manifest manifests/kmufed.csv`. KMU-FED is then never trained on and is reported as its own test source, showing how a model trained on internet faces does on real in-car faces.
* **Try bigger backbones** for the paper's comparison table: `--backbone efficientnet_b0`, `convnext_tiny`. Any timm model name works.

### 5.2 Cabin behaviour model + leakage study (Kaggle, notebook section 4)
```bash
python -m dms.training.train_image --kind cabin --manifest manifests/statefarm.csv --backbone efficientnet_b0 --epochs 20 --out runs/cabin
python -m dms.training.train_image --kind cabin --manifest manifests/statefarm.csv --backbone efficientnet_b0 --epochs 20 --split random --out runs/cabin_randomsplit
```
The random split will look much better (often ~99%) than the driver split (typically ~80–90%). That gap *is* the finding: report both.

Note: State Farm images are a **side view** of the driver. For a front-facing webcam demo, phone detection comes from YOLO. The cabin model is mainly for the paper experiments, or for a camera mounted to the side.

### 5.3 Temporal drowsiness model on UTA-RLDD (Mac)
1. **Extract features fold by fold** (keeps disk use low):
   ```bash
   # download Fold1 into data/uta-rldd/Fold1_part1/... (folders 01, 02, ... with 0.mov, 5.mov, 10.mov)
   python -m dms.data.sequences data/uta-rldd features/rldd --fps 10
   # check features/rldd/*.npz exist, then delete that fold's videos and download the next fold
   ```
   Re-running skips videos already processed. Expect several hours for all 180 videos; leave it running overnight.
2. **Train and compare** (each takes minutes):
   ```bash
   python -m dms.training.train_temporal features/rldd --model transformer --folds 5 --fit-all --out runs/temporal
   python -m dms.training.train_temporal features/rldd --model gru --folds 5 --out runs/temporal_gru
   python -m dms.training.train_temporal features/rldd --model rf  --folds 5 --out runs/temporal_rf
   ```
   `--fit-all` also trains a final model on all subjects → `runs/temporal/deploy.pt`. Use that one in the demo.
3. **Ablations (E5):**
   * `--seq-len 100` / `300` / `600` (10 s / 30 s / 60 s windows)
   * Without per-driver calibration: add `--no-calibration` (everyone gets one population-average baseline). The difference is the evidence for contribution C2.
4. **Cross-dataset (E6):** extract NTHU-DDD/DMD clips into `features/nthu` with the same 0/5/10 naming, then
   `python -m dms.training.train_temporal features/rldd --test-dir features/nthu --out runs/cross`.
5. **What to expect:** 3-class, subject-independent is hard. The original RLDD baseline is ≈ 65% (video level), so 70–80% is a good result.

### 5.4 YOLO: phone / cigarette / seatbelt
1. On https://universe.roboflow.com, open a driver-behaviour dataset (e.g. "Reckless driver detection") → **Download Dataset** → format **YOLOv11** → *show download code*.
2. **Kaggle:** put the workspace/project/version into notebook section 5 (API key as a Secret).
   **Mac:** download the zip to `data/driver_yolo/`, then:
   ```bash
   yolo detect train data=data/driver_yolo/data.yaml model=weights/yolo11n.pt epochs=60 imgsz=640 batch=16 device=mps project=runs name=yolo
   ```
3. Use it: `python -m dms.web.server --yolo runs/yolo/weights/best.pt`. Class names containing phone, cigarette/smok, bottle/cup/drink or no_seatbelt are picked up automatically.
4. Its `results.csv` / `results.png` (mAP50, precision, recall) go into the paper.

### 5.5 Export + speed numbers (E8)
```bash
python -m dms.training.export runs/face/best.pt
python -m dms.training.export runs/temporal/deploy.pt --int8
```
Prints parameters, MB, and milliseconds per inference (PyTorch and ONNX). Run it on the Mac for the paper's speed table.

### 5.6 Use the trained models in the demo
```bash
python -m dms.web.server --face-ckpt runs/face/best.pt
```
* `--face-ckpt` loads your **trained eye-state network** (97.9% on unseen people). It is fused with the eye-geometry measure: the network decides when it is confident (> 70% either way), the geometry settles uncertain cases. The **Eye state** row of *How it works* and the eyes card ("CNN closed …%") show it is active.
* Emotion stays on the pretrained AffectNet model by default, because it is usually better on live webcams. To use your own emotion head instead: add `--emotion trained`.
* `--temporal-ckpt runs/temporal/deploy.pt` adds the drowsiness-over-time model (after UTA-RLDD training).
* `--cabin-ckpt runs/cabin/best.pt` is optional (side-view cameras).
* `--yolo runs/yolo/weights/best.pt` uses a fine-tuned YOLO.
* Calibration (temperature scaling) stored in the checkpoints is applied automatically.

### 5.7 Train better models (next Kaggle run)
In the notebook's first cell set `PRESET = "best"`: EfficientNet-B2 for both models, 15 epochs, test-time augmentation, about 5 hours on a T4, so run it overnight with **Save & Run All**. Results go to `runs_best/` (download `dms_runs_best.zip`). Every run also reports calibrated confidence (`ECE`, plus `ece_uncalibrated` for comparison).

The biggest accuracy gains come from **data, not epochs**:

| Model | Now | Realistic next step | How |
|---|---|---|---|
| Eye state | 97.9% | 98–99% | already near the label-noise ceiling of MRL; add your own recordings (§6) for your camera |
| Emotion (FER2013) | 69.2% | 72–75% on FER2013; much better live | FER2013's own human agreement is only ~65%. Train on **AffectNet / RAF-DB** (request access) and test on KMU-FED |
| Behaviour (unseen drivers) | 92.1% | 94–96% | `PRESET="best"`; add **100-Driver** (100 drivers instead of 26) |

"Human-level" is a moving target: for eye state we are already there; for facial emotion, models beat the average human on FER2013, and the real limit is that expressions are ambiguous.

### 5.8 Optional features: cabin audio and accident detection
```bash
python -m dms.web.server --face-ckpt runs/face/best.pt --audio --mishap
```
* `--audio`: an **🎤 Audio** switch appears. It listens to the microphone (webcam mode) or the uploaded video's own soundtrack (file mode) and detects **snoring** (alarm), **shouting/screaming** (stress), **sirens** ("give way"), **phone ringing**, **crash/skid sounds**. Audio is analysed in memory in 1-second windows and never recorded. To test: play a siren or snoring sound from YouTube near the laptop.
* `--mishap`: **possible-accident detection**. A sudden camera jolt or a crash/skid sound, followed by the driver not responding (face gone or eyes closed for 3 s), triggers a full-screen **"Possible accident! Are you OK?"** with a 15-second countdown and an **I'm OK** button. If nobody presses it, an SOS event is raised and logged. In the planned phone app this would send an SMS with the location. To test: start, then give the laptop lid a sharp knock or jerk (a jolt) and close your eyes for 3 s.
* Everything else (alerts, trip reports) is also written to the offline event store `~/.dms/events.db`; see `docs/PRODUCTION.md`.

## 6. Fine-tuning on your own data (makes the demo robust)

### 6.1 Record and label
* At least **3 different people** (5–10 is better). The trainer needs ≥ 3 people to keep test people unseen.
* Same laptop and camera as the demo. Record a mix of normal light and dim light.
* Script per person (~3 min): 40 s normal → eyes closed 2–3 s (×3, with normal in between) → 3 yawns → look left/right/down 4–5 s each → phone to ear 6 s → texting 6 s → drinking → 8 s each of angry, happy, surprised, sad faces.
* `recordings/labels.csv` (times in seconds; use QuickTime's timeline or VLC):
  ```
  video,start_s,end_s,label
  riya_01.mov,0,40,normal
  riya_01.mov,41.0,43.5,eyes_closed
  riya_01.mov,50,54.5,look_away
  riya_01.mov,60,66,phone
  riya_01.mov,70,73,yawn
  riya_01.mov,80,88,anger
  ```
  Labels: `normal, eyes_closed, yawn, look_away, phone, drinking, smoking, drowsy, neutral, happy, sad, surprise, fear, disgust, anger`.

### 6.2 Fine-tune the face model
```bash
python -m dms.tools.extract_crops --videos recordings --labels recordings/labels.csv --out data/own
python -m dms.training.train_image --kind face --manifest data/own/manifest.csv \
  --init-ckpt runs/face/best.pt --freeze-epochs 2 --epochs 8 --lr 1e-4 --bs 32 --out runs/face_own
```
* `--init-ckpt` starts from the model trained on public data.
* `--freeze-epochs 2` trains only the output layers first, so a small dataset can't wreck the backbone; then everything trains at a low learning rate.
* To avoid forgetting the public data, add the public manifests too: `--manifest data/own/manifest.csv manifests/fer.csv`.

### 6.3 Fine-tune YOLO on your own frames
Upload ~200 frames from your recordings (phone, bottle, cigarette prop) to a Roboflow project → draw boxes → **Generate version** → export **YOLOv11** → merge it with a public driver dataset in Roboflow → train as in §5.4 starting from your previous `best.pt` (`model=runs/yolo/weights/best.pt`).

### 6.4 Measure the improvement
Re-run §2.5 with the new checkpoints (`python -m dms.app.run ... --face-ckpt runs/face_own/best.pt`) and compare `eval_own.json` before and after. Keep at least one person's recordings out of fine-tuning so the comparison is fair.

---

## 7. Results checklist for the paper

| Experiment | Command (section) | Output to put in the paper |
|---|---|---|
| E1 eye state | §5.1 | `runs/face/metrics.json` → `test.eye`, `confusion_eye.png` |
| E2 emotion (in-car) | §5.1 with `--test-manifest manifests/kmufed.csv` | `test.emotion` |
| E3 leakage study | §5.2 | `runs/cabin` vs `runs/cabin_randomsplit` |
| E4 drowsiness models | §5.3 | `runs/temporal*/metrics.json` → `summary` (mean ± std) |
| E5 ablations | §5.3 step 3 | same, per setting |
| E6 cross-dataset | §5.3 step 4 | `runs/cross/metrics.json` |
| E7 system-level alerts | §2.5 | `runs/eval_own.json` (recall, delay, false alarms/hour) |
| E8 speed | §5.5 | `runs/export/benchmarks.jsonl` |
| YOLO | §5.4 | `runs/yolo/results.csv` (mAP50) |

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `pip install` fails on mediapipe/onnxruntime | Use Python 3.12 (`python3.12 -m venv .venv`) |
| Web page stuck on "connecting…" | The server isn't running, or a model file is missing: run `bash scripts/download_models.sh` |
| Webcam black in the CLI | macOS camera permission for VS Code/Terminal (§1.4) |
| `MPS` errors during training | `export PYTORCH_ENABLE_MPS_FALLBACK=1`, or add `--workers 0` |
| `source ... has <3 subjects` | That dataset needs ≥ 3 people for a subject split; add people, or give it a fixed `split` column |
| Kaggle: no GPU option | Verify your phone number in Kaggle settings |
| Kaggle: dataset not found in notebook | Check the *Input* panel; the notebook prints what is in `/kaggle/input` (section 1) |
| Too many / too few alerts | Recalibrate; then adjust thresholds in `dms/features/temporal.py` and cooldowns in `dms/fusion/alerts.py` |
| `model integrity check failed` | A file in `weights/` was changed or corrupted: `git checkout weights/` or `bash scripts/download_models.sh` |
| `access token required` | You started with `--host 0.0.0.0`: open the full URL the server printed (it ends with `?token=…`) |
| Audio switch missing | Start the server with `--audio` |
| "Only rtsp://, http(s)://…" when pasting a file path in the stream box | Use **Video file** instead, or start with `--allow-files` |
