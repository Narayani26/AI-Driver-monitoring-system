"""Generates notebooks/kaggle_train.ipynb (keeps the notebook reviewable as code).
    python scripts/build_notebooks.py
"""
import json
from pathlib import Path

cells = []
md = lambda s: cells.append({"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")})
code = lambda s: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")})

md("""
# AI Driver Monitoring System: training on Kaggle

Runs the **GPU-heavy training** for the project. The live demo and the temporal drowsiness model run on your Mac (see `docs/GUIDE.md`).

**Before running**
1. Notebook settings (right panel): **Accelerator → GPU T4 x2**, **Internet → On**. Both need a phone-verified Kaggle account.
2. **Add Input** (right panel → *Add Input* → search) and attach whichever you want to train with:
   * Competition **State Farm Distracted Driver Detection** (`state-farm-distracted-driver-detection`); join the competition first on its page.
   * Dataset **FER2013** (`msambare/fer2013`)
   * Dataset **MRL Eye Dataset** (e.g. `imadeddinedjerarda/mrl-eye-dataset`)
   * Optional: a **yawn/eye** dataset (search "yawn"), your private uploads (AffectNet, KMU-FED, 100-Driver, your own crops)
3. Optional, for YOLO: *Add-ons → Secrets* → add `ROBOFLOW_API_KEY`. If the GitHub repo is private, also add `GITHUB_TOKEN`.
4. **Run All**, or run the sections you need. Each section is skipped automatically if its dataset isn't attached.

When it finishes, download `/kaggle/working/dms_runs_better.zip` (`dms_runs.zip` for "fast", `dms_runs_best.zip` for "best") from the Output panel, unzip it into the project folder on your Mac, and the web app shows the results. Every model is calibrated (temperature scaling on the validation set) and the calibration is used automatically at runtime.
""")
code("""
REPO = "github.com/Narayani26/AI-Driver-monitoring-system.git"
BRANCH = "feature/dms-implementation"   # or "main" once the branch is merged
PRESET = "better"  # "fast":   MobileNetV3 / EfficientNet-B0, 8 epochs           (~2-3 h on T4 with all datasets)
                   # "better": EfficientNet-B0 for both, 12 epochs + test-time aug  (~4-5 h)  <- recommended
                   # "best":   EfficientNet-B2 for both, 15 epochs + test-time aug  (~10-12 h: may hit Kaggle's 12 h limit)
PRESETS = {
    "fast":   ("mobilenetv3_large_100", "efficientnet_b0", 8, 8, 128, ""),
    "better": ("efficientnet_b0", "efficientnet_b0", 12, 12, 96, "--tta"),
    "best":   ("efficientnet_b2", "efficientnet_b2", 15, 15, 64, "--tta"),
}
BACKBONE_FACE, BACKBONE_CABIN, EPOCHS_FACE, EPOCHS_CABIN, BS, EXTRA = PRESETS[PRESET]
EPOCHS_YOLO = 60
OUT = "runs" if PRESET == "fast" else f"runs_{PRESET}"
""")
code("""
import os, subprocess
token = ""
try:
    from kaggle_secrets import UserSecretsClient
    token = UserSecretsClient().get_secret("GITHUB_TOKEN")
except Exception:
    pass  # public repo: no token needed
url = f"https://{token + '@' if token else ''}{REPO}"
if not os.path.exists("/kaggle/working/dms"):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, url, "/kaggle/working/dms"], check=True)
os.chdir("/kaggle/working/dms")
!pip install -q timm ultralytics onnx onnxruntime roboflow
!git log --oneline -1
import torch; print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - enable the GPU accelerator!")
""")
md("""
## 1–2. Find every attached dataset and build manifests
Datasets are recognised by their **contents**, so any Kaggle copy works. Attach as many as you like (right panel → Add Input):
* **Faces/emotions:** FER2013 (`msambare/fer2013`), **AffectNet+** (`mohammadmahoor/affectnetplus`) or another AffectNet copy, **RAF-DB** (search "raf-db")
* **Eyes/yawns:** MRL Eye, a **Yawn-Eye** dataset (search "yawn eye")
* **KMU-FED** (if you upload it): used as an in-car **test-only** set
* **State Farm** competition → cabin behaviour
* **NTHU-DDD** frames (`ikhlaselhamly/nthu-ddd`) → frame-level drowsiness, **only if** person IDs are in the file names (otherwise skipped: train and test would share people). Same rule skips "Driver Drowsiness Dataset (DDD)" copies.

The first cell below prints every dataset's real folder structure. If something is marked `?` or `!`, paste that output to get it supported.
""")
code("""
import sys
sys.path.insert(0, "/kaggle/working/dms")
from dms.data.probe import main as probe
probe(["/kaggle/input"])          # what is really in each dataset
from dms.data.discover import discover
found = discover("/kaggle/input", "manifests")
face_manifests, face_test, cabin_manifests, drowsy_manifests = found["face"], found["face_test"], found["cabin"], found["drowsy"]
""")
md("""
## 3. Multi-task face model (eye state + yawn + emotion)
One backbone, three heads, trained on the union of datasets (each image only has some labels). People in the test split never appear in training.
""")
code("""
if face_manifests:
    test_arg = ("--test-manifest " + " ".join(face_test)) if face_test else ""
    !python -m dms.training.train_image --kind face --manifest {" ".join(face_manifests)} {test_arg} \\
        --backbone {BACKBONE_FACE} --epochs {EPOCHS_FACE} --bs {BS} --lr 5e-4 --workers 4 {EXTRA} --out {OUT}/face
""")
md("""
## 4. Cabin behaviour model + leakage study (paper experiment E3)
The same model trained twice: **driver-independent split** (honest) vs **random image split** (what many papers do). The gap between the two is a result for the paper.
""")
code("""
if cabin_manifests:
    !python -m dms.training.train_image --kind cabin --manifest manifests/statefarm.csv --backbone {BACKBONE_CABIN} \\
        --epochs {EPOCHS_CABIN} --bs 64 --workers 4 --out {OUT}/cabin
    !python -m dms.training.train_image --kind cabin --manifest manifests/statefarm.csv --backbone {BACKBONE_CABIN} \\
        --epochs {EPOCHS_CABIN} --bs 64 --workers 4 --split random --out {OUT}/cabin_randomsplit
""")
md("""
## 4b. Frame-level drowsiness (NTHU-DDD) + second leakage study
Same idea as section 4: train once with **unseen people** in the test set (honest) and once with a random split (what many papers do).
""")
code("""
if drowsy_manifests:
    !python -m dms.training.train_image --kind drowsy --manifest {" ".join(drowsy_manifests)} --backbone {BACKBONE_CABIN} \\
        --epochs {EPOCHS_CABIN} --bs 64 --workers 4 --out {OUT}/drowsy
    !python -m dms.training.train_image --kind drowsy --manifest {" ".join(drowsy_manifests)} --backbone {BACKBONE_CABIN} \\
        --epochs {EPOCHS_CABIN} --bs 64 --workers 4 --split random --out {OUT}/drowsy_randomsplit
else:
    print("no usable drowsy-frame dataset attached (or it had no person IDs) - section skipped")
""")
md("""
## 5. YOLO fine-tuning: phone / cigarette / seatbelt (optional)
Open a driver-behaviour dataset on Roboflow Universe → *Download Dataset* → format **YOLOv11** → *show download code*. Copy the workspace, project and version numbers into the cell below.
""")
code("""
RF_WORKSPACE, RF_PROJECT, RF_VERSION = "", "", 1     # e.g. from the Roboflow download snippet
if RF_WORKSPACE:
    from kaggle_secrets import UserSecretsClient
    from roboflow import Roboflow
    rf = Roboflow(api_key=UserSecretsClient().get_secret("ROBOFLOW_API_KEY"))
    ds = rf.workspace(RF_WORKSPACE).project(RF_PROJECT).version(RF_VERSION).download("yolov11", location="/kaggle/working/yolo_data")
    !yolo detect train data=/kaggle/working/yolo_data/data.yaml model=yolo11n.pt epochs={EPOCHS_YOLO} imgsz=640 batch=64 device=0,1 project=runs name=yolo exist_ok=True
""")
md("## 6. Export to ONNX + speed benchmark")
code("""
for ck in [f"{OUT}/face/best.pt", f"{OUT}/cabin/best.pt", f"{OUT}/drowsy/best.pt"]:
    if os.path.exists(ck):
        !python -m dms.training.export {ck} --out {OUT}/export
""")
md("## 7. Results summary + download")
code("""
import json, glob
for p in sorted(glob.glob(f"{OUT}/**/metrics.json", recursive=True)):
    m = json.load(open(p))
    for task, v in list(m.get("test", {}).items()) + [(t + " (held-out)", v) for t, v in m.get("test_heldout", {}).items()]:
        print(f"{p:45s} {task:8s} acc={v['accuracy']:.4f}  macroF1={v['macro_f1']:.4f}  ECE={v['ece']:.3f} (uncalibrated {v.get('ece_uncalibrated', v['ece']):.3f})  n={v['n']}")
!cd /kaggle/working/dms && zip -qr /kaggle/working/dms_{OUT}.zip {OUT} -x "{OUT}/**/last.pt" && ls -lh /kaggle/working/dms_{OUT}.zip
""")

def write_notebook(cells, name):
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
        c["source"] = [l + "\n" for l in c["source"].split("\n")]
        c["source"][-1] = c["source"][-1].rstrip("\n")
    out = Path(__file__).resolve().parents[1] / "notebooks" / name
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


write_notebook(cells, "kaggle_train.ipynb")

# =========================================================== temporal (UTA-RLDD) notebook
cells = []
md("""
# AI Driver Monitoring System: drowsiness-over-time model on Kaggle (UTA-RLDD)

Turns UTA-RLDD videos (60 people × alert / low-vigilant / drowsy) into face-feature sequences with MediaPipe, then trains the
**Conv-Transformer** with **subject-independent 5-fold cross-validation**, plus baselines (GRU, Random Forest) and the
**calibration ablation** (contribution C2).

**Before running**
1. Session options: **Accelerator → None (CPU)** is enough and saves your GPU quota (the work is face tracking on the CPU). **Internet → On.**
2. **Add Input** → the UTA-RLDD dataset (e.g. `thuannn18022005/uta-rldd-full`).
3. **Save Version → Save & Run All**. It takes about 2-4 h; the log shows progress per video.
4. Download `dms_temporal.zip` from Output and unzip it in the project (creates `runs_temporal/…` and `features/rldd/…`).
""")
code("""
REPO = "github.com/Narayani26/AI-Driver-monitoring-system.git"
BRANCH = "feature/dms-implementation"   # or "main" once merged
MAX_SECONDS = 360   # use the first 6 min of each ~10 min video (set None for all; ~2x longer)
WORKERS = 4         # Kaggle sessions have 4 CPU cores
EPOCHS = 20
""")
code("""
import os, subprocess, sys
if not os.path.exists("/kaggle/working/dms"):
    subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, f"https://{REPO}", "/kaggle/working/dms"], check=True)
os.chdir("/kaggle/working/dms")
sys.path.insert(0, "/kaggle/working/dms")
!apt-get -qq update > /dev/null 2>&1; apt-get -qq install -y libegl1 libgles2 > /dev/null 2>&1 || true
!pip install -q mediapipe timm scikit-learn
!git log --oneline -1
""")
md("## 1. Find the videos")
code("""
from pathlib import Path
from dms.data.probe import main as probe
from dms.data.discover import dataset_roots
from dms.data.sequences import find_videos
probe(["/kaggle/input"])
roots = [(len(find_videos(r)), r) for r in dataset_roots(Path("/kaggle/input"))]
n, RLDD = max(roots) if roots else (0, None)
print(f"using {RLDD} ({n} videos)" if n else "NO VIDEOS FOUND - paste the probe output above to get the layout supported")
""")
md("## 2. Video → face-feature sequences (resumable, parallel)")
code("""
if RLDD:
    ms = f"--max-seconds {MAX_SECONDS}" if MAX_SECONDS else ""
    !python -m dms.data.sequences "{RLDD}" features/rldd --fps 10 --workers {WORKERS} {ms}
""")
md("## 3. Train and compare (subject-independent 5-fold CV)")
code("""
if RLDD:
    common = f"--folds 5 --epochs {EPOCHS} --steps-per-epoch 100"
    !python -m dms.training.train_temporal features/rldd --model transformer {common} --fit-all --out runs_temporal/transformer
    !python -m dms.training.train_temporal features/rldd --model gru {common} --out runs_temporal/gru
    !python -m dms.training.train_temporal features/rldd --model rf --folds 5 --out runs_temporal/rf
    !python -m dms.training.train_temporal features/rldd --model transformer {common} --no-calibration --out runs_temporal/transformer_nocalib
    !python -m dms.training.train_temporal features/rldd --model transformer {common} --seq-len 100 --out runs_temporal/transformer_10s
""")
md("## 4. Results + download")
code("""
import json, glob
print(f"{'run':40s} {'window macro-F1':>18s} {'video accuracy':>18s}")
for p in sorted(glob.glob("runs_temporal/*/metrics.json")):
    s = json.load(open(p))["summary"]
    f1, va = s["window_macro_f1"], s["video_accuracy"]
    print(f"{p.split('/')[1]:40s} {f1['mean']:.3f} ± {f1['std']:.3f}     {va['mean']:.3f} ± {va['std']:.3f}")
!cd /kaggle/working/dms && zip -qr /kaggle/working/dms_temporal.zip runs_temporal features && ls -lh /kaggle/working/dms_temporal.zip
""")
write_notebook(cells, "kaggle_temporal.ipynb")
