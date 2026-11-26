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

When it finishes, download `/kaggle/working/dms_runs.zip` (or `dms_runs_best.zip` for the "best" preset) from the Output panel, unzip it into the project folder on your Mac, and the web app shows the results. Every model is calibrated (temperature scaling on the validation set) and the calibration is used automatically at runtime.
""")
code("""
REPO = "github.com/Narayani26/AI-Driver-monitoring-system.git"
BRANCH = "feature/dms-implementation"   # or "main" once the branch is merged
PRESET = "fast"   # "fast": MobileNetV3/EfficientNet-B0, 8 epochs (~2 h on T4)
                  # "best": EfficientNet-B2 for both, 15 epochs + test-time augmentation (~5 h; run it overnight)
if PRESET == "best":
    BACKBONE_FACE, BACKBONE_CABIN, EPOCHS_FACE, EPOCHS_CABIN, BS = "efficientnet_b2", "efficientnet_b2", 15, 15, 64
else:
    BACKBONE_FACE, BACKBONE_CABIN, EPOCHS_FACE, EPOCHS_CABIN, BS = "mobilenetv3_large_100", "efficientnet_b0", 8, 8, 128
EPOCHS_YOLO = 60
EXTRA = "--tta" if PRESET == "best" else ""
OUT = "runs" if PRESET == "fast" else "runs_best"
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
* **Faces/emotions:** FER2013 (`msambare/fer2013`), **AffectNet** (search "affectnet"), **RAF-DB** (search "raf-db")
* **Eyes/yawns:** MRL Eye, a **Yawn-Eye** dataset (search "yawn eye")
* **KMU-FED** (if you upload it): used as an in-car **test-only** set
* **State Farm** competition → cabin behaviour
Copies of the "Driver Drowsiness Dataset (DDD)" are skipped on purpose: they are frames from the same videos without person IDs, so any train/test split would leak people and inflate the score.
""")
code("""
import sys
sys.path.insert(0, "/kaggle/working/dms")
from dms.data.discover import discover
found = discover("/kaggle/input", "manifests")
face_manifests, face_test, cabin_manifests = found["face"], found["face_test"], found["cabin"]
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
for ck in [f"{OUT}/face/best.pt", f"{OUT}/cabin/best.pt"]:
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

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"
    c["source"] = [l + "\n" for l in c["source"].split("\n")]
    c["source"][-1] = c["source"][-1].rstrip("\n")
out = Path(__file__).resolve().parents[1] / "notebooks" / "kaggle_train.ipynb"
out.write_text(json.dumps(nb, indent=1))
print("wrote", out)
