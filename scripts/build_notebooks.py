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

When it finishes, download `/kaggle/working/dms_runs.zip` (Output panel), unzip it into the project folder on your Mac (it creates `runs/...`), and the web app shows the results.
""")
code("""
REPO = "github.com/Narayani26/AI-Driver-monitoring-system.git"
BRANCH = "main"          # use "feature/dms-implementation" if the pull request isn't merged yet
EPOCHS_FACE, EPOCHS_CABIN, EPOCHS_YOLO = 15, 20, 60
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
md("## 1. Find the attached datasets")
code("""
from pathlib import Path
INPUT = Path("/kaggle/input")
for d in sorted(INPUT.iterdir()):
    print(d)

def find_dir(*markers):
    \"\"\"First folder under /kaggle/input that contains one of the marker files/folders.\"\"\"
    for m in markers:
        hit = next(iter(sorted(INPUT.rglob(m), key=lambda p: len(p.parts))), None)
        if hit is not None:
            return hit.parent
    return None

STATEFARM = find_dir("driver_imgs_list.csv")
FER       = find_dir("angry")            # FER2013 class folders (train/angry, test/angry, ...)
MRL       = find_dir("s0001_*.png")      # MRL eye crops are named s0001_00001_...
YAWN      = find_dir("yawn")             # yawn / no_yawn folders
print("State Farm:", STATEFARM, "\\nFER:", FER and FER.parent, "\\nMRL:", MRL, "\\nYawn:", YAWN and YAWN.parent)
""")
md("## 2. Build manifests (one CSV per dataset, with person IDs for leakage-free splits)")
code("""
os.makedirs("manifests", exist_ok=True)
face_manifests = []
if MRL:
    !python -m dms.data.manifests mrl "{MRL.parent}" manifests/mrl.csv
    face_manifests.append("manifests/mrl.csv")
if YAWN:
    !python -m dms.data.manifests folders "{YAWN.parent}" manifests/yawn.csv --map yawn=yawn:1 no_yawn=yawn:0 Closed=eye:0 Open=eye:1 --source yawn
    face_manifests.append("manifests/yawn.csv")
if FER:
    !python -m dms.data.manifests fer "{FER.parent}" manifests/fer.csv
    face_manifests.append("manifests/fer.csv")
# Private uploads (AffectNet / KMU-FED): attach them, then uncomment and fix the path
# !python -m dms.data.manifests affectnet /kaggle/input/<your-affectnet>/ manifests/affectnet.csv ; face_manifests.append("manifests/affectnet.csv")
if STATEFARM:
    !python -m dms.data.manifests statefarm "{STATEFARM}" manifests/statefarm.csv
print(face_manifests)
""")
md("""
## 3. Multi-task face model (eye state + yawn + emotion)
One backbone, three heads, trained on the union of datasets (each image only has some labels). People in the test split never appear in training.
""")
code("""
if face_manifests:
    !python -m dms.training.train_image --kind face --manifest {" ".join(face_manifests)} \\
        --backbone mobilenetv3_large_100 --epochs {EPOCHS_FACE} --bs 128 --lr 5e-4 --workers 4 --out runs/face
""")
md("""
## 4. Cabin behaviour model + leakage study (paper experiment E3)
The same model trained twice: **driver-independent split** (honest) vs **random image split** (what many papers do). The gap between the two is a result for the paper.
""")
code("""
if STATEFARM:
    !python -m dms.training.train_image --kind cabin --manifest manifests/statefarm.csv --backbone efficientnet_b0 \\
        --epochs {EPOCHS_CABIN} --bs 64 --workers 4 --out runs/cabin
    !python -m dms.training.train_image --kind cabin --manifest manifests/statefarm.csv --backbone efficientnet_b0 \\
        --epochs {EPOCHS_CABIN} --bs 64 --workers 4 --split random --out runs/cabin_randomsplit
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
for ck in ["runs/face/best.pt", "runs/cabin/best.pt"]:
    if os.path.exists(ck):
        !python -m dms.training.export {ck} --out runs/export
""")
md("## 7. Results summary + download")
code("""
import json, glob
for p in sorted(glob.glob("runs/**/metrics.json", recursive=True)):
    m = json.load(open(p))
    for task, v in m.get("test", {}).items():
        print(f"{p:45s} {task:8s} acc={v['accuracy']:.4f}  macroF1={v['macro_f1']:.4f}  n={v['n']}")
!cd /kaggle/working/dms && zip -qr /kaggle/working/dms_runs.zip runs -x "runs/**/last.pt" && ls -lh /kaggle/working/dms_runs.zip
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
