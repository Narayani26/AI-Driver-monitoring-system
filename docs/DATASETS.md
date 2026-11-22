# Datasets: what to download, from where, and in what order

Each dataset below maps to one model in the system. Start with **Tier 1**: all of it is downloadable today without approval, and it is enough for a complete paper. Send the **Tier 2** access requests on day 1, because approval takes days to weeks.

> Licences: almost all of these are *research / non-commercial only*. That is fine for the project and paper. Cite each one, and don't redistribute the images.

## Tier 1: download now (no approval needed)

| # | Dataset | Trains | Size | Where | Manifest / prep command |
|---|---|---|---|---|---|
| 1 | **MRL Eye Dataset**: 84,898 NIR eye crops, 37 people, labelled open/closed + glasses + reflections + lighting | `FaceStateNet` eye head | ~330 MB | official: mrl.cs.vsb.cz/eyedataset · also mirrored on Kaggle (search "MRL eye dataset") and [Roboflow](https://universe.roboflow.com/mrl-eye-dataset-rwrm0) | `python -m dms.data.manifests mrl data/mrlEyes_2018_01 manifests/mrl.csv` |
| 2 | **Yawn-Eye dataset** (4 folders: `yawn`, `no_yawn`, `Open`, `Closed`) | yawn head | ~150 MB | Kaggle (search "yawn eye dataset") or [Roboflow driver-yawn](https://universe.roboflow.com/universidad-carlos-3-madrid/driver-yawn/dataset/30) | `python -m dms.data.manifests folders data/yawn_eye manifests/yawn.csv --map yawn=yawn:1 no_yawn=yawn:0` |
| 3 | **YawDD**: 322 mirror-view + 29 dashboard-view videos of drivers talking/singing/yawning | yawn test set (videos; realistic) | ~5 GB | IEEE DataPort (search "YawDD yawning detection dataset") | extract frames, then `folders` manifest |
| 4 | **UTA-RLDD**: 60 people, 180 videos (alert / low-vigilant / drowsy), ~30 h | **temporal drowsiness model** (the core experiment) | **111 GB** | [sites.google.com/view/utarldd](https://sites.google.com/view/utarldd/home) | `python -m dms.data.sequences data/uta-rldd features/rldd` |
| 5 | **State Farm Distracted Driver**: 22,424 images, 26 drivers, 10 classes | `CabinNet` behaviour classifier + the leakage study (E3) | ~4 GB | [Kaggle competition](https://www.kaggle.com/c/state-farm-distracted-driver-detection) (join the competition to download) | `python -m dms.data.manifests statefarm data/state-farm manifests/statefarm.csv` |
| 6 | **FER2013**: 35,887 grey 48×48 faces, 7 emotions | emotion head (fallback if AffectNet is slow to arrive) | ~60 MB | Kaggle (search "fer2013") | `python -m dms.data.manifests fer data/fer2013 manifests/fer.csv` |
| 7 | **KMU-FED**: 1,106 in-vehicle NIR images, 12 drivers, 6 emotions | **driver-domain emotion test set** | small | Keimyung Univ. CVPR lab page (search "KMU-FED") | `python -m dms.data.manifests kmufed data/kmu-fed manifests/kmufed.csv` |
| 8 | **DMD**: 37 drivers, 41 h, RGB+IR+depth, face/body/hands views | cross-dataset test (distraction + drowsiness) | large; download a subset | [dmd.vicomtech.org](https://dmd.vicomtech.org) | test-only |
| 9 | **Roboflow driver-behaviour sets** (YOLO format): phone, cigarette, seatbelt, drinking | **YOLO fine-tuning** | 1–10k images each | e.g. [Reckless driver detection](https://universe.roboflow.com/yolo-v8rn5/reckless-driver-detection-iz7ob) (phone, seatbelt, cigarette, drinking, eating, hands on/off wheel), [Abnormal Driver Behaviour](https://universe.roboflow.com/aoop-vgwve/abnormal-driver-behaviour-nttpp), [Seatbelt and Mobile Phone](https://universe.roboflow.com/my-workspace-fndo6/seatbelt-and-mobile-phone) | export as "YOLOv11" → `yolo detect train …` (below) |

**UTA-RLDD is 111 GB.** You don't need it all at once. It is split into 5 folds of ~22 GB each. Download one fold, run `dms.data.sequences` on it (this keeps only ~1 MB of features per video), delete the videos, then do the next fold. The script skips videos it has already processed. The authors ask that results use their 5-fold split, which our `--folds 5` cross-validation mirrors.

## Tier 2: request today (approval required)

| Dataset | Why it matters | How |
|---|---|---|
| **AffectNet** (~440k labelled faces) | best emotion training data; the pretrained HSEmotion model was trained on it | request form on Mohammad Mahoor's lab site (academic e-mail required) |
| **RAF-DB** (~30k) | cleaner labels than FER2013 | e-mail request via the RAF-DB website |
| **NTHU-DDD** (36 people, IR + RGB, glasses/sunglasses/night) | second drowsiness dataset → cross-dataset experiment E6 | request from the NTHU Computer Vision Lab |
| **100-Driver** (470k images, 100 drivers, 4 cameras, day + night, 22 classes) | much better behaviour data than State Farm | fill in the licence PDF at [100-driver.github.io](https://100-driver.github.io) and e-mail it |
| **Drive&Act** (9.6M frames, 83 activities) | optional, fine-grained activities | request via the dataset website |

## Your own data: the step that makes the model work for *your* demo

Public datasets give you the paper numbers. **Your demo camera, lighting and face are a different domain.** The single most effective improvement is a small set of recordings of your own:

1. Record 5–10 consenting volunteers (classmates) on the **same laptop/camera you'll demo with**, 3–5 min each. Follow a script: normal driving posture → slow blinks → eyes closed 2 s (×3) → yawns → look left/right/down for 4 s → phone to ear → texting → drinking → angry/stressed face.
2. Write down the time ranges of each action in a CSV (`video,start_s,end_s,label`).
3. Use it as (a) **a test set**: report "accuracy on our own recordings" separately; reviewers like this; (b) **fine-tuning data** for the eye/yawn heads and YOLO, using a subject split.
4. Run `python -m dms.app.run --source rec.mp4 --log rec.jsonl --headless` to get per-frame predictions, then compare them against your CSV to measure detection delay and false alarms per hour (experiment E7).

## Fine-tuning YOLO for driver behaviour (phone, cigarette, seatbelt)

COCO-pretrained `yolo11n` already detects `cell phone`, `bottle` and `cup`, which is what the demo uses now. To add cigarettes, seatbelts and hands-on-wheel, and to get better phone detection inside a car:

```bash
# Roboflow: Download Dataset → format "YOLOv11" → unzip to data/driver_yolo (contains data.yaml)
yolo detect train data=data/driver_yolo/data.yaml model=yolo11n.pt epochs=80 imgsz=640 batch=32 device=mps   # Mac
yolo detect train data=data/driver_yolo/data.yaml model=yolo11s.pt epochs=80 imgsz=640 batch=64 device=0     # Kaggle GPU
python -m dms.web.server --yolo runs/detect/train/weights/best.pt
```
Class names are matched by keyword (`phone`, `cigarette`/`smok`, `bottle`/`cup`/`drink`, `no_seatbelt`), so a fine-tuned model drops in without code changes.

## Which dataset is used for which result in the paper

| Paper experiment | Train | Test |
|---|---|---|
| E1 eye state | MRL (subject split) | MRL test subjects; CEW; own recordings |
| E2 emotion | AffectNet/RAF-DB/FER (+ multi-task) | **KMU-FED** (in-car, never trained on) |
| E3 behaviour + leakage study | State Farm (random split vs driver split) | State Farm unseen drivers; 100-Driver |
| E4/E5 drowsiness + ablations | UTA-RLDD 5-fold | UTA-RLDD held-out fold |
| E6 cross-dataset | UTA-RLDD | NTHU-DDD / DMD |
| E7 system-level alerts | – | own recordings |
| E8 speed | – | Mac M5 CPU / Neural Engine, Raspberry Pi / phone if available |
