# Step by step: update, use the last results, train with the new datasets

Commands are for the Mac terminal inside the project folder (`cd ~/"Honors Project"/AI-Driver-monitoring-system`, or wherever yours is). Keep the quotes: the folder name has a space.

## A. Update the code (2 min)
```bash
source .venv/bin/activate
git pull
pip install -r requirements.txt
python -m pytest -q
```
The last line should say `… passed`.

## B. Use the previous Kaggle run (run #3) (5 min)
1. Kaggle → your notebook → **Output** tab → click the **⬇** next to `dms_runs.zip` (77.75 MB).
2. Unzip it into the project folder (creates `runs/face`, `runs/cabin`, `runs/cabin_randomsplit`, `runs/export`):
   ```bash
   unzip -o ~/Downloads/dms_runs.zip -d .
   ls runs
   ```
3. Start the demo with your trained eye model:
   ```bash
   python -m dms.web.server --face-ckpt runs/face/best.pt --sensitivity low
   ```
   Open http://localhost:8000. The **How it works** tab now shows your results, and the eyes card shows "CNN closed …%".

## C. Image models with more data: AffectNet+, RAF-DB, NTHU-DDD (Kaggle, GPU)
1. Download the new notebook to your Mac: it's `notebooks/kaggle_train.ipynb` in the project (updated by `git pull`).
2. Kaggle → **Create → New Notebook** → **File → Import Notebook** → upload that file.
   A new notebook is needed because Kaggle keeps its own copy of the old one.
3. Right panel → **Add Input**:
   * Competition: **State Farm Distracted Driver Detection**
   * Datasets: **fer2013** (msambare), **MRL Eye Dataset** (imadeddinedjerarda), **affectnetplus** (mohammadmahoor), **nthu-ddd** (ikhlaselhamly)
   * Optional: **raf-db** (any copy with folders 1–7), a **yawn eye** dataset
4. **Session options:** Accelerator **GPU T4 x2**, Internet **On**.
5. In the first code cell set `PRESET = "fast"` (~3–4 h with this much data) or `"best"` (~7–9 h).
6. **Save Version → Save & Run All (Commit)** → **Save**.
7. After ~10 minutes open the run → **Logs** and find the dataset list:
   * `✓` = used (shows images, groups and labels)
   * `!` or `?` = not used, with the folder structure printed underneath. **Copy that block and paste it to me**, and I'll add support for that layout.
   * NTHU-DDD is used only if person IDs are found in the file names; the log says how many.
8. When it finishes: **Output** → download `dms_runs.zip` (fast) or `dms_runs_best.zip` (best) → `unzip -o ~/Downloads/dms_runs*.zip -d .`

## D. Drowsiness-over-time model: UTA-RLDD (Kaggle, CPU, runs in parallel with C)
1. Import `notebooks/kaggle_temporal.ipynb` into a **second** new notebook (same steps as C1–C2).
2. **Add Input** → **uta-rldd-full** (thuannn18022005).
3. **Session options:** Accelerator **None** (CPU; doesn't use GPU quota), Internet **On**.
4. **Save Version → Save & Run All**. About 2–4 h.
5. Log check after ~10 min: `using /kaggle/input/... (N videos)` then `[1/N] 01 label=0: … steps, face 9x%`.
   If it says `NO VIDEOS FOUND`, paste the probe block printed above it.
6. Download `dms_temporal.zip` → `unzip -o ~/Downloads/dms_temporal.zip -d .`

## D2. AffectNet+ as a held-out emotion test (Mac, after C finishes, ~10 min)
The Kaggle AffectNet+ upload contains only the **validation set** (5,500 images with `.json` labels; `train_set/` holds just a readme: the training images are given out by the authors on request). AffectNet's validation set is what papers report as their test set, so we use it **only to test** (new notebook runs do this automatically).

For the run that's already going, score it on the Mac:
```bash
kaggle datasets download -d mohammadmahoor/affectnetplus -p data/affectnetplus --unzip
python -m dms.data.discover data manifests
python -m dms.training.evaluate_image runs/face/best.pt manifests/affectnet_plus.csv
python -m dms.training.evaluate_image hsemotion manifests/affectnet_plus.csv
```
The two lines print emotion accuracy for **your model** (trained on FER2013 + RAF-DB, never saw AffectNet) and for the **pretrained demo model** (trained on AffectNet's training set: its home turf, so it has an advantage). Both appear in the web app's results table.

## E. Use everything in the demo
```bash
python -m dms.web.server --face-ckpt runs_best/face/best.pt \
    --temporal-ckpt runs_temporal/transformer/deploy.pt \
    --drowsy-ckpt runs_best/drowsy/best.pt \
    --sensitivity low --audio --mishap
```
(use `runs/` instead of `runs_best/` if you ran the "fast" preset; leave out any checkpoint you don't have yet)

## F. Get the numbers for the paper
* **How it works** tab: all runs with accuracy and macro-F1.
* `runs*/…/metrics.json` and the `confusion_*.png` images.
* Paste the last cells' output from both notebooks to me and I'll update `docs/RESULTS.md`.

## What to expect
| Experiment | What is new | Realistic outcome |
|---|---|---|
| Emotion with AffectNet+ (+RAF-DB) | ~10× more emotion data, cleaner labels | clearly better live emotions; AffectNet test accuracy typically 60–65% (8-class papers ~60–65%; 7-class a bit higher) |
| NTHU frame drowsiness: unseen people vs random split (the Kaggle copy has **only 4 people**, 18k frames, so the honest test is a single unseen person: report as indicative) | second leakage study | random split ~95–99%, unseen people much lower: that gap is the finding |
| UTA-RLDD Conv-Transformer vs GRU vs RF, ± calibration | the core temporal experiment | 3-class subject-independent is hard: original baseline ≈ 65% video-level; 70–80% is good |
