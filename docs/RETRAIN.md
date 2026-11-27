# Retrain: steps (round 5)

All commands run in the **VS Code terminal** (`Ctrl+\``) inside the project, with the environment active: the prompt starts with `(.venv)`. If it doesn't, run `source .venv/bin/activate`.

## 1. Get the new code
```bash
git pull
python -m pytest -q
```

## 2. Download and unzip the previous run's outputs (in parallel with step 4)
1. Kaggle → image notebook → **Output** → **⬇ `dms_runs.zip`**; temporal notebook → **Output** → **⬇ `dms_temporal.zip`**. They land in `~/Downloads`.
2. Unzip into the project:
   ```bash
   unzip -o ~/Downloads/dms_runs.zip -d .
   unzip -o ~/Downloads/dms_temporal.zip -d .
   ls runs runs_temporal features/rldd | head
   ```
   You should see `runs/face`, `runs/drowsy`, `runs/cabin` (from run 3), `runs_temporal/…`, and 178 `rldd_*.npz` files.
   In VS Code's Explorer (left sidebar) the new folders appear too; click any `confusion_*.png` to view it.

## 3. AffectNet+ held-out emotion test (~10 min)
```bash
kaggle datasets download -d mohammadmahoor/affectnetplus -p data/affectnetplus --unzip
python -m dms.data.discover data manifests
python -m dms.training.evaluate_image runs/face/best.pt manifests/affectnet_plus.csv
python -m dms.training.evaluate_image hsemotion manifests/affectnet_plus.csv
```
If the discover line for `affectnetplus` shows `!` or `?`, copy that block to Claude.

## 4. Kaggle: image models, "better" preset (GPU, ~4–5 h)
1. Kaggle → **Create → New Notebook** → **File → Import Notebook** → upload `notebooks/kaggle_train.ipynb` (updated by `git pull`).
2. **Add Input**: competition **State Farm Distracted Driver Detection** (it was missing last time); datasets **fer2013** (msambare), **mrl-eye-dataset** (imadeddinedjerarda), **raf-db-dataset** (shuvoalok), **yawn-eye-dataset-new** (serenaraju), **affectnetplus** (mohammadmahoor; used as a held-out test automatically), **nthu-ddd** (ikhlaselhamly).
3. **Session options**: **GPU T4 x2**, Internet **On**.
4. Leave `PRESET = "better"` in the first code cell.
5. **Save Version → Save & Run All**. After ~10 min check the log: every dataset should have a `✓`; affectnetplus should say "held-out TEST set".
6. When done: **Output → ⬇ `dms_runs_better.zip`** → `unzip -o ~/Downloads/dms_runs_better.zip -d .`

## 5. Mac: improved drowsiness-over-time models (CPU, ~1–1.5 h, no Kaggle needed)
The videos were already converted to features on Kaggle (inside `dms_temporal.zip`), so this trains locally:
```bash
bash scripts/train_temporal_all.sh
```
It trains 7 models (subject-independent 5-fold each) and prints a table:
* `rf`, `hybrid` (new: Conv-Transformer + PERCLOS/blink/pose statistics), `hybrid_nocalib` (calibration ablation), `transformer` (longer training): 3 classes
* `rf_binary`, `hybrid_binary`, `transformer_binary`: alert vs drowsy
Quicker run: `EPOCHS=20 bash scripts/train_temporal_all.sh`. Keep the Mac plugged in.

## 6. When everything is done
* Paste to Claude: the table from step 5, the last cell's output of step 4 (lines starting with `runs_better/...`), and the two lines from step 3. `docs/RESULTS.md` gets updated from them.
* Demo with the best models (round-5 results: `hybrid_binary` reached 80.5% video accuracy on unseen people, so it is used):
  ```bash
  python -m dms.web.server --face-ckpt runs/face/best.pt --cabin-ckpt runs/cabin/best.pt \
      --temporal-ckpt runs_temporal2/hybrid_binary/deploy.pt --sensitivity low
  ```
  (If the Kaggle run used the "better" preset, its folder is `runs_better/` instead of `runs/`.)
