# Pretrained models (included in the repo)

These are needed by the live demo and are committed so a fresh `git clone` works offline.
`bash scripts/download_models.sh` re-downloads them if they are ever missing.

| File | Size | What it does | Source | Licence |
|---|---|---|---|---|
| `face_landmarker.task` | 3.7 MB | 478 face landmarks, 52 blendshapes, head pose | Google MediaPipe Face Landmarker | Apache-2.0 |
| `emotion_enet_b0_8.onnx` | 16 MB | facial emotion (EfficientNet-B0, AffectNet, 8 classes) | HSEmotion, A. Savchenko (github.com/HSE-asavchenko/face-emotion-recognition) | Apache-2.0 |
| `yolo11n.pt` | 5.6 MB | object detection (COCO: cell phone, bottle, cup) | Ultralytics YOLO11n | AGPL-3.0 |
| `yamnet.tflite` | 4.1 MB | cabin audio: 521 sound classes (snoring, shouting, sirens, crash, ...) | Google YAMNet via MediaPipe Audio Classifier | Apache-2.0 |

Your own trained checkpoints go in `runs/` (not committed; they come from Kaggle as `dms_runs.zip` or from training on your Mac).

`SHA256SUMS` records each file's checksum. The server verifies them at start-up and refuses to run with a modified model (a tampered model could silently disable safety alerts). After deliberately replacing a model, regenerate it:

```bash
python -c "from pathlib import Path; from dms.security import write_sums; write_sums([p for p in Path('weights').iterdir() if p.suffix in ('.task','.onnx','.pt','.tflite')])"
```
