# Pretrained models (included in the repo)

These are needed by the live demo and are committed so a fresh `git clone` works offline.
`bash scripts/download_models.sh` re-downloads them if they are ever missing.

| File | Size | What it does | Source | Licence |
|---|---|---|---|---|
| `face_landmarker.task` | 3.7 MB | 478 face landmarks, 52 blendshapes, head pose | Google MediaPipe Face Landmarker | Apache-2.0 |
| `emotion_enet_b0_8.onnx` | 16 MB | facial emotion (EfficientNet-B0, AffectNet, 8 classes) | HSEmotion, A. Savchenko (github.com/HSE-asavchenko/face-emotion-recognition) | Apache-2.0 |
| `yolo11n.pt` | 5.6 MB | object detection (COCO: cell phone, bottle, cup) | Ultralytics YOLO11n | AGPL-3.0 |

Your own trained checkpoints go in `runs/` (not committed; they come from Kaggle as `dms_runs.zip` or from training on your Mac).
