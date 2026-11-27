#!/usr/bin/env bash
# Pretrained models used out of the box (all downloaded into weights/):
#  - MediaPipe Face Landmarker: 478 landmarks + 52 blendshapes + head-pose matrix (Apache-2.0)
#  - HSEmotion EfficientNet-B0, AffectNet 8-class, ONNX (Apache-2.0, A. Savchenko)
#  - Ultralytics YOLO11n, COCO (AGPL-3.0): phone / bottle / cup detection
#  - YAMNet (MediaPipe audio classifier, Apache-2.0): 521 sound classes for cabin audio
set -euo pipefail
W="$(cd "$(dirname "$0")/.." && pwd)/weights"
mkdir -p "$W"
get() { [ -s "$W/$1" ] && echo "have $1" || { echo "downloading $1"; curl -fL --retry 3 -o "$W/$1" "$2"; }; }
get face_landmarker.task   https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
get emotion_enet_b0_8.onnx https://raw.githubusercontent.com/HSE-asavchenko/face-emotion-recognition/main/models/affectnet_emotions/onnx/enet_b0_8_best_afew.onnx
get yolo11n.pt             https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
get yolo11s.pt             https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt
get yolo11n-pose.pt        https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt
get yamnet.tflite          https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite
# integrity check: models must match weights/SHA256SUMS (the server refuses to start otherwise)
(cd "$W" && if command -v sha256sum >/dev/null; then sha256sum -c --ignore-missing SHA256SUMS; else shasum -a 256 -c SHA256SUMS; fi)
