# Where to run what, and the path from laptop demo to dash cam + phone

## 1. Compute: MacBook M5 vs Kaggle vs Colab

**Short answer:** develop, demo, and run the light training on the **MacBook**. Run the heavy image-model training on **Kaggle**.

| Job | Where | Why |
|---|---|---|
| Live demo (web app, webcam, video upload) | **MacBook M5** | Everything here is real-time on a laptop CPU. The M5 is far faster than the 4-core cloud container this was tested on (~30 FPS there with emotion + YOLO). |
| UTA-RLDD feature extraction (MediaPipe over 180 videos) | **MacBook**, overnight | CPU-bound; no GPU benefit; the dataset is local after download |
| Temporal model (Conv-Transformer, GRU, RF) | **MacBook** | 174k parameters; trains in minutes on CPU/MPS |
| Eye/yawn heads (MRL ~85k small images) | **MacBook (MPS)** or Kaggle | MobileNetV3 on 85k crops is fine on Apple GPU |
| Emotion on AffectNet (~440k images), CabinNet on 100-Driver, YOLO fine-tuning | **Kaggle** | Free GPUs (T4×2 or P100, ~30 h/week), many datasets are already hosted there (attach them, no download), sessions run up to 12 h in the background |
| Same as above, alternative | Colab | Free T4 disconnects when idle or after a few hours; fine with Colab Pro. Data must be copied from Drive each session. |

The training code picks the device automatically (`cuda` → Kaggle/Colab, `mps` → Mac, else CPU).

### Mac setup

```bash
brew install python@3.12                # 3.11–3.12 has the widest wheel support (MediaPipe, ONNX Runtime)
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh
python -m dms.web.server                # open http://localhost:8000
```

Tips: plug in the charger during training (macOS throttles on battery). Use `--workers 4` for data loading. If an MPS op is unsupported, `PYTORCH_ENABLE_MPS_FALLBACK=1` runs it on CPU.

### Kaggle setup (heavy training)

1. New Notebook → Settings → Accelerator **GPU T4 ×2**, Internet **on**.
2. "Add Input" → attach the datasets (State Farm, FER2013, MRL mirrors, your uploaded AffectNet/100-Driver zip as a *private* dataset).
3. In a cell:
   ```bash
   !git clone <your GitHub repo URL> dms && cd dms && pip install -q timm mediapipe onnx onnxruntime ultralytics
   !cd dms && python -m dms.data.manifests statefarm /kaggle/input/state-farm-distracted-driver-detection manifests/statefarm.csv
   !cd dms && python -m dms.training.train_image --kind cabin --manifest manifests/statefarm.csv --epochs 25 --out /kaggle/working/runs/cabin
   ```
4. Download `/kaggle/working/runs/...` (checkpoint + `metrics.json` + confusion PNGs) into `runs/` on the Mac. The web app's "How it works" tab then shows the numbers automatically.

## 2. From laptop demo to the product: dash cam → phone → alerts

### What you described
A dash cam watches the driver. It connects to the phone, and the phone runs the AI **even when locked and without mobile network**, giving voice and buzzer alerts plus suggestions.

### Facts that shape the design
1. **The camera must face the driver.** Normal dash cams face the road. You need a *dual-channel* dash cam with a cabin camera, ideally **infrared** so it works at night. Many ride-share/taxi models have one. Some expose an **RTSP** live stream over their own Wi-Fi; many only work with their own app. Check RTSP support before buying, or use a camera you control (see option C).
2. **No internet needed.** The dash cam makes its own Wi-Fi hotspot and the phone joins it locally. All models run on-device, so mobile coverage doesn't matter. Android keeps mobile data for other apps while joined to a no-internet Wi-Fi.
3. **Locked phone, Android:** possible. A *foreground service* (with the mandatory persistent notification "Driver monitoring active") keeps running with the screen locked. It reads the RTSP stream, runs the models, speaks alerts (TextToSpeech), vibrates and plays an alarm.
4. **Locked phone, iPhone:** iOS suspends normal apps soon after locking, and doesn't allow continuous video processing in the background. Workarounds are fragile (the background-audio trick risks App Store rejection). For iPhone users the robust design is option C.
5. **Same models on all platforms.** MediaPipe Face Landmarker has Android/iOS SDKs with the same 478 landmarks, blendshapes and head-pose matrix. Our CNNs/transformer export to ONNX (`dms.training.export`) → ONNX Runtime Mobile, or TFLite/Core ML. The trackers, calibration, fusion and alert rules are plain logic, about 600 lines to port to Kotlin/Swift. `dms/fusion/alerts.py` is written to port 1:1.

### Three tiers

```
A. Demo (now)          laptop webcam / video file / IP stream ──► web app on the Mac
B. Prototype (next)    IR dash cam ──Wi-Fi RTSP──► Android app (foreground service, on-device AI) ──► voice · alarm · vibration
C. Product             IR camera + small computer (Raspberry Pi 5 / Jetson / phone-class SoC) runs the AI + buzzer
                         └── Bluetooth LE ──► phone app (iOS + Android, works while locked) ──► notification · voice · trip log
```

* **A** is built (`python -m dms.web.server`). Choose **Dash cam / IP stream** to simulate B today: install an IP-camera app on a spare Android phone (it serves an `http://<phone-ip>:8080/video` stream), mount it facing you, and enter that URL. This is the same Wi-Fi-stream data path a dash cam uses.
* **B** is the right next step for the paper's "deployment" section: Kotlin + CameraX/RTSP (ExoPlayer or FFmpeg) + MediaPipe Tasks + ONNX Runtime Mobile.
* **C** is how commercial driver monitoring is built: the safety-critical alarm sounds from the in-car device itself, even if the phone is dead, locked or forgotten. BLE notifications reach locked iPhones and Android phones reliably and need no network.

### Alerts and suggestions (already implemented, `dms/fusion/alerts.py`)

| Trigger | Severity | Says | Suggests |
|---|---|---|---|
| Eyes closed ≥ 1 s (microsleep) | critical, alarm | "Wake up! Your eyes were closed." | Pull over safely; don't continue driving |
| Drowsiness score high / temporal model "drowsy" | warning | "You seem drowsy." | 15–20 min break; coffee + short nap |
| ≥ 3 yawns in 5 min | caution | "Frequent yawning detected." | Fresh air, plan a break |
| Eyes off road ≥ 3 s | warning | "Please look ahead at the road." | Keep glances under 2 s |
| Repeated glances ≥ 10 s in 30 s | caution | "You keep looking away from the road." | Set up navigation/music before driving |
| Phone in use | warning | "Please put the phone down." | Hands-free or pull over |
| Sustained anger/fear | caution | "You seem stressed." | Deep breath, slow down, keep distance |
| No face for 5 s | caution | "Driver not visible." | Check the camera |

Cooldowns stop repeated nagging, and short spikes are ignored (debounce). The text lives in one table, so adding **Hindi/Marathi** voice prompts means translating that table and setting the speech language (`u.lang = "hi-IN"` / `"mr-IN"` in the web app; Android TTS supports both).

## 3. Running the web app from a phone on the same Wi-Fi

Browsers only allow the camera on `https://` or `localhost`. To open the demo on a phone:

```bash
brew install mkcert && mkcert -install && mkcert <your-mac-ip> localhost
python -m dms.web.server --host 0.0.0.0 --ssl-certfile <your-mac-ip>+1.pem --ssl-keyfile <your-mac-ip>+1-key.pem
# phone: install mkcert's root CA on the phone once (`mkcert -CAROOT` shows it), then open https://<your-mac-ip>:8000
```
