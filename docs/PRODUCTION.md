# From demo to real use: architecture, deployment, non-functional requirements

## 1. How it runs in a real car

```
 Driver-facing IR camera  ──►  Edge device: runs ALL the AI, offline  ──►  Driver feedback (in the car)
 (dash cam cabin lens,          · face landmarks, eye/emotion CNNs            · voice + alarm sound
  USB IR camera, or phone)      · YOLO phone detection, YAMNet audio          · (optional) buzzer / seat vibration
                                · risk fusion, alerts, accident check
                                · offline event store (SQLite)
                                          │  when online (Wi-Fi at home / mobile data)
                                          ▼
                                 Sync server (yours): trip reports, fleet dashboard, family app, SOS relay
```

The key decision is **edge-first**: all processing happens in the car, so the system:
- works with **no or poor internet** (tunnels, highways, rural roads);
- reacts in **milliseconds** (no network round trip before a "wake up" alarm);
- never sends **camera footage** anywhere (privacy, `docs/SECURITY.md`).

The cloud is optional, used only for reports and SOS forwarding, and everything waits in a queue until a connection exists (store-and-forward).

## 2. Deployment options

| Option | Who it's for | Status in this repo |
|---|---|---|
| **A. Laptop / browser dashboard** (`python -m dms.web.server`) | demos, research, testing | ✅ built and tested |
| **B. Headless edge box**: Raspberry Pi 5 / Jetson / mini PC + USB IR camera or the dash cam's RTSP stream, started by systemd or Docker, speaks alerts (`python -m dms.app.run --headless --speak --mishap --store …`) | retrofit for any car; fleets; **works with existing dash cams that expose RTSP** | ✅ software built (CLI tested; Dockerfile and systemd unit written but **not build-tested here**, because this sandbox blocks the package mirrors) |
| **C. Android phone app**: phone mounted facing the driver *or* receiving the dash cam's Wi-Fi stream; runs as a foreground service, so it **keeps working with the screen locked** | consumers; the "dash cam → phone" vision | 📐 designed (§4), not built |
| **D. Dash cam with built-in AI**: the same models compiled for the camera's chip (e.g. via ONNX/TFLite on Ambarella/Novatek/Rockchip NPUs) | OEM / product company | 📐 future: needs a hardware partner |

### Using existing dash cams
1. The dash cam needs a **cabin-facing lens** (dual-channel / "taxi" models with an IR interior camera).
2. It must offer a **live stream** on its Wi-Fi, usually `rtsp://192.168.x.1/...` (check the manual or try VLC). Cameras that only work with their own app can't be used without the maker's SDK.
3. Point option B (or the web app's **Dash cam / IP stream**) at that URL. The software reconnects by itself when the stream drops.

## 3. Non-functional requirements

| Requirement | How it is met | Status |
|---|---|---|
| **Real-time latency** | Landmarks every frame; heavy models throttled (emotion 5 Hz, eye CNN 15 Hz, YOLO 4 Hz, temporal model 1 Hz). Measured ~25–30 FPS on a 4-core cloud CPU with everything on; face CNN 4.6 ms, cabin CNN 9.6 ms (ONNX, 2 threads) | ✅ measured |
| **Alert reaction time** | Microsleep alert 1.0 s after eyes close; eyes-off-road at 3.0 s (Euro NCAP rule); measured exactly on scripted test clips (`dms.tools.evaluate_alerts`) | ✅ measured |
| **Works offline / poor network** | All inference local; SQLite event store in WAL mode; background sync with exponential backoff (30 s → 15 min); unsynced data never deleted | ✅ tested (`test_event_store_offline_then_sync`) |
| **Availability / self-healing** | Live camera streams reconnect automatically (1 s → 30 s backoff); systemd `Restart=always`; Docker `HEALTHCHECK` on `/api/health` | ✅ code; reconnect tested via CLI logic, not on a real dash cam |
| **Observability** | `/api/health` (uptime, sessions, model integrity, store and sync status); per-frame JSON logs (`--log`); trip reports | ✅ |
| **Security & privacy** | `docs/SECURITY.md`: loopback default, tokens, Origin/Host checks, HTTPS, model checksums, no footage stored, least-privilege service | ✅ tested |
| **Low false-alarm rate** | Per-driver calibration; smoothing + hysteresis; debounce and cooldowns per alert type; strong evidence required for spoken "impact" alerts; jolt detector needs a coherent camera shift (ignores tunnels, covered lens, a face leaving) | ✅ unit-tested; ⏳ false alarms per hour on real drives not yet measured |
| **Trustworthy confidence** | Temperature scaling on validation data, stored in checkpoints, applied at runtime | ✅ |
| **Resource use** | CPU-only; models 4–17 MB each; ~25 MB total pretrained; ONNX export for mobile/edge runtimes | ✅ |
| **Maintainability** | Modular package, 33 unit/integration tests, GitHub Actions CI on every push, self-describing checkpoints | ✅ |
| **Portability** | Python on macOS/Linux/ARM; ONNX models; MediaPipe has Android/iOS SDKs with the same outputs | ✅ desktop; 📐 mobile |
| **Configurability / localisation** | All alert texts in one table (`dms/fusion/alerts.py MESSAGES`), so Hindi/Marathi is a translation; thresholds are constructor arguments | ✅ |
| **Functional safety** | A driver-assistance warning system in a product would need ISO 26262 / Euro NCAP test-protocol validation | ❌ out of scope for a student project (state this clearly) |

## 4. Android app design (option C)

- **Kotlin**, CameraX (phone camera) **or** an RTSP player (ExoPlayer/FFmpeg) for the dash cam's stream.
- **MediaPipe Tasks Android** (FaceLandmarker, AudioClassifier with the same `.task`/`.tflite` files) plus **ONNX Runtime Mobile** for the eye/emotion and YOLO models exported by `dms.training.export`.
- Port the logic files 1:1: `features/temporal.py`, `fusion/risk.py`, `fusion/alerts.py`, `fusion/mishap.py`. They are plain arithmetic, about 600 lines.
- **Foreground service** (`foregroundServiceType="camera|microphone|connectedDevice"`) with a persistent notification. It keeps running with the screen locked.
- Alerts: `TextToSpeech` (works offline, supports hi-IN / mr-IN), alarm audio stream, vibration.
- SOS: after the countdown, an SMS with the GPS location to the emergency contact via `SmsManager`. SMS needs no mobile data, only signal.
- Storage: Room/SQLite with the same schema as `dms/storage.py`; WorkManager for sync when online.
- **iPhone:** iOS stops background video processing, so on iPhone use option B, with the phone receiving Bluetooth LE alerts (allowed in the background).

## 5. What "production-ready" still needs (honest gap list)

1. **Real-world validation:** night driving with an IR camera, sunglasses, different skin tones and ages, 10+ hours of real driving to measure false alarms per hour.
2. **Drowsiness-over-time model** trained on UTA-RLDD (code ready, dataset pending).
3. **Accident detection on real crash data:** the logic is tested on simulated jolts and real crash *sounds* (ESC-50), but not on real collisions. Phones should add **accelerometer** data (a crash is a > 4 g spike), which is far more reliable than the camera.
4. **Native mobile app** (§4) and Bluetooth buzzer hardware.
5. **Per-user authentication** for multi-driver fleets; signed OTA model updates.
6. **Regulatory work** (functional safety, data-protection impact assessment) before any commercial use.
