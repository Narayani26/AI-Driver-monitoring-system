# POE / viva preparation: demo, explanation, code, questions

## 0. Before 10:00 (checklist)
- [ ] `git pull`, then restart both apps (latest fixes: dozing alert, yawns at a distance, calmer score, printable report).
- [ ] Charger plugged in, Chrome open, camera and microphone allowed for localhost.
- [ ] Terminal 1 (backend): `source .venv/bin/activate`, then `set -a; source .env; set +a; python -m dms.cloud.app`.
- [ ] Terminal 2 (car app; replace `<key>` with your device key):
  `DMS_SYNC_TOKEN=<key> python -m dms.web.server --face-ckpt runs/face/best.pt --temporal-ckpt runs_temporal2/hybrid_binary/deploy.pt --audio --mishap --sync-url http://127.0.0.1:9000/api/ingest --share-training-data`
- [ ] Tabs: http://localhost:8000 (car app), http://127.0.0.1:9000 (dashboard, logged in), Supabase Table Editor.
- [ ] Demo videos in one folder (see the end of this file). Sensitivity: **Normal**.
- [ ] Open `docs/submission/`: Report, Paper and Slides (PDF and DOCX/PPTX).
- [ ] Backup: a 2-minute screen recording of a working demo (Cmd+Shift+5), in case Wi-Fi, the camera or the laptop misbehaves.

## 1. Demo script (5-6 minutes)
1. **Start (webcam, sit normally).** Say: "The first 5 seconds learn *my* open-eye shape and neutral head pose from whatever angle the camera has. That is per-driver calibration, worth 16 points of accuracy in our tests." The badge shows "front view · monitoring".
2. **Close your eyes for 2 s.** "Wake up! Your eyes were closed." with an alarm and CRITICAL. Explain: eye opening relative to my calibration, plus the eye CNN; a microsleep is 1 s or more.
3. **Doze: drop your head forward with your eyes closed.** The same alarm within about 1-2 s.
4. **Tilt your head back with your eyes open.** Eyes show "not judged" and no false alarm. Explain why: from a low camera, open eyes look closed.
5. **Look left for 4 s.** "Please look ahead at the road" (Euro NCAP: 3 s). A quick mirror glance under 1.5 s does nothing.
6. **Hold the phone to your ear.** "Please put the phone down" within about 1 s (YOLO, only near the driver).
7. **Yawn twice, slowly.** The yawn counter goes up and the drowsiness score rises.
8. **Tap "✗ Not true"** on an alert. The score drops at once and that alert pauses. "Driver feedback acts immediately and becomes training data with consent."
9. **Move the laptop or camera to a side angle.** "Camera moved, re-learning", then monitoring resumes (side view if no frontal face).
10. **Video file:** play the side-view or night clip. Show side view (head points, head-drop value) and zoom search for small or dark faces.
11. **Trip report**, then click sleepiness 1-9, then **Print / save PDF** (one clean page).
12. **Dashboard:** the trip appears within about 30 s, with alerts and your ratings; the Admin section shows data quality and the model registry. **Supabase:** the same rows in `trips` and `events`.

If something misfires, say so plainly: "this is why drivers can tap 'Not true', and why learned models are only supporting evidence."

## 2. Explaining the system (2-minute version)
"A camera feeds our in-car app. Every frame, MediaPipe gives 478 face points. From them we compute eye opening, mouth opening, head angles and gaze, always relative to the driver's own calibration. In a background thread, neural networks add evidence: our own eye-state CNN, YOLO for phones, a pose model for side cameras, emotion, cabin sounds, and our Transformer that looks at the last 30 seconds. Trackers turn this into blinks, PERCLOS, microsleeps, yawns, nods and glances, following Euro NCAP rules. A fusion model gives one 0-100 risk score with reasons, and the alert manager speaks a warning with a suggestion. Everything works offline. Trips and alerts sync to a FastAPI backend on Supabase PostgreSQL. With consent, ratings become training data; a weekly job retrains and only deploys a model if it is better on drivers it has never seen."

## 3. Code walkthrough (where things are)
| File | What it does | Key names |
|---|---|---|
| `dms/web/server.py` | FastAPI app, WebSocket per browser tab; receives frames and audio, returns results | `ws_endpoint`, `Session.process`, `handle_text` (feedback, KSS) |
| `dms/web/static/app.js` | browser UI: camera capture, overlay, gauges, alerts, voice, report | `captureLoop`, `onAlert`, `toast`, `showReport` |
| `dms/pipeline.py` | **the heart**: one frame in, a full `FrameResult` out | `DriverMonitor.process`, `_schedule_models`, `_frontal`, `_zoom_search`, `dismiss` |
| `dms/perception/face.py` | MediaPipe wrapper, driver selection, zoom search | `FaceLandmarker`, `select_driver`, `detect_zoomed` |
| `dms/perception/pose.py` | side-view head keypoints | `HeadPoseDetector`, `head_from_keypoints` |
| `dms/features/geometry.py` | EAR, MAR, head pose, gaze | `compute_features` |
| `dms/features/temporal.py` | calibration and trackers | `Calibrator`, `EyeClosureTracker`, `YawnTracker`, `GlanceTracker`, `HeadNodTracker` |
| `dms/features/camera.py` | camera-move detection | `CameraMoveDetector` |
| `dms/fusion/risk.py` | risk score, levels, reasons | `RiskFusion.update`, `_drowsiness`, `_distraction` |
| `dms/fusion/alerts.py` | messages, cooldowns, debounce, mute | `MESSAGES`, `AlertManager.update` |
| `dms/fusion/mishap.py` | jolt + crash sound + responsiveness, SOS | `JoltDetector`, `MishapDetector` |
| `dms/models/networks.py` | FaceStateNet, CabinNet, Transformer, Hybrid | `build_temporal` |
| `dms/training/*` | training and evaluation | `train_image.py`, `train_temporal.py` |
| `dms/storage.py` | offline SQLite and the sync thread | `EventStore`, `Syncer` |
| `dms/cloud/*` | backend, database, auth, quality gate, export, retrain | `app.py`, `db.py`, `auth.py`, `quality.py`, `retrain.py` |

**One frame, step by step (`DriverMonitor.process`):**
1. Face landmarks are found; if none, a zoomed second look.
2. Features are computed.
3. Due models are scheduled in the background thread.
4. Camera-move check.
5. Calibration (first seconds).
6. Eye, yawn, nod and glance trackers update.
7. Risk fusion.
8. Mishap check.
9. Alert manager.
10. Trip recorder.

## 4. Questions you are likely to get (with answers)
**Basics**
- *What is EAR?* Eye Aspect Ratio = (sum of the two vertical eye distances) / (2 × the horizontal distance). It is about 0.25-0.3 when open and drops towards 0.1 when closed. We use it relative to the driver's calibrated open eye.
- *What is PERCLOS?* The percentage of time the eyes are at least 80% closed over 60 s, the most validated drowsiness measure (NHTSA). Above roughly 15% is drowsy.
- *Microsleep vs blink?* A blink is under about 0.4 s; a microsleep is closure of 1 s or more (Normal sensitivity).
- *How is a yawn detected?* Mouth open (jaw blendshape > 0.45 or mouth ratio > 0.55) held for 1.5 s; talking does not hold it that long.
- *How do you know where the driver is looking?* Yaw and pitch from MediaPipe's facial transformation matrix, plus iris position (gaze), compared with the calibrated neutral pose.

**Deep learning**
- *Which deep-learning models did you train?* FaceStateNet (eye/yawn/emotion, MobileNetV3 or EfficientNet-B2), CabinNet (State Farm behaviours), the frame-drowsiness CNN (NTHU) and the Hybrid Conv-Transformer (UTA-RLDD).
- *Why multi-task?* One backbone, three answers, one forward pass: cheaper and it shares features. Missing labels are masked, and losses are balanced by learned uncertainty (Kendall 2018).
- *Why a Transformer for drowsiness?* Drowsiness is a pattern over time (long blinks, then yawns, then nodding). Self-attention relates every moment of the last 30 s to every other. The statistics branch adds classic PERCLOS and blink rate.
- *What is transfer learning?* We start from ImageNet weights and fine-tune on driver data, which needs far less data.
- *What is temperature scaling / ECE?* It divides logits by a fitted temperature so that "90% sure" really is right about 90% of the time; ECE measures that gap (0.006 for eyes).
- *Why are there pretrained models too?* MediaPipe, YOLO, HSEmotion and YAMNet are trained on huge datasets we cannot match. We use the best model per task.

**Evaluation**
- *Why not report 99%?* With a random split the same people are in train and test: State Farm gives 99.6% that way, but 92.1% on new drivers. We always split by person and check it in code.
- *What is subject-independent / GroupKFold?* Folds are formed by person, so test people are never seen in training.
- *Why is drowsiness accuracy "only" 80-87%?* It is the hard problem: self-reported labels and people differ. The dataset's published baseline is about 65% for three levels; we match it and reach 87% for alert vs drowsy.
- *What does calibration give?* +15.8 points (53.4% vs 37.6%) on unseen people.
- *How did you measure the sound detection?* ESC-50: 560 clips, live alert rule; snoring 95%, siren 93%, crash 78%; 0% false shouting alarms.
- *False alarms?* Learned models alone can't exceed CAUTION; glances under 1.5 s don't count; cooldowns; "Not true" feedback. On-road false alarms per hour are now measurable from the ratings.

**Real situations**
- *Glasses?* Works (tested on our own webcam footage with glasses). *Sunglasses?* The eyes are hidden: eyes are not judged, but head nodding and attention still work.
- *Night?* An ordinary camera can't see the eyes in the dark; an infrared camera is needed (future work). The zoom search helps on dark dash cams (0% to 71% face detection).
- *Passenger?* The driver is the largest face, then tracked by position; passengers are ignored.
- *Camera moved mid-drive?* The background and the driver's position change together, detected in 0.75-1.5 s, re-learned in about 3 s. A driver turning the head leaves the background unchanged, so it is still distraction.
- *Side-mounted camera?* The face detector finds nothing at 90°, so the pose model tracks nose, eye and ear: head-drop alarm and eye check.
- *Crash?* A jolt or crash sound starts a silent check; if the driver doesn't respond for 3 s, "Possible accident" and a 15 s countdown to SOS. A crash sound from the radio alone gives only a calm notice.
- *Internet lost?* Everything runs locally; alerts queue in SQLite and upload later.
- *Talking or singing?* Mouth open must be held 1.5 s to count as a yawn; occasional miscounts only nudge the score.

**Architecture and backend**
- *Why FastAPI + WebSocket?* Asynchronous, fast, one persistent connection per tab for frames and results; Python, so it sits next to the ML code.
- *Why does lag not grow?* The browser sends a new frame only after the previous result arrives, and heavy models run in a background thread.
- *Database?* PostgreSQL 16 on Supabase via SQLAlchemy. Tables: users, devices, trips, events, samples, model_versions. RLS is on, with no public policies.
- *Security?* scrypt passwords, signed 12 h tokens, login rate limit, per-device hashed keys, account isolation, SHA-256 model checks, CSP, no video stored.
- *Is retraining automatic?* The pipeline is built and tested end to end. It runs with `python -m dms.cloud.retrain` or the weekly systemd timer on a server; on the laptop it is not scheduled yet, and it only trains once enough drivers (5) have shared data.
- *What does the JSON/CSV trip report contain?* Summary KPIs, every alert with its time, and a per-second timeline of risk, drowsiness, distraction and PERCLOS: no video, a few KB.

**Novelty (if asked "what is new?")**
1. Measured identity leakage on two driver datasets.
2. Measured per-driver calibration.
3. View-adaptive monitoring: any camera position, moved camera, small or dark faces.
4. Evidence-gated fusion against false alarms.
5. Privacy-preserving learning from driver feedback with a safe promotion rule.
6. Multi-modal accident check.

## 5. Demo videos (free)
- **Your own recordings are the best demo**, because they're legal and realistic: normal driving posture, closing the eyes, yawning, phone, look-away, head drop. Record 20-40 s each with QuickTime and play them with **Video file**.
- Free stock sites: Pexels and Pixabay (free licence) and Coverr (royalty-free). Search for "tired driver", "driver sleeping", "driver yawning", "driver phone", "driving at night car interior", or the [Coverr fall-asleep page](https://coverr.co/stock-video-footage/fall-asleep) and [Coverr yawning page](https://coverr.co/stock-video-footage/yawning).
- Research datasets with real driver videos: UTA-RLDD (already used), the [DMD dataset](https://dmd.vicomtech.org/) (free for academic use after a form; drowsiness, phone, drinking, yawning) and [YawDD on IEEE DataPort](https://ieee-dataport.org/taxonomy/term/3469) (yawning).
- **Crash:** no free licensed in-car crash video exists for this. Demonstrate it live: play a car-crash sound (Pixabay sound effects, "car crash") and tap or shake the laptop, then don't respond, which starts the countdown; press "I'm OK" to cancel. Enable it with `--mishap --audio`.
