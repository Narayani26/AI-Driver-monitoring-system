# Project overview: what it does, how it works, how well, and what is new

## 1. What the system does

A camera watches the driver. Video stays on the device and is analysed in real time (25–30 frames/s on a laptop CPU, no internet needed). The system warns the driver with a voice, buzzer tones and a suggestion when it sees:

| Hazard | How it is detected |
|---|---|
| **Microsleep / eyes closed** | eye opening vs the driver's own calibrated open eye, confirmed by an eye-state CNN; ≥ 1 s closed (Normal sensitivity) |
| **Drowsiness building up** | PERCLOS (eyes-closed percentage over 60 s), long blinks, yawns, head nodding, and a learned 30 s drowsiness model |
| **Head dropping (side cameras too)** | nose dropping relative to the ear, in head lengths, compared with the driver's normal position |
| **Eyes off the road** | head pose and gaze. Euro NCAP rules: one look away ≥ 3 s, or 10 s within 30 s |
| **Phone, drinking, smoking** | object detector near the driver's head, hands and lap |
| **Stress** | facial emotion (anger/fear); raises risk, never alerts alone |
| **Cabin sounds** | snoring, sirens, crash/skid, shouting, phone ringing, crying baby |
| **Possible accident** | camera jolt or crash sound, then the driver doesn't respond → 15 s countdown → SOS |

Everything feeds one **0–100 risk score**, with smoothing and levels that don't flicker. Every alert carries its reasons (explainable). Trips, alerts and the driver's feedback go to a cloud backend when online.

## 2. Tech stack

| Part | Technology |
|---|---|
| In-car web app | HTML/CSS/JavaScript (no framework), Canvas, WebSocket, getUserMedia, AudioWorklet, Web Speech (voice), Web Audio (buzzer) |
| In-car server | Python 3.12, FastAPI + Uvicorn, a background thread for the AI models, SQLite (offline store), HTTPS sync |
| Cloud backend | FastAPI, SQLAlchemy 2, psycopg 3, **PostgreSQL 16 on Supabase** (or Docker / SQLite); dashboard in HTML/CSS/JS |
| Machine learning | PyTorch, timm, Ultralytics YOLO, MediaPipe, ONNX Runtime, scikit-learn, OpenCV |
| Training | Kaggle T4 GPUs (image models), laptop CPU (temporal models); pandas, matplotlib |
| DevOps | Git/GitHub, GitHub Actions (unit tests, PostgreSQL tests, Docker build), Docker, Docker Compose, systemd |

## 3. The deep-learning models and how they work

**Deep learning** here means neural networks with many layers that learn their own features from labelled examples, instead of hand-written rules. The system uses nine networks:

| # | Network | Type | Trained by | What it does live |
|---|---|---|---|---|
| 1 | **MediaPipe Face Landmarker** | CNN face detector + mesh regressor | Google (pretrained) | 478 3-D face points, 52 expression scores, head rotation |
| 2 | **FaceStateNet** (ours) | CNN: MobileNetV3-Large / EfficientNet-B2 backbone with 3 output heads | **us** (Kaggle T4) | eye open/closed (live); yawn and emotion heads evaluated |
| 3 | **HSEmotion** | CNN EfficientNet-B0 | pretrained on AffectNet | facial emotion (live) |
| 4 | **CabinNet** (ours) | CNN EfficientNet-B0 | **us** on State Farm | 10 driver behaviours from side-view cameras |
| 5 | **Frame-drowsiness CNN** (ours) | CNN EfficientNet-B2 | **us** on NTHU-DDD | supporting drowsiness evidence (optional) |
| 6 | **Hybrid Conv-Transformer** (ours) | 1-D convolution + Transformer encoder + statistics branch | **us** on UTA-RLDD (60 people) | drowsiness from 30 s of face measurements |
| 7 | **YOLO11n** | single-shot object detector (CNN) | pretrained on COCO | phone, bottle, cup |
| 8 | **YOLO11n-pose** | detector + keypoint regressor | pretrained on COCO | nose, eyes, ears, shoulders from **any angle** (side cameras) |
| 9 | **YAMNet** | CNN on audio spectrograms | pretrained on AudioSet | 521 sound classes, grouped into cabin events |

**How they work:**
- **CNNs** (2–5, 7–9) slide small learned filters over the image. Early layers find edges, middle layers find parts (an eyelid, a phone corner), and late layers find objects or states.
  - *MobileNetV3* and *EfficientNet* are CNN families designed to be accurate but cheap, so they run in real time on a CPU.
  - *Transfer learning:* we start from ImageNet weights and fine-tune them on driver data.
- **Multi-task learning (FaceStateNet):**
  - One shared backbone feeds three small heads: eye, yawn and emotion. Each dataset labels only one task, so the loss of a missing label is masked out.
  - The three losses are balanced automatically with learned uncertainty weights (Kendall et al. 2018).
  - One forward pass answers all three questions.
- **Temporal model:** drowsiness is a process, not a single frame. Every 0.1 s we record 9 face measurements: eye opening, mouth, pitch, yaw, roll, gaze, blink and jaw scores, all *relative to the driver's own calibration*.
  - A 1-D convolution turns 30 s (300 steps) into short "patches".
  - A Transformer encoder uses self-attention to relate every moment to every other (e.g. a long blink 20 s before a yawn).
  - A parallel branch computes classic statistics (PERCLOS, blink rate, pose variance). Both are combined to output *alert* or *drowsy*.
- **Calibration of confidence:** every trained model has *temperature scaling* fitted on validation data. A reported 90% then really means about 90%, measured with ECE (expected calibration error).
- **YOLO** predicts boxes and classes in one pass over the image; the pose variant also predicts 17 body keypoints per person.

## 4. Accuracy (all on data the models never saw; people never seen in training unless stated)

| Task | Test set | Result | Notes |
|---|---|---|---|
| Eye open/closed | MRL Eye, **unseen people** (12,576 images) | **98.0%** | ECE 0.006 after calibration |
| Yawn | Yawn-Eye test (219) | 99–100% | small set without person IDs: upper bound |
| Emotion (7 classes) | FER2013 + RAF-DB test (10,246) | **75.4%** (EfficientNet-B2), 73.8% (MobileNetV3) | |
| Emotion, other dataset | AffectNet+ (3,500, never trained on) | 48.4% ours · **63.1% HSEmotion** | why the live demo uses HSEmotion |
| Driver behaviour (10 classes) | State Farm, **unseen drivers** (3,440) | **92.1%** (99.6% with the usual random split) | |
| Frame drowsiness | NTHU-DDD, unseen person (5,953) | 66.7% (94–97% random split) | only 4 people in this copy: indicative |
| **Drowsy vs alert over time** | UTA-RLDD, 5-fold, **unseen people** | **87.3%** videos (Random Forest), **80.5%** (Hybrid, deployed) | chance 50% |
| 3 levels (alert/low/drowsy) | UTA-RLDD, unseen people | 64.7% (RF), 55.6% (Transformer) | chance 33%; published baseline ≈ 65% |
| Calibration ablation | UTA-RLDD, hybrid model | 53.4% **with** vs 37.6% **without** per-driver calibration | **+15.8 points** |
| Snoring / siren / baby / crash | ESC-50 (560 clips) | caught 95% / 93% / 82% / 78% (90% at a lower threshold) | false shouting alarms 0% |
| Side camera (90° profile) | stock clip, 77 frames | face found 0/77 (MediaPipe) → head found 77/77 (pose model); nod alert raised | qualitative |
| Speed | laptop CPU | 12–16 ms per frame end to end (≈ 30 FPS); models in the background | |

**Honest reading:**
- Per-frame tasks (eyes, yawns, behaviour) are at or near the state of the art.
- Drowsiness *over time* on unseen people is the hard problem: 80–87% for two classes.
- Emotion from one photo does not transfer well between datasets. That's why emotion only modifies the risk score.
- Not measured yet: false alarms per hour on real drives. The retraining pipeline (§7) is built to measure this from driver feedback.

## 5. What already exists vs what this project adds

**Existing (we build on it):**
- PERCLOS (Wierwille 1994) and the eye aspect ratio (Soukupová & Čech 2016).
- Euro NCAP's driver-state-monitoring rules.
- MediaPipe, YOLO, HSEmotion, YAMNet.
- Public datasets: MRL, FER2013, RAF-DB, AffectNet, State Farm, NTHU-DDD, UTA-RLDD.
- Commercial systems (Seeing Machines, Smart Eye, built-in car DMS) use dedicated infrared cameras at fixed mounting positions.

**What is new here (contributions suitable for a paper):**
1. **Leakage quantified on two driver datasets.**
   - The common random split overstates accuracy: State Farm 99.6% vs 92.1% on new drivers (17× more errors), and up to 41× with a larger network. NTHU-DDD shows 7–11× more errors.
   - Larger models *widen* the gap: they memorise who the person is.
2. **Per-driver calibration, measured:**
   - Normalising every feature to the driver's own baseline, in training and live, adds +9.6 to +15.8 points on unseen people.
   - Without it, the model drops to near chance.
3. **View-adaptive monitoring for arbitrary camera placement:**
   - Pose-gated eye measurement: eyes are not judged when the head is tilted, which removes false microsleeps from low webcams.
   - A body-pose fallback for side mounts: head-nod detection and an eye-crop CNN.
   - **Camera-move detection:** background change + driver geometry change → automatic re-calibration. It is kept separate from a driver turning their head, which still counts as distraction.
   - Phone and dash-cam installs are rarely fixed or frontal; most DMS work assumes a fixed infrared camera.
4. **Evidence-gated risk fusion against false alarms:**
   - Learned models trained on other people and cameras act only as *supporting* evidence: alone they reach at most CAUTION.
   - A WARNING needs measurable signs. Every alert lists its reasons.
5. **Privacy-preserving, human-in-the-loop continual learning:**
   - Only numeric face measurements leave the car (no video), and only with consent.
   - Labels come from the driver: rating alerts, plus a Karolinska sleepiness rating.
   - An auditable quality gate (driver visible, frontal, same person, moving).
   - Evaluation split by driver, and promotion only if the false-alarm rate doesn't rise.
6. **Multi-modal mishap detection:**
   - Camera jolt (phase correlation) + crash sound + driver responsiveness.
   - Escalation from a silent check, to a calm notice, to a question, to an SOS countdown, so a sound from the radio doesn't trigger an alarm.

## 6. Situations from a driver's point of view

| Situation | What the system does |
|---|---|
| Calibrated looking at the camera, later the camera is moved to the side | notices the move in about 3 s, says "Camera moved, re-learning your position", re-calibrates in side view (5 s) and keeps monitoring. The trip and report are kept |
| Driver turns to look at a phone or a passenger | **not** a camera move (the background is unchanged): "Please look ahead" after 3 s |
| Mirror checks (< 2 s each) | nothing; many glances (10 s within 30 s): a gentle caution |
| Head tilted back or down in front of a low webcam | eyes "not judged" (open eyes look closed from below); no false microsleep |
| Real eyes closed ≥ 1 s | "Wake up! Your eyes were closed." with an alarm |
| Side camera, driver nodding off | head-lowering caution, then "Wake up! Your head is dropping." |
| Passenger in the back leans in | ignored; the system follows the driver's face |
| Phone held up or to the ear | "Please put the phone down" within about 1 s |
| Driver leaves the seat | "Looking for the driver"; "Driver not visible" after 5 s; never "safe" |
| Crash sound from the radio or another car | calm "Loud crash sound nearby" notice, no alarm |
| Real jolt and the driver doesn't respond | "Possible accident" → 15 s countdown → SOS, unless "I'm OK" is pressed |
| Tunnel or clouds (brightness change) | no re-calibration (the background comparison ignores brightness) |
| Too many alerts | the same alert can't repeat within 30 s (warnings) or 90 s (cautions); only warnings and critical alerts are spoken |

Known limits:
- **Night driving needs an infrared camera.** A normal webcam can't see the eyes in the dark.
- Dark sunglasses hide the eyes. Head nodding and attention still work.
- Audio can't tell *who* is snoring.
- The false-alarm rate on real drives still has to be measured.

## 7. Data and retraining pipeline

```
car: alert ─► driver taps ✓/✗, rates sleepiness ─► 30 s of numbers (with consent) ─► sync (offline-first)
cloud: quality gate ─► samples table ─► weekly: export split by driver ─► enough data? ─► train candidate
       ─► test candidate AND current model on held-out real drivers ─► approve only if F1 ≥ current and
          false alarms don't rise ─► model registry ─► devices download + verify SHA-256 ─► admin can roll back
```
Commands: `python -m dms.cloud.retrain` (or the weekly systemd timer `deploy/dms-retrain.timer`), and on the device `python -m dms.cloud.fetch_model`.

## 8. Security

| Area | What was done |
|---|---|
| Footage | never stored or uploaded; only numbers and alert metadata leave the car |
| In-car web app | listens on this computer only by default; when shared on a network: random access token (HttpOnly, SameSite cookie), Origin/Host checks against cross-site WebSocket hijacking and DNS rebinding, a strict Content-Security-Policy, size limits, and an allow-list for stream URLs |
| Model files | SHA-256 checked at start; the server refuses tampered models; downloaded model updates are verified before use |
| Accounts | passwords hashed with scrypt + salt (≥ 10 characters, upper case and a digit); HMAC-signed login tokens valid for 12 h; login blocked after 5 failures in 15 min; the same error for wrong email or password |
| Devices | each dash cam or phone has its own random 256-bit key; only its hash is stored; revocable; one account can never write into another's data (tested) |
| Database | Supabase PostgreSQL with Row Level Security on every table and no public policies: the public API can read nothing. Only the backend (owner) connects, over TLS |
| Privacy rights | training data only with opt-in consent; withdrawing consent deletes it; deleting the account erases everything; exported training data uses pseudonyms, not emails |
| Transport | sync and model download require HTTPS (plain http only for localhost testing); store-and-forward retries with backoff |
| Containers / services | non-root Docker user, database not exposed outside Docker Compose, hardened systemd units (no new privileges, read-only system) |
| Supply chain | CI runs unit tests, tests on a real PostgreSQL and a Docker build on every push |
