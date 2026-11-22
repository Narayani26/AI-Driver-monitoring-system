# Research study: AI-based Driver Monitoring System

*Drowsiness, emotion and unsafe-behaviour detection in real time. Study date: November 2026.*

This document is the research basis for the project. It covers the state of the art, public datasets and repos, what is weak in existing work, the novelty claims, the evaluation protocol, and the plan for the paper, report and slides.

> Numbers marked **(verify)** come from memory or secondary sources. Check them against the original paper or dataset page before they go into the paper.

---

## 1. Problem framing

A camera-based Driver Monitoring System (DMS) watches the driver through an in-cabin camera and decides, in real time, whether the driver is fit to drive. We cover three dimensions:

| Dimension | What we detect | Signals |
|---|---|---|
| **Drowsiness / fatigue** | alert → low-vigilant → drowsy; microsleeps | eye closure (EAR, PERCLOS, blink duration), yawns, head nods, learned temporal patterns |
| **Distraction / unsafe behaviour** | eyes-off-road (long and repeated short glances), phone use, texting, drinking, reaching back, talking to a passenger | head pose, gaze, cabin-view behaviour classifier |
| **Affect (emotion)** | neutral, happy, sad, surprise, fear, disgust, anger → negative high-arousal "stress/anger" state | facial expression network |

**Why now: regulation.** From 1 January 2026 the Euro NCAP *Safe Driving – Driver Engagement* protocol scores driver monitoring heavily. Reports describe up to 25 points available, versus about 2 previously, and real-time eye and head tracking is required for top scores. The protocol distinguishes **long distraction** (one long glance away) from **visual attention time sharing** (repeated short glances), and wants drowsiness severity assessed against recognised sleepiness scales. Steering-only (indirect) monitoring is no longer enough ([Euro NCAP SD-202 v1.1](https://www.euroncap.com/media/91719/sd-202-driver-monitoring-test-procedure-v11.pdf), [Smart Eye summary](https://smarteye.se/blog/euro-ncap-2026-whats-changing/), [ETSC](https://etsc.eu/euro-ncap-new-2026-protocols/)). Our alert logic is built around these definitions, which gives the paper a concrete, current motivation.

**Scope decision.** Camera-only (single RGB or NIR webcam). Vehicle-signal fusion (CAN bus, IMU) is listed as future work. Emotion is treated as a **risk modifier**, not a standalone hazard, because the evidence links negative high-arousal states to degraded driving, not emotions in general.

---

## 2. State of the art

### 2.1 Approaches, roughly chronological

1. **Geometric / threshold methods.** Eye Aspect Ratio (EAR, Soukupová & Čech 2016), Mouth Aspect Ratio (MAR), PERCLOS (percentage of eyelid closure over time; the P80 criterion = eyes ≥80% closed). Fast and explainable, but they use fixed thresholds that ignore individual eye shape and camera placement. Most GitHub projects are still here.
2. **Per-frame CNN classifiers.** Eye open/closed on MRL-Eye/CEW, yawn/no-yawn, distraction classes on State Farm. They report very high accuracy, but usually with random image splits (see §4).
3. **Temporal deep models.** LSTM / HM-LSTM on landmark features (UTA-RLDD baseline, Ghoddoosian et al. 2019), 3D CNNs, two-stream networks, Video Swin / ViT ([arXiv 2207.12148](https://arxiv.org/pdf/2207.12148)), LSTM autoencoders for anomaly-style drowsiness ([arXiv 2209.05269](https://arxiv.org/pdf/2209.05269)). Frame-level ViT results of 98.89% (NTHU-DDD) and 99.4% (UTA-RLDD) are reported ([IEEE Access 2024](https://ieeexplore.ieee.org/document/10815964)), but appear to be within-dataset.
4. **Combined systems.** A January 2026 *Sensors* paper (Zambrano et al.) combines fatigue, distraction and emotion using facial landmarks + DL. It reports EAR 88.89%, MAR 85.19% and 100% for head pose/gaze, and **notes that emotion has traditionally been studied separately from fatigue and distraction** ([Sensors 26(3):889](https://doi.org/10.3390/s26030889)). This is our closest prior work and the main paper to compare against. An edge-AI facial-expression DMS is in [PMC12611093](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12611093/).
5. **Vision-language models (2026 frontier).** [arXiv 2606.02273](https://arxiv.org/abs/2606.02273) (June 2026) builds natural-language descriptions for Drive&Act. Zero-shot VLMs cannot reliably describe fine-grained driver activity; fine-tuning raises their ACCR score from 66 to 76 and generalises to DMD. VLMs are far too heavy for a real-time in-car loop, but are an interesting optional "explanation" layer (§6, future work).
6. **Personalised thresholds (closest to our calibration idea).** Ersoy et al., April 2026 ([arXiv 2604.22479](https://arxiv.org/abs/2604.22479)): a short pre-drive calibration sets EAR/MAR thresholds at 75% / 140% of the driver's own baseline. This improved eye-state accuracy from 91.70% to 93.23% and yawning from 95.90% to 97.23%. Their CNN on 64×64 eye/mouth crops scored higher still (~98–99%, verify in the PDF). **We must cite this and position against it** (see C2).
7. **Lightweight driver FER.** Shuffle Vision Transformer ([arXiv 2409.03438](https://arxiv.org/pdf/2409.03438)) and a dual-attention network on KMU-FED ([MDPI Designs 2025](https://www.mdpi.com/2411-9660/9/2/45)).

Surveys: drowsiness-only survey ([AI Review 2026](https://link.springer.com/article/10.1007/s10462-026-11505-w)), [arXiv 2206.07489](https://arxiv.org/pdf/2206.07489), distracted-driving modalities review ([arXiv 2501.11758](https://arxiv.org/pdf/2501.11758)). **No 2025–2026 survey found covers all three dimensions together**, which supports the "unified system" framing.

### 2.2 Open-source projects

| Repo | What it does | What it lacks |
|---|---|---|
| [dejely/manobela](https://github.com/dejely/manobela) | Phone-camera DMS: eye closure, yawn, head pose, phone use (MediaPipe + YOLO) | Threshold-based; no learned temporal model or emotion; no evaluation protocol |
| [jhan15/driver_monitoring](https://github.com/jhan15/driver_monitoring) | MediaPipe face mesh + YOLOv5 phone detection | Pinned to 2022 mediapipe; fixed thresholds |
| [lpianta/SafeDrive](https://github.com/lpianta/SafeDrive) | Drowsiness, head orientation, phone detection (separate scripts) | Not integrated; no fusion |
| [albertnew2012/Driver-Attention-Monitor-System](https://github.com/albertnew2012/Driver-Attention-Monitor-System) | 468-pt mesh, yawn, head movement | No phone or emotion detection |
| DrowSAFE ([Hackster](https://www.hackster.io/MohamedAliBedair/drowsafe-real-time-driver-drowsiness-detection-system-aeee70)) | Raspberry Pi; PERCLOS + EAR + head pose + yawn composite score | No learning; no distraction classes |
| [arXiv 2511.13618](https://arxiv.org/pdf/2511.13618) | MediaPipe + EAR real-time drowsiness | Single dimension |

**Pattern:** public repos are threshold pipelines with no rigorous evaluation. Published papers have learned models but often weak evaluation protocols. Nobody combines both well.

> **License decision:** Ultralytics YOLO11 is **AGPL-3.0**, so the project as a whole must be released under AGPL-compatible terms if distributed. This was accepted, and YOLO11n is used for phone/drink detection (fine-tunable for cigarettes/seatbelts). The pretrained emotion baseline (HSEmotion) and MediaPipe are Apache-2.0.

---

## 3. Datasets

| Dataset | Task in our system | Size / subjects | Access |
|---|---|---|---|
| **MRL Eye** | eye open/closed (NIR crops) | 84,898 images, 37 subjects (verify) | free download |
| **CEW** (Closed Eyes in the Wild) | eye state, in-the-wild | ≈2.4k faces (verify) | free |
| **YawDD** | yawning (videos, mirror + dash cam) | videos of ~100 drivers (verify) | free, request form |
| **UTA-RLDD** | **3-level drowsiness, temporal** (alert / low-vigilant / drowsy) | 60 subjects, ~30 h, 180 videos | free (research) |
| **NTHU-DDD** | drowsiness (IR + RGB, glasses, night) | 36 subjects (verify) | on request from NTHU |
| **DMD** (Vicomtech) | distraction + drowsiness, 3 views, RGB/IR/depth | 37 drivers, 41 h | free, [dmd.vicomtech.org](https://dmd.vicomtech.org) |
| **Drive&Act** | 83 fine-grained activities, 6 views | 9.6M frames, 12 h | on request ([ICCV'19](https://openaccess.thecvf.com/content_ICCV_2019/html/Martin_DriveAct_A_Multi-Modal_Dataset_for_Fine-Grained_Driver_Behavior_Recognition_in_ICCV_2019_paper.html)) |
| **State Farm Distracted Driver** | 10 cabin behaviour classes | 22,424 labelled images, 26 drivers | Kaggle |
| **100-Driver** | 22 distraction classes, multi-camera, day/night | ~470k images, 100 drivers (verify) | on request |
| **AUC Distracted Driver v2** | 10 classes (same scheme as State Farm) | ~14k images, 44 drivers (verify) | on request |
| **AffectNet** | 8 emotions (we use 7) | ~440k manually labelled (verify) | on request (academic) |
| **RAF-DB** | 7 emotions | ~30k (verify) | on request |
| **FER2013** | 7 emotions, 48×48 grey | 35,887 | Kaggle |
| **KMU-FED** | driver emotion, in-vehicle NIR | 1,106 images, 12 subjects | free |
| **DEFE** | spontaneous driver emotion while driving | 60 participants | on request ([arXiv 2005.08626](https://arxiv.org/abs/2005.08626)) |

**Recommended minimal set for a student project** (all obtainable within ~1–2 weeks):
MRL Eye + a yawn set → eye/yawn heads · AffectNet or RAF-DB (FER2013 if access is slow) + **KMU-FED as driver-domain test** → emotion · **UTA-RLDD** → temporal drowsiness · **State Farm** (+100-Driver if granted) → behaviour · **DMD** → cross-dataset distraction/drowsiness test.

---

## 4. Gaps in existing work (what we exploit)

1. **Inflated accuracy from subject leakage.** Random image splits put near-identical consecutive frames of the same person in both train and test. State Farm models reach ~99% that way, but drop sharply when tested on unseen drivers. Most reported 98–99% drowsiness numbers are within-dataset. Searching for cross-dataset, subject-independent temporal evaluation on NTHU-DDD ↔ UTA-RLDD found little.
2. **Fixed thresholds.** EAR = 0.25 does not work across people (eye shape, ethnicity, glasses) or camera positions (a dash camera off to the side shifts the neutral yaw by 20°+; our own end-to-end test showed exactly this). Ersoy et al. 2026 personalise EAR/MAR thresholds, but not head pose, and not the input of a learned model.
3. **Per-frame decisions.** Drowsiness is a process (blink duration drifts, PERCLOS builds up, head nods appear), but many systems classify single frames.
4. **Emotion in isolation.** Noted explicitly by Zambrano et al. 2026.
5. **Lab-to-cabin domain gap.** FER datasets are frontal, well-lit, RGB; cabins have NIR at night, strong side light, occlusion (sunglasses, hands, masks).
6. **Missing deployment metrics.** Latency, model size, false alarms per hour and calibration (are the confidences trustworthy?) are rarely reported.
7. **Alerts not tied to standards.** Ad-hoc rules instead of Euro NCAP's long-distraction / VATS definitions.

---

## 5. Our contributions (paper claims)

**C1. A unified, explainable, standards-aligned DMS.** Drowsiness, distraction and affect are fused into a *Driver Risk Index* (0–100, four levels). It uses time-based smoothing, hysteresis and hard overrides (microsleep, long distraction). Every alert carries human-readable reasons. Distraction logic implements Euro NCAP 2026's long-distraction and visual-attention-time-sharing definitions.

**C2. Per-driver calibration extended beyond thresholds.** Personalised EAR/MAR *thresholds* already exist (Ersoy et al. 2026). Our extension: (a) we also calibrate the **neutral head pose**, which matters because a dash cam mounted off to the side shifts the apparent yaw by 20°+ and breaks fixed "looking away" rules; (b) the calibration **normalises the input of a learned temporal model**, and the training-time baseline of each subject is taken from their *alert* recording (normalising on the drowsy video would erase the signal); (c) the same transform is used in training and at runtime. **Ablation:** none / thresholds-only (≈ Ersoy et al.) / full calibration.

**C3. Lightweight conv-transformer temporal model over landmark features.** A conv stem models individual blinks and patchifies time (0.5 s tokens). A Transformer models 30 s trends. 174k parameters, **0.64 ms per window on CPU (ONNX, measured)**. Because it runs on landmark features, not pixels, it is privacy-preserving (no face images need to leave the device or be stored) and naturally more robust across datasets than pixel models. **Compared against:** GRU, Random Forest on hand-crafted PERCLOS/blink features, and threshold rules.

**C4. Partial-label multi-task face network.** One backbone (MobileNetV3/EfficientNet) with eye-state, yawn and emotion heads. It is trained on the *union* of datasets where each image has only some labels (masked loss), with learned uncertainty task weighting (Kendall et al. 2018). Cabin-domain augmentation (grayscale/NIR simulation, occlusion, motion blur) is used. **Test:** does multi-task + augmentation improve driver-domain emotion (KMU-FED) over single-task training on AffectNet?

**C5. A rigorous evaluation protocol.** Subject-independent splits enforced in code (`assert_no_leakage`), GroupKFold cross-validation with mean ± std, cross-dataset tests, window- *and* video-level metrics, macro-F1, balanced accuracy, Cohen's κ, Expected Calibration Error, false alarms per hour on alert driving, and latency/size on CPU (and a Pi/Jetson if available). Reporting **both** the leaky random-split score and the honest subject-independent score on the same model is itself a useful finding.

---

## 6. Methodology (as implemented)

```
camera ─► MediaPipe Face Landmarker (478 pts + 52 blendshapes + 4×4 head-pose matrix)
            │
            ├─► geometry: EAR (L/R), MAR, pitch/yaw/roll, iris gaze ratio, blink & jawOpen blendshapes
            ├─► calibration (first 10 s) → DriverProfile
            ├─► trackers: blinks, PERCLOS(60 s, P80), microsleep(≥1 s), yawns(≥1.5 s),
            │             long distraction(≥3 s), VATS(≥10 s cumulative in 30 s)
            ├─► 10 Hz feature sequence (30 s) ─► Conv-Transformer ─► alert / low-vigilant / drowsy  (1 Hz)
            ├─► face crop ─► FaceStateNet ─► emotion (5 Hz)
frame ──────┴─► CabinNet ─► 10 behaviour classes (3 Hz)
                        ▼
                  RiskFusion ─► index 0-100, OK / CAUTION / WARNING / CRITICAL + reasons
```

Fusion: `risk = 1 − (1 − drowsiness)(1 − distraction)`, scaled by `(1 + 0.3·affect)`, where affect = anger + 0.7·fear + 0.4·disgust (smoothed over 5 s). Distraction also takes YOLO behaviour confidences (phone ×1.0, smoking ×0.6, drinking ×0.4). Hysteresis: enter at 30/55/80, leave 10 points lower.

Two design details found during testing:
* **PERCLOS needs enough history.** Over a 60 s window it is divided by at least 30 s. Otherwise a single 2.5 s eye closure 10 s into a drive reads as 25–43% PERCLOS and drowsiness saturates. (Observed in our first end-to-end test.)
* **Microsleeps have an after-effect.** After a microsleep, drowsiness keeps a floor of 0.65 that decays with τ = 45 s. A microsleep is strong evidence of sleepiness, so risk should not drop to OK the moment the eyes reopen.

Alerts (`dms/fusion/alerts.py`): each hazard maps to a spoken message, a suggestion and a sound, with debounce (e.g. phone ≥ 1 s, stress ≥ 5 s) and per-severity cooldowns (6 s critical … 120 s info). Nagging is the main reason drivers switch DMS off.

All thresholds are configurable so you can match the exact protocol version you cite.

---

## 7. Experiments for the paper

| # | Experiment | Data | Metric(s) |
|---|---|---|---|
| E1 | Eye state: single-task vs multi-task | MRL (subject split) → CEW (cross) | acc, macro-F1 |
| E2 | Emotion: AffectNet-only vs multi-task + cabin aug | AffectNet/RAF → **KMU-FED** (cross-domain) | macro-F1, ECE |
| E3 | Behaviour: random split vs **driver split** (leakage study) | State Farm (+100-Driver) | acc, macro-F1 |
| E4 | Drowsiness: threshold vs RF vs GRU vs Conv-Transformer | UTA-RLDD, 5-fold subject-independent | window & video macro-F1, κ |
| E5 | Ablation: calibration on/off, blendshapes on/off, window 10/30/60 s | UTA-RLDD | as E4 |
| E6 | Cross-dataset drowsiness | train RLDD → test NTHU-DDD / DMD | macro-F1 |
| E7 | System-level: false alarms/hour, detection latency | own recordings (5–10 volunteers, consented) + DMD | FA/h, seconds-to-alert |
| E8 | Efficiency | all models | params, MB, ms (CPU, Pi/Jetson), FPS end-to-end |

**Realistic targets** (typical literature ranges; verify when citing):
eye state on MRL ≈ 97–99% · FER2013 ≈ 70–75%, RAF-DB ≈ 88–92% · State Farm driver-split ≈ 80–92% (vs ~99% random split) · UTA-RLDD 3-class subject-independent is **hard**: the original baseline is ≈ 65% video-level, so 70–80% is a good result, and binary alert-vs-drowsy will be considerably higher.

Beating the 3-class RLDD baseline subject-independently is a *stronger* paper than claiming 99% on a leaky split. Reviewers increasingly know the difference.

**Measured so far** (this container, 4-core CPU, no GPU):
landmarks + features + fusion ≈ **81 FPS** · full pipeline with emotion (5 Hz) + YOLO11n (4 Hz) ≈ **30 FPS** · web demo ≈ 15–30 ms per frame · Conv-Transformer **0.64 ms** (ONNX FP32), 174k params, 1.0 MB · MobileNetV3-L face net **3.2 ms** (ONNX FP32), 4.2M params.
Finding: *dynamic* INT8 on CNNs made inference ~16× slower in ONNX Runtime (ConvInteger kernels). Use static QDQ quantisation with calibration data for CNNs.

---

## 8. Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.10+ | ecosystem |
| Face landmarks | **MediaPipe Face Landmarker** (Tasks API) | 478 pts incl. iris, 52 blendshapes, head-pose matrix, real-time on CPU, Apache-2.0 |
| DL framework | **PyTorch** + **timm** | any pretrained backbone with one string |
| Vision I/O | OpenCV | camera, video, drawing |
| Classical ML / metrics | scikit-learn | GroupKFold, RF baseline, metrics |
| Deployment | **ONNX + ONNX Runtime** (FP32 / INT8) | CPU, Raspberry Pi 5, Jetson (TensorRT EP) |
| Training compute | Kaggle / Colab GPU (T4/P100) | free; datasets are small enough |
| Experiment tracking (optional) | Weights & Biases or TensorBoard | plots for the paper |
| Object detection | **Ultralytics YOLO11n** (AGPL-3.0) | phone / bottle / cup out of the box; fine-tune for cigarette, seatbelt |
| Emotion baseline | **HSEmotion EfficientNet-B0** (ONNX, AffectNet) | working emotions before our own training; comparison baseline |
| Demo UI | **FastAPI + WebSocket + vanilla JS** (no CDN; works offline) | side-by-side live footage + analysis, voice/buzzer alerts, trip report |
| Hardware for demo | laptop webcam; better: USB NIR camera + 850 nm IR LEDs for night | Euro NCAP-style cabins use NIR |

---

## 9. Roadmap

| Phase | Weeks | Output |
|---|---|---|
| 0. Setup & data requests | 1 | Request AffectNet, NTHU-DDD, 100-Driver, DEFE now (approval takes time); download MRL, State Farm, UTA-RLDD, DMD, KMU-FED |
| 1. Baselines | 2–3 | Threshold system running on webcam (✅ done); manifests; E3 leakage study |
| 2. Models | 4–6 | FaceStateNet (E1, E2), CabinNet, temporal model (E4, E5) |
| 3. Cross-dataset + system eval | 7–8 | E6, E7, E8; own consented recordings |
| 4. Writing | 9–10 | Paper, report, slides, demo video |

### Paper outline (IEEE conference format, 6–8 pages)
1. Introduction: road-safety statistics, Euro NCAP 2026, contributions C1–C5
2. Related work: §2 of this document
3. Method: pipeline figure, features, calibration, temporal model, multi-task net, fusion
4. Experimental setup: datasets, protocol (subject-independent, cross-dataset), metrics
5. Results: tables E1–E8, confusion matrices, ablations, latency
6. Discussion: leakage finding, failure cases (sunglasses, night, extreme pose), ethics
7. Conclusion & future work: VLM explanations, CAN/IMU fusion, NIR hardware, rPPG heart-rate

### Slides outline (12–15 slides)
Problem & stats → Euro NCAP 2026 → existing work & gaps → our system (diagram) → each module (1 slide each) → evaluation protocol → results → live demo → limitations & ethics → future work.

---

## 10. Ethics, privacy and limitations

* Process on-device; store only features and alert logs, not video (the temporal model never needs pixels).
* Facial emotion recognition is contested: expressions ≠ felt emotions, with cultural and demographic variance. Present affect as a *weak risk modifier*, report per-group performance if metadata allows, and state this limitation in the paper.
* Get informed consent for any recordings of volunteers; check whether your institution requires ethics-board approval.
* Known failure modes to test and report: dark sunglasses (no eye signal; fall back to head pose and yawns), strong backlight, extreme head pose, faces partly outside the frame.
