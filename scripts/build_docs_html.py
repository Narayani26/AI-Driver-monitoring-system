"""Builds the project report and the IEEE paper as PDFs (HTML + headless Chromium), in the department's
reference layout (InferPilot report/paper). Run from the repository root:

    PYTHONPATH=scripts python scripts/build_docs_html.py
"""
from __future__ import annotations

import base64
import html
import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image

import submission_content as C

A = Path("docs/report_assets").resolve()
OUT = Path("docs/submission").resolve()
OUT.mkdir(parents=True, exist_ok=True)
SHORT = "AI-Based Driver Monitoring System: Real-Time Drowsiness and Unsafe-Behaviour Detection"
MONTH = "NOVEMBER 2026"
e = html.escape


def ui(name: str) -> Path:
    """Browser screenshots without the browser's tab bar."""
    src, dst = A / name, A / ("crop_" + Path(name).stem + ".png")
    if not dst.exists():
        im = Image.open(src).convert("RGB")
        im.crop((0, int(im.height * 0.125), im.width, im.height)).save(dst)
    return dst


def img(path: Path) -> str:
    mime = "image/png" if path.suffix == ".png" else "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


# ============================================================== shared CSS
REPORT_CSS = """
@page { size: A4; margin: 24mm 22mm 22mm 28mm; }
body { font-family: 'Times New Roman', 'Liberation Serif', Times, serif; font-size: 12pt; line-height: 1.75; color: #000; margin: 0; }
p { text-align: justify; margin: 0 0 9pt; }
h1 { font-size: 15pt; margin: 0 0 12pt; page-break-before: always; }
h2 { font-size: 12.5pt; margin: 14pt 0 6pt; }
h3 { font-size: 12pt; margin: 10pt 0 4pt; font-style: italic; }
.center { text-align: center; }
ul { margin: 0 0 9pt 0; padding-left: 22pt; } li { text-align: justify; margin-bottom: 3pt; }
table.t { border-collapse: collapse; width: 100%; margin: 4pt 0 12pt; font-size: 10pt; line-height: 1.3; page-break-inside: avoid; }
table.t th { border-top: 1.2pt solid #000; border-bottom: 0.8pt solid #000; padding: 4pt 5pt; text-align: left; }
table.t td { padding: 3.5pt 5pt; border-bottom: 0.4pt solid #bbb; vertical-align: top; }
table.t tr:last-child td { border-bottom: 1.2pt solid #000; }
.tcap { text-align: center; font-size: 10pt; font-weight: bold; margin: 10pt 0 2pt; text-transform: uppercase; letter-spacing: .02em; }
figure { margin: 10pt 0 12pt; text-align: center; page-break-inside: avoid; }
figure img { max-width: 100%; max-height: 118mm; border: 0.4pt solid #ccc; }
figcaption { font-size: 10pt; margin-top: 4pt; text-align: center; }
.formula { text-align: center; font-style: italic; margin: 6pt 0 10pt; }
code { font-family: 'Courier New', monospace; font-size: 10pt; }
.page { page-break-after: always; }
.toc { width: 100%; border-collapse: collapse; font-size: 11.5pt; line-height: 1.45; }
.toc th { text-align: left; border-bottom: 1pt solid #000; padding: 3pt 4pt; font-size: 11pt; }
.toc td { padding: 1.5pt 4pt; vertical-align: top; }
.toc td.pg { text-align: right; width: 18mm; }
.toc .ch td { font-weight: bold; padding-top: 6pt; }
.toc .sub td.ti { padding-left: 14pt; }
.dots { display: flex; } .dots span.f { flex: 1; border-bottom: 1pt dotted #555; margin: 0 4pt 5pt; }
"""

# ============================================================== report body
figs, tabs = [], []


def P(t):
    return f"<p>{t}</p>"


def UL(items):
    out = []
    for it in items:
        if isinstance(it, tuple):
            out.append(f"<li><b>{e(it[0])}:</b> {e(it[1])}</li>")
        else:
            out.append(f"<li>{e(it)}</li>")
    return "<ul>" + "".join(out) + "</ul>"


def FIG(path, cap, width="100%"):
    figs.append(cap)
    return f'<figure><img src="{img(path)}" style="width:{width}"><figcaption>Fig. {len(figs)}. {e(cap)}</figcaption></figure>'


ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI", "XVII"]


def TAB(cap, headers, rows):
    tabs.append(cap)
    h = "".join(f"<th>{e(x)}</th>" for x in headers)
    b = "".join("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in r) + "</tr>" for r in rows)
    return f'<div style="page-break-inside:avoid"><div class="tcap">Table {ROMAN[len(tabs) - 1]}. {e(cap)}</div><table class="t"><tr>{h}</tr>{b}</table></div>'


CH = []  # (number, title, [(sub number, sub title)])


def H1(num, title):
    CH.append((num, title, []))
    return f'<h1 id="c{num}">{num}. {e(title.upper())}</h1>'


def H2(num, title):
    CH[-1][2].append((num, title))
    return f'<h2 id="s{num}">{num} {e(title)}</h2>'


def body() -> str:
    s = []
    s += [H1("1", "Introduction"), H2("1.1", "Background and Motivation"),
          P("Road traffic crashes kill about 1.19 million people every year [22]. Drowsiness and distraction are among the main human "
            "causes: a driver whose eyes close for two seconds at 80 km/h travels more than 44 metres without seeing the road, and a glance "
            "at a phone has the same effect. Unlike alcohol, fatigue has no roadside test, and drivers notice their own impairment late."),
          P("Driver Monitoring Systems (DMS) observe the driver with a camera and warn before an attention lapse becomes a crash. Regulation "
            "and Euro NCAP ratings [19] now require them in new cars, but they rely on infrared cameras fitted at the factory in a fixed position. "
            "The many vehicles already on the road, and drivers who use a phone or a dash camera, have no comparable protection. Two further "
            "problems limit practical use: published accuracies are often measured with the same people in the training and test data, and "
            "systems that raise false alarms are switched off by drivers."),
          H2("1.2", "System Overview"),
          P("<b>The problem.</b> Detect drowsiness, distraction and unsafe behaviour from an ordinary camera, for any driver and camera position, "
            "in real time, with few false alarms, without sending video anywhere."),
          P("<b>What we built.</b> An offline-first in-car application (Python, FastAPI, browser interface) that analyses every frame with "
            "MediaPipe face landmarks and a set of neural networks running in a background thread, estimates the driver state with calibrated "
            "trackers and Euro NCAP rules, fuses everything into an explainable 0-100 risk score and speaks proportionate alerts. A cloud "
            "back end (FastAPI, PostgreSQL on Supabase) stores trips, alerts and driver feedback and runs a retraining pipeline."),
          P("<b>What we found.</b> Eye state reaches 98.0% accuracy and driver behaviour 92.1% on people never seen in training; alert-versus-"
            "drowsy classification reaches 87.3% (Random Forest) and 80.5% (deployed hybrid network) of videos of 60 unseen people. Random "
            "image splits hide most errors (17-41 times fewer), and per-driver calibration adds 15.8 percentage points."),
          FIG(A / "architecture.png", "System overview: in-car edge application (offline-first) and cloud back end"),
          H2("1.3", "Contributions"),
          "<ol>" + "".join(f"<li><b>{e(a)}.</b> {e(b)}</li>" for a, b in C.CONTRIBUTIONS) + "</ol>",
          H2("1.4", "Organisation of the Report"),
          P("Chapter 2 reviews related work. Chapter 3 states the problem and requirements and Chapter 4 the technology stack. Chapter 5 "
            "describes the architecture and database, Chapter 6 the datasets and models and Chapter 7 the implementation. Chapter 8 shows the "
            "user interface, Chapters 9 and 10 the testing and results, Chapter 11 the challenges, and Chapter 12 concludes.")]

    s += [H1("2", "Related Work"), H2("2.1", "Eye-Based Drowsiness Measures"),
          P("Soukupov&aacute; and &Ccaron;ech introduced the Eye Aspect Ratio (EAR), the ratio of vertical to horizontal eye-landmark "
            "distances, which falls towards zero as the eye closes [1]. PERCLOS, the percentage of time the eyes are at least 80% closed, is "
            "the most validated drowsiness measure [2]. Both depend on the individual's eye shape, which motivates per-driver calibration."),
          H2("2.2", "Facial Landmarks and Head Pose"),
          P("MediaPipe regresses 468-478 three-dimensional face landmarks from a single RGB camera in real time [5], together with blendshape "
            "coefficients (eye blink, jaw open) and a facial transformation matrix that gives yaw, pitch and roll. Its detector is designed "
            "for frontal or moderately turned faces."),
          H2("2.3", "Deep Learning for Driver State"),
          P("Convolutional networks such as MobileNetV3 [6] and EfficientNet [7] classify eye state, facial expression and driver behaviour "
            "efficiently. Multi-task learning shares one backbone between tasks, with losses balanced by learned uncertainty [9]; temperature "
            "scaling calibrates the confidence of the outputs [10]. For drowsiness over time, recurrent and Transformer [8] models process "
            "sequences; UTA-RLDD [3] provides realistic self-labelled videos of 60 people with a published video-level baseline of about 65% "
            "for three levels, while NTHU-DDD [4] is widely used but contains few people."),
          H2("2.4", "Object, Pose and Audio Models"),
          P("Single-stage YOLO detectors [12] locate phones, bottles and cups in real time; the pose variant predicts body keypoints such as "
            "nose, eyes, ears and shoulders from any viewing angle. YAMNet [13] classifies 521 AudioSet sound classes from spectrograms."),
          H2("2.5", "Evaluation Practice and Research Gap"),
          P("Driver datasets often contain few people and many near-identical frames per person, so random image splits place the same person "
            "in training and test sets. Commercial systems use infrared cameras in fixed positions. Work on consumer cameras at arbitrary "
            "positions, on limiting false alarms, and on learning from real drivers without collecting video is limited; this project "
            "addresses these gaps."),
          TAB("Comparison of related approaches", ["Approach", "Strength", "Limitation", "This project"],
              [["Geometric thresholds (EAR/MAR)", "Light, explainable", "Fixed thresholds, pose-sensitive", "Per-driver calibration, pose gating"],
               ["Single-frame CNN", "Accurate per image", "Identity leakage, no time context", "Subject-independent tests, temporal model"],
               ["Temporal deep models", "Use time context", "Need much data", "Hybrid model with statistics branch"],
               ["Commercial infrared DMS", "Robust, works at night", "Fixed hardware, factory-fitted", "Any camera, view-adaptive"]])]

    s += [H1("3", "Problem Statement and Requirements"), H2("3.1", "Problem Definition"),
          P("Design and evaluate a real-time system that, from an ordinary camera and optionally a microphone, detects driver drowsiness, "
            "distraction and unsafe behaviour for any driver and camera position, alerts proportionately with few false alarms, keeps video on "
            "the device, and improves from real use without compromising privacy."),
          H2("3.2", "Functional Requirements"),
          UL(["FR1: accept webcam, video file and RTSP/HTTP camera streams (laptop, phone, Wi-Fi dash cam).",
              "FR2: calibrate to the driver automatically from any camera angle; re-calibrate when the camera is moved.",
              "FR3: detect microsleep, PERCLOS, blinks, yawns, head nodding and eyes-off-road (Euro NCAP rules).",
              "FR4: detect phone use, drinking and smoking; recognise emotion; classify cabin sounds.",
              "FR5: compute a 0-100 risk score with reasons; voice and buzzer alerts with suggestions and cooldowns; driver feedback.",
              "FR6: detect a possible accident and run an SOS countdown.",
              "FR7: produce a trip report (screen, JSON, CSV, PDF) without video.",
              "FR8: store events offline and synchronise them; accounts, devices, trips and alert feedback in the cloud.",
              "FR9: collect consented training windows, check their quality, retrain, compare and redeploy models."]),
          H2("3.3", "Non-Functional Requirements"),
          TAB("Non-functional requirements", ["Requirement", "Target", "Achieved"],
              [["Latency", "< 50 ms per frame on a laptop CPU", "12-16 ms (about 30 FPS)"],
               ["Offline operation", "No internet needed", "Yes; store-and-forward synchronisation"],
               ["Privacy", "No video leaves the device", "Only numbers and alert metadata are uploaded"],
               ["Security", "Authenticated, least privilege", "Tokens, hashed keys, Row Level Security, CSP, checksums"],
               ["Reliability", "Recover from camera or network loss", "Automatic reconnection, local queue, CI tests"],
               ["Portability", "Mac, Linux, Docker, edge boxes", "Docker image and systemd units"]]),
          H2("3.4", "Hardware and Software Requirements"),
          UL(["Laptop or PC with 4 or more CPU cores, webcam and microphone (tested on a MacBook Air with Apple Silicon).",
              "Optional: Android phone with an IP-camera app or a Wi-Fi dash cam; an NVIDIA GPU only for training (Kaggle T4).",
              "Python 3.12, Chrome, optional Docker, and a PostgreSQL database (Supabase) for the cloud part."])]

    s += [H1("4", "Technology Stack"),
          TAB("Technology stack", ["Layer", "Technologies"], C.TECH),
          P("Python was chosen for its machine-learning ecosystem; FastAPI for asynchronous HTTP and WebSocket handling; MediaPipe and "
            "Ultralytics for real-time, CPU-friendly perception; PyTorch and timm for training; ONNX Runtime for fast inference of the "
            "emotion model. The browser front end needs no installation and uses the device camera, microphone and speech synthesis "
            "directly. SQLAlchemy lets the same back end run on SQLite during development and on PostgreSQL (Supabase) in production.")]

    s += [H1("5", "System Architecture and Design"), H2("5.1", "Overall Architecture"),
          P("The in-car application is offline-first. Camera frames are analysed by the perception models; geometric indicators and "
            "temporal trackers estimate the driver state; risk fusion and the alert manager decide what to tell the driver. Events and trip "
            "summaries are written to a local SQLite store and synchronised to the cloud when a network is available (Fig. 1)."),
          H2("5.2", "Processing Pipeline per Frame"),
          UL([("Face landmarks", "MediaPipe on the full frame; if no face is found, a zoomed second look around the driver's head"),
              ("Features", "EAR, MAR, yaw, pitch, roll, gaze, blink and jaw blendshapes"),
              ("Calibration", "open-eye EAR (80th percentile) and neutral head pose (median); re-learned when the camera moves"),
              ("Trackers", "eye closure (blinks, PERCLOS, microsleep), yawns, head nods and glances (Euro NCAP)"),
              ("Learned models (background thread)", "eye-state CNN, emotion, YOLO objects, YOLO pose (side view), temporal model"),
              ("Decision", "risk score with hysteresis and reasons, alerts with cooldowns, voice and buzzer")]),
          H2("5.3", "Use Cases"),
          TAB("Use cases", ["Actor", "Use case"],
              [["Driver", "Start monitoring; receive alerts; tap 'Not true'; rate sleepiness; view and print the trip report"],
               ["Driver (cloud)", "Register, log in, add devices, view trips and alerts, rate alerts, give or withdraw data consent"],
               ["Administrator", "View data quality, false alarms and the model registry; approve or roll back a model"],
               ["Device", "Upload events, trips and samples with its key; download the approved model"],
               ["Scheduler", "Weekly retraining job"]]),
          H2("5.4", "Database Design"),
          TAB("Cloud database schema (PostgreSQL on Supabase, Row Level Security on all tables)", ["Table", "Main fields", "Purpose"],
              [["users", "id, email, password_hash (scrypt), role, consent_training, consent_at", "Accounts and consent"],
               ["devices", "id, user_id, name, kind, key_hash (SHA-256), revoked, last_seen", "Dash cams, phones, laptops"],
               ["trips", "id, user_id, device_id, t_start, t_end, summary (JSON), kss", "One drive"],
               ["events", "id, trip_id, kind, severity, t_session, payload, feedback", "Alerts and driver feedback"],
               ["samples", "id, label, label_source, data (numbers only), quality, accepted, reject_reason", "Training windows"],
               ["model_versions", "version, status, sha256, metrics, data, path", "Model registry"]]),
          P("No column holds images, video or audio. Row Level Security is enabled on every table with no public policies, so Supabase's "
            "public API can read nothing; only the back end, connected as the owner over TLS, reads and writes.")]

    s += [H1("6", "Datasets and Deep-Learning Models"), H2("6.1", "Datasets"),
          TAB("Datasets used", ["Dataset", "Size used", "Task", "Split"], C.DATASETS),
          P("All splits are subject-independent: images or videos of one person are never in both training and test sets, and an assertion "
            "in the code stops training if this is violated. AffectNet+ and ESC-50 are used for testing only."),
          H2("6.2", "Models"),
          TAB("Neural networks in the system", ["Model", "Architecture", "Source", "Role"], C.MODELS),
          H2("6.3", "How the Models Work"),
          P("Convolutional neural networks slide learned filters over the image: early layers respond to edges, middle layers to parts such as "
            "an eyelid or the corner of a phone, and late layers to states and objects. MobileNetV3 and EfficientNet are designed for high "
            "accuracy per computation, so they run in real time on a CPU. We start from ImageNet weights (transfer learning) and fine-tune on "
            "driver data."),
          P("FaceStateNet shares one backbone between three heads (eye, yawn, emotion). Each dataset labels only one task; missing labels are "
            "masked out of the loss and the task losses are balanced with learned uncertainty weights [9]. After training, temperature "
            "scaling [10] is fitted on validation data so that predicted confidence matches observed accuracy (expected calibration error, ECE)."),
          P("The temporal model receives, every 0.1 s, nine facial measurements normalised to the driver's calibration (eye opening, mouth, "
            "pitch, yaw, roll, two gaze values, blink and jaw blendshapes). A one-dimensional convolution groups 30 s (300 steps) into patches, "
            "a Transformer encoder relates every moment to every other through self-attention, and a parallel branch computes classic "
            "statistics (PERCLOS, blink rate, pose variance). The combined representation is classified as alert or drowsy."),
          H2("6.4", "Training"),
          UL(["Image models: Kaggle T4 GPU, AdamW, cosine schedule, augmentation; 'fast' (MobileNetV3, 8 epochs) and 'best' (EfficientNet-B2, 15 epochs, test-time augmentation) presets.",
              "Temporal models: 5-fold GroupKFold by person, 40 epochs, dropout 0.3; a Random Forest on engineered features as baseline.",
              "The best model per task is deployed, not the largest: fast face network live, best frame-drowsiness network, hybrid binary temporal network."])]

    s += [H1("7", "Implementation"), H2("7.1", "Geometric Indicators"),
          P("For the six landmarks p1-p6 around an eye:"),
          '<div class="formula">EAR = ( |p2 &minus; p6| + |p3 &minus; p5| ) / ( 2 |p1 &minus; p4| ),&nbsp;&nbsp; openness = ( EAR &minus; 0.3 EAR<sub>open</sub> ) / ( 0.7 EAR<sub>open</sub> )</div>',
          P("An eye is closed below 20% openness (the P80 criterion), confirmed by the eye-state CNN when it is confident. A closure of one "
            "second or more is a microsleep; PERCLOS is the time-weighted fraction of closed-eye time over 60 s. The mouth aspect ratio and the "
            "jaw-open blendshape detect yawns held for 1.5 s; faces far from the camera are re-measured in an enlarged crop."),
          H2("7.2", "Per-Driver and Any-Angle Calibration"),
          P("During the first seconds the system learns the driver's open-eye EAR (80th percentile) and neutral head pose (median) as the camera "
            "sees them, so a side-mounted or diagonal camera simply has a different neutral pose. Eye shape is judged only near that pose, because "
            "open eyes look closed when the head tilts back in front of a low camera, except when the head drops forward with the eyelids "
            "closing, which is how people doze off. If no frontal face is found (side camera), YOLO11n-pose tracks the head: the drop of the "
            "nose relative to the ear, in head lengths, detects nodding, and the visible eye is checked by the eye CNN. Small or dark faces "
            "(wide-angle or night dash cams) are re-detected in an enlarged crop around the driver's head."),
          H2("7.3", "Camera-Move Detection"),
          P("When the background (cabin interior with the driver masked out, compared by normalised cross-correlation, which ignores "
            "brightness) no longer matches and the driver's position, size or view changes for 0.75-1.5 s, the camera has been moved: the "
            "position is re-learned in about three seconds and the trip continues. A driver who turns the head changes the face but not the "
            "background, so that case remains a distraction alert."),
          H2("7.4", "Risk Fusion"),
          '<div class="formula">R = 100 &middot; ( 1 &minus; (1 &minus; d)(1 &minus; x) ) &middot; ( 1 + 0.3 a )</div>',
          P("Drowsiness evidence d (PERCLOS, long blinks, yawns, head nods, snoring) and distraction evidence x (time off the road beyond 1.5 s, "
            "phone, behaviour) are smoothed over time and combined with the negative high-arousal affect a. Learned models are supporting "
            "evidence: alone they cannot exceed 28, below the CAUTION level (30). Levels (OK, CAUTION 30, WARNING 55, CRITICAL 80) use "
            "hysteresis so they do not flicker; a microsleep or head drop forces CRITICAL. When the driver taps 'Not true', the evidence is "
            "reset and that alert is paused for three minutes."),
          H2("7.5", "Alerts and Sensitivity"),
          TAB("Alert sensitivity presets", ["Setting", "Low", "Normal", "High"],
              [["Microsleep (eyes closed)", "1.5 s", "1.0 s", "0.8 s"], ["Looking away", "4 s", "3 s (Euro NCAP)", "2.5 s"],
               ["Glances within 30 s", "12 s", "10 s", "8 s"], ["Drowsy score threshold", "70%", "60%", "50%"],
               ["Repeat cooldown", "x1.5", "warning 30 s, caution 90 s", "x0.75"]]),
          H2("7.6", "Real-Time Engineering"),
          P("Face landmarks run on every frame; the heavier networks run in a background thread on the newest frame, so a slow model only "
            "updates less often and never delays the video. Model files are verified by SHA-256 at start-up and a per-model timing log keeps "
            "performance visible. Audio is processed in 0.5 s windows from a 16 kHz AudioWorklet stream."),
          H2("7.7", "Cloud Back End and Retraining Pipeline"),
          UL(["The car application stores alerts and trips in SQLite and uploads them with a device key over HTTPS when online (idempotent).",
              "Drivers rate alerts and their sleepiness (Karolinska Sleepiness Scale [20]). With consent, the 30 s of numeric measurements before a rated alert become a labelled training window.",
              "A quality gate keeps only windows where the driver's face is visible (>= 90%), frontal (>= 70%), the same person, calibrated and, if known, moving.",
              "The retraining job (weekly timer or by hand) exports windows split by driver, trains a candidate on UTA-RLDD plus real data, and compares it with the current model on held-out drivers; it is approved only if macro-F1 is not lower and the false-alarm rate does not rise.",
              "Devices download the approved model and verify its SHA-256 before use; an administrator can roll back."]),
          H2("7.8", "Security and Privacy"),
          TAB("Security and privacy measures", ["Area", "Measure"],
              [["Footage", "Never stored or uploaded"], ["In-car application", "Loopback by default; token, Origin/Host checks, CSP, size limits when exposed"],
               ["Models", "SHA-256 verified at start; downloaded updates verified"], ["Accounts", "scrypt password hashing, signed 12 h tokens, login rate limit"],
               ["Devices", "Per-device random key, only its hash stored, revocable"], ["Database", "Supabase PostgreSQL, Row Level Security, TLS"],
               ["Consent", "Opt-in; withdrawal deletes shared data; account erasure; pseudonymised exports"]])]

    s += [H1("8", "User Interface"),
          P("The in-car interface shows the camera and the AI view side by side, the risk gauge with its reasons, the driver-state cards and a "
            "live timeline. Alerts appear as a banner with a suggestion, a voice message and a buzzer tone, and can be rejected with one tap. "
            "The figures below use a publicly available night dash-cam clip played as a video file."),
          FIG(ui("ui_idle.jpg"), "Start screen: source selection (webcam, video file, dash cam / IP stream), sensitivity and settings"),
          FIG(ui("ui_night_monitoring.jpg"), "Night wide-angle dash cam: driver found by the zoom search, front view monitoring, risk OK"),
          FIG(ui("ui_camera_moved.jpg"), "Camera moved: the system notices the new view and re-learns the driver's position"),
          FIG(ui("ui_trip_report.jpg"), "Trip report: safety score, KPIs, time in each risk level, alerts and the sleepiness rating"),
          FIG(A / "cloud_login.png", "Cloud back end: login page", width="85%"),
          FIG(A / "cloud_dashboard.png", "Cloud dashboard: KPIs, devices, safety-score trend, alerts by type and trips (sample data)", width="70%"),
          FIG(A / "cloud_admin.png", "Administrator view: training-data quality, false alarms and the model registry (sample data)", width="70%")]

    s += [H1("9", "Testing and Quality Assurance"), H2("9.1", "Automated Tests"),
          P("The repository contains 60 automated tests (pytest) covering feature extraction, calibration, PERCLOS, microsleep, yawns, glance "
            "rules, risk fusion and hysteresis, alert cooldowns and muting, audio rules, mishap escalation, camera-move detection, zoom search, "
            "driver selection among several faces, the cloud API (accounts, brute-force limit, device keys, isolation between accounts, "
            "consent, quality gate, export) and an end-to-end retraining run. GitHub Actions runs them on every push, again against a real "
            "PostgreSQL 16 server, and builds the Docker image."),
          H2("9.2", "Scenario Tests"),
          TAB("Scenario tests", ["Scenario", "Expected behaviour", "Result"],
              [["Eyes closed for 1 s or more", "Microsleep alarm", "Pass"], ["Head drops forward with eyes closing", "Microsleep / head-drop alarm in about 1-2 s", "Pass"],
               ["Head tilted back (low webcam)", "Eyes 'not judged', no false microsleep", "Pass"], ["Looking away for 3 s", "'Please look ahead' alert", "Pass"],
               ["Mirror glance under 1.5 s", "No change", "Pass"], ["Phone held to the ear or in front of the face", "Phone alert in about 1 s", "Pass (21/22 frames)"],
               ["Side-mounted camera, driver nodding", "Side view, head-lowering caution, then alarm", "Pass"],
               ["Night wide-angle dash cam", "Driver found, no false alerts", "Pass (71% of frames)"],
               ["Camera moved after calibration", "'Camera moved' and quick re-learning", "Pass (0.75-1.5 s)"],
               ["Passenger leans in", "Ignored, driver followed", "Pass"], ["Crash sound from the radio", "Calm notice, no alarm", "Pass"],
               ["'Not true' tapped", "Risk drops, alert paused 3 min", "Pass"], ["No internet", "Works; uploads later", "Pass"]]),
          H2("9.3", "Performance"),
          P("End-to-end processing takes 12-16 ms per frame on a laptop CPU (about 30 frames per second), measured in the browser and in the "
            "server's per-model timing log.")]

    s += [H1("10", "Results and Evaluation"), H2("10.1", "Headline Results"),
          TAB("Results on data never seen in training", ["Task", "Test data", "Result", "Notes"], C.RESULTS),
          FIG(A / "results_accuracy.png", "Accuracy per task on unseen data", width="92%"),
          H2("10.2", "Identity Leakage"),
          P("On State Farm a random split gives 99.6% but new drivers only 92.1% (17 times more errors; 41 times with EfficientNet-B2). On "
            "NTHU-DDD random splits give 94-97% against 61-67% for an unseen person. Larger models widen the gap because they memorise who is "
            "in the image."),
          FIG(A / "results_leakage.png", "Identity leakage: random split versus unseen people", width="72%"),
          H2("10.3", "Drowsiness over Time and Calibration"),
          P("Alert-versus-drowsy reaches 87.3% (Random Forest) and 80.5% (hybrid network) of videos of unseen people; three levels reach "
            "64.7%, on par with the published baseline [3]. Removing per-driver calibration drops the hybrid model from 53.4% to 37.6%, close "
            "to chance."),
          FIG(A / "results_temporal.png", "Drowsiness over time on UTA-RLDD (60 people, 5-fold, unseen subjects)", width="80%"),
          H2("10.4", "Cabin Sounds"),
          FIG(A / "results_audio.png", "Cabin-sound detection on ESC-50", width="72%"),
          H2("10.5", "Discussion"),
          P("Per-frame tasks are at or near the state of the art when measured on unseen people. Drowsiness over time is the hard problem: "
            "80-87% for alert versus drowsy. Emotion from a single image transfers poorly between datasets (48% on AffectNet+), so emotion only "
            "modifies the risk score. The false-alarm rate on real drives is now measurable through driver feedback.")]

    s += [H1("11", "Challenges and Solutions"),
          TAB("Challenges encountered and how they were solved", ["Challenge", "Solution", "Outcome"], C.CHALLENGES)]

    s += [H1("12", "Conclusion and Future Scope"), H2("12.1", "Conclusion"),
          P("We built and evaluated a complete driver monitoring system that runs in real time on ordinary hardware, adapts to each driver and "
            "to arbitrary camera placement, explains its alerts, limits false alarms, keeps video on the device and improves from consented "
            "driver feedback through a safe retraining pipeline. Subject-independent evaluation shows strong per-frame accuracy (98% eye state, "
            "92% behaviour on new drivers) and 80-87% for drowsiness over time, and quantifies identity leakage and the value of per-driver "
            "calibration."),
          H2("12.2", "Limitations"),
          UL(["Ordinary cameras cannot see the eyes at night; an infrared camera is needed.", "Dark sunglasses hide the eyes (head nodding and attention still work).",
              "Audio cannot attribute snoring to a person.", "The false-alarm rate per hour on real drives still has to be measured."]),
          H2("12.3", "Future Scope"),
          UL(["Infrared camera support for night driving.", "Native Android application running when the phone is locked; the cloud API is ready.",
              "Vehicle data (speed, steering) via OBD-II to pause alerts when parked and to strengthen drowsiness estimates.",
              "On-device quantisation (INT8) for low-cost dash-cam processors.", "A larger real-drive dataset through the feedback loop and scheduled retraining."])]

    refs = "".join(f"<p style='text-align:left;padding-left:22pt;text-indent:-22pt;margin-bottom:5pt'>[{i}] {e(r)}</p>" for i, r in enumerate(C.REFERENCES, 1))
    s += ['<h1 id="cref">REFERENCES</h1>', refs]
    CH.append(("", "References", []))
    s += ['<h1 id="capA">APPENDIX A: PROJECT STRUCTURE AND HOW TO RUN</h1>',
          TAB("Source code layout", ["Folder", "Contents"],
              [["dms/perception", "face landmarks, zoom search, pose, eye state, emotion, objects, audio"],
               ["dms/features", "geometry (EAR, MAR, pose), temporal trackers, camera-move detection"],
               ["dms/fusion", "risk fusion, alerts, mishap detection"], ["dms/models, dms/training", "networks, training and evaluation"],
               ["dms/web", "in-car server (FastAPI + WebSocket) and browser interface"],
               ["dms/cloud", "cloud back end, database, quality gate, export, retraining, model download"],
               ["tests, db, deploy", "60 automated tests, PostgreSQL migration, systemd units"]]),
          P("<b>Live demo:</b> <code>python -m dms.web.server --face-ckpt runs/face/best.pt --temporal-ckpt runs_temporal2/hybrid_binary/deploy.pt --audio --mishap</code>"),
          P("<b>Cloud back end:</b> <code>set -a; source .env; set +a; python -m dms.cloud.app</code>"),
          P(f"<b>Source code:</b> {C.REPO}")]
    CH.append(("", "Appendix A: Project Structure and How to Run", []))
    return "\n".join(s)


SYMBOLS = [("EAR", "Eye Aspect Ratio"), ("EAR<sub>open</sub>", "Driver's calibrated open-eye EAR (80th percentile)"),
           ("MAR", "Mouth Aspect Ratio"), ("PERCLOS", "Percentage of time the eyes are at least 80% closed (60 s window)"),
           ("d", "Drowsiness evidence (0-1)"), ("x", "Distraction evidence (0-1)"), ("a", "Negative high-arousal affect (0-1)"),
           ("R", "Risk score (0-100)"), ("T", "Softmax temperature (calibration)"), ("ECE", "Expected Calibration Error"),
           ("KSS", "Karolinska Sleepiness Scale (1-9)"), ("NCC", "Normalised cross-correlation (camera-move detection)"),
           ("F1", "Harmonic mean of precision and recall; macro-F1 averages over classes"),
           ("CNN", "Convolutional Neural Network"), ("RLS", "Row Level Security"), ("CI", "Continuous Integration")]


def render(html_doc: str, pdf: Path, footer: str | None, header: str | None = None):
    tmp = Path(tempfile.mkdtemp()) / "doc.html"
    tmp.write_text(html_doc)
    js = f"""
const {{ chromium }} = require('playwright');
(async () => {{
  const b = await chromium.launch(); const p = await b.newPage();
  await p.goto('file://{tmp}', {{ waitUntil: 'load' }});
  await p.pdf({{ path: '{pdf}', preferCSSPageSize: true, printBackground: true,
    displayHeaderFooter: {str(bool(footer or header)).lower()},
    headerTemplate: {repr(header or '<span></span>')}, footerTemplate: {repr(footer or '<span></span>')} }});
  await b.close();
}})();"""
    jsf = tmp.parent / "r.js"
    jsf.write_text(js)
    subprocess.run(["node", str(jsf)], check=True)


def to_docx(html_doc: str, name: str):
    """Editable Word version of the same document (LibreOffice: HTML -> DOCX)."""
    d = Path(tempfile.mkdtemp())
    src = d / f"{name}.html"
    # Word has no CSS columns/@page: give it plain flow with page breaks and a fixed image width
    # Word-friendly flow: tighter line spacing so each front-matter page fits one page, smaller margins inside blocks
    fix = ("<style>body{line-height:1.35 !important} p{line-height:1.5 !important} div{line-height:1.35}"
           " .cols{column-count:1} figure img{max-width:15cm}</style></head>")
    html_doc = re.sub(r"""<img src=(['"])(data:[^'"]+)\1 style=(['"])width:(\d+)%\3>""",
                      lambda m: f'<img src="{m.group(2)}" width="{int(6.0 * int(m.group(4)))}">', html_doc)
    html_doc = html_doc.replace("<div class='page'>", "<p style='page-break-before: always; margin:0'></p><div>")
    html_doc = html_doc.replace('<h1 id=', "<p style='page-break-before: always; margin:0'></p><h1 id=")
    src.write_text(html_doc.replace("</head>", fix, 1).replace("margin-top:28mm", "margin-top:10mm").replace("margin-top:20mm", "margin-top:8mm")
                   .replace("margin-top:14mm", "margin-top:6mm").replace("margin-top:12mm", "margin-top:6mm").replace("margin-top:10mm;", "margin-top:5mm;")
                   .replace("line-height:2'", "line-height:1.6'"))
    subprocess.run(["soffice", "--headless", "--infilter=HTML (StarWriter)", "--convert-to", "docx:MS Word 2007 XML",
                    "--outdir", str(OUT), str(src)], check=True, capture_output=True, timeout=300)
    print("saved", OUT / f"{name}.docx")


def page_of(pdf: Path, needles: list[str]) -> dict[str, int]:
    out, n = {}, int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True).stdout).group(1))
    for i in range(1, n + 1):
        txt = subprocess.run(["pdftotext", "-f", str(i), "-l", str(i), "-layout", str(pdf), "-"], capture_output=True, text=True).stdout
        flat = re.sub(r"\s+", " ", txt)
        for nd in needles:
            if nd not in out and nd in flat:
                out[nd] = i
    return out


def doc(inner, css):
    return f"<!doctype html><html><head><meta charset='utf-8'><style>{css}</style></head><body>{inner}</body></html>"


def build_report():
    figs.clear(); tabs.clear(); CH.clear()
    main_html = body()
    hdr = (f"<div style='width:100%;font-family:Times New Roman,serif;font-size:8.5pt;font-style:italic;text-align:right;"
           f"margin:0 22mm 0 28mm;color:#333'>{e(SHORT)}</div>")
    ftr_num = "<div style='width:100%;text-align:center;font-family:Times New Roman,serif;font-size:10pt'><span class='pageNumber'></span></div>"
    main_pdf = OUT / "_main.pdf"
    render(doc(main_html, REPORT_CSS), main_pdf, ftr_num, hdr)

    # page numbers for contents / figures / tables
    needles = {}
    for num, title, subs in CH:
        needles[("c", num, title)] = (f"{num}. {title.upper()}" if num else title.upper()[:40])
        for sn, st in subs:
            needles[("s", sn, st)] = f"{sn} {st}"
    for i, cap in enumerate(figs, 1):
        needles[("f", i, cap)] = f"Fig. {i}. {cap[:40]}"
    for i, cap in enumerate(tabs, 1):
        needles[("t", i, cap)] = f"TABLE {ROMAN[i - 1]}. {cap[:30].upper()}"
    pages = page_of(main_pdf, list(needles.values()))
    pg = {k: pages.get(v, "") for k, v in needles.items()}

    def row(no, title, page, cls=""):
        return f"<tr class='{cls}'><td style='width:22mm'>{no}</td><td class='ti'><div class='dots'><span>{e(title)}</span><span class='f'></span></div></td><td class='pg'>{page}</td></tr>"

    front = ["DECLARATION", "ACKNOWLEDGEMENT", "ABSTRACT", "LIST OF TABLES", "LIST OF FIGURES", "LIST OF SYMBOLS"]
    front_pages = dict(zip(front, ["ii", "iii", "iv", "vii", "viii", "ix"]))
    toc = "<table class='toc'><tr><th>CHAPTER NO.</th><th>TITLE</th><th style='text-align:right'>PAGE NO.</th></tr>"
    toc += "".join(row("", f, front_pages[f], "ch") for f in front)
    for num, title, subs in CH:
        toc += row(f"{num}." if num else "", title.upper(), pg[("c", num, title)], "ch")
        toc += "".join(row("", f"{sn}  {st}", pg[("s", sn, st)], "sub") for sn, st in subs)
    toc += "</table>"
    lot = "<table class='toc'><tr><th>TABLE NO.</th><th>TITLE</th><th style='text-align:right'>PAGE NO.</th></tr>" + "".join(
        row(f"Table {ROMAN[i - 1]}", cap, pg[("t", i, cap)]) for i, cap in enumerate(tabs, 1)) + "</table>"
    lof = "<table class='toc'><tr><th>FIGURE NO.</th><th>TITLE</th><th style='text-align:right'>PAGE NO.</th></tr>" + "".join(
        row(f"Fig. {i}", cap, pg[("f", i, cap)]) for i, cap in enumerate(figs, 1)) + "</table>"
    los = "<table class='toc'><tr><th>SYMBOL</th><th>DESCRIPTION</th></tr>" + "".join(
        f"<tr><td style='width:32mm'>{a}</td><td>{e(b)}</td></tr>" for a, b in SYMBOLS) + "</table>"

    names = "".join(f"<div style='font-size:15pt;font-weight:bold;margin:4pt 0'>{e(n.upper())} (Roll No. {r})</div>" for n, r, _ in C.STUDENTS)
    college = "KOLHAPUR INSTITUTE OF TECHNOLOGY'S<br>COLLEGE OF ENGINEERING KOLHAPUR<br>(EMPOWERED AUTONOMOUS)"
    title_pg = f"""<div class='center' style='line-height:1.5'>
      <div style='font-size:19pt;font-weight:bold;margin-top:8mm'>{e(C.TITLE.upper())}</div>
      <div style='font-size:13pt;font-weight:bold;margin-top:12mm'>A PROJECT REPORT</div>
      <div style='font-size:12.5pt;font-style:italic;font-weight:bold;margin-top:8mm'>Submitted by</div>
      <div style='margin-top:4mm'>{names}</div>
      <div style='font-size:12.5pt;font-style:italic;font-weight:bold;margin-top:8mm'>in partial fulfillment for the award of the degree of</div>
      <div style='font-size:15pt;font-weight:bold;margin-top:6mm'>BACHELOR OF TECHNOLOGY</div>
      <div style='font-size:12pt;font-weight:bold;margin-top:3mm'>IN</div>
      <div style='font-size:13pt;font-weight:bold;margin-top:3mm'>COMPUTER SCIENCE AND ENGINEERING</div>
      <img src='{img(A / "kit_logo.png")}' style='width:40%'>
      <div style='font-size:17pt;font-weight:bold;margin-top:8mm;line-height:1.6'>{college}</div>
      <div style='font-size:13pt;font-weight:bold;margin-top:8mm'>{MONTH}</div></div>"""
    head = f"<div class='center' style='font-size:17pt;font-weight:bold;line-height:1.5'>{college}</div>"
    names_plain = " and ".join(f"“{n.upper()}” (Roll No. {r})" for n, r, _ in C.STUDENTS)
    sig = f"""<table style='width:100%;margin-top:28mm;font-size:11pt;line-height:1.4'><tr>
      <td style='width:50%;vertical-align:top'><b>SIGNATURE</b><br><br><br><b>{e(C.HOD)}</b><br><b>HEAD OF THE DEPARTMENT</b><br>Department of Computer Science and Engineering<br>KIT's College of Engineering (Empowered Autonomous),<br>Kolhapur &ndash; 416234, Maharashtra, INDIA</td>
      <td style='vertical-align:top'><b>SIGNATURE</b><br><br><br><b>{e(C.GUIDE)}</b><br><b>SUPERVISOR</b><br>Department of Computer Science and Engineering<br>KIT's College of Engineering (Empowered Autonomous),<br>Kolhapur &ndash; 416234, Maharashtra, INDIA</td></tr></table>"""
    cert = f"""{head}<div class='center' style='font-size:14pt;font-weight:bold;margin:12mm 0 8mm'>CERTIFICATE</div>
      <p style='line-height:2'>This is to certify that the Project report entitled, <b>“{e(C.TITLE.upper())}”</b> submitted by {names_plain} to
      Kolhapur Institute of Technology's College of Engineering Kolhapur (Empowered Autonomous), Kolhapur, Maharashtra, INDIA, in partial
      fulfillment for the award of the degree of <b>“BACHELOR OF TECHNOLOGY”</b> in <b>“COMPUTER SCIENCE AND ENGINEERING”</b> is a
      record of their own work carried out under my supervision and guidance.</p>{sig}"""
    decl = f"""{head}<div class='center' style='font-size:14pt;font-weight:bold;margin:12mm 0 8mm'>DECLARATION</div>
      <p style='line-height:2'>We hereby declare that the Project entitled, <b>“{e(C.TITLE.upper())}”</b> submitted to Kolhapur Institute of
      Technology's College of Engineering Kolhapur (Empowered Autonomous), Kolhapur, Maharashtra, INDIA in partial fulfillment of the award of
      the Degree of <b>“BACHELOR OF TECHNOLOGY”</b> in <b>“COMPUTER SCIENCE AND ENGINEERING”</b> is a bonafide work carried out by us.
      The material contained in this Project has not been submitted to any University or Institution for the award of any degree.</p>
      <div style='margin-top:20mm;font-weight:bold;line-height:1.8'>{'<br>'.join(f'{e(n.upper())} (Roll No. {r})' for n, r, _ in C.STUDENTS)}</div>
      <div style='margin-top:10mm;line-height:1.8'>Place: Kolhapur<br>Date: November 28, 2026</div>"""
    ack = f"""{head}<div class='center' style='font-size:14pt;font-weight:bold;margin:12mm 0 8mm'>ACKNOWLEDGEMENT</div>
      <p style='line-height:2'>We express our sincere gratitude to our project supervisor, <b>{e(C.GUIDE)}</b>, for his continuous guidance, constructive
      feedback and encouragement throughout the design, implementation and evaluation of this project. His technical insights and critical
      review were instrumental in shaping our experimental methodology and analysis.</p>
      <p style='line-height:2'>We also extend our heartfelt thanks to the Head of the Department, <b>{e(C.HOD)}</b>, and all faculty and staff members of the
      Department of Computer Science and Engineering at Kolhapur Institute of Technology's College of Engineering Kolhapur (Empowered
      Autonomous), Kolhapur, for providing the academic environment, computing infrastructure and support required to carry out this work.
      We thank the creators of the public datasets and open-source software used in this project, and our peers and families for their
      constant support and encouragement.</p>
      <div style='margin-top:14mm;font-weight:bold;line-height:1.8'>{'<br>'.join(f'{e(n.upper())} (Roll No. {r})' for n, r, _ in C.STUDENTS)}</div>
      <div style='margin-top:8mm;line-height:1.8'>Place: Kolhapur<br>Date: November 28, 2026</div>"""
    abstract = f"<div class='center' style='font-size:14pt;font-weight:bold;margin:4mm 0 8mm'>ABSTRACT</div><p style='line-height:2'>{e(C.ABSTRACT)}</p><p><b><i>Keywords:</i></b> <i>{e(', '.join(C.KEYWORDS))}.</i></p>"

    def sec_title(t):
        return f"<div class='center' style='font-size:14pt;font-weight:bold;margin:2mm 0 6mm'>{t}</div>"

    fronts = [("title", title_pg, None), ("cert", cert, "i"), ("decl", decl, "ii"), ("ack", ack, "iii"), ("abs", abstract, "iv"),
              ("toc", sec_title("TABLE OF CONTENTS") + toc, "v"), ("lot", sec_title("LIST OF TABLES") + lot, "vii"),
              ("lof", sec_title("LIST OF FIGURES") + lof, "viii"), ("los", sec_title("LIST OF SYMBOLS") + los, "ix")]
    parts = []
    for key, inner, num in fronts:
        f = OUT / f"_f_{key}.pdf"
        foot = None if num is None else f"<div style='width:100%;text-align:center;font-family:Times New Roman,serif;font-size:10pt'>{num}</div>"
        if key == "toc":  # contents runs over two pages: number them v, vi
            foot = "<div style='width:100%;text-align:center;font-family:Times New Roman,serif;font-size:10pt'><span class='pageNumber'></span></div>"
            render(doc(inner, REPORT_CSS.replace("@page {", "@page { ") + "@page { } body{counter-reset:none}"), f, foot)
            # pageNumber restarts at 1 in this separate render: replace with roman by re-rendering with explicit footers per page
            n = int(re.search(r"Pages:\s+(\d+)", subprocess.run(["pdfinfo", str(f)], capture_output=True, text=True).stdout).group(1))
            romans = ["v", "vi", "vii"][:n]
            pieces = []
            for k in range(n):
                pf = OUT / f"_f_toc_{k}.pdf"
                render(doc(inner, REPORT_CSS), pf, f"<div style='width:100%;text-align:center;font-family:Times New Roman,serif;font-size:10pt'>{romans[k]}</div>")
                one = OUT / f"_f_toc_only_{k}.pdf"
                subprocess.run(["pdfseparate", "-f", str(k + 1), "-l", str(k + 1), str(pf), str(one)], check=True)
                pieces.append(one)
            parts += pieces
            continue
        render(doc(inner, REPORT_CSS), f, foot)
        parts.append(f)
    combined = "".join(f"<div class='page'>{inner}</div>" for _, inner, _ in fronts) + main_html
    to_docx(doc(combined, REPORT_CSS), "DMS_Project_Report")
    final = OUT / "DMS_Project_Report.pdf"
    subprocess.run(["pdfunite", *map(str, parts), str(main_pdf), str(final)], check=True)
    for p in OUT.glob("_*.pdf"):
        p.unlink()
    print("saved", final)


# ============================================================== IEEE paper
PAPER_CSS = """
@page { size: Letter; margin: 19mm 16mm 25mm 16mm; }
body { font-family: 'Times New Roman', 'Liberation Serif', Times, serif; font-size: 10pt; line-height: 1.18; margin: 0; color: #000; }
.title { font-size: 23pt; text-align: center; margin: 2mm 0 6mm; line-height: 1.15; }
.authors { display: flex; justify-content: space-around; text-align: center; font-size: 10pt; margin-bottom: 8mm; line-height: 1.25; }
.authors .n { font-size: 11pt; } .authors i { font-size: 9pt; }
.cols { column-count: 2; column-gap: 6mm; }
p { text-align: justify; margin: 0 0 4pt; text-indent: 1em; }
p.ni { text-indent: 0; }
.abs { font-weight: bold; font-size: 9pt; text-indent: 0; }
h2 { font-size: 10pt; font-weight: normal; font-variant: small-caps; text-align: center; margin: 9pt 0 4pt; letter-spacing: .02em; }
h3 { font-size: 10pt; font-style: italic; font-weight: normal; margin: 6pt 0 3pt; }
figure { margin: 6pt 0 8pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; }
figure.wide { column-span: all; }
figcaption { font-size: 8pt; text-align: justify; margin-top: 3pt; }
.tcap { font-size: 8pt; text-align: center; font-variant: small-caps; margin: 6pt 0 2pt; }
.tbl { break-inside: avoid; page-break-inside: avoid; display: block; }
h2, h3 { break-after: avoid; page-break-after: avoid; }
.ref { break-inside: avoid; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin-bottom: 8pt; break-inside: avoid; line-height: 1.15; }
th { border-top: 1pt solid #000; border-bottom: .6pt solid #000; padding: 2pt 3pt; text-align: left; }
td { padding: 1.6pt 3pt; vertical-align: top; } tr:last-child td { border-bottom: 1pt solid #000; }
.formula { text-align: center; font-style: italic; margin: 4pt 0 6pt; text-indent: 0; }
ol.contrib { margin: 0 0 4pt; padding-left: 14pt; } ol.contrib li { text-align: justify; margin-bottom: 2pt; }
.ref { font-size: 8pt; text-indent: -16pt; padding-left: 16pt; margin-bottom: 2pt; text-align: left; }
"""


def build_paper():
    def fig(path, cap, wide=False, width="100%"):
        return f"<figure class='{'wide' if wide else ''}'><img src='{img(path)}' style='width:{width}'><figcaption>{cap}</figcaption></figure>"

    def table(cap, headers, rows):
        h = "".join(f"<th>{e(x)}</th>" for x in headers)
        b = "".join("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in r) + "</tr>" for r in rows)
        return f"<div class='tbl'><div class='tcap'>{cap}</div><table><tr>{h}</tr>{b}</table></div>"

    authors = "".join(f"<div><div class='n'>{e(n)}</div><i>Roll No. {r}</i><br><i>Dept. of Computer Science and Engineering<br>Kolhapur Institute of Technology's<br>College of Engineering (Empowered Autonomous)<br>Kolhapur &ndash; 416234, Maharashtra, India</i></div>" for n, r, _ in C.STUDENTS)
    authors += (f"<div><div class='n'>{e(C.GUIDE)}</div><i>Project Supervisor</i><br><i>Dept. of Computer Science and Engineering<br>Kolhapur Institute of Technology's<br>"
                "College of Engineering (Empowered Autonomous)<br>Kolhapur &ndash; 416234, Maharashtra, India</i></div>")
    contrib = "".join(f"<li><i>{e(a)}:</i> {e(b)}</li>" for a, b in C.CONTRIBUTIONS)
    refs = "".join(f"<p class='ref'>[{i}] {e(r)}</p>" for i, r in enumerate(C.REFERENCES, 1))
    t1 = table("Table I<br>Neural networks in the system", ["Model", "Type", "Role"], [(m, a, r) for m, a, _, r in C.MODELS])
    t2 = table("Table II<br>Datasets (all splits subject-independent)", ["Dataset", "Size", "Task"], [(d, s, t) for d, s, t, _ in C.DATASETS])
    t3 = table("Table III<br>Results on data never seen in training", ["Task", "Test data", "Result"], [(a, b, c) for a, b, c, _ in C.RESULTS])
    t4 = table("Table IV<br>Alert sensitivity presets", ["Setting", "Low", "Normal", "High"],
               [["Microsleep", "1.5 s", "1.0 s", "0.8 s"], ["Looking away", "4 s", "3 s", "2.5 s"], ["Glances in 30 s", "12 s", "10 s", "8 s"],
                ["Drowsy threshold", "70%", "60%", "50%"]])
    t5 = table("Table V<br>Challenges and solutions", ["Challenge", "Solution", "Outcome"], C.CHALLENGES[:5])

    inner = f"""
<div class='title'>{e(C.TITLE)}</div>
<div class='authors'>{authors}</div>
<div class='cols'>
<p class='abs'><i>Abstract</i>&mdash;{e(C.ABSTRACT)}</p>
<p class='abs'><i>Index Terms</i>&mdash;{e(', '.join(C.KEYWORDS))}.</p>
<h2>I. Introduction</h2>
<p>Road crashes kill about 1.19 million people every year [22], and drowsiness and distraction are leading human causes. Driver Monitoring
Systems (DMS) observe the driver and warn before a lapse becomes a crash; regulation and Euro NCAP [19] now require them in new cars, using
factory-fitted infrared cameras. Vehicles already on the road, and drivers relying on a phone or dash camera, lack such protection.</p>
<p>Two further problems limit practical use. First, accuracies are frequently reported with the same people in the training and test sets,
which overstates how a model performs on a new driver. Second, systems that raise false alarms are switched off. We present a complete,
real-time DMS for ordinary cameras that runs offline on a laptop CPU and addresses both problems. Our contributions are:</p>
<ol class='contrib'>{contrib}</ol>
<p>Section II reviews related work, Section III the architecture, Section IV the method, Section V the experimental setup and Section VI the
results; Sections VII and VIII discuss limitations and conclude.</p>
<h2>II. Related Work</h2>
<p><i>Geometric measures.</i> The Eye Aspect Ratio (EAR) [1] and PERCLOS [2] are the classic drowsiness measures; both depend on the individual's
eye shape. MediaPipe [5] provides real-time 3-D face landmarks and blendshapes from a monocular camera.</p>
<p><i>Learned models.</i> CNNs such as MobileNetV3 [6] and EfficientNet [7] classify eye state, expression and behaviour; multi-task learning
with uncertainty weighting [9] shares a backbone between tasks; temperature scaling [10] calibrates confidences. For temporal drowsiness,
UTA-RLDD [3] offers 60 people with realistic self-labelled videos and a baseline of about 65% for three levels; NTHU-DDD [4] is widely used
but contains few people. YOLO [12] detects objects and body keypoints; YAMNet [13] classifies audio events.</p>
<p><i>Gap.</i> Few works evaluate strictly on unseen drivers, address arbitrary camera placement, or learn from real drivers without collecting video.</p>
<h2>III. System Architecture</h2>
{fig(A / "architecture.png", "Fig. 1. Architecture of the system: in-car edge application (offline-first, left to right: input, perception, driver state, decision and action) and the cloud back end with the retraining pipeline (bottom).", wide=True, width="96%")}
<p>Frames from a webcam, phone, dash camera (RTSP) or video file are processed by MediaPipe on every frame. The heavier networks (eye-state
CNN, emotion, YOLO objects and pose, temporal model) run in a background thread on the newest frame, so a slow model only updates less often
and the video stays real time. Trackers estimate blinks, PERCLOS, microsleeps, yawns, head nods and glances; a risk-fusion model and an alert
manager decide the response. Events are written to a local SQLite store and uploaded over HTTPS with a per-device key to a FastAPI back end on
PostgreSQL (Supabase) with Row Level Security.</p>
<h2>IV. Method</h2>
<h3>A. Driver-Normalised Features and Calibration</h3>
<p>Per frame we compute EAR, MAR, yaw, pitch, roll, two gaze values and the blink and jaw-open blendshapes. During the first seconds the system
learns the driver's open-eye EAR (80th percentile) and neutral head pose (median) as the camera sees them; every feature is normalised to this
profile, both live and when training the temporal model.</p>
<div class='formula'>EAR = (|p<sub>2</sub>&minus;p<sub>6</sub>| + |p<sub>3</sub>&minus;p<sub>5</sub>|) / (2|p<sub>1</sub>&minus;p<sub>4</sub>|)&nbsp;&nbsp;&nbsp;(1)</div>
<p>An eye is closed when its openness, (EAR &minus; 0.3 EAR<sub>open</sub>)/(0.7 EAR<sub>open</sub>), falls below 0.2 (P80), confirmed by the
eye-state CNN when it is confident; a closure of 1 s or more is a microsleep.</p>
<h3>B. View-Adaptive Perception</h3>
<p>Eye shape is judged only near the calibrated pose, because open eyes look closed when the head tilts back in front of a low camera; a head
dropped forward with closing eyelids is still judged, since that is how drivers doze. Small or dark faces are re-detected in an enlarged crop
around the driver's head. When no frontal face exists (side camera), YOLO11n-pose keypoints give the nose drop relative to the ear in head
lengths (alert &minus;0.15, nodding +0.25 on a test clip) and an eye crop for the CNN. A camera move is declared when the masked background no
longer matches (normalised cross-correlation &lt; 0.6) and the driver's position, size or view changes for 0.75&ndash;1.5 s; the position is then
re-learned in about 3 s. A driver turning the head leaves the background unchanged and remains a distraction.</p>
<h3>C. Models</h3>
{t1}
<p>FaceStateNet shares a MobileNetV3/EfficientNet backbone between eye, yawn and emotion heads with masked losses and learned uncertainty weights.
The hybrid temporal model applies a 1-D convolution that patches 300 steps (30 s at 10 Hz), a Transformer encoder [8], and a parallel statistics
branch (PERCLOS, blink rate, pose variance), then classifies alert versus drowsy.</p>
<h3>D. Evidence-Gated Risk Fusion</h3>
<div class='formula'>R = 100 &middot; (1 &minus; (1&minus;d)(1&minus;x)) &middot; (1 + 0.3a)&nbsp;&nbsp;&nbsp;(2)</div>
<p>Drowsiness evidence d (PERCLOS, long blinks, yawns, head nods, snoring) and distraction evidence x (off-road time beyond 1.5 s, phone,
behaviour) are smoothed and combined with the negative high-arousal affect a. Learned models are supporting evidence: alone they cannot exceed
28, below CAUTION (30). Levels use hysteresis; a microsleep or head drop forces CRITICAL. A driver's &ldquo;Not true&rdquo; resets the evidence
and pauses that alert for three minutes.</p>
{t4}
<h3>E. Privacy-Preserving Continual Learning</h3>
<p>With consent, the 30 s of numeric features before a rated alert, or at a Karolinska Sleepiness Scale [20] rating, become a labelled window. A
quality gate requires the driver's face in at least 90% of frames, a frontal pose in at least 70%, the same tracked person, a plausible
calibration and, if known, a moving vehicle. Retraining exports windows split by driver, trains on UTA-RLDD plus the real data and compares
candidate and current model on held-out drivers; the candidate is approved only if macro-F1 is not lower and the false-alarm rate does not
rise. Devices verify the SHA-256 of the approved model.</p>
<h2>V. Experimental Setup</h2>
{t2}
<p>Image models were trained on a Kaggle T4 GPU (AdamW, cosine schedule; MobileNetV3 for 8 and EfficientNet-B2 for 15 epochs). Temporal models
used 5-fold GroupKFold by person on UTA-RLDD (178 videos, first 6 min at 10 Hz, 30 s windows), 40 epochs, with a Random Forest baseline on
engineered features. A leakage assertion aborts training if any person appears in two splits.</p>
<h2>VI. Results</h2>
{t3}
{fig(A / "results_accuracy.png", "Fig. 2. Accuracy per task on data never seen in training.", width="92%")}
<p><i>Per-frame tasks.</i> Eye state reaches 98.0% (ECE 0.006 after temperature scaling) and driver behaviour 92.1% on unseen drivers.</p>
<p><i>Identity leakage.</i> Random splits overstate accuracy: 99.6% versus 92.1% on State Farm (17&times; more errors on new drivers, 41&times;
with EfficientNet-B2) and 94&ndash;97% versus 61&ndash;67% on NTHU-DDD (Fig. 3).</p>
{fig(A / "results_leakage.png", "Fig. 3. Identity leakage: random split versus unseen people.", width="80%")}
<p><i>Drowsiness over time.</i> Alert versus drowsy reaches 87.3% (Random Forest) and 80.5% (hybrid network) of videos of unseen people; three
levels reach 64.7%, on par with the published baseline [3]. Removing per-driver calibration drops the hybrid model from 53.4% to 37.6%,
close to chance (Fig. 4).</p>
{fig(A / "results_temporal.png", "Fig. 4. Drowsiness over time on UTA-RLDD (60 people, 5-fold, unseen subjects).", width="88%")}
<p><i>Audio and robustness.</i> Cabin sounds are detected with 95% (snoring), 93% (siren) and 78% (crash) recall with no false shouting alarms.
On a side-mounted clip the face detector found 0/77 frames and the pose fallback 77/77; on a night wide-angle dash cam the zoom search raised
face detection from 0% to 71% (Fig. 5); when the camera is moved mid-trip, the new view is detected and the position re-learned. The full pipeline runs in 12&ndash;16 ms per frame.</p>
{fig(ui("ui_night_monitoring.jpg"), "Fig. 5. Live interface on a night wide-angle dash cam: the zoom search finds the small, dark face; front view monitoring with risk 0 and the driver-state cards.", wide=True, width="66%")}
{fig(ui("ui_trip_report.jpg"), "Fig. 6. Trip report without video: safety score, KPIs, time per risk level, alerts and the driver's sleepiness rating.", width="82%")}
<h2>VII. Discussion and Limitations</h2>
{t5}
<p>Learned per-frame drowsiness models trained on few people transfer poorly to new drivers and cameras; treating them as supporting evidence
keeps their benefit while limiting false alarms. Limitations: ordinary cameras cannot see the eyes at night (infrared is needed), dark
sunglasses hide the eyes, audio cannot attribute snoring to a person, and the false-alarm rate on real drives must still be measured; the
feedback pipeline is designed to provide exactly that measurement.</p>
<h2>VIII. Conclusion</h2>
<p>We built and evaluated a real-time, privacy-preserving DMS for ordinary cameras that adapts to each driver and to camera placement,
explains and limits its alerts, and improves from consented driver feedback. Subject-independent evaluation shows strong per-frame accuracy
and 80&ndash;87% for drowsiness over time, and quantifies identity leakage and the value of per-driver calibration. Future work: infrared night
operation, a native mobile application, vehicle data and on-road false-alarm measurement.</p>
<h2>Acknowledgment</h2>
<p>The authors thank the Department of Computer Science and Engineering, KIT's College of Engineering, Kolhapur, for its support, and the
creators of the public datasets and open-source software used in this work.</p>
<h2>References</h2>
{refs}
</div>"""
    out = OUT / "DMS_IEEE_Research_Paper.pdf"
    render(doc(inner, PAPER_CSS), out, None)
    to_docx(doc(inner, PAPER_CSS), "DMS_IEEE_Research_Paper")
    print("saved", out)


if __name__ == "__main__":
    import sys

    if "--paper" not in sys.argv:
        build_report()
    build_paper()
