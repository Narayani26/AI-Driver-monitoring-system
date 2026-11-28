"""Additional report sections (diagrams, algorithms, detailed tables, deployment manual).
Every number here comes from the project's own logs, code or measurements."""
from html import escape as e


def CODE(title, body):
    return (f"<div style='page-break-inside:avoid;margin:6pt 0 12pt'><div style='font-size:10pt;font-weight:bold;margin-bottom:2pt'>{e(title)}</div>"
            f"<pre style='font-family:\"Courier New\",monospace;font-size:9.4pt;line-height:1.35;background:#f6f8fa;border:0.6pt solid #c9d1d9;"
            f"border-radius:3pt;padding:7pt 9pt;margin:0;white-space:pre-wrap'>{e(body)}</pre></div>")


def ch1(H2, P, UL, FIG, TAB, A):
    return [H2("1.5", "Scope of the Project"),
            P("The project covers the complete path from camera to driver alert and from alert to model improvement: real-time perception on "
              "an ordinary laptop CPU, per-driver calibration, drowsiness, distraction, unsafe-behaviour and cabin-sound detection, explainable "
              "risk fusion, voice and buzzer alerts with suggestions, a possible-accident check, trip reports, an offline event store with "
              "synchronisation, a cloud back end with accounts and devices, and a consent-based retraining pipeline. It deliberately excludes "
              "vehicle control (braking, steering), proprietary vehicle buses and infrared hardware, which are discussed as future work."),
            H2("1.6", "Expected Outcomes"),
            UL(["A working, demonstrable system for webcam, video file, phone and dash-cam input.",
                "Trained models evaluated honestly on people never seen in training, with the effect of identity leakage and calibration measured.",
                "A secure, deployable cloud back end with a database, dashboard and retraining pipeline.",
                "Documentation, automated tests and continuous integration so that the work can be reproduced and extended."])]


def ch2(H2, P, UL, FIG, TAB, A):
    return [H2("2.6", "Commercial Driver Monitoring Systems"),
            P("Production systems from automotive suppliers place a near-infrared camera with infrared LEDs on the steering column, the "
              "instrument cluster or the rear-view mirror. Infrared illumination makes the eyes visible at night and through most sunglasses, "
              "and the fixed position lets the supplier calibrate the geometry at the factory. Euro NCAP scores such systems on how quickly they "
              "detect long distraction (a glance of 3 s or more), visual attention time-sharing (repeated short glances) and drowsiness, and on "
              "whether the vehicle reacts. Their limitations are cost, the need for factory integration and the absence of any solution for "
              "the large existing fleet; these motivated a design that works with consumer cameras in arbitrary positions."),
            H2("2.7", "Public Driver-Monitoring Datasets"),
            TAB("Public datasets considered", ["Dataset", "Content", "People", "Strength", "Weakness"],
                [["MRL Eye", "84,898 near-infrared eye crops", "37", "Large, person IDs in file names", "Eye crops only"],
                 ["FER2013", "35,887 grey 48x48 faces, 7 emotions", "unknown", "Large, standard benchmark", "Low resolution, noisy labels"],
                 ["RAF-DB", "15,339 aligned faces, 7 emotions", "unknown", "Cleaner labels, in the wild", "Not driving"],
                 ["AffectNet+", "Validation set, human and soft labels", "unknown", "Different source: true cross-dataset test", "Training set not public"],
                 ["State Farm", "22,424 images, 10 behaviours", "26", "Real cars, driver IDs", "Side camera only"],
                 ["NTHU-DDD (Kaggle)", "18,000 frames, drowsy / not drowsy", "4", "Driving simulator", "Very few people"],
                 ["UTA-RLDD", "Videos: alert, low-vigilant, drowsy", "60", "Realistic, self-labelled, many people", "Labels are subjective"],
                 ["Yawn-Eye", "2,900 face crops", "unknown", "Yawn labels", "No person IDs"],
                 ["ESC-50", "2,000 sound clips, 50 classes", "-", "Standard audio benchmark", "Not recorded in cars"]]),
            H2("2.8", "Summary of the Reviewed Literature"),
            TAB("Summary of the reviewed literature", ["Ref.", "Work", "Key idea", "Relevance to this project"],
                [["[1]", "Soukupova and Cech", "Eye Aspect Ratio from landmarks", "Eye-closure measure, normalised per driver"],
                 ["[2]", "Wierwille et al.", "PERCLOS", "Main drowsiness indicator (60 s window)"],
                 ["[3]", "Ghoddoosian et al.", "UTA-RLDD and a temporal baseline", "Temporal training data and baseline (~65%)"],
                 ["[5]", "Kartynnik et al.", "Real-time 3-D face mesh", "Landmarks, blendshapes, head pose"],
                 ["[6], [7]", "MobileNetV3, EfficientNet", "Efficient CNN backbones", "FaceStateNet, CabinNet"],
                 ["[8]", "Vaswani et al.", "Transformer self-attention", "Temporal drowsiness model"],
                 ["[9]", "Kendall et al.", "Uncertainty-weighted multi-task loss", "Training three heads together"],
                 ["[10]", "Guo et al.", "Temperature scaling", "Calibrated confidences"],
                 ["[12]", "Ultralytics YOLO11", "Real-time detection and pose", "Phones, bottles; side-view head"],
                 ["[13]", "Hershey et al. (YAMNet)", "Audio event classification", "Snoring, siren, crash sounds"],
                 ["[19]", "Euro NCAP", "Driver-state monitoring protocol", "Distraction timing rules"],
                 ["[23]", "Ersoy et al.", "Personalised EAR/MAR thresholds", "Related to per-driver calibration"],
                 ["[24]", "Bianco et al.", "Cross-camera distraction classification", "Related to camera-placement robustness"]])]


def ch3(H2, P, UL, FIG, TAB, A):
    return [H2("3.5", "Feasibility Study"),
            TAB("Feasibility study", ["Aspect", "Assessment"],
                [["Technical", "Pretrained real-time models (MediaPipe, YOLO) run on a laptop CPU; training fits on free Kaggle GPUs; measured 12-16 ms per frame"],
                 ["Economic", "Open-source software and public datasets; hardware already owned (laptop, phone); free tiers of Kaggle and Supabase"],
                 ["Operational", "Runs in a browser with no installation for the driver; works offline; alerts in natural language with suggestions"],
                 ["Legal and ethical", "No video stored or uploaded; explicit, revocable consent for training data; dataset licences respected; alerts are advisory"],
                 ["Schedule", "Delivered within the semester with iterative milestones (Table below)"]]),
            H2("3.6", "Development Methodology and Timeline"),
            P("The project followed an iterative, test-driven process: each feature was implemented together with automated tests, demonstrated "
              "on recorded clips, and refined from observed failures (for example false eye closures under head tilt, missed phones, a camera "
              "moved during a trip). Training experiments were tracked as numbered runs with stored metrics."),
            TAB("Project timeline", ["Phase", "Work", "Outcome"],
                [["1. Study", "Literature, datasets, Euro NCAP rules", "Requirements and design"],
                 ["2. Core pipeline", "Landmarks, EAR/MAR, pose, trackers, risk fusion, alerts, web UI", "Working geometric DMS"],
                 ["3. Learning", "Dataset readers, subject-independent splits, image and temporal models on Kaggle", "Runs 1-6, temporal rounds 1-5"],
                 ["4. Robustness", "Background model thread, pose gating, side view, zoom search, camera-move detection", "Real-time and any camera angle"],
                 ["5. Production", "Offline store, sync, security, cloud back end, Supabase, Docker, CI", "Deployable system"],
                 ["6. Learning loop", "Driver feedback, quality gate, retraining, model registry", "Continuous improvement"],
                 ["7. Documentation", "Report, paper, presentation, demo", "Submission"]]),
            H2("3.7", "Risk Analysis"),
            TAB("Project risks and mitigations", ["Risk", "Impact", "Mitigation"],
                [["Too little drowsiness data", "Weak temporal model", "UTA-RLDD (60 people), engineered-feature baseline, feedback loop for real data"],
                 ["Optimistic evaluation", "Wrong conclusions", "Subject-independent splits enforced by an assertion in code"],
                 ["Real-time performance", "Laggy alerts", "Background model thread, throttling, per-model timing log"],
                 ["False alarms", "Drivers switch the system off", "Evidence-gated fusion, cooldowns, sensitivity presets, 'Not true' feedback"],
                 ["Privacy of cabin video", "Loss of trust, legal issues", "On-device processing, no footage stored, consent for numeric samples only"],
                 ["Camera placement varies", "Missed detections", "Any-angle calibration, side-view fallback, zoom search, camera-move detection"]])]


def ch4(H2, P, UL, FIG, TAB, A):
    return [H2("4.1", "Python and the Scientific Stack"),
            P("Python 3.12 is the implementation language for the server, perception, training and cloud code. NumPy provides array maths "
              "for the geometric features and trackers; pandas handles dataset manifests; scikit-learn provides the Random Forest baseline, "
              "GroupKFold splitting and metrics; matplotlib produces the evaluation figures."),
            H2("4.2", "MediaPipe"),
            P("Google's MediaPipe Face Landmarker runs a face detector followed by a mesh regressor and returns, for every frame, 478 3-D "
              "landmarks, 52 blendshape coefficients and a facial transformation matrix. It runs in VIDEO mode, which tracks the face between "
              "frames, and in IMAGE mode for the zoomed second look. MediaPipe's audio classifier runs the YAMNet model."),
            H2("4.3", "PyTorch, timm and ONNX Runtime"),
            P("PyTorch is used for all trained networks. The timm library supplies ImageNet-pretrained MobileNetV3 and EfficientNet backbones. "
              "Models are exported to ONNX for benchmarking, and ONNX Runtime executes the pretrained HSEmotion model."),
            H2("4.4", "Ultralytics YOLO11"),
            P("YOLO11n detects phones, bottles and cups (COCO classes) and YOLO11n-pose predicts 17 body keypoints. Both run on the CPU at "
              "reduced input size (480 px) in the background thread."),
            H2("4.5", "FastAPI, Uvicorn and WebSocket"),
            P("FastAPI provides the HTTP and WebSocket endpoints of both the in-car server and the cloud back end; Uvicorn is the ASGI server. "
              "Each browser tab opens one WebSocket over which JPEG frames and audio chunks go to the server and JSON results come back. "
              "Pydantic validates request bodies in the cloud API."),
            H2("4.6", "Browser APIs"),
            P("getUserMedia opens the camera and microphone; an AudioWorklet resamples audio to 16 kHz; Canvas 2D draws the overlay and charts; "
              "the Web Speech API speaks alerts (a female voice is preferred); the Web Audio API generates buzzer tones. A strict Content "
              "Security Policy forbids inline scripts."),
            H2("4.7", "SQLAlchemy, SQLite and PostgreSQL (Supabase)"),
            P("SQLAlchemy 2 maps the cloud tables to Python classes and generates SQL for the configured database. SQLite is used in the car "
              "(offline store) and for local development; PostgreSQL 16 hosted on Supabase is the production database, reached through "
              "psycopg 3 over TLS. A migration file reproduces the schema with Row Level Security."),
            H2("4.8", "Docker, GitHub Actions and systemd"),
            P("A Docker image packages the in-car server; Docker Compose runs the cloud back end with PostgreSQL; systemd units start the in-car "
              "service with the vehicle and run weekly retraining. GitHub Actions runs the test suite, the PostgreSQL tests and the Docker "
              "build on every push.")]


def ch5(H2, P, UL, FIG, TAB, A):
    return [H2("5.5", "Context Diagram"),
            FIG(A / "diag_dfd0.png", "Context diagram (DFD level 0)", width="85%"),
            P("At the highest level the system receives frames and audio from the cabin, gives alerts and receives feedback from the driver, "
              "and exchanges trips, alerts, training samples and approved models with the cloud back end, which the administrator manages."),
            H2("5.6", "Data-Flow Diagram (Level 1)"),
            FIG(A / "diag_dfd1.png", "Data-flow diagram (level 1)", width="100%"),
            P("Process 1.0 decodes frames; 2.0 runs the perception models; 3.0 turns measurements into driver-state indicators using the "
              "driver profile (D1); 4.0 fuses them into risk and alerts; 5.0 stores events locally (D2) and synchronises them to the cloud "
              "database (D3); 6.0 retrains and approves models from consented data in D3."),
            H2("5.7", "Use-Case Diagram"),
            FIG(A / "diag_usecase.png", "Use-case diagram", width="95%"),
            H2("5.8", "Sequence of One Frame"),
            FIG(A / "diag_sequence.png", "Sequence diagram for one video frame", width="100%"),
            P("The browser sends a new frame only after the previous result has arrived, so a slow step can never build up a backlog. The "
              "heavier models run in a background thread on the newest frame and their latest outputs are reused, which keeps the frame "
              "path at 12-16 ms."),
            H2("5.9", "Entity-Relationship Diagram"),
            FIG(A / "diag_er.png", "Entity-relationship diagram of the cloud database", width="100%"),
            H2("5.10", "Module Description"),
            TAB("Software modules", ["Module", "Responsibility"],
                [["dms/perception/face.py", "Face landmarker wrapper, driver selection among several faces, zoomed second look"],
                 ["dms/perception/pose.py", "Side-view head keypoints, head drop, eye crop"],
                 ["dms/perception/eyes.py, emotion.py, objects.py, audio.py", "Eye-state CNN, emotion, YOLO objects, YAMNet audio groups"],
                 ["dms/features/geometry.py", "EAR, MAR, head pose, gaze, blendshape features"],
                 ["dms/features/temporal.py", "Calibration, eye closure (blinks, PERCLOS, microsleep), yawns, glances, head nods"],
                 ["dms/features/camera.py", "Camera-move detection"],
                 ["dms/fusion/risk.py, alerts.py, mishap.py", "Risk score, alert manager, possible-accident logic"],
                 ["dms/pipeline.py", "DriverMonitor: one frame in, FrameResult out; model scheduling"],
                 ["dms/web/server.py, static/", "In-car FastAPI server and browser interface"],
                 ["dms/storage.py", "Offline SQLite store and synchronisation thread"],
                 ["dms/cloud/", "Cloud API, database, authentication, quality gate, export, retraining, model download"],
                 ["dms/training/, dms/data/", "Dataset discovery and manifests, image and temporal training, evaluation, export"]])]


def ch6(H2, P, UL, FIG, TAB, A):
    return [H2("6.5", "Dataset Preparation and Preprocessing"),
            P("A discovery tool inspects every downloaded dataset by content (folder names, CSV and JSON annotation files) and writes one "
              "manifest format with the image path, the task label and a subject identifier. Subject identifiers come from file names (MRL "
              "Eye, NTHU-DDD), from annotation files (State Farm) or are unique per image where no person information exists; the splitter "
              "groups by subject so that no person crosses splits. Face images are cropped with a margin, resized to 224 x 224 and normalised "
              "with ImageNet statistics; training uses random resized crops, horizontal flips and colour jitter. MRL eye crops are near-infrared "
              "and grey, so live eye crops are converted to grey before inference."),
            P("For the temporal model every UTA-RLDD video is processed at 10 Hz: MediaPipe features are extracted for the first six minutes, "
              "the subject's profile is computed from the alert video only (calibrating on the drowsy video would normalise away the signal), "
              "and 30 s windows (300 steps) are sampled. Binary experiments use the alert (0) and drowsy (10) videos and leave out the "
              "ambiguous low-vigilant videos."),
            H2("6.6", "Network Details"),
            TAB("Network details", ["Network", "Input", "Backbone / layers", "Outputs", "Parameters"],
                [["FaceStateNet (fast)", "224x224 RGB face/eye crop", "MobileNetV3-Large + 3 MLP heads", "eye 2, yawn 2, emotion 7", "4.2 M"],
                 ["FaceStateNet (best)", "224x224 RGB", "EfficientNet-B2 + 3 heads", "eye 2, yawn 2, emotion 7", "7.7 M"],
                 ["CabinNet", "224x224 full frame", "EfficientNet-B0", "10 behaviours", "4.0 M"],
                 ["Frame-drowsiness CNN", "224x224 full frame", "EfficientNet-B2", "alert / drowsy", "7.7 M"],
                 ["Hybrid Conv-Transformer", "300 x 9 feature sequence", "1-D conv patches + Transformer encoder + statistics MLP", "alert / drowsy", "~0.2 M"],
                 ["Random Forest (baseline)", "window statistics", "400 trees, min. 5 samples per leaf, balanced", "3 or 2 classes", "-"]]),
            H2("6.7", "Training Settings"),
            TAB("Temporal-model training settings (from train_temporal.py)", ["Setting", "Value"],
                [["Optimiser", "AdamW, learning rate 1e-3, weight decay 0.05"], ["Schedule", "100-step warm-up, then cosine decay"],
                 ["Epochs x steps", "40 x 100 (batch 64 windows, sampled uniformly over videos)"], ["Window / evaluation stride", "300 steps (30 s) / 50 steps"],
                 ["Loss", "cross-entropy with label smoothing 0.1"], ["Regularisation", "dropout 0.3, gradient clipping at 1.0"],
                 ["Augmentation", "Gaussian noise 0.02, per-feature gain 5%, 5% dropped steps"], ["Validation", "5-fold GroupKFold by person; leakage assertion"]]),
            P("Image models were trained on a Kaggle T4 GPU with AdamW and a cosine schedule from ImageNet weights: the 'fast' preset used "
              "MobileNetV3 for 8 epochs and the 'best' preset EfficientNet-B2 for 15 epochs with test-time augmentation. A temperature was "
              "fitted per head on the validation set after training and is stored in the checkpoint and applied at run time."),
            H2("6.8", "Evaluation Metrics"),
            UL([("Accuracy", "fraction of correct predictions"),
                ("Macro-F1", "mean over classes of the harmonic mean of precision and recall; robust to class imbalance"),
                ("Balanced accuracy", "mean recall over classes"),
                ("Video accuracy", "for UTA-RLDD, the class of a video is the mean of its window probabilities"),
                ("Expected Calibration Error (ECE)", "average gap between confidence and accuracy over confidence bins"),
                ("Recall at threshold (audio)", "share of target clips that trigger the live alert rule; false-alarm rate on other sounds")])]


def ch7(H2, P, UL, FIG, TAB, A):
    return [H2("7.9", "Algorithms"),
            CODE("Algorithm 1: per-driver calibration (any camera angle)",
                 "input: frames of the first T seconds (T = 5 s live, 3 s after a camera move)\n"
                 "for each frame with a detected face:\n    store (EAR, MAR, yaw, pitch)\n"
                 "when T has elapsed and >= 60 samples are stored:\n"
                 "    EAR_open <- 80th percentile of EAR        # robust to blinks\n"
                 "    MAR_rest <- median of MAR\n"
                 "    yaw0, pitch0 <- median of yaw, pitch       # the camera's view of 'looking at the road'\n"
                 "side camera (no frontal face): learn the median head drop of the pose keypoints instead\n"
                 "the view that finishes first (front or side) becomes the primary view"),
            CODE("Algorithm 2: eye closure, blinks, PERCLOS and microsleep",
                 "openness <- (EAR - 0.3*EAR_open) / (0.7*EAR_open)\n"
                 "if head is near the calibrated pose OR (head dropped forward AND eyelid blendshape > 0.55):\n"
                 "    closed <- CNN_p > 0.7  or  (CNN_p >= 0.3 and openness < 0.2)\n"
                 "else: closed episode paused                 # head tilted back: eye shape unreliable\n"
                 "if closed: closed_since <- closed_since or now\n"
                 "    if now - closed_since >= microsleep_s (1.0 s Normal): MICROSLEEP\n"
                 "else if closed_since: duration < microsleep_s -> blink;  closed_since <- none\n"
                 "PERCLOS <- time-weighted share of closed samples over 60 s (gaps capped at 0.5 s, span >= 30 s)"),
            CODE("Algorithm 3: Euro NCAP glance rules",
                 "off_road <- |yaw - yaw0| > 25 deg  or  |pitch - pitch0| > 20 deg  or  face lost (head turned away)\n"
                 "long distraction  <- continuous off_road >= 3 s (Normal)\n"
                 "VATS              <- sum of off_road time within the last 30 s >= 10 s\n"
                 "glances < 1.5 s do not raise the distraction score (mirror checks)"),
            CODE("Algorithm 4: risk fusion and alerts",
                 "d <- max(ramp(PERCLOS, 0.08, 0.30), 0.6*ramp(blink duration), 0.7*ramp(yawns), head drop, nods, snoring)\n"
                 "m <- weighted mean of learned models (temporal 0.6, frame CNN 0.4)\n"
                 "if d < 0.25: d <- max(d, 0.28*m)  else: d <- max(d, d + (1-d)*0.3*m)\n"
                 "x <- max(0.8*ramp(off_road, 1.5, 3.0), phone, 0.6*smoking, min(behaviour CNN, 0.5))\n"
                 "d, x <- exponential smoothing in time\n"
                 "R <- 100 * (1 - (1-d)(1-x)) * (1 + 0.3*affect)\n"
                 "if microsleep or head drop: R <- 100\n"
                 "if no measured sign at all: R <- min(R, 54)        # learned models alone never warn\n"
                 "level <- hysteresis(R; enter 30/55/80, leave 10 points lower)\n"
                 "alert if condition holds long enough AND not in cooldown AND not muted by 'Not true'"),
            CODE("Algorithm 5: camera-move detection",
                 "every 0.25 s:\n"
                 "    ncc <- normalised cross-correlation(reference background, current background), driver masked out\n"
                 "    scene_changed  <- ncc < 0.6\n"
                 "    driver_changed <- view changed (front/side/none) or centre moved > 0.25 or size ratio > 1.4\n"
                 "    if scene_changed and driver_changed for 1.5 s (0.75 s if ncc < 0.3 and the view changed):\n"
                 "        re-learn the driver's position (quick 3 s calibration); keep the trip and its report\n"
                 "    while a move is suspected, 'look at the road' alerts wait"),
            CODE("Algorithm 6: retraining and promotion",
                 "export accepted windows; rank drivers by hash; hold out 30% of drivers (at least one) for testing\n"
                 "if training drivers < 5 or test drivers < 2 or a class is missing: stop\n"
                 "train candidate on UTA-RLDD + real training windows\n"
                 "evaluate candidate and current model on held-out real drivers\n"
                 "approve if F1_cand >= max(F1_cur, 0.6) and FA_cand <= FA_cur + 0.02 else reject\n"
                 "store version, metrics and SHA-256; devices download only the approved version and verify it"),
            H2("7.10", "Alert-Level and Possible-Accident Logic"),
            FIG(A / "diag_levels.png", "Risk-level state machine with hysteresis", width="95%"),
            FIG(A / "diag_mishap.png", "Possible-accident state machine", width="95%"),
            P("A crash or skid sound or a camera jolt only starts a silent check. If the driver then looks normal, a strong sound alone produces "
              "a calm notice (it may come from another car or the radio) and a real jolt produces 'Sudden jolt detected'. If the face disappears "
              "or the eyes stay closed for three seconds, the system announces a possible accident and starts a fifteen-second countdown; "
              "pressing 'I'm OK' cancels it, otherwise an SOS event is raised."),
            H2("7.11", "Cabin-Audio Pipeline"),
            P("The browser streams microphone audio at 16 kHz in 0.25 s blocks. Every 0.5 s the last second is classified by YAMNet; the 521 "
              "class scores are grouped into snoring, distress (shouting, screaming), siren, horn, phone ringing, baby crying, impact and skid. "
              "Because sounds are bursty, an audio alert needs two detections within a short window instead of a continuous hold. Snoring "
              "raises drowsiness strongly; distress raises the affect term; crash and skid feed the possible-accident logic."),
            H2("7.12", "Selected Code Excerpts"),
            CODE("Risk combination (dms/fusion/risk.py, simplified)",
                 "base = 1.0 - (1.0 - self.w_drowsy * d) * (1.0 - self.w_distract * x)\n"
                 "index = 100.0 * float(np.clip(base * (1.0 + self.affect_gain * a), 0.0, 1.0))\n"
                 "if r.microsleep:\n    reasons.insert(0, f\"MICROSLEEP: eyes closed {r.closed_duration:.1f}s\")\n    index = 100.0"),
            CODE("Choosing the driver among several faces (dms/perception/face.py)",
                 "areas = [ptp(f[:, 0]) * ptp(f[:, 1]) for f in faces]\nbiggest = argmax(areas)                       # closest to the camera\n"
                 "if prev_centre is not None:\n    near = argmin(distance of each face to prev_centre)\n"
                 "    if distance[near] < 0.2 * width and areas[near] >= 0.5 * areas[biggest]:\n        return near                           # keep following the same driver\n"
                 "return biggest")]


def ch8(H2, P, UL, FIG, TAB, A):
    return [H2("8.1", "Design Principles"),
            UL(["Glanceable: the risk gauge, level colour and one-line reasons are readable in under a second.",
                "Proportionate: cautions show a message and a soft tone; only warnings and critical alerts are spoken.",
                "Explainable: every alert carries its reasons and a practical suggestion.",
                "Correctable: the driver can reject an alert with one tap and rate sleepiness on a 1-9 colour scale.",
                "Private: video stays in the browser and the local server; reports contain numbers only."]),
            H2("8.2", "Screens"),
            P("The start screen selects the source (webcam, video file or dash cam / IP stream), the sensitivity (Low, Normal, High) and "
              "settings such as voice, buzzer, object detection and cabin sound. While monitoring, the AI view shows the face box and eye and "
              "mouth contours (or the head keypoints in side view), the status badge shows the view and state, and the side panel shows the "
              "risk gauge with drowsiness, distraction and stress bars, the eye, PERCLOS, blink, yawn, attention and behaviour cards, cabin "
              "audio and emotion. The trip report summarises the session and can be downloaded as JSON or CSV or printed to PDF. The cloud "
              "dashboard shows the driver's trips, safety-score trend, alerts by type and devices; the administrator view adds data quality, "
              "false alarms and the model registry.")]


def ch9(H2, P, UL, FIG, TAB, A):
    tests = [
        ("Geometry", "test_ear_matches_definition, test_euler_identity_and_yaw, test_compute_features_without_iris_or_pose"),
        ("Eye closure", "test_blinks_counted_and_perclos, test_microsleep_detected, test_perclos_not_inflated_early, test_eye_cnn_fusion, test_eyes_not_judged_when_head_tilted"),
        ("Calibration and sequences", "test_calibrator, test_feature_sequence_resamples, test_training_window_snapshot"),
        ("Yawns and glances", "test_yawn_needs_duration, test_glance_long_and_vats"),
        ("Risk and alerts", "test_risk_levels_and_hysteresis, test_alert_cooldown_and_debounce, test_sensitivity_presets_change_alert_timing, test_learned_drowsiness_alone_is_only_a_caution, test_risk_accepts_binary_temporal_model, test_not_true_drops_risk_and_pauses_the_alert, test_risk_uses_audio"),
        ("Objects and drivers", "test_phone_showing_face_is_ignored, test_phone_held_in_front_of_face_counts, test_driver_is_tracked_and_passengers_ignored"),
        ("Any camera angle", "test_head_from_side_view_keypoints_and_nod, test_camera_move_vs_driver_turning, test_zoomed_face_maps_back_to_the_frame"),
        ("Audio and mishap", "test_audio_group_scorer, test_jolt_detector_fires_on_whole_frame_change_only, test_mishap_driver_ok_after_impact, test_mishap_unresponsive_leads_to_sos_unless_acknowledged, test_mishap_needs_evidence, test_audio_alerts_and_mishap_alerts, test_crash_sound_alone_is_a_calm_notice"),
        ("Models and training", "test_multitask_masked_loss, test_temporal_models, test_checkpoint_roundtrip, test_temperature_scaling_fixes_overconfidence, test_hybrid_temporal_model_and_binary"),
        ("Evaluation hygiene", "test_subject_split_no_leakage, test_random_split_is_leaky_and_subject_split_is_not, test_same_file_name_in_train_and_test_is_not_one_subject, test_evaluate_alerts_scoring"),
        ("Dataset discovery", "test_dataset_discovery_by_content, test_discovery_of_official_affectnet_and_csv_layouts, test_nthu_frames_need_person_ids, test_rldd_video_layouts, test_affectnet_plus_json_validation_only_is_heldout"),
        ("Real time", "test_async_models_keep_frames_fast"),
        ("Edge storage and security", "test_event_store_offline_then_sync, test_security_rules, test_web_server_security_end_to_end, test_edge_store_queues_samples_for_sync"),
        ("Cloud back end", "test_register_login_and_bad_passwords, test_ingest_trips_events_and_idempotency, test_accounts_are_isolated, test_training_samples_need_consent_and_pass_quality, test_export_matches_training_format"),
        ("Retraining", "test_retraining_pipeline_end_to_end, test_promotion_rule"),
    ]
    return [H2("9.4", "Catalogue of Automated Tests"),
            TAB("Automated tests by area", ["Area", "Tests (pytest)"], tests),
            H2("9.5", "Continuous Integration"),
            P("GitHub Actions runs three jobs on every push: the full test suite with a training smoke test on synthetic data; the cloud "
              "back-end tests against a PostgreSQL 16 service container; and a Docker build with a health-check and an authentication test "
              "of the running container. A red pipeline blocks a release."),
            H2("9.6", "Testing on Recorded Clips"),
            P("Beyond unit tests, every robustness feature was checked on recorded clips by replaying them through the pipeline or the "
              "browser with a virtual camera: head tilts in front of a low webcam (no false microsleep), a phone held up (21 of 22 frames), a "
              "side-mounted driver falling asleep (head found in 77 of 77 frames, head-drop alarm), a night wide-angle dash cam (face in 71% "
              "of frames, no false alerts) and a camera moved from a frontal to a side view (detected, re-learned, no false re-calibration while "
              "the driver nodded).")]


def ch10(H2, P, UL, FIG, TAB, A):
    folds = [["RF, 3 classes", "66.7", "61.1", "58.3", "62.9", "74.3", "64.7"],
             ["Hybrid, 3 classes", "58.3", "47.2", "50.0", "45.7", "65.7", "53.4"],
             ["Transformer, 3 classes", "61.1", "50.0", "55.6", "45.7", "65.7", "55.6"],
             ["Hybrid without calibration, 3 classes", "44.4", "41.7", "36.1", "28.6", "37.1", "37.6"],
             ["RF, alert vs drowsy", "75.0", "91.7", "91.7", "82.6", "95.7", "87.3"],
             ["Hybrid, alert vs drowsy", "79.2", "79.2", "79.2", "73.9", "91.3", "80.5"],
             ["Transformer, alert vs drowsy", "75.0", "70.8", "70.8", "78.3", "82.6", "75.5"]]
    return [H2("10.6", "Per-Fold Results on UTA-RLDD"),
            TAB("Video accuracy (%) per fold, 12 unseen people per fold", ["Model", "Fold 0", "Fold 1", "Fold 2", "Fold 3", "Fold 4", "Mean"], folds),
            P("The spread between folds (for example 75-96% for the binary Random Forest) shows how much results depend on which people "
              "are in the test set, and why a single random split is not a reliable estimate."),
            H2("10.7", "Larger Network versus Faster Network"),
            TAB("Fast preset (MobileNetV3/EfficientNet-B0) versus best preset (EfficientNet-B2)", ["Task", "Fast", "Best", "Change"],
                [["Eye open/closed (unseen people)", "98.0%", "97.95%", "same"], ["Yawn", "99.1%", "100%", "+0.9"],
                 ["Emotion (FER2013 + RAF-DB)", "73.8%", "75.4%", "+1.6"], ["Emotion on AffectNet+", "46.9%", "48.4%", "+1.5"],
                 ["Behaviour, unseen drivers", "92.1%", "89.0%", "-3.1"], ["Behaviour, random split", "99.6%", "99.7%", "same"],
                 ["Frame drowsiness, unseen person", "60.6%", "66.7%", "+6.1"], ["Image time on CPU (ONNX)", "4.4-9.3 ms", "14.8-15.0 ms", "slower"]]),
            P("The larger network fits the training drivers better but generalises worse to new drivers on State Farm: the gap between random "
              "split and unseen drivers grows from 17 to 41 times more errors. We therefore deploy the best model per task, not the largest."),
            H2("10.8", "Processing Time per Component"),
            TAB("Processing time per component", ["Component", "Time", "Rate"],
                [["Face landmarks (every frame)", "9-13 ms", "every frame"], ["Zoomed second look (small/dark faces)", "~29 ms", "only when needed"],
                 ["Eye-state CNN", "4.4 ms (ONNX, MobileNetV3)", "8 Hz"], ["Emotion (HSEmotion, ONNX)", "13-19 ms", "5 Hz"],
                 ["YOLO11n objects", "41-63 ms", "8 Hz, background"], ["YOLO11n-pose (side view)", "~45 ms", "10 Hz when no frontal face"],
                 ["Temporal model", "5-6 ms", "1 Hz"], ["End-to-end frame path", "12-16 ms", "about 30 FPS"]]),
            P("CPU times were measured with the server's per-model timing log on a laptop-class CPU; ONNX times come from the export benchmark "
              "with two threads."),
            H2("10.9", "Comparison with Existing Work"),
            TAB("Comparison with existing work", ["Aspect", "Typical academic work", "Commercial DMS", "This project"],
                [["Camera", "Webcam, fixed", "Infrared, factory-fitted", "Any camera, any position, moved"],
                 ["Evaluation", "Often random split", "Proprietary", "Subject-independent, leakage measured"],
                 ["Per-driver adaptation", "Rare", "Yes", "Yes, measured (+15.8 points)"],
                 ["False-alarm control", "Thresholds", "Tuned", "Evidence gating, cooldowns, driver feedback"],
                 ["Privacy", "Varies", "On-device", "On-device, consent, no video uploaded"],
                 ["Learning from use", "No", "Fleet data (proprietary)", "Consented, quality-checked, safe promotion"],
                 ["Night", "Limited", "Yes (infrared)", "Needs infrared camera (future work)"]])]


def ch11_deploy(H1, H2, P, UL, FIG, TAB, A):
    return [H1("11", "Deployment and User Manual"),
            H2("11.1", "Deployment Overview"),
            FIG(A / "diag_deploy.png", "Deployment diagram", width="100%"),
            H2("11.2", "Installation"),
            UL(["Clone the repository and create a virtual environment: python3.12 -m venv .venv; source .venv/bin/activate.",
                "Install the dependencies: pip install -r requirements.txt.",
                "Download the pretrained models if needed: bash scripts/download_models.sh (checked against SHA-256 sums).",
                "Place the trained checkpoints in runs/ and runs_temporal2/ (from the Kaggle and laptop training runs)."]),
            H2("11.3", "Running the In-Car Application"),
            UL(["Start: python -m dms.web.server --face-ckpt runs/face/best.pt --temporal-ckpt runs_temporal2/hybrid_binary/deploy.pt --audio --mishap.",
                "Open http://localhost:8000, choose Webcam, Video file or Dash cam / IP stream, and press Start.",
                "Sit normally for a few seconds while the position is learned; monitoring then starts automatically.",
                "React to alerts; tap 'Not true' on a wrong alert; open Trip report at the end and rate sleepiness."]),
            H2("11.4", "Cloud Back End on Supabase"),
            UL(["Create a Supabase project; the schema in db/migrations/0001_init.sql creates the tables with Row Level Security.",
                "Put the session-pooler connection string (postgresql+psycopg://..., sslmode=require) and a random DMS_CLOUD_SECRET in .env.",
                "Start: set -a; source .env; set +a; python -m dms.cloud.app, then register at http://localhost:9000.",
                "Make an administrator: python -m dms.cloud.app --make-admin <email>.",
                "Add a device in the dashboard and start the car app with DMS_SYNC_TOKEN=<device key> and --sync-url <backend>/api/ingest."]),
            H2("11.5", "Containers and Services"),
            P("docker compose up -d starts the back end with a private PostgreSQL database whose port is not exposed. On an in-car Linux box "
              "the deploy/dms-edge.service unit starts the headless monitor with the vehicle (reading an RTSP dash cam, speaking alerts, "
              "storing events offline) with systemd hardening options. deploy/dms-retrain.timer runs the retraining job weekly on the server."),
            H2("11.6", "Retraining in Operation"),
            FIG(A / "diag_retrain.png", "Retraining and safe-deployment flow", width="80%"),
            P("Retraining starts only when enough drivers have shared consented, quality-checked data, so that a new model is judged on several "
              "real people rather than one. Every candidate is stored with its metrics; only an approved model is offered to devices, which verify "
              "its SHA-256 and keep the previous model for rollback."),
            H2("11.7", "Maintenance"),
            UL(["Monitor the [speed] log line for per-model timing and the /api/health endpoint for the store and synchronisation status.",
                "Review the false alarms per alert type in the administrator view and adjust sensitivity presets if needed.",
                "Rotate device keys by removing a device and adding a new one; revoke lost devices immediately.",
                "Keep the model checksums (weights/SHA256SUMS) under version control; the server refuses modified models."])]


def appendices(TAB, P, CH):
    out = ['<h1 id="capB">APPENDIX B: CLOUD API REFERENCE</h1>',
           TAB("Cloud back-end endpoints", ["Method and path", "Authentication", "Purpose"],
               [["POST /api/auth/register", "-", "Create an account; returns a login token"],
                ["POST /api/auth/login", "-", "Log in (rate-limited); returns a token"],
                ["GET, PATCH, DELETE /api/me", "user token", "Profile, consent, account erasure"],
                ["GET, POST /api/devices; DELETE /api/devices/{id}", "user token", "List, add (key shown once), revoke devices"],
                ["POST /api/ingest", "device key", "Upload events, trips and samples (idempotent)"],
                ["GET /api/trips; GET /api/trips/{id}", "user token", "Trips and their alerts"],
                ["POST /api/trips/{id}/kss", "user token", "Sleepiness rating"],
                ["POST /api/events/{id}/feedback", "user token", "Mark an alert correct or a false alarm"],
                ["GET /api/stats", "user token", "Personal statistics"],
                ["GET /api/admin/dataset, /api/admin/overview, /api/admin/models", "admin token", "Data quality, overview, model registry"],
                ["POST /api/admin/models/{id}/status", "admin token", "Approve, reject or roll back a model"],
                ["GET /api/models/latest; GET /api/models/{id}/download", "device key", "Approved model and its SHA-256"],
                ["GET /api/health", "-", "Health check"]]),
           '<h1 id="capC">APPENDIX C: CONFIGURATION PARAMETERS</h1>',
           TAB("Main run-time parameters", ["Parameter", "Default", "Meaning"],
               [["calib_seconds", "5 s (3 s after a camera move)", "Calibration duration"], ["sensitivity", "normal", "Alert timing preset (low / normal / high)"],
                ["eye_hz / face_hz / object_hz / head_hz", "8 / 5 / 8 / 10 Hz", "Rates of the background models"],
                ["eye_max_pitch / eye_max_yaw", "15 / 25 deg", "Pose range in which eye shape is judged"],
                ["seq_len, seq_rate_hz", "300, 10 Hz", "Temporal-model window (30 s)"],
                ["learned_cap", "0.28", "Maximum drowsiness from learned models alone"],
                ["COOLDOWN_S", "critical 8 s, warning 30 s, caution 90 s", "Minimum time between repeats of one alert"],
                ["--audio / --mishap", "off", "Enable cabin sounds / possible-accident detection"],
                ["--sync-url, DMS_SYNC_TOKEN", "-", "Cloud upload endpoint and device key"],
                ["--share-training-data", "off", "Opt-in collection of numeric training windows"]])]
    CH.append(("", "Appendix B: Cloud API Reference", []))
    CH.append(("", "Appendix C: Configuration Parameters", []))
    return out
