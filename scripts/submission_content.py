"""Shared facts for the report, paper and slides: one place, so all three say the same thing."""

TITLE = "AI-Based Driver Monitoring System for Real-Time Detection of Drowsiness and Unsafe Behaviour"
STUDENTS = [("Purva Manoj Jadhav", "B09", "2324000121"), ("Narayani Prithviraj Ghorpade", "C18", "2324000414")]
GUIDE = "Dr. Abhinandan P. Shirahatti"
HOD = "Dr. Lingaraj A. Hadimani"
COLLEGE = "KOLHAPUR INSTITUTE OF TECHNOLOGY'S COLLEGE OF ENGINEERING KOLHAPUR (EMPOWERED AUTONOMOUS)"
DEPT = "Department of Computer Science & Engineering"
YEAR = "2026-27"
COURSE = "Honors Mini Project (UCSHN0751)"
REPO = "github.com/Narayani26/AI-Driver-monitoring-system"

ABSTRACT = (
    "Driver drowsiness and distraction cause a large share of road accidents, yet most monitoring systems either need a "
    "dedicated infrared camera at a fixed position or report accuracy figures that do not hold for new drivers. We present an "
    "AI-based Driver Monitoring System (DMS) that runs in real time (about 30 frames per second) on an ordinary laptop CPU with "
    "a webcam, phone or dash camera, and needs no internet connection. MediaPipe face landmarks, a multi-task convolutional "
    "network for eye state, a Conv-Transformer over 30 s of driver-normalised facial measurements, YOLO object and pose "
    "detectors and the YAMNet audio classifier feed an explainable risk-fusion model with Euro NCAP distraction rules, voice and "
    "buzzer alerts with suggestions, and a possible-accident check. Every evaluation is subject-independent. Eye state reaches "
    "98.0% accuracy on unseen people, driver behaviour 92.1% on unseen drivers, and alert-versus-drowsy classification 87.3% "
    "(Random Forest) and 80.5% (deployed hybrid network) of videos of 60 unseen people. Per-driver calibration adds 15.8 "
    "percentage points, and the common random split overstates accuracy by up to 41 times fewer errors. View-adaptive "
    "monitoring handles side-mounted, wide-angle and moved cameras; learned models act only as supporting evidence to limit "
    "false alarms; and a privacy-preserving cloud backend (FastAPI, PostgreSQL on Supabase) collects consented, "
    "quality-checked driver feedback to retrain and safely redeploy the drowsiness model."
)
KEYWORDS = ["driver monitoring", "drowsiness detection", "distraction", "computer vision", "deep learning",
            "PERCLOS", "subject-independent evaluation", "edge AI", "privacy"]

FEATURES = [
    ("Microsleep / eyes closed", "Eye opening relative to the driver's own calibrated open eye (EAR), confirmed by an eye-state CNN; alert after 1 s closed (Normal sensitivity)"),
    ("Drowsiness build-up", "PERCLOS over 60 s, long blinks, yawns, head nodding and a learned 30 s drowsiness model"),
    ("Head dropping (any camera)", "Frontal: head pitched down with eyelids closing; side camera: nose dropping relative to the ear (pose keypoints)"),
    ("Eyes off the road", "Head pose and gaze; Euro NCAP rules: one glance away of 3 s, or 10 s within 30 s"),
    ("Phone, drinking, smoking", "YOLO11n object detection near the driver's head, hands and lap"),
    ("Stress / affect", "Facial emotion (HSEmotion); raises risk, never alerts alone"),
    ("Cabin sounds", "YAMNet: snoring, siren, crash/skid, shouting, phone ringing, crying baby"),
    ("Possible accident", "Camera jolt or crash sound, then an unresponsive driver: 15 s countdown, then SOS"),
    ("Camera moved", "Background and driver position change together: position re-learned in about 3 s"),
    ("Driver feedback", "'Not true' lowers the risk at once and pauses that alert; KSS sleepiness rating per trip"),
]

TECH = [
    ("In-car web app (front end)", "HTML5, CSS3, JavaScript (no framework), Canvas 2D, WebSocket, getUserMedia, AudioWorklet, Web Speech API, Web Audio"),
    ("In-car server", "Python 3.12, FastAPI, Uvicorn, background model thread, SQLite (offline event store), HTTPS store-and-forward sync"),
    ("Cloud back end", "FastAPI, SQLAlchemy 2, psycopg 3, PostgreSQL 16 (Supabase), dashboard in HTML/CSS/JS"),
    ("Computer vision / ML", "PyTorch, timm, MediaPipe, Ultralytics YOLO11, ONNX Runtime, OpenCV, scikit-learn, NumPy, pandas"),
    ("Training infrastructure", "Kaggle (NVIDIA T4 GPU) for image models, MacBook CPU for temporal models, matplotlib"),
    ("DevOps", "Git/GitHub, GitHub Actions CI (unit tests, PostgreSQL tests, Docker build), Docker, Docker Compose, systemd"),
]

DATASETS = [
    ("MRL Eye", "84,898 near-infrared eye crops, 37 people", "Eye open/closed", "split by person"),
    ("Yawn-Eye (Kaggle)", "2,900 face images", "Yawn / eye", "official train/test"),
    ("FER2013", "35,887 face images, 7 emotions", "Emotion", "official test set"),
    ("RAF-DB", "15,339 aligned faces, 7 emotions", "Emotion", "official test set"),
    ("AffectNet+ (validation)", "3,500 human-labelled faces used", "Emotion, cross-dataset test only", "never trained on"),
    ("State Farm Distracted Driver", "22,424 images, 26 drivers, 10 classes", "Driver behaviour (side camera)", "split by driver"),
    ("NTHU-DDD (Kaggle copy)", "18,000 frames, 4 people", "Frame drowsiness", "split by person"),
    ("UTA-RLDD", "178 videos, 60 people, alert / low-vigilant / drowsy", "Drowsiness over time", "5-fold, split by person"),
    ("ESC-50", "560 clips used (40 classes)", "Cabin-sound alerts", "evaluation only"),
]

MODELS = [
    ("MediaPipe Face Landmarker", "CNN detector + mesh regressor", "Google, pretrained", "478 3-D face points, 52 blendshapes, head rotation"),
    ("FaceStateNet (ours)", "MobileNetV3-L / EfficientNet-B2 backbone, 3 heads", "Trained by us", "Eye open/closed (live); yawn and emotion heads evaluated"),
    ("HSEmotion", "EfficientNet-B0 (ONNX)", "Pretrained on AffectNet", "Facial emotion (live)"),
    ("CabinNet (ours)", "EfficientNet-B0", "Trained by us (State Farm)", "10 driver behaviours, side-view cameras"),
    ("Frame-drowsiness CNN (ours)", "EfficientNet-B2", "Trained by us (NTHU-DDD)", "Supporting drowsiness evidence"),
    ("Hybrid Conv-Transformer (ours)", "1-D conv + Transformer encoder + statistics branch", "Trained by us (UTA-RLDD)", "Alert vs drowsy from 30 s of features"),
    ("YOLO11n", "Single-stage object detector", "Pretrained on COCO", "Phone, bottle, cup"),
    ("YOLO11n-pose", "Detector + keypoint regressor", "Pretrained on COCO", "Head keypoints from any angle (side view)"),
    ("YAMNet", "CNN on audio spectrograms", "Pretrained on AudioSet", "521 sound classes -> cabin events"),
]

RESULTS = [
    ("Eye open/closed", "MRL Eye, unseen people (12,576)", "98.0%", "ECE 0.006 after temperature scaling"),
    ("Yawn", "Yawn-Eye test (219)", "99.1-100%", "small set, no person IDs: upper bound"),
    ("Emotion (7 classes)", "FER2013 + RAF-DB test (10,246)", "75.4% (B2) / 73.8% (MobileNetV3)", ""),
    ("Emotion, other dataset", "AffectNet+ (3,500, never trained on)", "48.4% ours / 63.1% HSEmotion", "why HSEmotion is used live"),
    ("Driver behaviour (10 classes)", "State Farm, unseen drivers (3,440)", "92.1%", "99.6% with a random split"),
    ("Frame drowsiness", "NTHU-DDD, unseen person (5,953)", "66.7%", "94-97% with a random split"),
    ("Alert vs drowsy (videos)", "UTA-RLDD, 5-fold, unseen people", "87.3% RF / 80.5% Hybrid", "chance 50%"),
    ("3 levels (videos)", "UTA-RLDD, unseen people", "64.7% RF / 55.6% Transformer", "chance 33%; published baseline ~65%"),
    ("Calibration ablation", "UTA-RLDD, hybrid model", "53.4% with vs 37.6% without", "+15.8 points"),
    ("Snoring / siren / baby / crash", "ESC-50 (560 clips)", "95 / 93 / 82 / 78% recall", "0% false shouting alarms"),
    ("Side camera (90-degree profile)", "stock clip, 77 frames", "head found 77/77 (face 0/77)", "nod alert raised"),
    ("Night wide-angle dash cam", "45-frame clip", "face found 71% (0% before zoom search)", "no false alerts"),
    ("Speed", "laptop CPU, end to end", "12-16 ms/frame (~30 FPS)", "models in a background thread"),
]

CONTRIBUTIONS = [
    ("Identity leakage quantified", "On two driver datasets the usual random split hides most errors: State Farm 99.6% vs 92.1% on new drivers (17x more errors; 41x with a larger network) and NTHU-DDD 7-11x. Larger models widen the gap."),
    ("Per-driver calibration measured", "Normalising every feature to the driver's own baseline, in training and live, adds +9.6 to +15.8 points on unseen people."),
    ("View-adaptive monitoring", "Pose-gated eye measurement, zoom search for small/dark faces, body-pose fallback for side cameras and automatic re-calibration when the camera is moved (kept separate from a driver turning the head)."),
    ("Evidence-gated risk fusion", "Learned models trained on other people and cameras are supporting evidence only; a warning needs measurable signs; every alert lists its reasons; the driver can reject an alert."),
    ("Privacy-preserving continual learning", "Only numeric measurements, with consent; driver-given labels; an auditable quality gate; evaluation split by driver; a new model is deployed only if it is not worse and does not raise false alarms."),
    ("Multi-modal mishap detection", "Camera jolt + crash sound + driver responsiveness with step-wise escalation."),
]

CHALLENGES = [
    ("Lag of 1.4 s per frame with large models", "Models moved to a background thread; frames answered from landmarks; per-model timing log", "12-16 ms per frame"),
    ("False 'eyes closed' when the head tilts back", "Eyes judged only near the calibrated pose; head-down + closed eyelids still counts (dozing)", "no false microsleeps on test clip"),
    ("Learned models wrong on new people", "Supporting evidence only; capped below CAUTION without measured signs", "fewer false alarms"),
    ("Side-mounted camera: no face found", "YOLO pose fallback: head nodding + eye crop CNN", "head found 77/77 frames"),
    ("Small/dark face in wide-angle night dash cam", "Zoom search: crop around the head, enlarge, re-detect", "0% -> 71% of frames"),
    ("Camera moved after calibration", "Background + driver-geometry change detector, quick re-learning", "detected in 0.75-1.5 s"),
    ("Phone held in front of the face ignored", "Ignore only a screen that contains the whole face", "21/22 frames detected"),
    ("Score stayed high after a false alarm", "'Not true' resets evidence and pauses the alert; faster decay", "drops immediately"),
    ("Passengers in view", "Driver = largest face, then tracked by position", "passengers ignored"),
    ("Optimistic accuracy in literature", "Subject-independent splits with leakage assertions", "honest numbers"),
]

REFERENCES = [
    "T. Soukupova and J. Cech, \"Real-time eye blink detection using facial landmarks,\" in Proc. 21st Computer Vision Winter Workshop, 2016.",
    "W. W. Wierwille et al., \"Research on vehicle-based driver status/performance monitoring,\" NHTSA Report DOT HS 808 247, 1994 (PERCLOS).",
    "R. Ghoddoosian, M. Galib and V. Athitsos, \"A realistic dataset and baseline temporal model for early drowsiness detection,\" in Proc. IEEE CVPR Workshops, 2019.",
    "C.-H. Weng, Y.-H. Lai and S.-H. Lai, \"Driver drowsiness detection via a hierarchical temporal deep belief network,\" in ACCV Workshops, 2016 (NTHU-DDD).",
    "Y. Kartynnik, A. Ablavatski, I. Grishchenko and M. Grundmann, \"Real-time facial surface geometry from monocular video on mobile GPUs,\" CVPR Workshops, 2019.",
    "A. Howard et al., \"Searching for MobileNetV3,\" in Proc. IEEE/CVF ICCV, 2019.",
    "M. Tan and Q. V. Le, \"EfficientNet: Rethinking model scaling for convolutional neural networks,\" in Proc. ICML, 2019.",
    "A. Vaswani et al., \"Attention is all you need,\" in Advances in Neural Information Processing Systems, 2017.",
    "A. Kendall, Y. Gal and R. Cipolla, \"Multi-task learning using uncertainty to weigh losses,\" in Proc. IEEE CVPR, 2018.",
    "C. Guo, G. Pleiss, Y. Sun and K. Q. Weinberger, \"On calibration of modern neural networks,\" in Proc. ICML, 2017.",
    "A. V. Savchenko, \"HSEmotion: High-speed emotion recognition library,\" Software Impacts, 2022.",
    "G. Jocher et al., \"Ultralytics YOLO11,\" software, 2024.",
    "S. Hershey et al., \"CNN architectures for large-scale audio classification,\" in Proc. IEEE ICASSP, 2017 (YAMNet family).",
    "A. Mollahosseini, B. Hasani and M. H. Mahoor, \"AffectNet: A database for facial expression, valence, and arousal computing in the wild,\" IEEE Trans. Affective Computing, 2017.",
    "S. Li, W. Deng and J. Du, \"Reliable crowdsourcing and deep locality-preserving learning for expression recognition in the wild,\" in Proc. IEEE CVPR, 2017 (RAF-DB).",
    "I. J. Goodfellow et al., \"Challenges in representation learning: A report on three machine learning contests,\" ICONIP, 2013 (FER2013).",
    "R. Fusek, \"Pupil localization using geodesic distance,\" in Proc. ISVC, 2018 (MRL Eye dataset).",
    "State Farm, \"Distracted Driver Detection,\" Kaggle competition dataset, 2016.",
    "Euro NCAP, \"Assessment Protocol - Safety Assist: Driver State Monitoring,\" 2023.",
    "T. Akerstedt and M. Gillberg, \"Subjective and objective sleepiness in the active individual,\" Int. J. Neuroscience, 1990 (KSS).",
    "K. J. Piczak, \"ESC: Dataset for environmental sound classification,\" in Proc. ACM Multimedia, 2015.",
    "World Health Organization, \"Global status report on road safety 2023,\" Geneva, 2023.",
]
