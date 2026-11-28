"""Builds the project report (DOCX; convert to PDF with LibreOffice). Run from the repository root:
    python scripts/build_report.py && soffice --headless --convert-to pdf --outdir docs/submission docs/submission/DMS_Project_Report.docx
"""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

import submission_content as C

A = Path("docs/report_assets")
OUT = Path("docs/submission/DMS_Project_Report.docx")
NAVY = RGBColor(0x1E, 0x3A, 0x8A)

doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Times New Roman", Pt(12)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(6)
st.paragraph_format.line_spacing = 1.3
for name, size in (("Heading 1", 16), ("Heading 2", 13.5), ("Heading 3", 12)):
    h = doc.styles[name]
    h.font.name, h.font.size, h.font.bold, h.font.color.rgb = "Times New Roman", Pt(size), True, NAVY
    h.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
for s in doc.sections:
    s.top_margin, s.bottom_margin, s.left_margin, s.right_margin = Cm(2.5), Cm(2.2), Cm(3.0), Cm(2.2)


def header_footer():
    sec = doc.sections[-1]
    hp = sec.header.paragraphs[0]
    hp.text = f"AI-BASED DRIVER MONITORING SYSTEM\tB.Tech CSE | KIT's COEK | {C.YEAR}"
    hp.runs[0].font.size = Pt(9)
    fp = sec.footer.paragraphs[0]
    fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = fp.add_run()
    for kind, text in (("begin", None), (None, "PAGE"), ("end", None)):
        if kind:
            el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), kind); r._r.append(el)
        else:
            el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = text; r._r.append(el)


def P(text="", bold=False, size=None, align=None, italic=False, after=None):
    p = doc.add_paragraph()
    if align == "c":
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    elif align == "j":
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    r = p.add_run(text)
    r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def J(text):
    return P(text, align="j")


def B(items):
    for it in items:
        p = doc.add_paragraph(style="List Bullet")
        if isinstance(it, tuple):
            r = p.add_run(it[0] + ": "); r.bold = True
            p.add_run(it[1])
        else:
            p.add_run(it)
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY


fig_n = [0]
tab_n = [0]


def FIG(path, caption, width=15.5):
    fig_n[0] += 1
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(f"Figure {fig_n[0]}: {caption}", italic=True, size=10.5, align="c")


def shade(cell, hexcolor):
    tcPr = cell._tc.get_or_add_tcPr()
    sh = OxmlElement("w:shd"); sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), hexcolor)
    tcPr.append(sh)


def T(headers, rows, caption, widths=None):
    tab_n[0] += 1
    P(f"Table {tab_n[0]}: {caption}", bold=True, size=10.5, align="c", after=2)
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, h in enumerate(headers):
        c = t.rows[0].cells[i]; c.text = ""
        r = c.paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(10)
        shade(c, "DBEAFE")
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(10)
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph()


def H1(text):
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
    doc.add_heading(text, level=1)


def H2(text):
    doc.add_heading(text, level=2)


# ------------------------------------------------------------------ front matter
P("", after=40)
P(C.TITLE.upper(), bold=True, size=18, align="c")
P("A PROJECT REPORT", bold=True, size=13, align="c", after=18)
P("Submitted by", italic=True, align="c")
t = doc.add_table(rows=1, cols=3); t.style = "Table Grid"; t.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(("Name of Student", "Roll No", "PRN Number")):
    t.rows[0].cells[i].text = h; t.rows[0].cells[i].paragraphs[0].runs[0].bold = True
for n, r, prn in C.STUDENTS:
    c = t.add_row().cells; c[0].text, c[1].text, c[2].text = n, r, prn
P("", after=14)
P("in partial fulfillment for the award of the degree of", italic=True, align="c")
P("BACHELOR OF TECHNOLOGY IN COMPUTER SCIENCE & ENGINEERING", bold=True, size=13, align="c")
P(f"({C.COURSE})", align="c", after=20)
P(f"Under the guidance of {C.GUIDE}", align="c", after=30)
P(C.COLLEGE, bold=True, size=13, align="c")
P(C.YEAR, bold=True, align="c")

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
P(C.COLLEGE, bold=True, size=12.5, align="c")
P("CERTIFICATE", bold=True, size=15, align="c", after=14)
J(f"This is to certify that the project report entitled “{C.TITLE}”, submitted by "
  + " and ".join(f"{n} ({r})" for n, r, _ in C.STUDENTS)
  + " in partial fulfillment of the requirements for the award of the degree of Bachelor of Technology in Computer Science & "
    "Engineering at Kolhapur Institute of Technology's College of Engineering Kolhapur (Empowered Autonomous), Kolhapur, "
    "Maharashtra, India, is a record of their work carried out under my / our supervision and guidance.")
J("The project develops and evaluates a real-time, privacy-preserving driver monitoring system that detects drowsiness, "
  "distraction and unsafe behaviour from an ordinary camera, alerts the driver, and improves itself from consented driver feedback.")
P("", after=50)
t = doc.add_table(rows=4, cols=2)
for i, (a, b) in enumerate((("SIGNATURE", "SIGNATURE"), (C.GUIDE, C.HOD), ("SUPERVISOR", "HEAD OF THE DEPARTMENT"), (C.DEPT, C.DEPT))):
    t.rows[i].cells[0].text, t.rows[i].cells[1].text = a, b
P("\nDate:")

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
P(C.COLLEGE, bold=True, size=12.5, align="c")
P("DECLARATION", bold=True, size=15, align="c", after=14)
J(f"We hereby declare that the project entitled “{C.TITLE}” submitted to Kolhapur Institute of Technology's College of "
  "Engineering Kolhapur (Empowered Autonomous), Kolhapur, Maharashtra, India in partial fulfillment of the award of the degree of "
  "“Bachelor of Technology” in “Computer Science & Engineering” is a bonafide work carried out by us. The system, "
  "experiments and results described in this report were implemented and measured by us; the source code is available at "
  f"{C.REPO}. This report has not been submitted to any other University or Institution for the award of any degree or diploma.")
P("\nNAME OF THE STUDENT(S)")
for i, (n, r, _) in enumerate(C.STUDENTS, 1):
    P(f"{i}. {n} ({r})")
P("\nPlace: KIT's College of Engineering, Kolhapur.\nDate:")

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
P("ACKNOWLEDGEMENT", bold=True, size=15, align="c", after=14)
J(f"We express our sincere gratitude to our project guide {C.GUIDE}, {C.DEPT}, and to the Head of the Department, {C.HOD}, for "
  "their guidance, encouragement and constructive suggestions throughout this project. We thank the faculty members and laboratory "
  "staff of the department for their support and facilities.")
J("We acknowledge the creators of the public datasets used in this work (MRL Eye, FER2013, RAF-DB, AffectNet+, State Farm "
  "Distracted Driver, NTHU-DDD, UTA-RLDD and ESC-50) and the open-source communities behind Python, PyTorch, MediaPipe, "
  "Ultralytics YOLO, OpenCV, FastAPI, SQLAlchemy and PostgreSQL. Finally, we thank our families and friends for their support.")

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
P("ABSTRACT", bold=True, size=15, align="c", after=14)
J(C.ABSTRACT)
P("Keywords: " + ", ".join(C.KEYWORDS), italic=True)

doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)
P("TABLE OF CONTENTS", bold=True, size=15, align="c", after=14)
TOC = ["1. Introduction", "2. Literature Review", "3. Problem Statement and Requirements", "4. Technology Stack",
       "5. System Architecture and Design", "6. Datasets and Deep-Learning Models", "7. Implementation",
       "8. User Interface and Screenshots", "9. Testing and Quality Assurance", "10. Results and Evaluation",
       "11. Challenges and Solutions", "12. Conclusion and Future Scope", "References",
       "Appendix A: Project Structure and How to Run", "Appendix B: Abbreviations"]
for x in TOC:
    P(x, after=2)

# ------------------------------------------------------------------ chapters
doc.add_section(WD_SECTION.NEW_PAGE)
header_footer()
doc.add_heading("CHAPTER 1: INTRODUCTION", level=1)
H2("1.1 Background")
J("Road traffic crashes kill about 1.19 million people every year (WHO, 2023). Drowsiness and distraction are among the "
  "main human causes: a driver who closes the eyes for two seconds at 80 km/h travels over 44 metres blind, and a glance at a "
  "phone has the same effect. Unlike alcohol, fatigue has no roadside test, and drivers notice their own impairment late.")
J("Driver Monitoring Systems (DMS) watch the driver with a camera and warn before attention lapses become a crash. From 2024, "
  "European regulations and Euro NCAP ratings require such systems in new cars, but they rely on dedicated infrared cameras "
  "fitted at the factory. The many vehicles already on the road, and the drivers who use a phone or a dash camera, have no "
  "comparable protection.")
H2("1.2 Motivation")
B(["A webcam, phone or dash camera is already present in most cars and on every laptop: monitoring should work with it.",
   "Published accuracies are often measured with the same people in training and test data and do not hold for a new driver.",
   "Systems that nag are switched off; false alarms matter as much as missed events.",
   "Cabin video is sensitive: processing should stay on the device, without internet."])
H2("1.3 Objectives")
B(["Detect microsleeps, drowsiness, yawning, head nodding, eyes-off-road, phone use, drinking and smoking in real time on a CPU.",
   "Adapt to each driver and to any camera position (front, diagonal, side, moved during the drive).",
   "Fuse all signals into one explainable 0-100 risk score and give proportionate voice, buzzer and suggestion alerts.",
   "Evaluate honestly: every test uses people the model has never seen.",
   "Provide a secure cloud back end with accounts, devices, trips and a retraining pipeline driven by consented driver feedback."])
H2("1.4 Scope")
J("The system targets passenger-car drivers using a laptop, phone or dash camera (including Wi-Fi dash cams over RTSP). It runs "
  "fully offline; the cloud is optional. Night driving with ordinary cameras, dark sunglasses and vehicle data (speed, steering) "
  "are outside the present scope and are discussed as future work.")
H2("1.5 Organisation of the Report")
J("Chapter 2 reviews related work, Chapter 3 defines the problem and requirements, Chapters 4-7 describe the technology, "
  "architecture, models and implementation, Chapter 8 shows the user interface, Chapters 9-10 present testing and results, "
  "Chapter 11 discusses challenges and Chapter 12 concludes.")

H1("CHAPTER 2: LITERATURE REVIEW")
H2("2.1 Eye-based drowsiness measures")
J("Soukupova and Cech introduced the Eye Aspect Ratio (EAR), the ratio of vertical to horizontal eye-landmark distances, which "
  "falls towards zero when the eye closes [1]. PERCLOS, the percentage of time the eyes are at least 80% closed over a window, "
  "is the most validated drowsiness measure from the NHTSA studies [2]. Both depend on the individual: people's open-eye EAR "
  "differs, which motivates per-driver calibration.")
H2("2.2 Facial landmarks and head pose")
J("MediaPipe Face Mesh / Face Landmarker regresses 468-478 3-D landmarks from a single RGB camera in real time on mobile hardware "
  "[5], and also outputs blendshape coefficients (e.g. eye blink, jaw open) and a facial transformation matrix from which yaw, pitch "
  "and roll are obtained. Its detector is designed for faces seen from the front or slightly turned.")
H2("2.3 Deep learning for driver state")
J("Convolutional networks such as MobileNetV3 [6] and EfficientNet [7] classify eye state, expressions and driver behaviour "
  "efficiently. Multi-task learning shares one backbone between tasks, with losses balanced by learned uncertainty [9]. For "
  "drowsiness over time, recurrent and Transformer models [8] process sequences; UTA-RLDD [3] provides 30 hours of realistic, "
  "self-labelled drowsiness videos of 60 people with a published video-level baseline of about 65% for three levels.")
H2("2.4 Object, pose and audio models")
J("Single-stage YOLO detectors [12] locate phones, bottles and cups in real time, and the pose variant predicts body keypoints "
  "(nose, eyes, ears, shoulders) from any viewing angle. YAMNet [13] classifies 521 AudioSet sound classes from spectrograms.")
H2("2.5 Evaluation practice and research gap")
J("Many driver datasets contain few people and many near-identical frames per person. Random image splits therefore place the "
  "same person in training and test sets, and reported accuracies above 95% often fall sharply on new drivers. Commercial "
  "systems use infrared cameras in fixed positions; work on arbitrary, consumer camera placement, on limiting false alarms, and "
  "on learning from real drivers without collecting video is limited. This project addresses these gaps.")
T(["Approach", "Strength", "Limitation", "This project"],
  [["Geometric (EAR/MAR thresholds)", "Light, explainable", "Fixed thresholds, pose-sensitive", "Per-driver calibration, pose gating"],
   ["Single-frame CNN", "Accurate per image", "Identity leakage, no time context", "Subject-independent tests, temporal model"],
   ["Temporal deep models", "Use time context", "Need much data", "Hybrid model + statistics, calibrated features"],
   ["Commercial IR DMS", "Robust, night", "Fixed hardware, expensive", "Any camera, view-adaptive"]],
  "Comparison of related approaches", widths=[3.6, 3.2, 3.6, 4.2])

H1("CHAPTER 3: PROBLEM STATEMENT AND REQUIREMENTS")
H2("3.1 Problem definition")
J("Design and evaluate a real-time system that, from an ordinary camera (and optionally a microphone), detects driver "
  "drowsiness, distraction and unsafe behaviour for any driver and camera position, alerts proportionately with few false alarms, "
  "keeps video on the device, and improves from real use without compromising privacy.")
H2("3.2 Functional requirements")
B(["FR1 Accept webcam, video file and RTSP/HTTP camera streams.", "FR2 Calibrate to the driver automatically from any camera angle.",
   "FR3 Detect microsleep, PERCLOS, blinks, yawns, head nodding, eyes off road (Euro NCAP rules).",
   "FR4 Detect phone, drinking and smoking; recognise emotion; classify cabin sounds.",
   "FR5 Compute a 0-100 risk score with reasons; issue voice/buzzer alerts with suggestions and cooldowns.",
   "FR6 Detect a possible accident and run an SOS countdown.", "FR7 Produce a trip report (screen, JSON, CSV, PDF).",
   "FR8 Store events offline and synchronise to the cloud; accounts, devices, trips and alert feedback in the cloud.",
   "FR9 Collect consented training windows, check quality, retrain, compare and redeploy models."])
H2("3.3 Non-functional requirements")
T(["Requirement", "Target", "Achieved"],
  [["Latency", "< 50 ms per frame on a laptop CPU", "12-16 ms (about 30 FPS)"],
   ["Offline operation", "No internet needed", "Yes; store-and-forward sync"],
   ["Privacy", "No video leaves the device", "Only numbers and alert metadata are uploaded"],
   ["Security", "Authenticated, least privilege", "Tokens, hashed keys, RLS, CSP, checksums"],
   ["Reliability", "Recover from camera/network loss", "Auto-reconnect, local queue, CI tests"],
   ["Portability", "Mac, Linux, Docker, edge boxes", "Docker image, systemd units"]],
  "Non-functional requirements", widths=[3.5, 5.5, 6.0])
H2("3.4 Hardware and software requirements")
B(["Laptop/PC with 4+ CPU cores, webcam and microphone (tested on a MacBook Air, Apple Silicon).",
   "Optional: Android phone with an IP-camera app or a Wi-Fi dash cam; NVIDIA GPU only for training (Kaggle T4 used).",
   "Python 3.12, a modern browser (Chrome), Docker (optional), a Supabase/PostgreSQL database for the cloud part."])

H1("CHAPTER 4: TECHNOLOGY STACK")
T(["Layer", "Technologies"], C.TECH, "Technology stack", widths=[4.0, 11.0])
J("Python was chosen for its machine-learning ecosystem; FastAPI for asynchronous HTTP and WebSocket handling; MediaPipe and "
  "Ultralytics for real-time, CPU-friendly perception; PyTorch/timm for training; ONNX Runtime for fast inference of the emotion "
  "model. The browser front end needs no installation and uses the device camera, microphone and speech synthesis directly. "
  "SQLAlchemy lets the same backend run on SQLite during development and on PostgreSQL (Supabase) in production.")

H1("CHAPTER 5: SYSTEM ARCHITECTURE AND DESIGN")
H2("5.1 Overall architecture")
FIG(A / "architecture.png", "System architecture: in-car edge application and cloud back end", width=16)
J("The in-car application is offline-first. Frames from the camera are analysed by the perception models; geometric indicators "
  "and temporal trackers estimate the driver state; risk fusion and the alert manager decide what to tell the driver. Events and "
  "trip summaries are written to a local SQLite store and synchronised to the cloud when a network is available.")
H2("5.2 Processing pipeline per frame")
B([("Face landmarks", "MediaPipe on the full frame; if no face, a zoomed second look around the driver's head."),
   ("Features", "EAR, MAR, yaw/pitch/roll, gaze, blink and jaw blendshapes."),
   ("Calibration", "first seconds: open-eye EAR (80th percentile), neutral head pose (median); re-learned when the camera moves."),
   ("Trackers", "eye closure (blinks, PERCLOS, microsleep), yawns, head nods, glances (Euro NCAP)."),
   ("Learned models (background thread)", "eye-state CNN, emotion, YOLO objects, YOLO pose (side view), temporal model."),
   ("Risk fusion and alerts", "score, level with hysteresis, reasons, alerts with cooldowns, voice and buzzer.")])
H2("5.3 Use cases")
T(["Actor", "Use case"],
  [["Driver", "Start monitoring; receive alerts; tap 'Not true'; rate sleepiness; view trip report"],
   ["Driver (cloud)", "Register, log in, add devices, view trips and alerts, rate alerts, give/withdraw data consent"],
   ["Admin", "View data quality, false alarms, model registry; approve or roll back a model"],
   ["Device", "Upload events, trips and samples with its key; download the approved model"],
   ["Scheduler", "Weekly retraining job"]], "Use cases", widths=[3.5, 11.5])
H2("5.4 Database design")
T(["Table", "Main fields", "Purpose"],
  [["users", "id, email, password_hash (scrypt), role, consent_training, consent_at", "accounts and consent"],
   ["devices", "id, user_id, name, kind, key_hash (SHA-256), revoked, last_seen", "dash cams / phones / laptops"],
   ["trips", "id, user_id, device_id, t_start, t_end, summary (JSON), kss", "one drive"],
   ["events", "id, trip_id, kind, severity, t_session, payload, feedback", "alerts and driver feedback"],
   ["samples", "id, label, label_source, data (numbers only), quality, accepted, reject_reason", "training windows"],
   ["model_versions", "version, status, sha256, metrics, data, path", "model registry"]],
  "Cloud database schema (PostgreSQL on Supabase; Row Level Security on all tables)", widths=[2.8, 8.0, 4.2])
J("The database has no column for images, video or audio. Row Level Security is enabled on every table with no public policies, "
  "so Supabase's public API can read nothing; only the backend, connected as the owner over TLS, reads and writes.")

H1("CHAPTER 6: DATASETS AND DEEP-LEARNING MODELS")
H2("6.1 Datasets")
T(["Dataset", "Size used", "Task", "Split"], C.DATASETS, "Datasets used", widths=[3.6, 4.8, 3.8, 2.8])
J("All splits are subject-independent: images or videos of one person are never in both training and test sets. A leakage "
  "assertion in the code stops training if this is violated. AffectNet+ and ESC-50 are used only for testing.")
H2("6.2 Models")
T(["Model", "Architecture", "Source", "Role"], C.MODELS, "Neural networks in the system", widths=[3.6, 4.2, 3.2, 4.0])
H2("6.3 How the models work")
J("Convolutional neural networks slide learned filters over the image: early layers respond to edges, middle layers to parts such "
  "as an eyelid or a phone corner, and late layers to states and objects. MobileNetV3 and EfficientNet are designed for a high "
  "accuracy per computation, so they run in real time on a CPU. We start from ImageNet weights (transfer learning) and fine-tune "
  "on driver data.")
J("FaceStateNet shares one backbone between three heads (eye, yawn, emotion). Each dataset labels only one task; missing labels "
  "are masked out of the loss, and the task losses are balanced with learned uncertainty weights [9]. After training, temperature "
  "scaling [10] is fitted on validation data so that predicted confidences match observed accuracy (measured by ECE).")
J("The temporal model receives, every 0.1 s, nine facial measurements normalised to the driver's calibration (eye opening, mouth, "
  "pitch, yaw, roll, two gaze values, blink and jaw blendshapes). A 1-D convolution groups 30 s (300 steps) into patches, a "
  "Transformer encoder relates every moment to every other with self-attention, and a parallel branch computes classic statistics "
  "(PERCLOS, blink rate, pose variance). The combined representation is classified as alert or drowsy.")
H2("6.4 Training")
B(["Image models: Kaggle T4 GPU, AdamW, cosine schedule, augmentation; 'fast' (MobileNetV3, 8 epochs) and 'best' (EfficientNet-B2, 15 epochs, test-time augmentation) presets.",
   "Temporal models: 5-fold GroupKFold by person, 40 epochs, dropout 0.3, Random Forest baseline on engineered features.",
   "Best model per task is deployed (not the biggest): fast face network live, best frame-drowsiness model, hybrid binary temporal model."])

H1("CHAPTER 7: IMPLEMENTATION")
H2("7.1 Geometric indicators")
J("Eye Aspect Ratio: EAR = (|p2-p6| + |p3-p5|) / (2 |p1-p4|), averaged over both eyes; openness = (EAR - 0.3 EAR_open) / "
  "(0.7 EAR_open), with EAR_open from calibration. An eye is closed below 20% openness (the P80 criterion), confirmed by the "
  "eye-state CNN when it is confident. A closure of 1 s or more is a microsleep; PERCLOS is the time-weighted fraction of closed "
  "eyes over 60 s. Mouth Aspect Ratio and the jaw-open blendshape detect yawns held for 1.5 s.")
H2("7.2 Per-driver and any-angle calibration")
J("During the first seconds the system learns the driver's open-eye EAR (80th percentile) and neutral head pose (median) as the "
  "camera sees them, so a side- or diagonally-mounted camera simply has a different neutral pose. Eye shape is judged only near "
  "that pose (open eyes look closed from a low camera when the head tilts back), except when the head drops forward with the "
  "eyelids closing, which is how people doze off. If no frontal face is found (side camera), YOLO11n-pose tracks the head: the "
  "drop of the nose relative to the ear, in head lengths, detects nodding, and the visible eye is checked by the eye CNN.")
H2("7.3 Camera-move detection")
J("When the background (cabin interior, driver masked out; normalised cross-correlation, insensitive to brightness) no longer "
  "matches AND the driver's position, size or view has changed for 0.75-1.5 s, the camera was moved: the position is re-learned "
  "in about 3 s and the trip continues. A driver turning the head changes the face but not the background, so it remains a "
  "distraction alert.")
H2("7.4 Risk fusion")
J("Drowsiness evidence d (PERCLOS, long blinks, yawns, head nods, snoring; learned models as bounded supporting evidence) and "
  "distraction evidence x (off-road time, phone, behaviour) are smoothed over time and combined as risk = 100 * (1 - (1-d)(1-x)) * "
  "(1 + 0.3 a), where a is negative high-arousal affect. Levels (OK, CAUTION 30, WARNING 55, CRITICAL 80) use hysteresis so they "
  "do not flicker. A microsleep or head drop forces CRITICAL; without any measured sign the score stays below CAUTION. When the "
  "driver taps 'Not true', the evidence is reset and that alert is paused for three minutes.")
H2("7.5 Alerts and sensitivity")
T(["Setting", "Low", "Normal", "High"],
  [["Microsleep (eyes closed)", "1.5 s", "1.0 s", "0.8 s"], ["Looking away", "4 s", "3 s (Euro NCAP)", "2.5 s"],
   ["Glances within 30 s", "12 s", "10 s", "8 s"], ["Drowsy score threshold", "70%", "60%", "50%"],
   ["Repeat cooldown", "x1.5", "warning 30 s, caution 90 s", "x0.75"]], "Alert sensitivity presets", widths=[5, 3, 4, 3])
H2("7.6 Real-time engineering")
J("Face landmarks run on every frame; the heavier networks run in a background thread on the newest frame, so a slow model "
  "only updates less often and never delays the video. Denormal floating-point flushing, model checksums at start-up and a "
  "per-model timing log keep performance predictable. Audio is processed in 0.5 s windows from a 16 kHz AudioWorklet stream.")
H2("7.7 Cloud back end and retraining pipeline")
B(["The car app stores alerts and trips in SQLite and uploads them with a device key over HTTPS when online (idempotent).",
   "Drivers rate alerts and their sleepiness (Karolinska Sleepiness Scale). With consent, the 30 s of numeric measurements before a rated alert become a labelled training window.",
   "A quality gate keeps only windows where the driver's face is visible (>= 90%), frontal (>= 70%), the same person, calibrated and, if known, moving.",
   "The retraining job (weekly systemd timer or by hand) exports windows split by driver, trains a candidate on UTA-RLDD plus real data, and compares it with the current model on held-out drivers. It is approved only if macro-F1 is not lower and the false-alarm rate does not rise.",
   "Devices download the approved model and verify its SHA-256 before use; an admin can roll back."])
H2("7.8 Security and privacy")
T(["Area", "Measure"],
  [["Footage", "Never stored or uploaded"], ["In-car app", "Loopback by default; token, Origin/Host checks, CSP, size limits when exposed"],
   ["Models", "SHA-256 verified at start; downloaded updates verified"],
   ["Accounts", "scrypt password hashing, signed 12 h tokens, login rate limit"],
   ["Devices", "Per-device random key, only its hash stored, revocable"],
   ["Database", "Supabase PostgreSQL, Row Level Security, TLS"],
   ["Consent", "Opt-in, withdrawal deletes shared data, account erasure, pseudonymised exports"]],
  "Security and privacy measures", widths=[3.5, 11.5])

H1("CHAPTER 8: USER INTERFACE AND SCREENSHOTS")
J("The in-car interface shows the live camera and the AI view side by side, the risk gauge with its reasons, the driver-state "
  "cards and a live timeline. Alerts appear as a large banner with a suggestion, a voice message and a buzzer tone, and can be "
  "rejected with one tap.")
FIG(A / "live_monitor.png", "Live monitor: camera, AI view with landmarks, risk gauge and driver-state cards")
FIG(A / "live_alerts.png", "Phone-use alert with suggestion and the one-tap 'Not true' feedback button")
FIG(A / "settings.png", "Controls: source selection, sensitivity and the settings menu")
FIG(A / "trip_report.png", "Trip report with KPIs, time in each risk level, alerts, sleepiness rating and risk timeline")
FIG(A / "cloud_login.png", "Cloud back end: login page")
FIG(A / "cloud_dashboard.png", "Cloud dashboard: KPIs, devices, safety-score trend, alerts by type and trips (sample data)", width=13)
FIG(A / "cloud_admin.png", "Admin view: training-data quality, false alarms reported by drivers and the model registry (sample data)", width=13)

H1("CHAPTER 9: TESTING AND QUALITY ASSURANCE")
H2("9.1 Automated tests")
J("The repository contains 60 automated tests (pytest) covering feature extraction, calibration, PERCLOS, microsleep, yawns, "
  "glance rules, risk fusion and hysteresis, alert cooldowns and muting, audio rules, mishap escalation, camera-move detection, "
  "zoom search, driver selection among several faces, the cloud API (accounts, brute-force limit, device keys, isolation between "
  "accounts, consent, quality gate, export) and an end-to-end retraining run. GitHub Actions runs them on every push, again "
  "against a real PostgreSQL 16 server, and builds the Docker image.")
H2("9.2 Scenario tests")
T(["Scenario", "Expected behaviour", "Result"],
  [["Eyes closed >= 1 s", "Microsleep alarm", "Pass"], ["Head drops forward, eyes closing", "Microsleep / head-drop alarm in ~1-2 s", "Pass"],
   ["Head tilted back (low webcam)", "Eyes 'not judged', no false microsleep", "Pass"],
   ["Look away 3 s", "'Please look ahead' alert", "Pass"], ["Mirror glance < 1.5 s", "No change", "Pass"],
   ["Phone held to ear / in front of face", "Phone alert in ~1 s", "Pass (21/22 frames)"],
   ["Side-mounted camera, driver nodding", "Side view, head-lowering caution then alarm", "Pass"],
   ["Night wide-angle dash cam", "Driver found, no false alerts", "Pass (71% frames)"],
   ["Camera moved after calibration", "'Camera moved' and quick re-learning", "Pass (0.75-1.5 s)"],
   ["Passenger leans in", "Ignored, driver followed", "Pass"], ["Crash sound from radio", "Calm notice, no alarm", "Pass"],
   ["'Not true' tapped", "Risk drops, alert paused 3 min", "Pass"], ["No internet", "Works; uploads later", "Pass"]],
  "Scenario tests", widths=[5.0, 6.5, 3.5])
H2("9.3 Performance")
J("End-to-end processing takes 12-16 ms per frame on a laptop CPU (about 30 frames per second), measured in the browser and in "
  "the server's per-model timing log.")

H1("CHAPTER 10: RESULTS AND EVALUATION")
T(["Task", "Test data", "Result", "Notes"], C.RESULTS, "Results on data never seen in training", widths=[3.6, 4.6, 3.8, 3.0])
FIG(A / "results_accuracy.png", "Accuracy per task on unseen data", width=15)
FIG(A / "results_leakage.png", "Identity leakage: random splits versus unseen people", width=12)
FIG(A / "results_temporal.png", "Drowsiness over time on UTA-RLDD (60 people, 5-fold, unseen subjects)", width=13)
FIG(A / "results_audio.png", "Cabin-sound detection on ESC-50", width=12)
H2("10.1 Discussion")
J("Per-frame tasks (eye state, yawns, behaviour) are at or near the state of the art when measured on unseen people. Drowsiness "
  "over time is the hard problem: 80-87% for alert versus drowsy on unseen people, in line with the dataset's published baseline. "
  "Removing per-driver calibration drops the hybrid model from 53.4% to 37.6% (three classes), close to chance, showing that "
  "personal baselines are essential. Emotion from a single image transfers poorly between datasets (48% on AffectNet+), so emotion "
  "only modifies the risk score. The false-alarm rate on real drives is now measurable through driver feedback.")
H2("10.2 Novel contributions")
for name, txt in C.CONTRIBUTIONS:
    B([(name, txt)])

H1("CHAPTER 11: CHALLENGES AND SOLUTIONS")
T(["Challenge", "Solution", "Outcome"], C.CHALLENGES, "Challenges encountered and how they were solved", widths=[4.5, 7.0, 3.5])

H1("CHAPTER 12: CONCLUSION AND FUTURE SCOPE")
H2("12.1 Conclusion")
J("We built and evaluated a complete driver monitoring system that runs in real time on ordinary hardware, adapts to each driver "
  "and to arbitrary camera placement, explains its alerts, limits false alarms, keeps video on the device and improves itself from "
  "consented driver feedback through a safe retraining pipeline. Honest, subject-independent evaluation shows strong per-frame "
  "accuracy (98% eye state, 92% behaviour on new drivers) and 80-87% for drowsiness over time, and quantifies two effects that "
  "matter for the field: identity leakage and the value of per-driver calibration.")
H2("12.2 Future scope")
B(["Infrared camera support for night driving; sunglasses-robust eye tracking.",
   "Native Android app (the in-car logic and the cloud API are ready to port) running when the phone is locked.",
   "Vehicle data (speed, steering) via OBD-II to pause alerts when parked and to strengthen drowsiness estimates.",
   "Measuring false alarms per hour on real drives and retraining as consented data grows.",
   "On-device quantisation (INT8) for low-cost dash-cam processors."])

H1("REFERENCES")
for i, r in enumerate(C.REFERENCES, 1):
    P(f"[{i}] {r}", size=11)

H1("APPENDIX A: PROJECT STRUCTURE AND HOW TO RUN")
for line in ["dms/perception   face landmarks, zoom search, pose, eyes, emotion, objects, audio",
             "dms/features     geometry (EAR, MAR, pose), temporal trackers, camera-move detection",
             "dms/fusion       risk fusion, alerts, mishap detection",
             "dms/models       network definitions and checkpoints",
             "dms/training     image and temporal training, evaluation, export",
             "dms/web          in-car server (FastAPI + WebSocket) and browser UI",
             "dms/cloud        cloud backend, database, quality gate, export, retraining, model download",
             "tests            60 automated tests;  db/migrations  PostgreSQL schema;  deploy  systemd units"]:
    P(line, size=10.5, after=0)
P("")
P("Run the live demo:", bold=True)
P("python -m dms.web.server --face-ckpt runs/face/best.pt --temporal-ckpt runs_temporal2/hybrid_binary/deploy.pt --audio --mishap", size=10)
P("Run the cloud backend:", bold=True)
P("set -a; source .env; set +a; python -m dms.cloud.app", size=10)

H1("APPENDIX B: ABBREVIATIONS")
T(["Abbreviation", "Meaning"],
  [["DMS", "Driver Monitoring System"], ["EAR / MAR", "Eye / Mouth Aspect Ratio"], ["PERCLOS", "Percentage of eye closure over time"],
   ["KSS", "Karolinska Sleepiness Scale"], ["CNN", "Convolutional Neural Network"], ["YOLO", "You Only Look Once (object detector)"],
   ["ECE", "Expected Calibration Error"], ["RLS", "Row Level Security"], ["RTSP", "Real Time Streaming Protocol"],
   ["CI", "Continuous Integration"], ["Euro NCAP", "European New Car Assessment Programme"]], "Abbreviations", widths=[4, 11])

doc.save(OUT)
print("saved", OUT)
