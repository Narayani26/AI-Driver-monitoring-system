"""IEEE-style two-column research paper (DOCX -> PDF with LibreOffice)."""
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt

import submission_content as C

A = Path("docs/report_assets")
OUT = Path("docs/submission/DMS_IEEE_Research_Paper.docx")
doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Times New Roman", Pt(10)
st.element.rPr.rFonts.set(qn("w:eastAsia"), "Times New Roman")
st.paragraph_format.space_after = Pt(3)
st.paragraph_format.line_spacing = 1.0
sec = doc.sections[0]
sec.page_height, sec.page_width = Cm(29.7), Cm(21.0)
sec.top_margin, sec.bottom_margin, sec.left_margin, sec.right_margin = Cm(1.9), Cm(2.5), Cm(1.6), Cm(1.6)


def P(text="", size=None, bold=False, italic=False, align="j", after=None):
    p = doc.add_paragraph()
    p.alignment = {"j": WD_ALIGN_PARAGRAPH.JUSTIFY, "c": WD_ALIGN_PARAGRAPH.CENTER, "l": WD_ALIGN_PARAGRAPH.LEFT}[align]
    r = p.add_run(text); r.bold, r.italic = bold, italic
    if size:
        r.font.size = Pt(size)
    if after is not None:
        p.paragraph_format.space_after = Pt(after)
    return p


def H(text):
    p = P(text.upper(), size=10, align="c", after=3)
    p.paragraph_format.space_before = Pt(8)


def SH(text):
    p = P(text, italic=True, align="l", after=2)
    p.paragraph_format.space_before = Pt(4)


def FIG(path, caption, width=8.4):
    doc.add_picture(str(path), width=Cm(width))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    P(caption, size=8, align="c")


def TABLE(caption, headers, rows):
    P(caption.upper(), size=8, align="c", after=1)
    t = doc.add_table(rows=1, cols=len(headers)); t.style = "Table Grid"
    for i, h in enumerate(headers):
        t.rows[0].cells[i].text = ""; r = t.rows[0].cells[i].paragraphs[0].add_run(h); r.bold = True; r.font.size = Pt(7.5)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""; r = cells[i].paragraphs[0].add_run(str(v)); r.font.size = Pt(7.5)
    P("", after=2)


# title block (single column)
P(C.TITLE, size=22, align="c", after=6)
P(" and ".join(n for n, _, _ in C.STUDENTS), size=11, align="c", after=0)
P(f"{C.DEPT}, KIT's College of Engineering (Empowered Autonomous), Kolhapur, Maharashtra, India", size=9.5, italic=True, align="c", after=0)
P(f"Guide: {C.GUIDE}", size=9.5, italic=True, align="c", after=10)

# two columns from here
new = doc.add_section(WD_SECTION.CONTINUOUS)
cols = new._sectPr.xpath("./w:cols")
c = cols[0] if cols else OxmlElement("w:cols")
c.set(qn("w:num"), "2"); c.set(qn("w:space"), "425")
if not cols:
    new._sectPr.append(c)

p = P("", align="j")
r = p.add_run("Abstract—"); r.bold = True; r.italic = True; r.font.size = Pt(9)
r = p.add_run(C.ABSTRACT); r.bold = True; r.font.size = Pt(9)
p = P("", align="j")
r = p.add_run("Index Terms—"); r.bold = True; r.italic = True; r.font.size = Pt(9)
r = p.add_run(", ".join(C.KEYWORDS) + "."); r.bold = True; r.font.size = Pt(9)

H("I. Introduction")
P("Road crashes kill about 1.19 million people every year [22], and drowsiness and distraction are leading human causes. Driver "
  "Monitoring Systems (DMS) observe the driver and warn before a lapse becomes a crash; regulation and Euro NCAP [19] now require "
  "them in new cars, using factory-fitted infrared cameras. Vehicles already on the road, and drivers relying on a phone or dash "
  "camera, lack such protection. Two further problems limit practical use: accuracies are frequently reported with the same people "
  "in training and test sets, and systems that raise false alarms are switched off.")
P("We present a complete, real-time DMS for ordinary cameras that runs offline on a laptop CPU. Our contributions are:")
for i, (name, txt) in enumerate(C.CONTRIBUTIONS, 1):
    p = P("", align="j"); r = p.add_run(f"{i}) {name}: "); r.italic = True; p.add_run(txt)

H("II. Related Work")
P("Eye Aspect Ratio (EAR) [1] and PERCLOS [2] are the classic geometric drowsiness measures; both depend on the individual's eye "
  "shape. MediaPipe [5] provides real-time 3-D face landmarks and blendshapes from a monocular camera. CNNs such as MobileNetV3 [6] "
  "and EfficientNet [7] classify eye state, expression and behaviour; multi-task learning with uncertainty weighting [9] shares a "
  "backbone between tasks; temperature scaling [10] calibrates confidences. For temporal drowsiness, UTA-RLDD [3] offers 60 people "
  "with realistic self-labelled videos and a baseline of about 65% for three levels; NTHU-DDD [4] is widely used but contains few "
  "people. YOLO [12] detects objects and body keypoints; YAMNet [13] classifies audio events. Few works evaluate strictly on unseen "
  "drivers, address arbitrary camera placement, or learn from real drivers without collecting video.")

H("III. System Architecture")
FIG(A / "architecture.png", "Fig. 1. Architecture: in-car edge application (offline-first) and cloud back end.")
P("Frames from a webcam, phone, dash camera (RTSP) or file are processed by MediaPipe on every frame; heavier networks (eye-state "
  "CNN, emotion, YOLO objects and pose, temporal model) run in a background thread on the newest frame, so a slow model only "
  "updates less often and the video stays real time. Trackers estimate blinks, PERCLOS, microsleeps, yawns, head nods and glances; "
  "a risk-fusion model and an alert manager decide the response. Events go to a local SQLite store and are uploaded over HTTPS "
  "with a per-device key to a FastAPI back end on PostgreSQL (Supabase) with Row Level Security.")

H("IV. Method")
SH("A. Driver-normalised features and calibration")
P("Per frame we compute EAR, MAR, yaw, pitch, roll, two gaze values and the blink and jaw-open blendshapes. During the first "
  "seconds the system learns the driver's open-eye EAR (80th percentile) and neutral head pose (median) as the camera sees them; "
  "every feature is normalised to this profile, both live and when training the temporal model. Openness = (EAR - 0.3 EAR_open) / "
  "(0.7 EAR_open); an eye is closed below 0.2 (P80), confirmed by the eye-state CNN when confident.")
SH("B. View-adaptive perception")
P("Eye shape is judged only near the calibrated pose, because open eyes look closed when the head tilts back in front of a low "
  "camera; a head dropped forward with closing eyelids (blendshape > 0.55) is still judged, since that is how drivers doze. Small "
  "or dark faces are re-detected in an enlarged crop around the driver's head. When no frontal face exists (side camera), "
  "YOLO11n-pose keypoints give the nose drop relative to the ear in head lengths (alert -0.15, nodding +0.25 on a test clip) and an "
  "eye crop for the CNN. A camera move is declared when the masked background no longer matches (normalised cross-correlation "
  "< 0.6) and the driver's position, size or view changes for 0.75-1.5 s; the position is then re-learned in about 3 s. A driver "
  "turning the head leaves the background unchanged and remains a distraction.")
SH("C. Models")
TABLE("Table I. Neural networks", ["Model", "Type", "Role"], [(m, a, r) for m, a, _, r in C.MODELS])
P("FaceStateNet shares a MobileNetV3/EfficientNet backbone between eye, yawn and emotion heads with masked losses and learned "
  "uncertainty weights. The hybrid temporal model applies a 1-D convolution that patches 300 steps (30 s at 10 Hz), a Transformer "
  "encoder, and a parallel statistics branch (PERCLOS, blink rate, pose variance), then classifies alert versus drowsy.")
SH("D. Evidence-gated risk fusion")
P("Drowsiness evidence d (PERCLOS, long blinks, yawns, head nods, snoring) and distraction evidence x (off-road time beyond 1.5 s, "
  "phone, behaviour) are smoothed and combined as R = 100 (1 - (1-d)(1-x)) (1 + 0.3a), with a the negative high-arousal affect. "
  "Learned models are supporting evidence: alone they cannot exceed 28, below CAUTION (30); with measured signs they add at most 30% "
  "of the remaining headroom. Levels use hysteresis; a microsleep or head drop forces CRITICAL. Alerts have cooldowns and three "
  "sensitivity presets; a driver's 'Not true' resets the evidence and pauses that alert for three minutes.")
SH("E. Privacy-preserving continual learning")
P("With consent, the 30 s of numeric features before a rated alert, or at a Karolinska Sleepiness Scale [20] rating, become a "
  "labelled window. A quality gate requires the driver's face in >= 90% of frames, frontal pose >= 70%, the same tracked person, "
  "a plausible calibration and, if known, a moving vehicle. Retraining exports windows split by driver, trains on UTA-RLDD plus "
  "real data, and compares candidate and current model on held-out drivers; the candidate is approved only if macro-F1 is not "
  "lower and the false-alarm rate does not rise. Devices verify the SHA-256 of the approved model.")

H("V. Experimental Setup")
TABLE("Table II. Datasets (all splits subject-independent)", ["Dataset", "Size", "Task"], [(d, s, t) for d, s, t, _ in C.DATASETS])
P("Image models were trained on a Kaggle T4 GPU (AdamW, cosine schedule; MobileNetV3 for 8 and EfficientNet-B2 for 15 epochs). "
  "Temporal models used 5-fold GroupKFold by person on UTA-RLDD (178 videos, first 6 min at 10 Hz, 30 s windows), 40 epochs, with "
  "a Random Forest baseline on engineered features. A leakage assertion aborts training if any person appears in two splits.")

H("VI. Results")
TABLE("Table III. Results on unseen data", ["Task", "Test data", "Result"], [(a, b, c) for a, b, c, _ in C.RESULTS])
FIG(A / "results_leakage.png", "Fig. 2. Identity leakage: random split vs unseen people.")
FIG(A / "results_temporal.png", "Fig. 3. Drowsiness over time on UTA-RLDD (unseen people).")
P("Per-frame tasks are near the state of the art on unseen people (eye state 98.0%, behaviour 92.1%). Random splits overstate "
  "accuracy: 99.6% vs 92.1% on State Farm (17x more errors on new drivers, 41x with EfficientNet-B2) and 94-97% vs 61-67% on "
  "NTHU-DDD. For drowsiness over time, alert-versus-drowsy reaches 87.3% (Random Forest) and 80.5% (hybrid network) of videos of "
  "unseen people; three levels reach 64.7%, on par with the published baseline [3]. Removing per-driver calibration drops the hybrid "
  "model from 53.4% to 37.6%, close to chance. Cabin sounds are detected with 95% (snoring), 93% (siren) and 78% (crash) recall "
  "with no false shouting alarms. On a side-mounted clip the face detector found 0/77 frames and the pose fallback 77/77; on a "
  "night wide-angle dash cam the zoom search raised face detection from 0% to 71%. The full pipeline runs in 12-16 ms per frame.")
FIG(A / "live_alerts.png", "Fig. 4. Live interface: AI view, risk gauge, driver state and an alert with one-tap feedback.")

H("VII. Discussion and Limitations")
P("Learned per-frame drowsiness models trained on few people transfer poorly to new drivers and cameras; treating them as "
  "supporting evidence keeps their benefit while limiting false alarms. Limitations: ordinary cameras cannot see the eyes at "
  "night (infrared is needed), dark sunglasses hide the eyes, audio cannot attribute snoring to a person, and the false-alarm rate "
  "on real drives must still be measured; the feedback pipeline is designed to provide exactly that measurement.")

H("VIII. Conclusion")
P("We built and evaluated a real-time, privacy-preserving DMS for ordinary cameras that adapts to each driver and to camera "
  "placement, explains and limits its alerts, and improves from consented driver feedback. Subject-independent evaluation shows "
  "strong per-frame accuracy and 80-87% for drowsiness over time, and quantifies identity leakage and the value of per-driver "
  "calibration. Future work: infrared night operation, a native mobile app, vehicle data and on-road false-alarm measurement.")

H("References")
for i, r in enumerate(C.REFERENCES, 1):
    P(f"[{i}] {r}", size=8, align="l", after=1)

doc.save(OUT)
print("saved", OUT)
