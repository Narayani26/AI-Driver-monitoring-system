"""Presentation (PPTX; convert to PDF with LibreOffice)."""
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

import submission_content as C

A = Path("docs/report_assets")
OUT = Path("docs/submission/DMS_Presentation.pptx")
NAVY, ACC, INK, MUT, LIGHT = RGBColor(0x0F, 0x1E, 0x46), RGBColor(0x25, 0x63, 0xEB), RGBColor(0x0F, 0x17, 0x2A), RGBColor(0x47, 0x55, 0x69), RGBColor(0xF1, 0xF5, 0xF9)
prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
W, Hh = prs.slide_width, prs.slide_height
BLANK = prs.slide_layouts[6]
num = [0]


def box(s, x, y, w, h, fill=None, line=None):
    sh = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    sh.adjustments[0] = 0.08
    if fill:
        sh.fill.solid(); sh.fill.fore_color.rgb = fill
    else:
        sh.fill.background()
    if line:
        sh.line.color.rgb = line; sh.line.width = Pt(1.25)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def text(s, x, y, w, h, t, size=18, bold=False, color=INK, align=PP_ALIGN.LEFT):
    tb = s.shapes.add_textbox(x, y, w, h); tf = tb.text_frame; tf.word_wrap = True
    lines = t if isinstance(t, list) else [t]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if isinstance(ln, tuple):
            r = p.add_run(); r.text = ln[0]; r.font.bold = True; r.font.size = Pt(size); r.font.color.rgb = ACC
            r = p.add_run(); r.text = "  " + ln[1]; r.font.size = Pt(size); r.font.color.rgb = color
        else:
            r = p.add_run(); r.text = ln; r.font.size = Pt(size); r.font.bold = bold; r.font.color.rgb = color
        p.space_after = Pt(6)
    return tb


def slide(title, kicker=None):
    s = prs.slides.add_slide(BLANK); num[0] += 1
    s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.18), Hh).fill.solid()
    s.shapes[-1].fill.fore_color.rgb = ACC; s.shapes[-1].line.fill.background()
    if kicker:
        text(s, Inches(0.6), Inches(0.35), Inches(11), Inches(0.4), kicker.upper(), 12, True, ACC)
    text(s, Inches(0.6), Inches(0.62), Inches(12.2), Inches(0.9), title, 30, True, INK)
    text(s, Inches(11.6), Inches(7.0), Inches(1.5), Inches(0.35), str(num[0]), 11, color=MUT, align=PP_ALIGN.RIGHT)
    text(s, Inches(0.6), Inches(7.0), Inches(9), Inches(0.35), "AI-Based Driver Monitoring System · KIT's COEK · " + C.YEAR, 10, color=MUT)
    return s


def pic(s, path, x, y, maxw, maxh):
    im = Image.open(path); r = im.width / im.height
    w, h = maxw, int(maxw / r)
    if h > maxh:
        h, w = maxh, int(maxh * r)
    s.shapes.add_picture(str(path), x + (maxw - w) // 2, y, w, h)


def cards(s, items, y=Inches(1.8), h=Inches(2.3), cols=None, size=14):
    n = len(items); cols = cols or n
    gap = Inches(0.25); w = (W - Inches(1.2) - gap * (cols - 1)) // cols
    for i, (t, d) in enumerate(items):
        r, c = divmod(i, cols)
        x, yy = Inches(0.6) + c * (w + gap), y + r * (h + gap)
        box(s, x, yy, w, h, fill=LIGHT)
        text(s, x + Inches(0.2), yy + Inches(0.15), w - Inches(0.4), Inches(0.6), t, size + 3, True, NAVY)
        text(s, x + Inches(0.2), yy + Inches(0.75), w - Inches(0.4), h - Inches(0.85), d, size, color=MUT)


# 1 title
s = prs.slides.add_slide(BLANK); num[0] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Hh); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, Inches(0.8), Inches(0.7), Inches(12), Inches(0.5), "HONORS MINI PROJECT (UCSHN0751) · B.TECH CSE · " + C.YEAR, 14, True, RGBColor(0x93, 0xC5, 0xFD))
text(s, Inches(0.8), Inches(1.5), Inches(11.5), Inches(2.4), "AI-Based Driver Monitoring System", 50, True, RGBColor(255, 255, 255))
text(s, Inches(0.8), Inches(3.0), Inches(11.5), Inches(1.2), "Real-time detection of drowsiness, distraction and unsafe behaviour from any camera, offline, private, and learning from drivers", 22, color=RGBColor(0xCB, 0xD5, 0xE1))
text(s, Inches(0.8), Inches(4.9), Inches(8), Inches(1.4), [f"{n} ({r})" for n, r, _ in C.STUDENTS], 18, True, RGBColor(255, 255, 255))
text(s, Inches(0.8), Inches(6.1), Inches(12), Inches(0.9), [f"Guide: {C.GUIDE}", "KIT's College of Engineering Kolhapur (Empowered Autonomous) · Dept. of CSE"], 14, color=RGBColor(0xCB, 0xD5, 0xE1))

# 2 problem
s = slide("Drowsy and distracted driving kills, and the driver notices last", "Problem")
cards(s, [("1.19 million", "road deaths every year worldwide (WHO 2023); fatigue and distraction are leading human causes"),
          ("44 m blind", "a 2-second microsleep at 80 km/h; a glance at a phone is the same"),
          ("No roadside test", "for fatigue; drivers underestimate their own sleepiness"),
          ("Built-in DMS only in new cars", "with infrared cameras; millions of cars, phones and dash cams have none")], cols=4, h=Inches(3.4), size=15)
text(s, Inches(0.6), Inches(5.6), Inches(12), Inches(1), "Goal: a monitoring system that works with the camera people already have, warns early, and does not nag.", 20, True, NAVY)

# 3 objectives
s = slide("Objectives", "What we set out to build")
cards(s, [("Detect", "microsleep, PERCLOS, yawns, head nodding, eyes off road, phone, drinking, smoking, cabin sounds, possible accidents"),
          ("Adapt", "to each driver and to any camera position: front, diagonal, side, moved mid-drive, wide-angle and dark"),
          ("Decide", "one explainable 0-100 risk score; proportionate voice and buzzer alerts with suggestions; few false alarms"),
          ("Improve", "secure cloud with accounts, trips and feedback; retrain from consented, quality-checked real data")], cols=2, h=Inches(2.35), size=15)

# 4 gap
s = slide("Existing systems: what is missing", "Literature and gap")
cards(s, [("Geometric thresholds (EAR/PERCLOS)", "Light and explainable, but fixed thresholds fail across people and head poses"),
          ("Single-image CNNs", "Report 95-99%, but often with the same people in train and test: optimistic on a new driver"),
          ("Commercial infrared DMS", "Robust, but factory-fitted at a fixed position; not for existing cars, phones or dash cams"),
          ("Gap we address", "honest testing on unseen people, any camera, fewer false alarms, learning from real drivers without video")], cols=2, h=Inches(2.35), size=15)

# 5 architecture
s = slide("System architecture", "Design")
pic(s, A / "architecture.png", Inches(0.5), Inches(1.6), Inches(12.3), Inches(5.3))

# 6 tech stack
s = slide("Technology stack", "Built with")
cards(s, [(a, b) for a, b in C.TECH], cols=3, h=Inches(2.45), size=13)

# 7 datasets
s = slide("Datasets: 9 public datasets, always split by person", "Data")
rows = C.DATASETS
tbl = s.shapes.add_table(len(rows) + 1, 4, Inches(0.6), Inches(1.6), Inches(12.1), Inches(4.8)).table
for i, h in enumerate(("Dataset", "Size used", "Task", "Split")):
    tbl.cell(0, i).text = h
for r, row in enumerate(rows, 1):
    for c, v in enumerate(row):
        tbl.cell(r, c).text = v
for r in range(len(rows) + 1):
    for c in range(4):
        for p in tbl.cell(r, c).text_frame.paragraphs:
            for run in p.runs:
                run.font.size = Pt(13)

# 8 models
s = slide("Deep-learning models: 9 networks, 4 trained by us", "Models")
tbl = s.shapes.add_table(len(C.MODELS) + 1, 3, Inches(0.6), Inches(1.6), Inches(12.1), Inches(5)).table
for i, h in enumerate(("Model", "Architecture / source", "What it does live")):
    tbl.cell(0, i).text = h
for r, (m, a, src, role) in enumerate(C.MODELS, 1):
    tbl.cell(r, 0).text, tbl.cell(r, 1).text, tbl.cell(r, 2).text = m, f"{a} · {src}", role
for r in range(len(C.MODELS) + 1):
    for c in range(3):
        for p in tbl.cell(r, c).text_frame.paragraphs:
            for run in p.runs:
                run.font.size = Pt(12.5)

# 9 how it works
s = slide("How a frame becomes an alert", "Method")
cards(s, [("1 · Calibrate", "first seconds learn YOUR open eye and neutral head pose, from whatever angle the camera has"),
          ("2 · Measure", "EAR, mouth, head pose, gaze, blendshapes every frame; CNN eye state; YOLO phone; pose keypoints for side cameras"),
          ("3 · Track over time", "blinks, PERCLOS (60 s), microsleep >= 1 s, yawns, head nods, Euro NCAP glance rules (3 s, 10 s in 30 s)"),
          ("4 · Fuse", "risk = 100·(1-(1-d)(1-x))·(1+0.3a); learned models = supporting evidence only; hysteresis; reasons"),
          ("5 · Act", "voice + buzzer + suggestion; cooldowns; 'Not true' button; SOS countdown after a possible crash"),
          ("6 · Learn", "ratings + sleepiness score -> quality gate -> retrain -> deploy only if better")], cols=3, h=Inches(2.45), size=13)

# 10 demo screenshots
s = slide("Live system", "Working demonstration")
pic(s, A / "live_monitor.png", Inches(0.5), Inches(1.55), Inches(6.1), Inches(5.3))
pic(s, A / "live_alerts.png", Inches(6.8), Inches(1.55), Inches(6.1), Inches(5.3))

# 11 trip report
s = slide("Trip report without any video", "Output")
pic(s, A / "trip_report.png", Inches(0.5), Inches(1.5), Inches(7.6), Inches(5.4))
text(s, Inches(8.4), Inches(1.8), Inches(4.5), Inches(5), ["Safety score, peak risk, PERCLOS, longest eye closure and look-away",
     "Time in each risk level and alerts by type", "Driver's sleepiness rating (Karolinska scale)",
     "Download as JSON / CSV / PDF: a few KB, no footage, syncs on weak networks"], 16, color=MUT)

# 12 cloud
s = slide("Cloud back end and retraining pipeline", "Production")
pic(s, A / "cloud_dashboard.png", Inches(0.5), Inches(1.5), Inches(5.2), Inches(5.4))
text(s, Inches(6.0), Inches(1.6), Inches(6.9), Inches(5.4), [
    ("Stack", "FastAPI + SQLAlchemy + PostgreSQL on Supabase"),
    ("Data", "users, devices, trips, alerts + feedback, samples, model registry"),
    ("Offline-first", "car stores locally, syncs with a device key"),
    ("Learning loop", "driver ratings + consent -> quality gate -> split by driver -> train -> compare on unseen drivers -> approve only if not worse and no extra false alarms -> devices verify SHA-256"),
    ("Admin", "data quality, false alarms per alert type, roll back")], 16, color=INK)

# 13 results
s = slide("Results on people the models never saw", "Evaluation")
pic(s, A / "results_accuracy.png", Inches(0.5), Inches(1.5), Inches(8.2), Inches(5.4))
text(s, Inches(8.9), Inches(1.7), Inches(4.1), Inches(5.3), ["Eye state 98.0%", "Behaviour 92.1% (new drivers)", "Alert vs drowsy 87.3% / 80.5%",
     "Emotion 75.4%", "Snoring 95% · siren 93%", "~30 FPS on a laptop CPU"], 19, True, NAVY)

# 14 research findings
s = slide("Research findings", "What is new")
pic(s, A / "results_leakage.png", Inches(0.5), Inches(1.5), Inches(6.1), Inches(4.2))
pic(s, A / "results_temporal.png", Inches(6.8), Inches(1.5), Inches(6.1), Inches(4.2))
text(s, Inches(0.6), Inches(5.9), Inches(12.2), Inches(1.1), ["Random splits hide most errors (17-41x on new drivers)  ·  per-driver calibration adds +15.8 points"], 18, True, NAVY)

# 15 contributions
s = slide("Our contributions", "Novelty")
cards(s, [(a, b) for a, b in C.CONTRIBUTIONS], cols=3, h=Inches(2.5), size=12.5)

# 16 real-world
s = slide("Real driving situations", "Robustness")
cards(s, [("Camera moved", "detected in 0.75-1.5 s, position re-learned in ~3 s"),
          ("Side-mounted camera", "pose fallback: head found 77/77 frames, nod alarm"),
          ("Night wide-angle dash cam", "zoom search: face 0% -> 71% of frames"),
          ("Head tilted back", "eyes not judged: no false microsleep"),
          ("Passenger leans in", "ignored, driver tracked"),
          ("Radio crash sound", "calm notice; jolt + no response -> SOS")], cols=3, h=Inches(2.35), size=14)

# 17 security
s = slide("Security and privacy by design", "Trust")
cards(s, [("No footage leaves the device", "only numbers and alert metadata; consent for training data"),
          ("Strong accounts", "scrypt hashing, signed tokens, login rate limit"),
          ("Device keys", "per-device, hashed, revocable; accounts isolated (tested)"),
          ("Database", "Supabase PostgreSQL, Row Level Security, TLS"),
          ("Integrity", "SHA-256 model checks; verified model updates"),
          ("Engineering", "60 automated tests, CI on every push, Docker")], cols=3, h=Inches(2.35), size=14)

# 18 limits & future
s = slide("Limitations and future scope", "Next")
cards(s, [("Limitations", "night needs an infrared camera; dark sunglasses hide the eyes; audio cannot tell who snores; on-road false-alarm rate still to be measured"),
          ("Future scope", "infrared support; native Android app when the phone is locked; OBD-II speed; INT8 for dash-cam chips; larger real-drive dataset via the feedback loop")], cols=2, h=Inches(3.2), size=17)

# 19 conclusion
s = prs.slides.add_slide(BLANK); num[0] += 1
bg = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Hh); bg.fill.solid(); bg.fill.fore_color.rgb = NAVY; bg.line.fill.background()
text(s, Inches(0.8), Inches(1.0), Inches(11.5), Inches(1), "Conclusion", 40, True, RGBColor(255, 255, 255))
text(s, Inches(0.8), Inches(2.2), Inches(11.5), Inches(3.5), [
    "A complete, real-time driver monitoring system for ordinary cameras: offline, private, explainable",
    "Honest results on unseen people: 98% eye state, 92% behaviour, 80-87% drowsiness over time",
    "Works with any camera position, learns from consented driver feedback, deploys only better models"], 22, color=RGBColor(0xE2, 0xE8, 0xF0))
text(s, Inches(0.8), Inches(5.6), Inches(11.5), Inches(1), "Thank you  ·  Questions?", 30, True, RGBColor(0x93, 0xC5, 0xFD))

prs.save(OUT)
print("saved", OUT, num[0], "slides")
