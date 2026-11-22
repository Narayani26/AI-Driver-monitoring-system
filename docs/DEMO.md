# Demo script (~15 min presentation + live demo)

This follows the flow you planned: **explain → open the app → live webcam → uploaded footage → trip report → product vision**.

## Before you present (the day before)

- [ ] `python -m dms.web.server` runs on the Mac; open `http://localhost:8000` and allow the camera.
- [ ] Copy the trained checkpoints into `runs/` and start with them, e.g. `python -m dms.web.server --temporal-ckpt runs/temporal/deploy.pt --face-ckpt runs/face/best.pt`. The "How it works" tab shows their accuracy automatically.
- [ ] Record **2–3 demo clips** of yourself (or a friend) acting out the scenarios below, ~60 s each. Use QuickTime → New Movie Recording.
- [ ] Render a **backup side-by-side video** of each clip, so you have something to show if anything fails live:
  `python -m dms.app.run --source clip1.mov --out clip1_demo.mp4 --side-by-side --headless --report clip1_report.json`
- [ ] Turn system volume up (voice + buzzer are part of the demo). Turn off notifications and Wi-Fi-dependent things; the app works fully offline.
- [ ] Lighting: light from the front, no bright window behind you.

## 1. Explain (slides, ~7 min)

| Slide | Say |
|---|---|
| Problem | Drowsy/distracted driving statistics; Euro NCAP 2026 made driver monitoring worth 25 points (from ~2) |
| What exists | Threshold-based GitHub projects; papers with 98–99% that test on the same people they trained on |
| Our system | The pipeline diagram (also on the "How it works" tab) |
| Models | MediaPipe landmarks → indicators; Conv-Transformer (temporal drowsiness); EfficientNet emotion; YOLO11 phone/objects; risk fusion |
| Datasets | MRL Eye, UTA-RLDD, AffectNet/FER → KMU-FED, State Farm/100-Driver, Roboflow driver sets; own recordings |
| Training | Subject-independent splits, 5-fold CV, Mac (MPS) + Kaggle GPUs |
| Results | Table: random split vs subject split (the leakage finding), drowsiness F1, emotion on in-car data, speed (ms/FPS) |
| What's different | The 5 contributions (one risk score with reasons; Euro NCAP rules; per-driver calibration; tiny privacy-friendly temporal model; honest evaluation) |
| What can improve | More in-car data, IR camera for night, fine-tuned YOLO, mobile app |

## 2. Open the app (~30 s)

Show the **How it works** tab: models currently loaded, results table, datasets. Then switch to **Live monitor**.

## 3. Live webcam demo (~3 min)

Press **Start**. Narrate each step while the right-hand panel reacts:

1. **Calibration (5 s)**: "It first learns *my* normal eye shape and head position, so it works for any driver and any camera angle." Sit normally and look at the screen centre (that's "the road").
2. **Normal driving (10 s)**: risk stays green, blinks counted, emotion neutral.
3. **Microsleep**: close your eyes for 2 s → red CRITICAL, alarm + voice: *"Wake up! Your eyes were closed… pull over safely."*
4. **Drowsiness after-effect**: point out that risk *stays elevated* and the system says *"You seem drowsy…"*. A microsleep is strong evidence of fatigue, so the system doesn't forget it the moment your eyes reopen.
5. **Yawn 3 times** (hold each ~2 s) → yawn counter, *"Frequent yawning detected."*
6. **Look away** (turn to the side) for 4 s → *"Please look ahead at the road."* Mention Euro NCAP's 3-second long-distraction rule.
7. **Phone to your ear / texting at chest height** → red YOLO box, *"Please put the phone down."*
8. **Angry face for ~6 s** → emotion bars shift to anger, *"You seem stressed…"*
9. Press **Stop** → the **trip report** opens: safety score, alerts by type, time per risk level, risk curve. Show **Download JSON/CSV**.

## 4. Pre-recorded footage (~2 min)

Two options; use either or both:

- **Upload (most reliable):** *Video file* → pick a recorded clip → it plays on the left while detection runs on the right, synchronised to the video's own clock. Seeking backwards resets cleanly.
- **Phone in front of the webcam (what you suggested):** play the clip on your phone and hold it up to the laptop camera. Tips: max brightness, hold it close so the face fills a good part of the frame, tilt slightly to avoid glare. The system recognises that a phone *showing* a face is a screen, not phone use (that YOLO box is drawn dashed as "ignored"), so it won't trigger a false phone alert.

## 5. Product vision (~1 min)

Switch to **Dash cam / IP stream** and explain: a driver-facing IR dash cam streams over its own Wi-Fi to the phone; the phone (Android foreground service) runs the same models offline, even when locked, and speaks the same alerts. If you have a spare Android phone running an IP-camera app, enter its URL and show it live. See `docs/DEPLOYMENT.md`.

## If something goes wrong

| Problem | Fix |
|---|---|
| Camera blocked / black | Chrome → site settings → allow camera; or use the upload path |
| "connecting…" never turns green | The server isn't running; check the terminal |
| Alerts too sensitive for your face/lighting | Press **Recalibrate** while looking straight ahead |
| No voice | Check the Voice toggle and system volume; Safari needs one click on the page first |
| Anything else | Play the backup `*_demo.mp4` side-by-side video |
