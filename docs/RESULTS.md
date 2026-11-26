# Results so far

All numbers are on data the models never saw during training. Eye-state and behaviour results are also on **people never seen in training** (subject-independent splits).

## 1. Image models (Kaggle run, T4 GPU, "fast" preset: 8 epochs)

| Model | Task | Test data | Accuracy | Macro-F1 | Calibration error (ECE) |
|---|---|---|---|---|---|
| MobileNetV3-L, multi-task | Eye open/closed | MRL Eye, unseen people (12,360 images) | **97.9%** | 0.974 | 0.051 |
| same network | Emotion (7 classes) | FER2013 official test (7,178 images) | **69.2%** | 0.683 | 0.072 |
| EfficientNet-B0 | Driver behaviour (10 classes) | State Farm, **unseen drivers** (3,440) | **92.1%** | 0.914 | 0.154 |
| same, **random split** | Driver behaviour | State Farm, same drivers as training (3,345) | 99.6% | 0.995 | 0.090 |

**Leakage finding:** the same model makes **~17× more errors** on new drivers (7.9%) than when its test set contains the training drivers (0.45%). Papers reporting ~99% on random splits overstate real-world performance.

**Speed** (ONNX, CPU, 2 threads): face network 4.6 ms, cabin network 9.6 ms per image; whole live pipeline ~25–30 FPS on a 4-core CPU.

These runs predate temperature scaling; the next run reports calibrated ECE next to the uncalibrated value.

## 2. Cabin audio alerts (YAMNet + our alert rules), ESC-50

560 clips: all 40 clips of each target sound plus 8 clips of each of the other 45 everyday sound classes (breathing, coughing, laughing, engine, train, rain, wind, fireworks, alarm clocks, …). A clip "alerts" with the live rule (score ≥ threshold in ≥ 2 half-second windows; 1 window for crashes). Reproduce: `python -m dms.tools.evaluate_audio --csv ESC-50/meta/esc50.csv --audio ESC-50/audio`. Raw numbers: [results/audio_eval_esc50.json](results/audio_eval_esc50.json).

| Alert | Caught (recall) at 0.5 | False alarms on other sounds | Notes |
|---|---|---|---|
| Snoring | **95%** | 0.2% | |
| Siren | **93%** | 0.4% | the only confusions were car horns |
| Baby crying | **82%** | 0.2% | |
| Crash (glass breaking) | 78% (90% at 0.3) | 3.8% | confusions: fireworks, thunderstorm. So a sound alone only starts a *silent* check; the spoken warning needs a camera jolt or a very clear sound (≥ 0.7) |
| Shouting | (no shouting class in ESC-50) | **0.0%** | was 6.6% before the fix: YAMNet's "crying" class fired on babies, so it was removed from the shouting group |
| Phone ringing | – | 0.9% | all from alarm clocks |
| Horn | 40% | 0% | used as context only, never alerts |

## 3. Alert timing on scripted clips

| Event | Alert | Measured delay |
|---|---|---|
| Eyes closed | microsleep | 1.0 s (Normal sensitivity) |
| Looking away | eyes off road | 3.0 s (Euro NCAP long-distraction rule) |
| Camera jolt + driver unresponsive | possible accident → SOS | 3 s + 15 s countdown |

## 4. Next results (pending)
* Image models with **AffectNet + RAF-DB + Yawn-Eye** added and the "best" preset (notebook `PRESET="best"`).
* KMU-FED (in-car emotions) as a held-out test.
* Drowsiness-over-time model on UTA-RLDD (5-fold, subject-independent).
* False alarms per hour on real drives (`dms.tools.evaluate_alerts`).
