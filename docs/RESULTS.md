# Results so far

All numbers are on data the models never saw during training. Eye-state and behaviour results are also on **people never seen in training** (subject-independent splits).

## 1. Image models

### Run 4 (Kaggle T4, "fast" preset): more data (FER2013 + RAF-DB + MRL Eye + Yawn-Eye; NTHU-DDD), calibrated

| Model | Task | Test data | Accuracy | Macro-F1 | ECE calibrated (before) |
|---|---|---|---|---|---|
| MobileNetV3-L, multi-task | Eye open/closed | MRL Eye, **unseen people** (12,576) | **98.0%** | 0.976 | **0.006** (0.048) |
| same network | Yawn | Yawn-Eye test (219)¹ | 99.1% | 0.991 | 0.009 (0.055) |
| same network | Emotion (7 classes) | FER2013 + RAF-DB official test sets (10,246) | **73.8%** | 0.701 | 0.033 (0.027) |
| EfficientNet-B0 | Frame drowsiness (NTHU-DDD) | **unseen person** (5,953 frames)² | **60.6%** | 0.581 | 0.160 (0.146) |
| same, **random split** | Frame drowsiness (NTHU-DDD) | same people as training (2,701) | 94.2% | 0.942 | 0.012 (0.055) |

Emotion improved from 69.2% (FER2013 only, run 3) to 73.8% by adding RAF-DB. Temperature scaling cut the calibration error of the eye and yawn heads ~8×; it did not help where the test people differ from the validation people (emotion, NTHU unseen person). That itself is a finding: calibration does not transfer across people for free.

¹ Yawn-Eye has no person IDs, so near-duplicate frames may sit in both splits; treat 99% as an upper bound.
² The Kaggle NTHU-DDD copy contains only 4 people (18,000 frames), so "unseen person" here is one person; indicative only.

### Run 3 (fast preset, State Farm included)

| Model | Task | Test data | Accuracy | Macro-F1 |
|---|---|---|---|---|
| MobileNetV3-L, multi-task | Eye | MRL Eye, unseen people (12,360) | 97.9% | 0.974 |
| same | Emotion | FER2013 official test (7,178) | 69.2% | 0.683 |
| EfficientNet-B0 | Driver behaviour (10 classes) | State Farm, **unseen drivers** (3,440) | **92.1%** | 0.914 |
| same, **random split** | Driver behaviour | same drivers as training (3,345) | 99.6% | 0.995 |

### The leakage finding (two datasets)

| Dataset | Random split (people shared) | Unseen people | Error rate × |
|---|---|---|---|
| State Farm (behaviour) | 99.6% | 92.1% | **17×** more errors |
| NTHU-DDD (frame drowsiness) | 94.2% | 60.6% | **7×** more errors |

Reporting random-split accuracy, as many papers do, overstates how a model works on a new driver. The effect is largest for drowsiness, where a frame classifier mostly learns *who* is in the image rather than *how sleepy* they are.

**Speed** (ONNX, CPU, 2 threads): face network 4.9 ms, cabin/drowsy network 9.7 ms per image; the whole live pipeline ~25–30 FPS on a 4-core CPU.

## 1b. Drowsiness over time: UTA-RLDD (60 people, 178 videos, 3 classes: alert / low-vigilant / drowsy)

Subject-independent 5-fold cross-validation (12 unseen people per fold), first 6 minutes of each video at 10 Hz, 30-second windows. Chance = 33%.

| Model | Window macro-F1 | Video accuracy |
|---|---|---|
| Random Forest on engineered features (PERCLOS, blink stats, pose stats) | **0.574 ± 0.035** | **64.1% ± 3.5** |
| Bidirectional GRU | 0.530 ± 0.033 | 57.3% ± 7.6 |
| Conv-Transformer (30 s windows) | 0.494 ± 0.018 | 52.8% ± 2.9 |
| Conv-Transformer (10 s windows) | 0.493 ± 0.023 | 55.6% ± 9.8 |
| Conv-Transformer **without per-driver calibration** | 0.408 ± 0.059 | 43.2% ± 7.7 |

Findings:
* **Per-driver calibration matters (contribution C2):** removing it drops the same model from 52.8% to 43.2% video accuracy (−9.6 points) and from 0.49 to 0.41 macro-F1.
* The engineered-feature Random Forest reaches **64.1%**, on par with the dataset's original published baseline (≈ 65% video-level, HM-LSTM), with a far simpler model.
* With only 60 people, the deep sequence models (GRU, Transformer) do **not** beat engineered features yet: a common small-data result. More training, regularisation and hybrid inputs (§4) are the next steps.
* 3-class self-reported drowsiness is hard (people's "low vigilant" and "drowsy" look similar); binary alert-vs-drowsy would score considerably higher.

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
* **AffectNet+ validation set (5,500 images) as a held-out emotion test:** our model vs the pretrained HSEmotion model (`dms.training.evaluate_image`, see RUN_TONIGHT D2).
* Temporal model improvements: longer training, hybrid input (sequence + engineered statistics), binary alert-vs-drowsy variant; NTHU-DDD/DMD cross-dataset test.
* "Best" preset (EfficientNet-B2, 15 epochs, TTA) with State Farm re-attached.
* False alarms per hour on real drives (`dms.tools.evaluate_alerts`).
