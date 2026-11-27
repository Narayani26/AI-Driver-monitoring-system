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

**Reproducibility check (run 5):** the whole Kaggle notebook was run again from a clean clone with the same seeds, now with State Farm and AffectNet+ attached. Every number above came out identical to the 4th decimal place (eye 0.9804, yawn 0.9909, emotion 0.7380, NTHU 0.6064), as did behaviour (92.1% unseen drivers / 99.6% random split), so the results are deterministic and reproducible.

### Emotion on a dataset never trained on: AffectNet+ (3,500 human-labelled validation images, 7 classes)

| Model | Trained on | Accuracy | Macro-F1 |
|---|---|---|---|
| Our multi-task network, run 3 | FER2013 | 34.4% | – |
| Our multi-task network, run 4/5 | FER2013 + RAF-DB | 46.9% | 0.443 |
| Pretrained HSEmotion EfficientNet-B0 | AffectNet (same source as the test) | **63.1%** | – |

Adding RAF-DB raised cross-dataset emotion accuracy by 12.5 points. Our model still trails a model trained on AffectNet itself, because the test images come from that dataset's distribution (an in-domain advantage). Emotion recognition from a single photo also does not transfer well between datasets, so the live demo uses HSEmotion for emotion (`--emotion pretrained`, the default), and emotion only modifies risk; it never triggers an alert by itself. Our 73.8% figure is on FER2013/RAF-DB test sets, 46.9% on a different dataset: both are reported.

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

### Round 5: longer training, hybrid model, binary task (same 5 folds, unseen people; reproduce: `bash scripts/train_temporal_all.sh`)

| Model | 3 classes: window macro-F1 | 3 classes: video accuracy | Alert vs drowsy: window macro-F1 | Alert vs drowsy: video accuracy |
|---|---|---|---|---|
| Random Forest on engineered features | **0.579 ± 0.020** | **64.7% ± 5.5** | **0.816 ± 0.028** | **87.3% ± 7.5** |
| Hybrid (Conv-Transformer + in-model statistics) | 0.479 ± 0.058 | 53.4% ± 7.5 | 0.758 ± 0.033 | 80.5% ± 5.8 |
| Conv-Transformer (40 epochs, dropout 0.3) | 0.491 ± 0.048 | 55.6% ± 7.2 | 0.746 ± 0.031 | 75.5% ± 4.5 |
| Hybrid **without per-driver calibration** | 0.379 ± 0.033 | 37.6% ± 5.4 | – | – |

Chance level: 33% for 3 classes, 50% for 2 classes.

* **Calibration effect confirmed on a second model:** removing per-driver calibration drops the hybrid from 53.4% to 37.6% video accuracy (**−15.8 points**, close to chance); for the transformer in the earlier round it was −9.6 points.
* **Binary alert-vs-drowsy** (the usual setting in the literature): RF 87.3%, hybrid 80.5% of unseen drivers' videos correctly classified.
* The RF stays best with 60 people. Hybrid inputs did not help the 3-class task, but they did help the binary task: 80.5% vs 75.5% for the plain transformer.
* **Deployed model:** the live app uses the hybrid binary network (`runs_temporal2/hybrid_binary/deploy.pt`, retrained on all 60 people), because it runs inside the same PyTorch/ONNX pipeline on every frame window. The RF is the offline benchmark.

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
* "Better"/"best" image presets (EfficientNet-B0/B2, 12–15 epochs, test-time augmentation): the 5th Kaggle run used the fast preset again.
* NTHU-DDD/DMD cross-dataset test of the temporal model.
* False alarms per hour on real drives (`dms.tools.evaluate_alerts`).
