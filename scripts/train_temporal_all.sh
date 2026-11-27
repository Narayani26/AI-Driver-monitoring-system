#!/usr/bin/env bash
# Train and compare the drowsiness-over-time models on already-extracted UTA-RLDD features.
# Runs on a laptop CPU (no GPU needed) because the slow video -> feature step is already done on Kaggle.
#
#   bash scripts/train_temporal_all.sh                 # features in features/rldd (from dms_temporal.zip)
#   EPOCHS=20 bash scripts/train_temporal_all.sh       # quicker
#
# Results: runs_temporal2/<experiment>/metrics.json + a summary table at the end.
set -euo pipefail
cd "$(dirname "$0")/.."
FEAT=${FEAT:-features/rldd}
OUT=${OUT:-runs_temporal2}
EPOCHS=${EPOCHS:-40}
[ -n "$(ls "$FEAT"/*.npz 2>/dev/null)" ] || { echo "no features in $FEAT - unzip dms_temporal.zip into the project first"; exit 1; }
export PYTHONUNBUFFERED=1  # show progress live (otherwise output appears only when each model finishes)
COMMON="--folds 5 --epochs $EPOCHS --steps-per-epoch 100 --drop 0.3"
run() { echo; echo "=== $1"; shift; python -m dms.training.train_temporal "$FEAT" "$@" 2>&1 | grep --line-buffered -E "fold [0-9]|window macroF1|epoch [0-9]+0/" || true; }

# 3 classes: alert / low-vigilant / drowsy
run "RF (3-class)"                         --model rf --folds 5 --out $OUT/rf
run "Hybrid (3-class)"                     --model hybrid $COMMON --fit-all --out $OUT/hybrid
run "Hybrid without calibration (3-class)" --model hybrid $COMMON --no-calibration --out $OUT/hybrid_nocalib
run "Transformer, longer (3-class)"        --model transformer $COMMON --out $OUT/transformer
# 2 classes: alert vs drowsy (low-vigilant left out) - the common setting in the literature
run "RF (binary)"                          --model rf --folds 5 --binary --out $OUT/rf_binary
run "Hybrid (binary)"                      --model hybrid $COMMON --binary --fit-all --out $OUT/hybrid_binary
run "Transformer (binary)"                 --model transformer $COMMON --binary --out $OUT/transformer_binary

python - "$OUT" <<'PY'
import glob, json, sys
print(f"\n{'experiment':28s} {'window macro-F1':>18s} {'video accuracy':>18s}")
for p in sorted(glob.glob(f"{sys.argv[1]}/*/metrics.json")):
    s = json.load(open(p))["summary"]
    f, v = s["window_macro_f1"], s["video_accuracy"]
    print(f"{p.split('/')[-2]:28s} {f['mean']:.3f} ± {f['std']:.3f}    {v['mean']:.3f} ± {v['std']:.3f}")
PY
