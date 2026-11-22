#!/usr/bin/env bash
# Download the Kaggle-hosted datasets into data/ and print instructions for the rest.
#
#   bash scripts/get_datasets.sh            # everything available from Kaggle
#   bash scripts/get_datasets.sh fer mrl    # just some
#
# Needs the Kaggle CLI + API token (see docs/GUIDE.md §4.1):
#   pip install kaggle && mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
# State Farm is a competition: click "Join Competition" / accept the rules on its Kaggle page once first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

command -v kaggle >/dev/null || { echo "kaggle CLI missing: pip install kaggle"; exit 1; }

dl_dataset() {  # slug dest
  if [ -d "data/$2" ] && [ -n "$(ls -A "data/$2" 2>/dev/null)" ]; then echo "have data/$2"; return; fi
  echo "== $1 -> data/$2"
  kaggle datasets download -d "$1" -p "data/$2" --unzip
}

want() { [ $# -eq 0 ] || [[ " ${ARGS} " == *" $1 "* ]]; }
ARGS="$*"

if [ -z "$ARGS" ] || want statefarm; then
  if [ ! -f data/state-farm/driver_imgs_list.csv ]; then
    echo "== State Farm (competition, ~4 GB)"
    kaggle competitions download -c state-farm-distracted-driver-detection -p data/state-farm
    (cd data/state-farm && unzip -q -o state-farm-distracted-driver-detection.zip && rm state-farm-distracted-driver-detection.zip)
  else echo "have data/state-farm"; fi
fi
if [ -z "$ARGS" ] || want fer; then dl_dataset msambare/fer2013 fer2013; fi
if [ -z "$ARGS" ] || want mrl; then dl_dataset imadeddinedjerarda/mrl-eye-dataset mrl; fi

cat <<'EOF'

Downloaded what Kaggle hosts. Next, build manifests:
  python -m dms.data.manifests statefarm data/state-farm manifests/statefarm.csv
  python -m dms.data.manifests fer       data/fer2013    manifests/fer.csv
  python -m dms.data.manifests mrl       data/mrl        manifests/mrl.csv

Manual downloads (forms / large files) - see docs/DATASETS.md:
  * Yawn-Eye: search Kaggle for "yawn eye dataset" (folders yawn/no_yawn/Open/Closed), then
      kaggle datasets download -d <owner>/<slug> -p data/yawn_eye --unzip
  * UTA-RLDD (111 GB, 5 folds): https://sites.google.com/view/utarldd/home -> data/uta-rldd/
  * YawDD: IEEE DataPort "YawDD yawning detection dataset"
  * KMU-FED (driver emotions, test set), DMD (dmd.vicomtech.org)
  * Request forms: AffectNet, RAF-DB, NTHU-DDD, 100-Driver (100-driver.github.io)
EOF
