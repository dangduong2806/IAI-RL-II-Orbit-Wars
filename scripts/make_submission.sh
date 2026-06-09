#!/usr/bin/env bash
set -e
SUB_DIR=${1:-$(pwd)}
OUT_ZIP=${2:-/kaggle/working/submission.zip}
cd "$SUB_DIR"
rm -f "$OUT_ZIP"
zip -r "$OUT_ZIP" main.py src checkpoints submission \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "*.ipynb_checkpoints*"
unzip -l "$OUT_ZIP" | head -40
