#!/usr/bin/env bash
set -euo pipefail

source /home/silass/.pyenv/versions/pytorch/bin/activate

for s in 0 15 30 45 60; do
  e=$((s + 15))
  echo "===== Window ${s}-${e}s ====="
  output=$(python /home/silass/Code/ASL-Project/infer_video_encdec.py \
    --video /home/silass/Code/how-to-sign/raw_videos/1EYhtLj97b8-5-rgb_front.mp4 \
    --checkpoint /home/silass/Code/ASL-Project/checkpoints_clean_v2/best_model.pt \
    --data /home/silass/Code/ASL-Project/how2sign_mediapipe_clip_1000vocab.h5 \
    --visual-backbone clip \
    --decode beam \
    --beam-size 5 \
    --length-penalty 1.2 \
    --start-sec "$s" \
    --end-sec "$e" \
    --chunk-size 512 \
    --chunk-overlap 0.0 \
    --max-glosses 24 2>&1 || true)

  matches=$(printf "%s\n" "$output" | grep -E "^Predicted glosses:|^Number of glosses:" || true)
  if [[ -n "$matches" ]]; then
    printf "%s\n" "$matches"
  else
    echo "No prediction lines matched filter for this window."
    echo "Last output lines:"
    printf "%s\n" "$output" | tail -n 20
  fi
  echo
done
