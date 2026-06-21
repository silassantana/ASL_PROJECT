#!/bin/bash
#
# Lightning.ai training script for ASL Transformer
# 
# Usage:
#   1. Upload this repo + checkpoints to Lightning workspace
#   2. Run: bash lightning_train.sh
#
# The script will:
#   - Set up the environment
#   - Download H5 data from HuggingFace if not present
#   - Resume from best checkpoint if available
#   - Run training with warm-restart recipe
#

set -e

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR="${DATA_DIR:-.}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-.}"
HF_REPO="silassantana/asl-how2sign-features"
H5_FILENAME="how2sign_mediapipe_clip_1000vocab.h5"

# ============================================================================
# Setup
# ============================================================================

echo "=========================================="
echo "Lightning.ai ASL Transformer Training"
echo "=========================================="
echo ""

# Create directories
mkdir -p "${DATA_DIR}"
mkdir -p "${CHECKPOINT_DIR}"

# ============================================================================
# Download H5 if not present
# ============================================================================

H5_PATH="${DATA_DIR}/${H5_FILENAME}"
if [ ! -f "$H5_PATH" ]; then
    echo "[1/4] Downloading H5 dataset from HuggingFace..."
    python3 << 'EOF'
import os
from huggingface_hub import hf_hub_download

data_dir = os.environ.get("DATA_DIR", ".")
hf_repo = os.environ.get("HF_REPO", "silassantana/asl-how2sign-features")
h5_filename = os.environ.get("H5_FILENAME", "how2sign_mediapipe_clip_1000vocab.h5")

print(f"Downloading {h5_filename} from {hf_repo}...")
hf_hub_download(
    repo_id=hf_repo,
    filename=h5_filename,
    repo_type="dataset",
    local_dir=data_dir,
)
print("Download complete.")
EOF
else
    echo "[1/4] H5 dataset already present at $H5_PATH"
fi

# ============================================================================
# Prepare checkpoint for resume
# ============================================================================

echo ""
echo "[2/4] Preparing checkpoint for resume..."

BEST_CKPT="${CHECKPOINT_DIR}/best_model.pt"
LATEST_CKPT="${CHECKPOINT_DIR}/latest_checkpoint.pt"

if [ -f "$BEST_CKPT" ]; then
    echo "Found best_model.pt, using as resume point..."
    cp "$BEST_CKPT" "$LATEST_CKPT"
    echo "Seeded latest_checkpoint.pt from best_model.pt"
elif [ -f "$LATEST_CKPT" ]; then
    echo "Found latest_checkpoint.pt, will resume from there"
else
    echo "No checkpoint found, starting from scratch"
fi

# ============================================================================
# Run training
# ============================================================================

echo ""
echo "[3/4] Starting training..."
echo ""

python3 train_transformer_encdec.py \
    --data "$H5_PATH" \
    --save-dir "$CHECKPOINT_DIR" \
    --hidden-dim 512 \
    --num-encoder-layers 3 \
    --num-decoder-layers 3 \
    --batch-size 16 \
    --eval-batch-size 32 \
    --grad-accum-steps 2 \
    --lr 5e-5 \
    --lr-restart-every 8 \
    --num-workers 8 \
    --use-channel-attention \
    --use-class-weights \
    --class-weight-power 0.5 \
    --label-smoothing 0.1 \
    --seed 42 \
    --eval-beam-size 2 \
    --eval-length-penalty 0.7 \
    --report-probe-samples 64 \
    --report-beam-size 4 \
    --report-length-penalty 0.7 \
    --early-stop-patience 20 \
    --early-stop-min-delta 0.005 \
    --resume

echo ""
echo "[4/4] Training complete!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
