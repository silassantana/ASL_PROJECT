# Lightning.ai Migration Guide

## Overview

Two launchers are provided to continue training on Lightning.ai after stopping Modal:

1. **`lightning_train.sh`** — Bash script (recommended if available)
2. **`lightning_train.py`** — Pure Python script (cross-platform)

Both scripts handle dataset download, checkpoint resume, and training execution.

## Prerequisites

### Local: Download Checkpoints from Modal

Stop the Modal run and pull your checkpoints:

```bash
# Stop Modal (if running)
Ctrl+C

# Download checkpoints
modal run modal_train.py::download_checkpoints --remote-dir checkpoints_new_clean --local-dir checkpoints_new_clean_modal
```

### Local: Prepare Repository

```bash
# Copy all files to Lightning workspace (via UI or CLI)
# Ensure these files are uploaded:
#   - train_transformer_encdec.py
#   - sign_transformer.py
#   - lightning_train.sh or lightning_train.py
#   - checkpoints_new_clean_modal/

# Optional: create a requirements.txt in Lightning workspace
pip install torch torchvision torchaudio torch-vision --index-url https://download.pytorch.org/whl/cu121
pip install h5py numpy tqdm huggingface_hub[hf_transfer]
```

## Launch on Lightning.ai

### Option 1: Bash Script

```bash
cd /workspace/your-project-name
bash lightning_train.sh
```

### Option 2: Python Script (Recommended for portability)

```bash
cd /workspace/your-project-name
python3 lightning_train.py \
    --data-dir /workspace/data \
    --checkpoint-dir /workspace/checkpoints
```

### Option 3: Direct Python (Full Control)

```bash
python3 train_transformer_encdec.py \
    --data /workspace/data/how2sign_mediapipe_clip_1000vocab.h5 \
    --save-dir /workspace/checkpoints \
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
```

## What the Launchers Do

1. **Data**: Downloads H5 from HuggingFace if not present (or uses local copy)
2. **Checkpoints**: Copies `best_model.pt` → `latest_checkpoint.pt` to resume from best state
3. **Training**: Runs the warm-restart recipe (same as Modal)
4. **Monitoring**: Prints epoch BLEU, WER, learning rate

## Checkpoint Recovery

If training is interrupted on Lightning, restart with:

```bash
python3 lightning_train.py --checkpoint-dir /workspace/checkpoints
```

The script will automatically resume from `latest_checkpoint.pt`.

## Troubleshooting

### "H5 file not found"
- Manually download and place at `/workspace/data/how2sign_mediapipe_clip_1000vocab.h5`
- Or set `HF_TOKEN` and re-run

### "No module named train_transformer_encdec"
- Ensure all `.py` files are in the working directory
- Run from the project root: `cd /workspace/your-project && python3 lightning_train.py`

### "Out of memory"
- Reduce `--batch-size` (default 16)
- Reduce `--num-workers` (default 8)
- Increase `--grad-accum-steps` (default 2) to maintain effective batch size

### "BLEU not improving"
- This is expected if you're resuming from a plateau
- Try `--lr-restart-every 4` for more frequent LR restarts
- Or disable warm restart: `--lr-restart-every 0`

## Next Steps

1. Monitor first few epochs for convergence
2. If BLEU improves past 2.36 (Modal best), continue running
3. When done, download final checkpoints via Lightning UI
