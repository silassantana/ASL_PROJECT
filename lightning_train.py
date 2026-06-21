#!/usr/bin/env python3
"""
Lightning.ai training launcher for ASL Transformer.

Usage:
    python3 lightning_train.py [--data-dir .] [--checkpoint-dir .]

Handles:
    - H5 dataset download from HuggingFace
    - Checkpoint resume setup
    - Training execution with warm-restart recipe
"""

import argparse
import os
import sys
import shutil
import subprocess
from pathlib import Path


PROFILES = {
    "warmrestart": {
        "default_checkpoint_dir": "./checkpoints_new_clean_modal",
        "default_patience": 20,
        "args": [
            "--hidden-dim", "512",
            "--num-encoder-layers", "3",
            "--num-decoder-layers", "3",
            "--batch-size", "16",
            "--eval-batch-size", "32",
            "--grad-accum-steps", "2",
            "--lr", "5e-5",
            "--lr-restart-every", "8",
            "--num-workers", "8",
            "--use-channel-attention",
            "--use-class-weights",
            "--class-weight-power", "0.5",
            "--label-smoothing", "0.1",
            "--eval-beam-size", "2",
            "--eval-length-penalty", "0.7",
            "--report-probe-samples", "64",
            "--report-beam-size", "4",
            "--report-length-penalty", "0.7",
            "--early-stop-min-delta", "0.005",
            "--resume",
        ],
    },
    "capacity_v1": {
        "default_checkpoint_dir": "./checkpoints_capacity_v1",
        "default_patience": 15,
        "args": [
            "--hidden-dim", "768",
            "--num-encoder-layers", "6",
            "--num-decoder-layers", "4",
            "--num-heads", "12",
            "--batch-size", "8",
            "--eval-batch-size", "16",
            "--grad-accum-steps", "4",
            "--lr", "2e-5",
            "--lr-restart-every", "6",
            "--num-workers", "8",
            "--use-channel-attention",
            "--use-class-weights",
            "--class-weight-power", "0.5",
            "--label-smoothing", "0.1",
            "--eval-beam-size", "2",
            "--eval-length-penalty", "0.7",
            "--report-probe-samples", "64",
            "--report-beam-size", "4",
            "--report-length-penalty", "0.7",
            "--early-stop-min-delta", "0.005",
            "--resume",
        ],
    },
}


def main():
    parser = argparse.ArgumentParser(description='Lightning.ai ASL Transformer Launcher')
    parser.add_argument('--profile', type=str, default='warmrestart', choices=sorted(PROFILES.keys()),
                        help='Training profile to run')
    parser.add_argument('--data-dir', type=str, default='.', help='Directory for H5 dataset')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory for checkpoints (defaults by profile)')
    parser.add_argument('--hf-repo', type=str, default='silassantana/asl-how2sign-features',
                        help='HuggingFace dataset repo')
    parser.add_argument('--h5-filename', type=str, default='how2sign_mediapipe_clip_1000vocab.h5',
                        help='H5 filename in repo')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--early-stop-patience', type=int, default=None, help='Early stopping patience override')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    profile = PROFILES[args.profile]
    early_stop_patience = (
        args.early_stop_patience
        if args.early_stop_patience is not None
        else int(profile['default_patience'])
    )

    # ========================================================================
    # Setup directories
    # ========================================================================

    print("=" * 70)
    print("Lightning.ai ASL Transformer Training")
    print("=" * 70)
    print()
    print(f"Profile: {args.profile}")

    data_dir = Path(args.data_dir)
    checkpoint_dir = Path(args.checkpoint_dir or profile['default_checkpoint_dir'])
    data_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    h5_path = data_dir / args.h5_filename

    # ========================================================================
    # Download H5 if not present
    # ========================================================================

    print(f"[1/4] Dataset preparation")
    if not h5_path.exists():
        print(f"  Downloading {args.h5_filename} from {args.hf_repo}...")
        try:
            from huggingface_hub import hf_hub_download
            hf_hub_download(
                repo_id=args.hf_repo,
                filename=args.h5_filename,
                repo_type="dataset",
                local_dir=str(data_dir),
            )
            print(f"  ✓ Download complete")
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            print(f"  Please manually download or set HF_TOKEN and try again.")
            sys.exit(1)
    else:
        print(f"  ✓ H5 dataset already present at {h5_path}")

    # ========================================================================
    # Prepare checkpoint for resume
    # ========================================================================

    print(f"\n[2/4] Checkpoint preparation")
    best_ckpt = checkpoint_dir / "best_model.pt"
    latest_ckpt = checkpoint_dir / "latest_checkpoint.pt"

    if best_ckpt.exists():
        print(f"  Found best_model.pt, seeding latest_checkpoint.pt...")
        shutil.copy2(best_ckpt, latest_ckpt)
        print(f"  ✓ Seeded latest_checkpoint.pt from best_model.pt")
    elif latest_ckpt.exists():
        print(f"  ✓ Found latest_checkpoint.pt, will resume from there")
    else:
        print(f"  ℹ No checkpoint found, starting from scratch")

    # ========================================================================
    # Run training
    # ========================================================================

    print(f"\n[3/4] Starting training...")
    print()

    cmd = [
        sys.executable,
        "train_transformer_encdec.py",
        "--data", str(h5_path),
        "--save-dir", str(checkpoint_dir),
        *profile['args'],
        "--seed", str(args.seed),
        "--early-stop-patience", str(early_stop_patience),
        "--epochs", str(args.epochs),
    ]

    print("Training command:")
    print(" ".join(cmd))
    print()

    try:
        result = subprocess.run(cmd, check=False)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()
