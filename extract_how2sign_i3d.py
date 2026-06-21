#!/usr/bin/env python3
"""
Extract I3D (Inflated 3D ConvNet) features from How2Sign videos.

I3D is a 3D CNN pretrained on Kinetics for action recognition.
It captures temporal motion patterns much better than CLIP or MediaPipe.

Based on: "Quo Vadis, Action Recognition?" (CVPR 2017)
"""

import argparse
import os
import cv2
import numpy as np
import h5py
import json
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import Counter
import pickle


# We'll use torchvision's pretrained models
try:
    from torchvision.models.video import r3d_18, R3D_18_Weights

    print("✓ Using torchvision R3D-18 (similar to I3D)")
except ImportError:
    print("ERROR: Need torchvision >= 0.13")
    print("Install: pip install torchvision")
    exit(1)


class I3DFeatureExtractor(nn.Module):
    """Wrapper for extracting features from I3D/R3D model."""

    def __init__(self):
        super().__init__()
        # Load pretrained R3D-18 (lightweight I3D variant)
        weights = R3D_18_Weights.KINETICS400_V1
        self.model = r3d_18(weights=weights)

        # Remove final classification layer
        self.model.fc = nn.Identity()

        # Set to eval mode
        self.model.eval()

        # Preprocessing
        self.transform = weights.transforms()

    def forward(self, video_clip):
        """
        Extract features from video clip.

        Args:
            video_clip: [batch, channels, frames, height, width]

        Returns:
            features: [batch, 512] feature vector
        """
        with torch.no_grad():
            features = self.model(video_clip)
        return features


def extract_video_clip(video_path, start_time, end_time, num_frames=16):
    """
    Extract video clip and preprocess for I3D.

    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        num_frames: Number of frames to sample

    Returns:
        clip: [num_frames, height, width, channels] or None if failed
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    total_frames = end_frame - start_frame

    if total_frames <= 0:
        cap.release()
        return None

    # Sample frame indices uniformly
    if total_frames < num_frames:
        frame_indices = list(range(start_frame, end_frame))
        frame_indices += [end_frame - 1] * (num_frames - len(frame_indices))
    else:
        step = total_frames / num_frames
        frame_indices = [int(start_frame + i * step) for i in range(num_frames)]

    # Extract frames
    frames = []
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            # Use zeros if frame read fails
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
            continue

        # Resize to 224x224 (I3D input size)
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    if len(frames) != num_frames:
        return None

    # Stack frames: [num_frames, height, width, channels]
    clip = np.stack(frames, axis=0)

    return clip


def preprocess_clip_for_i3d(clip):
    """
    Preprocess clip for I3D model.

    Args:
        clip: [T, H, W, C] numpy array (uint8, RGB)

    Returns:
        tensor: [1, C, T, H, W] tensor (float32, normalized)
    """
    # Convert BGR to RGB
    clip = clip[:, :, :, ::-1]

    # Convert to float and normalize to [0, 1]
    clip = clip.astype(np.float32) / 255.0

    # Convert to tensor: [T, H, W, C] -> [C, T, H, W]
    clip = torch.from_numpy(clip).permute(3, 0, 1, 2)

    # Normalize (ImageNet stats)
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
    clip = (clip - mean) / std

    # Add batch dimension: [1, C, T, H, W]
    clip = clip.unsqueeze(0)

    return clip


def build_vocabulary(csv_paths, vocab_size, curated_vocab_file=None):
    """Build vocabulary from CSVs."""
    if curated_vocab_file and os.path.exists(curated_vocab_file):
        with open(curated_vocab_file, "r") as f:
            curated_words = [
                line.strip()
                for line in f
                if line.strip() and not line.strip().startswith("#")
            ]

        all_words = set()
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, sep="\t")
            for sentence in df["SENTENCE"]:
                words = sentence.lower().split()
                all_words.update(words)

        idx_to_word = [word for word in curated_words if word in all_words]

        if len(idx_to_word) < vocab_size:
            word_counts = Counter()
            for csv_path in csv_paths:
                df = pd.read_csv(csv_path, sep="\t")
                for sentence in df["SENTENCE"]:
                    words = sentence.lower().split()
                    word_counts.update(words)

            existing = set(idx_to_word)
            for word, _ in word_counts.most_common():
                if word not in existing:
                    idx_to_word.append(word)
                    if len(idx_to_word) >= vocab_size:
                        break

        idx_to_word = idx_to_word[:vocab_size]
    else:
        word_counts = Counter()
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, sep="\t")
            for sentence in df["SENTENCE"]:
                words = sentence.lower().split()
                word_counts.update(words)

        most_common = word_counts.most_common(vocab_size)
        idx_to_word = [word for word, _ in most_common]

    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    return word_to_idx, idx_to_word


def process_split(
    csv_path,
    video_dir,
    word_to_idx,
    extractor,
    device,
    num_frames=16,
    max_samples=None,
    checkpoint_file=None,
):
    """Process one split with I3D feature extraction."""
    df = pd.read_csv(csv_path, sep="\t")

    sequences = []
    labels = []
    successful = 0
    failed = 0
    start_idx = 0

    # Load checkpoint if exists
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"  Loading checkpoint: {checkpoint_file}")
        with open(checkpoint_file, "rb") as f:
            checkpoint = pickle.load(f)
            sequences = checkpoint["sequences"]
            labels = checkpoint["labels"]
            successful = checkpoint["successful"]
            failed = checkpoint["failed"]
            start_idx = checkpoint["next_idx"]
        print(
            f"  Resuming from row {start_idx} (already: {successful} success, {failed} failed)"
        )

    if max_samples:
        df = df.head(max_samples)

    # Slice dataframe to skip already processed rows
    df_to_process = df.iloc[start_idx:] if start_idx > 0 else df

    pbar = tqdm(
        df_to_process.iterrows(),
        total=len(df_to_process),
        desc=f"Processing {Path(csv_path).stem}",
    )

    current_row = start_idx  # Track absolute position in original dataframe

    for idx, row in pbar:
        video_name = row["VIDEO_NAME"]
        start_time = row["START_REALIGNED"]
        end_time = row["END_REALIGNED"]
        sentence = row["SENTENCE"]

        video_path = os.path.join(video_dir, f"{video_name}.mp4")

        # Extract video clip
        clip = extract_video_clip(video_path, start_time, end_time, num_frames)

        if clip is None:
            failed += 1
            pbar.set_postfix(
                {"success": successful, "failed": failed, "row": current_row}
            )
            if checkpoint_file and (successful + failed) % 100 == 0:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(
                        {
                            "sequences": sequences,
                            "labels": labels,
                            "successful": successful,
                            "failed": failed,
                            "next_idx": current_row + 1,
                        },
                        f,
                    )
            current_row += 1
            continue

        # Preprocess for I3D
        clip_tensor = preprocess_clip_for_i3d(clip).to(device)

        # Extract I3D features
        with torch.no_grad():
            features = extractor(clip_tensor)

        # Convert to numpy
        features = features.cpu().numpy().flatten()  # [512]

        # Get word indices
        words = sentence.lower().split()
        word_indices = [word_to_idx[word] for word in words if word in word_to_idx]

        if len(word_indices) == 0:
            failed += 1
            pbar.set_postfix(
                {"success": successful, "failed": failed, "row": current_row}
            )
            if checkpoint_file and (successful + failed) % 100 == 0:
                with open(checkpoint_file, "wb") as f:
                    pickle.dump(
                        {
                            "sequences": sequences,
                            "labels": labels,
                            "successful": successful,
                            "failed": failed,
                            "next_idx": current_row + 1,
                        },
                        f,
                    )
            current_row += 1
            continue

        sequences.append(features)
        labels.append(word_indices)
        successful += 1

        pbar.set_postfix({"success": successful, "failed": failed, "row": current_row})

        if checkpoint_file and (successful + failed) % 100 == 0:
            with open(checkpoint_file, "wb") as f:
                pickle.dump(
                    {
                        "sequences": sequences,
                        "labels": labels,
                        "successful": successful,
                        "failed": failed,
                        "next_idx": current_row + 1,
                    },
                    f,
                )

        current_row += 1

    if checkpoint_file:
        with open(checkpoint_file, "wb") as f:
            pickle.dump(
                {
                    "sequences": sequences,
                    "labels": labels,
                    "successful": successful,
                    "failed": failed,
                    "next_idx": start_idx + len(df_to_process),
                },
                f,
            )

    return sequences, labels, successful, failed


def create_h5_dataset(output_path, train_data, val_data, test_data, idx_to_word):
    """Create HDF5 dataset with I3D features."""
    with h5py.File(output_path, "w") as f:
        f.attrs["num_classes"] = len(idx_to_word)
        f.attrs["feature_dim"] = 512  # I3D output dimension
        f.attrs["feature_type"] = "i3d_r3d18"

        gloss_names = np.array(idx_to_word, dtype="S")
        f.create_dataset("gloss_names", data=gloss_names)

        for split_name, (sequences, labels) in [
            ("train", train_data),
            ("val", val_data),
            ("test", test_data),
        ]:
            if len(sequences) == 0:
                continue

            # Stack sequences (all same length: 512)
            sequences_array = np.array(sequences, dtype=np.float32)  # [N, 512]

            # Pad labels
            max_label_len = max(len(label) for label in labels)
            num_samples = len(sequences)

            labels_padded = np.zeros((num_samples, max_label_len), dtype=np.int32)
            label_lengths = np.zeros(num_samples, dtype=np.int32)

            for i, label in enumerate(labels):
                label_len = len(label)
                labels_padded[i, :label_len] = label
                label_lengths[i] = label_len

            # Save to HDF5
            f.create_dataset(f"{split_name}_sequences", data=sequences_array)
            f.create_dataset(
                f"{split_name}_sequence_lengths",
                data=np.full(num_samples, 1, dtype=np.int32),
            )  # All length 1 (single feature vector)
            f.create_dataset(f"{split_name}_labels", data=labels_padded)
            f.create_dataset(f"{split_name}_label_lengths", data=label_lengths)

            print(f"{split_name}: {num_samples} samples")


def main():
    parser = argparse.ArgumentParser(description="Extract I3D features from How2Sign")
    parser.add_argument(
        "--video-dir",
        type=str,
        default="/home/silass/Code/how-to-sign/raw_videos",
        help="Directory containing videos",
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default="/home/silass/Code/how-to-sign",
        help="Directory containing CSV files",
    )
    parser.add_argument("--vocab-size", type=int, default=200, help="Vocabulary size")
    parser.add_argument(
        "--curated-vocab",
        type=str,
        default=None,
        help="Path to curated vocabulary file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="how2sign_i3d_{vocab_size}vocab.h5",
        help="Output HDF5 file",
    )
    parser.add_argument(
        "--num-frames", type=int, default=16, help="Number of frames per clip"
    )
    parser.add_argument(
        "--max-per-split", type=int, default=None, help="Maximum samples per split"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")

    args = parser.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output filename
    output_path = args.output.format(vocab_size=args.vocab_size)

    # CSV files
    csv_files = {
        "train": os.path.join(args.csv_dir, "how2sign_realigned_train.csv"),
        "val": os.path.join(args.csv_dir, "how2sign_realigned_val.csv"),
        "test": os.path.join(args.csv_dir, "how2sign_realigned_test.csv"),
    }

    for split, path in csv_files.items():
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            return

    print("=" * 70)
    print("I3D Feature Extraction for How2Sign")
    print("=" * 70)
    print(f"Video directory: {args.video_dir}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Frames per clip: {args.num_frames}")
    print(f"Output: {output_path}")
    print()

    # Build vocabulary
    print("Building vocabulary...")
    word_to_idx, idx_to_word = build_vocabulary(
        csv_files.values(), args.vocab_size, args.curated_vocab
    )
    print(f"Vocabulary: {idx_to_word[:20]}...")
    print()

    # Load I3D extractor
    print("Loading I3D feature extractor...")
    extractor = I3DFeatureExtractor().to(device)
    print("✓ Model loaded")
    print()

    # Checkpoints
    checkpoint_dir = ".extraction_checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    train_ckpt = (
        os.path.join(checkpoint_dir, f"i3d_train_{args.vocab_size}.pkl")
        if args.resume
        else None
    )
    val_ckpt = (
        os.path.join(checkpoint_dir, f"i3d_val_{args.vocab_size}.pkl")
        if args.resume
        else None
    )
    test_ckpt = (
        os.path.join(checkpoint_dir, f"i3d_test_{args.vocab_size}.pkl")
        if args.resume
        else None
    )

    # Process splits
    train_seqs, train_labels, train_success, train_fail = process_split(
        csv_files["train"],
        args.video_dir,
        word_to_idx,
        extractor,
        device,
        args.num_frames,
        args.max_per_split,
        train_ckpt,
    )
    print(f"Train: {train_success} successful, {train_fail} failed\n")

    val_seqs, val_labels, val_success, val_fail = process_split(
        csv_files["val"],
        args.video_dir,
        word_to_idx,
        extractor,
        device,
        args.num_frames,
        args.max_per_split,
        val_ckpt,
    )
    print(f"Val: {val_success} successful, {val_fail} failed\n")

    test_seqs, test_labels, test_success, test_fail = process_split(
        csv_files["test"],
        args.video_dir,
        word_to_idx,
        extractor,
        device,
        args.num_frames,
        args.max_per_split,
        test_ckpt,
    )
    print(f"Test: {test_success} successful, {test_fail} failed\n")

    # Create HDF5
    print(f"Creating HDF5: {output_path}")
    create_h5_dataset(
        output_path,
        (train_seqs, train_labels),
        (val_seqs, val_labels),
        (test_seqs, test_labels),
        idx_to_word,
    )

    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE!")
    print("=" * 70)
    print(f"Total: {train_success + val_success + test_success} samples")
    print(f"Output: {output_path}")
    print()

    if args.resume:
        for ckpt in [train_ckpt, val_ckpt, test_ckpt]:
            if ckpt and os.path.exists(ckpt):
                os.remove(ckpt)
        print("Checkpoints cleaned up.")

    print("Next step:")
    print(f"  python train_transformer_encdec.py --data {output_path} \\")
    print("      --hidden-dim 256 --epochs 100 --resume")


if __name__ == "__main__":
    main()
