#!/usr/bin/env python3
"""
Build a segment-level HDF5 from YouTube-ASL .npz pose files.

Each segment in the YouTube auto-caption becomes one training sample:
  pose clip [T_clip, 1629]  →  English sentence (str)

Splits are done at the VIDEO level so no signer leaks between train/val.

H5 layout:
  /train/clips/{i}   [T_clip, 1629] float32, gzip-4
  /train/texts/{i}   scalar bytes (UTF-8 sentence)
  /train/n           scalar int
  /val/clips/{i}     ...
  /val/texts/{i}     ...
  /val/n             scalar int
  /stats/mean        [1629] float32
  /stats/std         [1629] float32

Usage:
  python build_segment_h5.py \
    --npz-dir ~/youtube_asl_poses \
    --output  youtube_asl_segments.h5
"""

import argparse
import json
import re
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

TARGET_FPS = 15

# Segments whose text matches these patterns are noise, not ASL
NOISE_RE = re.compile(
    r"^\[.*\]$|^\(.*\)$|^♪.*♪$|^music$|^applause$|^laughter$",
    re.IGNORECASE,
)


def clean_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def is_usable_segment(seg: dict, min_dur: float, max_dur: float,
                       min_chars: int) -> bool:
    dur = seg["end"] - seg["start"]
    text = clean_text(seg["text"])
    if dur < min_dur or dur > max_dur:
        return False
    if len(text) < min_chars:
        return False
    if NOISE_RE.match(text):
        return False
    return True


def slice_clip(poses: np.ndarray, start_s: float, end_s: float) -> np.ndarray:
    f0 = int(start_s * TARGET_FPS)
    f1 = int(end_s   * TARGET_FPS)
    f0 = max(0, f0)
    f1 = min(len(poses), f1)
    return poses[f0:f1]


def compute_stats(clips: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    sample = rng.choice(len(clips), size=min(len(clips), 5000), replace=False)
    D = clips[0].shape[1]
    s, sq, n = np.zeros(D), np.zeros(D), 0
    for i in sample:
        c = clips[i].astype(np.float64)
        s  += c.sum(0)
        sq += (c**2).sum(0)
        n  += len(c)
    mean = (s / n).astype(np.float32)
    std  = np.sqrt(np.maximum(sq/n - (mean.astype(np.float64))**2, 1e-8)).astype(np.float32)
    return mean, std


def build(npz_dir: Path, output: Path,
          val_split: float, min_dur: float, max_dur: float, min_chars: int):

    npz_files = sorted(npz_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} .npz files")

    # Collect all usable segments grouped by video
    video_data = []   # list of (video_id, [(clip, text), ...])
    skipped_no_cap = 0
    skipped_seg = 0

    for npz_path in tqdm(npz_files, desc="Reading .npz files", unit="video"):
        d = np.load(npz_path, allow_pickle=True)
        poses   = d["poses"]                          # [T, 1629]
        caption = str(d["caption"]).strip()
        segs    = json.loads(str(d["segments"]))

        if len(caption) < min_chars or poses.shape[0] < 30:
            skipped_no_cap += 1
            continue

        clips = []
        for seg in segs:
            if not is_usable_segment(seg, min_dur, max_dur, min_chars):
                skipped_seg += 1
                continue
            clip = slice_clip(poses, seg["start"], seg["end"])
            min_frames = int(min_dur * TARGET_FPS)
            if len(clip) < min_frames:
                skipped_seg += 1
                continue
            text = clean_text(seg["text"])
            clips.append((clip, text))

        if clips:
            video_data.append((npz_path.stem, clips))

    print(f"Videos with usable segments: {len(video_data)}")
    print(f"Skipped (no caption):        {skipped_no_cap}")
    print(f"Skipped segments (too short/long/noise): {skipped_seg}")

    # Split at video level
    rng = np.random.RandomState(42)
    rng.shuffle(video_data)
    n_val = max(1, int(len(video_data) * val_split))
    val_videos   = video_data[:n_val]
    train_videos = video_data[n_val:]

    train_clips = [(c, t) for _, segs in train_videos for c, t in segs]
    val_clips   = [(c, t) for _, segs in val_videos   for c, t in segs]

    print(f"\nTrain videos: {len(train_videos)}  →  {len(train_clips)} segments")
    print(f"Val   videos: {len(val_videos)}    →  {len(val_clips)} segments")

    # Normalisation stats from training clips
    print("\nComputing normalisation stats from training clips …")
    mean, std = compute_stats([c for c, _ in train_clips])

    # Write H5
    print(f"\nWriting {output} …")
    with h5py.File(output, "w") as f:
        for split_name, split_clips in [("train", train_clips), ("val", val_clips)]:
            grp_clips = f.require_group(f"{split_name}/clips")
            grp_texts = f.require_group(f"{split_name}/texts")
            for i, (clip, text) in enumerate(tqdm(split_clips,
                                                   desc=f"  {split_name}", unit="seg")):
                grp_clips.create_dataset(
                    str(i), data=clip, dtype="float32",
                    compression="gzip", compression_opts=4,
                )
                grp_texts.create_dataset(str(i), data=text.encode("utf-8"))
            f[f"{split_name}/n"] = len(split_clips)

        f["stats/mean"] = mean
        f["stats/std"]  = std

    size_gb = output.stat().st_size / 1e9
    print(f"\nDone. {output}  ({size_gb:.2f} GB)")
    print(f"  Train segments: {len(train_clips)}")
    print(f"  Val   segments: {len(val_clips)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-dir",   default="~/youtube_asl_poses")
    parser.add_argument("--output",    default="youtube_asl_segments.h5")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--min-dur",   type=float, default=1.5,
                        help="Min segment duration in seconds")
    parser.add_argument("--max-dur",   type=float, default=20.0,
                        help="Max segment duration in seconds")
    parser.add_argument("--min-chars", type=int,   default=10,
                        help="Min caption characters")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir).expanduser().resolve()
    output  = Path(args.output).expanduser().resolve()

    if not npz_dir.exists():
        sys.exit(f"npz-dir not found: {npz_dir}")

    build(npz_dir, output, args.val_split, args.min_dur, args.max_dur, args.min_chars)


if __name__ == "__main__":
    main()
