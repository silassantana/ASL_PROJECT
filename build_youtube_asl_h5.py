#!/usr/bin/env python3
"""
Build or incrementally update a HDF5 file from YouTube-ASL .npz pose files.

Run any time — already-present videos are skipped, new ones are appended,
and the index is rebuilt at the end.

H5 layout:
  /poses/{video_id}      [T, 1629] float32, gzip-4
  /captions/{video_id}   scalar bytes (UTF-8)
  /segments/{video_id}   scalar bytes (JSON)
  /index                 [N] variable-length bytes  ← ordered ID list for DataLoader

Usage:
  python build_youtube_asl_h5.py \
    --npz-dir ~/youtube_asl_poses \
    --output  youtube_asl_poses.h5

  # On Modal, mount the volume and point --npz-dir there.
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def load_npz(path: Path):
    d = np.load(path, allow_pickle=True)
    poses = d["poses"]                          # [T, 1629] float32
    caption = str(d["caption"])
    segments = str(d["segments"])
    return poses, caption, segments


def build(npz_dir: Path, output: Path, min_frames: int, min_caption_chars: int):
    npz_files = sorted(npz_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} .npz files in {npz_dir}")

    with h5py.File(output, "a") as f:
        poses_grp    = f.require_group("poses")
        captions_grp = f.require_group("captions")
        segments_grp = f.require_group("segments")

        already = set(poses_grp.keys())
        print(f"Already in H5: {len(already)}  |  New candidates: {len(npz_files) - len(already)}")

        added = 0
        skipped_exist = 0
        skipped_short = 0
        skipped_nocap = 0

        for npz_path in tqdm(npz_files, desc="Building H5", unit="video"):
            vid = npz_path.stem

            if vid in already:
                skipped_exist += 1
                continue

            try:
                poses, caption, segments = load_npz(npz_path)
            except Exception as e:
                tqdm.write(f"  SKIP {vid}: load error — {e}")
                continue

            if poses.shape[0] < min_frames:
                skipped_short += 1
                continue

            if len(caption) < min_caption_chars:
                skipped_nocap += 1
                continue

            poses_grp.create_dataset(
                vid, data=poses, dtype="float32",
                compression="gzip", compression_opts=4,
            )
            captions_grp.create_dataset(vid, data=caption.encode("utf-8"))
            segments_grp.create_dataset(vid, data=segments.encode("utf-8"))
            added += 1

        # Rebuild index (ordered list of all valid IDs)
        all_ids = sorted(poses_grp.keys())
        dt = h5py.special_dtype(vlen=bytes)
        if "index" in f:
            del f["index"]
        idx_ds = f.create_dataset("index", shape=(len(all_ids),), dtype=dt)
        for i, vid in enumerate(all_ids):
            idx_ds[i] = vid.encode("utf-8")

        print(f"\nDone.")
        print(f"  Added:           {added}")
        print(f"  Skipped (exist): {skipped_exist}")
        print(f"  Skipped (short): {skipped_short}")
        print(f"  Skipped (no cap):{skipped_nocap}")
        print(f"  Total in H5:     {len(all_ids)}")
        print(f"  Output:          {output}  ({output.stat().st_size / 1e9:.2f} GB)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz-dir",  default="~/youtube_asl_poses",
                        help="Directory containing scraped .npz files")
    parser.add_argument("--output",   default="youtube_asl_poses.h5",
                        help="Output H5 file (created or appended)")
    parser.add_argument("--min-frames",       type=int, default=30,
                        help="Drop clips shorter than this many pose frames (~2s at 15fps)")
    parser.add_argument("--min-caption-chars", type=int, default=10,
                        help="Drop clips with fewer caption characters than this")
    args = parser.parse_args()

    npz_dir = Path(args.npz_dir).expanduser().resolve()
    output  = Path(args.output).expanduser().resolve()

    if not npz_dir.exists():
        sys.exit(f"npz-dir not found: {npz_dir}")

    build(npz_dir, output, args.min_frames, args.min_caption_chars)


if __name__ == "__main__":
    main()
