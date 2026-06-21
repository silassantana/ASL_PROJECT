#!/usr/bin/env python3
"""
YouTube-ASL Pose + Caption Extractor
=====================================
Downloads videos from the official YouTube-ASL dataset (11,095 videos,
984 hours, 610K English captions - Google Research / NeurIPS 2023),
extracts MediaPipe Holistic pose trajectories, and saves paired
(poses, captions) for supervised ASL→English translation training.

This is the real foundation. Not scraped. Not unlabeled.
Curated by Google Research. 2,500+ unique signers. English-aligned.

Output per video:
  <output_dir>/<video_id>.npz
    poses:    [T, 1629] float32  (543 landmarks × pos+vel+accel)
    captions: str                 (full English caption text)
    segments: list of {start, end, text} dicts (time-aligned)

  manifest.jsonl  — one line per completed video
  failed.jsonl    — failed downloads for retry

Usage:
  python scrape_youtube_asl.py \\
    --ids youtube_asl_video_ids.txt \\
    --output-dir ~/youtube_asl_poses \\
    --workers 1

  # Resume:
  python scrape_youtube_asl.py \\
    --ids youtube_asl_video_ids.txt \\
    --output-dir ~/youtube_asl_poses \\
    --resume
"""

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# ---------------------------------------------------------------------------
# MediaPipe landmark extraction (exact mirror of your existing pipeline)
# ---------------------------------------------------------------------------

FACE_KEY_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 191, 80, 81, 82, 13,
    312, 311, 310, 415, 308, 33, 160, 158, 133, 153, 144, 263, 387, 385, 362,
    380, 373, 70, 63, 105, 66, 107, 300, 293, 334, 296, 336, 1, 2, 98, 327,
    172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361,
    323, 454, 356, 389,
]


def create_holistic():
    import mediapipe as mp
    class Holistic:
        def __init__(self):
            self.h = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
            )
        def process(self, rgb):
            return self.h.process(rgb)
        def close(self):
            self.h.close()
    return Holistic()


def extract_frame_landmarks(frame_bgr, holistic):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    r = holistic.process(rgb)
    lm = []

    if r.pose_landmarks:
        for p in r.pose_landmarks.landmark:
            lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 99)

    if r.left_hand_landmarks:
        for p in r.left_hand_landmarks.landmark:
            lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 63)

    if r.right_hand_landmarks:
        for p in r.right_hand_landmarks.landmark:
            lm.extend([p.x, p.y, p.z])
    else:
        lm.extend([0.0] * 63)

    if r.face_landmarks:
        for idx in FACE_KEY_INDICES:
            if idx < len(r.face_landmarks.landmark):
                p = r.face_landmarks.landmark[idx]
                lm.extend([p.x, p.y, p.z])
            else:
                lm.extend([0.0, 0.0, 0.0])
    else:
        lm.extend([0.0] * (len(FACE_KEY_INDICES) * 3))

    arr = np.array(lm, dtype=np.float32)
    target = 543
    if arr.shape[0] < target:
        arr = np.concatenate([arr, np.zeros(target - arr.shape[0], dtype=np.float32)])
    else:
        arr = arr[:target]
    return arr


def add_motion_features(seq):
    """[T, 543] → [T, 1629] with velocity + acceleration."""
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    acc = np.zeros_like(seq)
    acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([seq, vel, acc], axis=1)


def extract_poses(video_path, holistic, target_fps=15, max_frames=8192):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, {}

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(src_fps / target_fps))

    landmarks = []
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            landmarks.append(extract_frame_landmarks(frame, holistic))
            if len(landmarks) >= max_frames:
                break
        frame_idx += 1

    cap.release()

    if len(landmarks) < 4:
        return None, {}

    seq = np.stack(landmarks, axis=0)
    seq = add_motion_features(seq)

    meta = {
        "src_fps": src_fps,
        "target_fps": target_fps,
        "stride": stride,
        "total_src_frames": total_frames,
        "extracted_frames": len(landmarks),
        "duration_s": len(landmarks) / target_fps,
    }
    return seq, meta


# ---------------------------------------------------------------------------
# Caption extraction
# ---------------------------------------------------------------------------

def parse_vtt(vtt_path):
    """Parse WebVTT subtitle file into list of {start, end, text} segments."""
    segments = []
    try:
        with open(vtt_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception:
        return segments

    # Remove WebVTT header
    content = re.sub(r'^WEBVTT.*?\n\n', '', content, flags=re.DOTALL)

    # Match timestamp blocks
    block_re = re.compile(
        r'(\d{2}:\d{2}:\d{2}[\.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[\.,]\d{3})[^\n]*\n(.*?)(?=\n\n|\Z)',
        re.DOTALL
    )

    def ts_to_sec(ts):
        ts = ts.replace(',', '.')
        parts = ts.split(':')
        h, m, s = int(parts[0]), int(parts[1]), float(parts[2])
        return h * 3600 + m * 60 + s

    for m in block_re.finditer(content):
        start = ts_to_sec(m.group(1))
        end = ts_to_sec(m.group(2))
        text = re.sub(r'<[^>]+>', '', m.group(3)).strip()
        text = re.sub(r'\s+', ' ', text)
        if text:
            segments.append({"start": start, "end": end, "text": text})

    # Deduplicate consecutive identical segments (YouTube auto-subs artifact)
    deduped = []
    for seg in segments:
        if not deduped or deduped[-1]["text"] != seg["text"]:
            deduped.append(seg)

    return deduped


def download_video_and_captions(video_id, tmpdir, max_height=480):
    """
    Download video + English captions using yt-dlp.
    Returns (video_path, vtt_path) or (None, None) on failure.
    Prefers manually uploaded captions; falls back to auto-generated.
    """
    url = f"https://www.youtube.com/watch?v={video_id}"
    out_template = str(Path(tmpdir) / f"{video_id}.%(ext)s")

    cmd = [
        "yt-dlp",
        url,
        "-f", f"bestvideo[height<={max_height}][ext=mp4]+bestaudio[ext=m4a]/best[height<={max_height}][ext=mp4]/best",
        "-o", out_template,
        "--write-subs",           # manually uploaded captions first
        "--write-auto-subs",      # fall back to auto-generated
        "--sub-lang", "en",
        "--sub-format", "vtt",
        "--convert-subs", "vtt",
        "--merge-output-format", "mp4",
        "--no-warnings",
        "--quiet",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None, None

    # Find downloaded files
    video_path = Path(tmpdir) / f"{video_id}.mp4"
    if not video_path.exists():
        return None, None

    # Look for VTT file (yt-dlp names it various ways)
    vtt_path = None
    for f in Path(tmpdir).glob(f"{video_id}*.vtt"):
        vtt_path = f
        break

    return video_path, vtt_path


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_done(manifest_path):
    done = set()
    if not Path(manifest_path).exists():
        return done
    with open(manifest_path) as f:
        for line in f:
            try:
                done.add(json.loads(line)["video_id"])
            except Exception:
                pass
    return done


def append_line(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extract poses + captions from YouTube-ASL dataset"
    )
    parser.add_argument("--ids", type=str, required=True,
                        help="Path to youtube_asl_video_ids.txt (one ID per line)")
    parser.add_argument("--output-dir", type=str, default="~/youtube_asl_poses",
                        help="Where to save .npz pose files")
    parser.add_argument("--target-fps", type=int, default=15)
    parser.add_argument("--max-frames", type=int, default=8192)
    parser.add_argument("--max-height", type=int, default=480)
    parser.add_argument("--sleep", type=float, default=1.0,
                        help="Seconds between downloads")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-processed videos")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start from this index in the ID list (for manual sharding)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="Stop at this index (exclusive)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    failed_path = output_dir / "failed.jsonl"

    # Load video IDs
    with open(args.ids) as f:
        all_ids = [line.strip() for line in f if line.strip()]

    ids = all_ids[args.start_idx:args.end_idx]
    print(f"YouTube-ASL Pose Extractor")
    print(f"  Total IDs in file: {len(all_ids)}")
    print(f"  Processing:        {len(ids)} (idx {args.start_idx}:{args.end_idx})")
    print(f"  Output:            {output_dir}")
    print()

    done = load_done(manifest_path) if args.resume else set()
    print(f"  Already done:      {len(done)}")
    print()

    holistic = create_holistic()
    saved = 0
    failed = 0

    try:
        for video_id in tqdm(ids, desc="YouTube-ASL", unit="video"):
            if video_id in done:
                continue

            npz_path = output_dir / f"{video_id}.npz"
            if args.resume and npz_path.exists():
                done.add(video_id)
                continue

            with tempfile.TemporaryDirectory() as tmpdir:
                video_path, vtt_path = download_video_and_captions(
                    video_id, tmpdir, max_height=args.max_height
                )

                if video_path is None:
                    append_line(failed_path, {
                        "video_id": video_id,
                        "reason": "download_failed",
                        "timestamp": datetime.now().isoformat(),
                    })
                    failed += 1
                    continue

                poses, pose_meta = extract_poses(
                    video_path, holistic,
                    target_fps=args.target_fps,
                    max_frames=args.max_frames,
                )

                if poses is None:
                    append_line(failed_path, {
                        "video_id": video_id,
                        "reason": "pose_extraction_failed",
                        "timestamp": datetime.now().isoformat(),
                    })
                    failed += 1
                    continue

                # Parse captions
                segments = parse_vtt(vtt_path) if vtt_path else []
                caption_text = " ".join(s["text"] for s in segments)

                # Save
                np.savez_compressed(
                    npz_path,
                    poses=poses,
                    caption=np.array(caption_text),
                    segments=np.array(json.dumps(segments)),
                )

                meta = {
                    "video_id": video_id,
                    "url": f"https://www.youtube.com/watch?v={video_id}",
                    "shape": list(poses.shape),
                    "has_captions": len(segments) > 0,
                    "n_caption_segments": len(segments),
                    "caption_chars": len(caption_text),
                    **pose_meta,
                    "timestamp": datetime.now().isoformat(),
                }

                append_line(manifest_path, meta)
                done.add(video_id)
                saved += 1

                tqdm.write(
                    f"  ✓ {video_id} | {poses.shape[0]}f "
                    f"({pose_meta.get('duration_s', 0):.0f}s) | "
                    f"captions={'yes' if segments else 'NO'} ({len(segments)} segs)"
                )

            time.sleep(args.sleep)

    finally:
        holistic.close()

    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Saved:  {saved}")
    print(f"  Failed: {failed}")
    print(f"  Total:  {len(done)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
