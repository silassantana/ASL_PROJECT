#!/usr/bin/env python3
"""
Representation-collapse probe (local, no GPU required for mediapipe part).

For a sample of How2Sign val clips, extract:
  - mean-pooled mediapipe holistic landmarks + motion (1629-D), and
  - mean-pooled CLIP ViT-B/32 image features (512-D)

then compare the spread of pairwise cosine distances for each
representation. A representation that "collapses" (everything maps to
nearly the same point) will have a tight, near-zero distribution of
pairwise distances -- which is what we measured for CLIP frame features
via the retrieval probe (diagonal - off-diagonal gap ~ 0.001).

This tells us whether mediapipe+motion features at least vary
meaningfully across different clips/sentences, independent of any
text/CLIP joint space (mediapipe has no pretrained text alignment, so a
retrieval-style probe isn't meaningful for it -- but a collapse check is).

Usage:
  python mediapipe_collapse_probe.py \
    --how2sign-dir ~/Code/how-to-sign \
    --n-samples 100 \
    --use-realigned
"""

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np


def _read_rows(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)
    return rows


def _find_split_csv(base_dir, split, use_realigned):
    preferred = f"how2sign_realigned_{split}.csv" if use_realigned else f"how2sign_{split}.csv"
    fallback = f"how2sign_{split}.csv" if use_realigned else f"how2sign_realigned_{split}.csv"
    p1 = base_dir / preferred
    p2 = base_dir / fallback
    if p1.exists():
        return p1
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Missing split CSV for {split}: checked {p1} and {p2}")


def _text_from_row(row):
    txt = (row.get("SENTENCE") or "").strip().lower()
    return " ".join(txt.split())


def _times_from_row(row, use_realigned):
    if use_realigned:
        s = row.get("START_REALIGNED", row.get("START", "0"))
        e = row.get("END_REALIGNED", row.get("END", "0"))
    else:
        s = row.get("START", row.get("START_REALIGNED", "0"))
        e = row.get("END", row.get("END_REALIGNED", "0"))
    return float(s), float(e)


def collect_aligned_rows(how2sign_dir, split, use_realigned, max_samples):
    """Replicate the row-filtering from prepare_how2sign_text_h5.py to recover
    the same (video_name, start_s, end_s, text) tuples that produced the H5."""
    video_dir = how2sign_dir / "raw_videos"
    csv_path = _find_split_csv(how2sign_dir, split=split, use_realigned=use_realigned)
    rows = _read_rows(csv_path)

    aligned = []
    for row in rows:
        video_name = (row.get("VIDEO_NAME") or "").strip()
        text = _text_from_row(row)
        if not video_name or not text:
            continue

        video_path = video_dir / f"{video_name}.mp4"
        if not video_path.exists():
            continue

        try:
            start_s, end_s = _times_from_row(row, use_realigned=use_realigned)
        except Exception:
            continue

        aligned.append((video_path, start_s, end_s, text))
        if len(aligned) >= max_samples:
            break

    return aligned


# ---------------------------------------------------------------------------
# Mediapipe landmark + motion extraction (mirrors infer_video_encdec.py)
# ---------------------------------------------------------------------------

def create_holistic_detector():
    import mediapipe as mp

    class OldStyleHolistic:
        def __init__(self):
            self.holistic = mp.solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=1,
                smooth_landmarks=True,
            )

        def process(self, image_rgb):
            return self.holistic.process(image_rgb)

        def close(self):
            self.holistic.close()

    return OldStyleHolistic()


FACE_KEY_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 191, 80, 81, 82, 13,
    312, 311, 310, 415, 308, 33, 160, 158, 133, 153, 144, 263, 387, 385, 362,
    380, 373, 70, 63, 105, 66, 107, 300, 293, 334, 296, 336, 1, 2, 98, 327,
    172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361,
    323, 454, 356, 389,
]


def extract_landmarks_from_frame(frame, holistic):
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)

    landmarks = []

    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)

    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)

    if results.face_landmarks:
        for idx in FACE_KEY_INDICES:
            if idx < len(results.face_landmarks.landmark):
                lm = results.face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0])
    else:
        landmarks.extend([0.0] * 318)

    arr = np.array(landmarks, dtype=np.float32)
    if arr.shape[0] != 543:
        if arr.shape[0] < 543:
            arr = np.concatenate([arr, np.zeros(543 - arr.shape[0], dtype=np.float32)])
        else:
            arr = arr[:543]
    return arr


def add_motion_features(landmarks_sequence):
    T, D = landmarks_sequence.shape
    velocity = np.zeros_like(landmarks_sequence)
    velocity[1:] = landmarks_sequence[1:] - landmarks_sequence[:-1]
    acceleration = np.zeros_like(landmarks_sequence)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    return np.concatenate([landmarks_sequence, velocity, acceleration], axis=1)


def extract_mediapipe_meanpooled(video_path, start_s, end_s, holistic, max_frames=128):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    start_frame = max(0, int(start_s * fps))
    end_frame = max(start_frame + 1, int(end_s * fps))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    landmarks_list = []
    cur = start_frame
    while cur <= end_frame and len(landmarks_list) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        landmarks_list.append(extract_landmarks_from_frame(frame, holistic))
        cur += 1

    cap.release()

    if not landmarks_list:
        return None

    landmarks_array = np.stack(landmarks_list, axis=0)  # [T, 543]
    enhanced = add_motion_features(landmarks_array)  # [T, 1629]
    return enhanced.mean(axis=0)  # [1629]


# ---------------------------------------------------------------------------
# CLIP mean-pooled extraction (for side-by-side comparison on same clips)
# ---------------------------------------------------------------------------

def extract_clip_meanpooled(video_path, start_s, end_s, clip_model, clip_processor, device, max_frames=32):
    import torch

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    start_frame = max(0, int(start_s * fps))
    end_frame = max(start_frame + 1, int(end_s * fps))
    total = end_frame - start_frame + 1
    take_n = min(max_frames, max(1, total))
    target_indices = set(int(x) for x in np.linspace(start_frame, end_frame, num=take_n, dtype=int))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    cur = start_frame
    while cur <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in target_indices:
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cur += 1
    cap.release()

    if not frames:
        return None

    from PIL import Image
    images = [Image.fromarray(f) for f in frames]
    inputs = clip_processor(images=images, return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(device)

    with torch.no_grad():
        feats = clip_model.get_image_features(pixel_values=pixel_values)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    return feats.mean(dim=0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Collapse statistics
# ---------------------------------------------------------------------------

def collapse_stats(embs, name):
    embs = np.asarray(embs, dtype=np.float64)
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    normed = embs / np.clip(norms, 1e-8, None)

    sim = normed @ normed.T  # cosine similarity
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    pairwise_sims = sim[iu]

    print(f"\n=== {name} ===")
    print(f"  N: {n}, dim: {embs.shape[1]}")
    print(f"  Pairwise cosine similarity: mean={pairwise_sims.mean():.4f}, "
          f"std={pairwise_sims.std():.4f}, min={pairwise_sims.min():.4f}, max={pairwise_sims.max():.4f}")

    # Also raw Euclidean distance spread (un-normalized), for reference.
    from scipy.spatial.distance import pdist
    try:
        eucl = pdist(embs, metric="euclidean")
        print(f"  Pairwise euclidean distance: mean={eucl.mean():.4f}, std={eucl.std():.4f}, "
              f"cv={eucl.std()/max(eucl.mean(),1e-8):.4f}")
    except ImportError:
        pass

    return pairwise_sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--how2sign-dir", type=str, default="~/Code/how-to-sign")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--use-realigned", action="store_true")
    parser.add_argument("--n-samples", type=int, default=100)
    parser.add_argument("--skip-clip", action="store_true", help="Skip CLIP side-by-side (faster)")
    args = parser.parse_args()

    how2sign_dir = Path(args.how2sign_dir).expanduser().resolve()

    print(f"Collecting aligned rows from {args.split} CSV...")
    aligned = collect_aligned_rows(how2sign_dir, args.split, args.use_realigned, args.n_samples)
    print(f"Got {len(aligned)} aligned (video, text) pairs.")

    holistic = create_holistic_detector()

    clip_model = clip_processor = device = None
    if not args.skip_clip:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading CLIP on {device}...")
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        clip_model.eval()

    mediapipe_embs = []
    clip_embs = []
    texts = []

    for i, (video_path, start_s, end_s, text) in enumerate(aligned):
        mp_emb = extract_mediapipe_meanpooled(video_path, start_s, end_s, holistic)
        if mp_emb is None:
            continue

        if not args.skip_clip:
            c_emb = extract_clip_meanpooled(video_path, start_s, end_s, clip_model, clip_processor, device)
            if c_emb is None:
                continue
            clip_embs.append(c_emb)

        mediapipe_embs.append(mp_emb)
        texts.append(text)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(aligned)}...")

    holistic.close()

    print(f"\nSuccessfully extracted features for {len(mediapipe_embs)} clips.")

    mediapipe_sims = collapse_stats(mediapipe_embs, "Mediapipe + motion (1629-D, mean-pooled)")
    if clip_embs:
        clip_sims = collapse_stats(clip_embs, "CLIP ViT-B/32 (512-D, mean-pooled, this script)")

        print("\n=== Comparison ===")
        print(f"  Mediapipe pairwise-sim std: {mediapipe_sims.std():.4f}")
        print(f"  CLIP      pairwise-sim std: {clip_sims.std():.4f}")
        if mediapipe_sims.std() > clip_sims.std() * 1.5:
            print("  -> Mediapipe features show meaningfully more spread (less collapsed) than CLIP.")
        elif mediapipe_sims.std() < clip_sims.std() * 0.67:
            print("  -> Mediapipe features show LESS spread than CLIP (unexpected).")
        else:
            print("  -> Similar spread; mediapipe doesn't obviously avoid the collapse seen with CLIP.")


if __name__ == "__main__":
    main()
