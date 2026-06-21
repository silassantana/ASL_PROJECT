#!/usr/bin/env python3
"""
Prepare a text-supervised How2Sign H5 from local videos and realigned CSVs.

Inputs expected under --how2sign-dir:
- raw_videos/<VIDEO_NAME>.mp4
- how2sign_realigned_train.csv (or how2sign_train.csv fallback)
- how2sign_realigned_val.csv (or how2sign_val.csv fallback)
- how2sign_realigned_test.csv (or how2sign_test.csv fallback)

Outputs:
- <output>.h5 with datasets:
  - {split}_sequences: [N, max_seq_len, 512] float32 CLIP frame features
  - {split}_sequence_lengths: [N] int32
  - {split}_texts: [N] utf-8 strings
"""

import argparse
import csv
import gc
import os
from datetime import datetime
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor


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


def _extract_frames(video_path, start_s, end_s, max_seq_len):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 25.0

    start_frame = max(0, int(start_s * fps))
    end_frame = max(start_frame + 1, int(end_s * fps))

    total = end_frame - start_frame + 1
    # Keep memory bounded by selecting target indices first and only retaining those frames.
    take_n = min(max_seq_len, max(1, total))
    target_indices = np.linspace(start_frame, end_frame, num=take_n, dtype=int)
    target_positions = set(int(x) for x in target_indices.tolist())

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    cur = start_frame

    while cur <= end_frame:
        ok, frame = cap.read()
        if not ok:
            break
        if cur in target_positions:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cur += 1

    cap.release()

    if not frames:
        return None

    return frames


def _encode_frames_clip(frames, model, processor, device, clip_batch_size=32):
    if not frames:
        return None

    all_feats = []
    for i in range(0, len(frames), clip_batch_size):
        batch = frames[i:i + clip_batch_size]
        inputs = processor(images=batch, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)

        with torch.no_grad():
            feats = model.get_image_features(pixel_values=pixel_values)
            feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-6)

        all_feats.append(feats.detach().cpu().numpy().astype(np.float32))

    return np.concatenate(all_feats, axis=0)


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


def build_split(
    split,
    rows,
    video_dir,
    h5f,
    model,
    processor,
    device,
    max_seq_len,
    clip_batch_size,
    max_samples,
    use_realigned,
    resume=False,
    checkpoint_every=25,
):
    if max_samples is not None:
        rows = rows[:max_samples]

    nmax = len(rows)
    seq_key = f"{split}_sequences"
    len_key = f"{split}_sequence_lengths"
    txt_key = f"{split}_texts"

    if seq_key in h5f and len_key in h5f and txt_key in h5f:
        if not resume:
            raise RuntimeError(
                f"Split '{split}' already exists in output. Use --resume to continue or choose a new --output file."
            )
        seq_ds = h5f[seq_key]
        len_ds = h5f[len_key]
        txt_ds = h5f[txt_key]
        total_rows_prev = int(h5f.attrs.get(f"{split}_total_rows", -1))
        if total_rows_prev not in (-1, nmax):
            raise RuntimeError(
                f"Row-count mismatch for split '{split}': previous={total_rows_prev}, current={nmax}. "
                "Use the same CSV/max-samples settings when resuming."
            )
        if int(seq_ds.shape[1]) != int(max_seq_len):
            raise RuntimeError(
                f"max_seq_len mismatch for split '{split}': existing={seq_ds.shape[1]}, requested={max_seq_len}."
            )
        complete = bool(h5f.attrs.get(f"{split}_complete", False))
        if complete:
            written = int(h5f.attrs.get(f"{split}_written", seq_ds.shape[0]))
            skipped = int(h5f.attrs.get(f"{split}_skipped", 0))
            print(f"{split}: already complete (written={written}, skipped={skipped}), skipping.")
            return
        row_cursor = int(h5f.attrs.get(f"{split}_row_cursor", 0))
        written = int(h5f.attrs.get(f"{split}_written", 0))
        skipped = int(h5f.attrs.get(f"{split}_skipped", 0))
        print(f"{split}: resuming from row {row_cursor}/{nmax} (written={written}, skipped={skipped})")
    else:
        seq_ds = h5f.create_dataset(
            seq_key,
            shape=(nmax, max_seq_len, 512),
            maxshape=(None, max_seq_len, 512),
            dtype="float32",
            chunks=(1, max_seq_len, 512),
            compression="gzip",
        )
        len_ds = h5f.create_dataset(
            len_key,
            shape=(nmax,),
            maxshape=(None,),
            dtype="int32",
        )
        txt_ds = h5f.create_dataset(
            txt_key,
            shape=(nmax,),
            maxshape=(None,),
            dtype=h5py.string_dtype("utf-8"),
        )
        row_cursor = 0
        written = 0
        skipped = 0
        h5f.attrs[f"{split}_total_rows"] = int(nmax)
        h5f.attrs[f"{split}_row_cursor"] = int(row_cursor)
        h5f.attrs[f"{split}_written"] = int(written)
        h5f.attrs[f"{split}_skipped"] = int(skipped)
        h5f.attrs[f"{split}_complete"] = False
        h5f.flush()

    for row_idx in tqdm(range(row_cursor, nmax), desc=f"{split} build"):
        row = rows[row_idx]
        video_name = (row.get("VIDEO_NAME") or "").strip()
        text = _text_from_row(row)
        if not video_name or not text:
            skipped += 1
            h5f.attrs[f"{split}_row_cursor"] = int(row_idx + 1)
            continue

        video_path = video_dir / f"{video_name}.mp4"
        if not video_path.exists():
            skipped += 1
            h5f.attrs[f"{split}_row_cursor"] = int(row_idx + 1)
            continue

        try:
            start_s, end_s = _times_from_row(row, use_realigned=use_realigned)
        except Exception:
            skipped += 1
            h5f.attrs[f"{split}_row_cursor"] = int(row_idx + 1)
            continue

        frames = _extract_frames(video_path, start_s, end_s, max_seq_len=max_seq_len)
        if not frames:
            skipped += 1
            h5f.attrs[f"{split}_row_cursor"] = int(row_idx + 1)
            continue

        feats = _encode_frames_clip(
            frames,
            model=model,
            processor=processor,
            device=device,
            clip_batch_size=clip_batch_size,
        )
        if feats is None or len(feats) == 0:
            skipped += 1
            h5f.attrs[f"{split}_row_cursor"] = int(row_idx + 1)
            continue

        slen = min(len(feats), max_seq_len)
        seq_ds[written, :slen, :] = feats[:slen]
        if slen < max_seq_len:
            seq_ds[written, slen:, :] = 0.0
        len_ds[written] = slen
        txt_ds[written] = text
        written += 1

        del frames
        del feats

        h5f.attrs[f"{split}_row_cursor"] = int(row_idx + 1)
        h5f.attrs[f"{split}_written"] = int(written)
        h5f.attrs[f"{split}_skipped"] = int(skipped)

        if ((row_idx + 1) % max(1, checkpoint_every)) == 0:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            h5f.flush()

    seq_ds.resize((written, max_seq_len, 512))
    len_ds.resize((written,))
    txt_ds.resize((written,))

    h5f.attrs[f"{split}_row_cursor"] = int(nmax)
    h5f.attrs[f"{split}_written"] = int(written)
    h5f.attrs[f"{split}_skipped"] = int(skipped)
    h5f.attrs[f"{split}_complete"] = True
    h5f.flush()

    print(f"{split}: written={written}, skipped={skipped}")


def main():
    parser = argparse.ArgumentParser(description="Prepare How2Sign text-supervised H5 from local videos")
    parser.add_argument("--how2sign-dir", type=str, default="~/Code/how-to-sign")
    parser.add_argument("--output", type=str, default="how2sign_clip_text_realigned.h5")
    parser.add_argument("--use-realigned", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--clip-batch-size", type=int, default=32)
    parser.add_argument("--max-samples-per-split", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resume", action="store_true", help="Resume from existing output H5 if present")
    parser.add_argument("--checkpoint-every", type=int, default=25, help="Flush progress every N rows")
    parser.add_argument(
        "--recreate-on-corrupt",
        action="store_true",
        help="If resume file is corrupt/unreadable, archive it and restart from scratch",
    )
    args = parser.parse_args()

    how2sign_dir = Path(args.how2sign_dir).expanduser().resolve()
    video_dir = how2sign_dir / "raw_videos"
    if not how2sign_dir.exists():
        raise FileNotFoundError(f"Missing how2sign dir: {how2sign_dir}")
    if not video_dir.exists():
        raise FileNotFoundError(f"Missing raw_videos dir: {video_dir}")

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")
    print(f"Using device: {device}")

    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    split_rows = {}
    for split in ["train", "val", "test"]:
        csv_path = _find_split_csv(how2sign_dir, split=split, use_realigned=args.use_realigned)
        rows = _read_rows(csv_path)
        split_rows[split] = rows
        print(f"Loaded {split}: {len(rows)} rows from {csv_path}")

    out_path = Path(args.output).resolve()
    mode = "a" if (args.resume and out_path.exists()) else "w"
    try:
        h5f_ctx = h5py.File(out_path, mode)
    except OSError as exc:
        if not (args.resume and args.recreate_on_corrupt and out_path.exists()):
            raise RuntimeError(
                f"Failed to open output H5 '{out_path}' in mode '{mode}': {exc}\n"
                "If this is a corrupted partial file, re-run with --recreate-on-corrupt."
            ) from exc

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = out_path.with_suffix(out_path.suffix + f".corrupt_{stamp}")
        os.replace(out_path, backup)
        print(f"Detected corrupt resume file. Moved to: {backup}")
        print("Recreating output H5 from scratch...")
        mode = "w"
        h5f_ctx = h5py.File(out_path, mode)

    with h5f_ctx as h5f:
        if mode == "w":
            h5f.attrs["feature_dim"] = 512
            h5f.attrs["feature_type"] = "clip_vit_b32"
            h5f.attrs["target_type"] = "text_sentence"
            h5f.attrs["text_source"] = "how2sign_realigned_sentence" if args.use_realigned else "how2sign_sentence"
            h5f.attrs["max_seq_len"] = int(args.max_seq_len)
        else:
            existing_max = int(h5f.attrs.get("max_seq_len", args.max_seq_len))
            if existing_max != int(args.max_seq_len):
                raise RuntimeError(
                    f"max_seq_len mismatch in resume mode: existing={existing_max}, requested={args.max_seq_len}"
                )

        for split in ["train", "val", "test"]:
            build_split(
                split=split,
                rows=split_rows[split],
                video_dir=video_dir,
                h5f=h5f,
                model=model,
                processor=processor,
                device=device,
                max_seq_len=args.max_seq_len,
                clip_batch_size=args.clip_batch_size,
                max_samples=args.max_samples_per_split,
                use_realigned=args.use_realigned,
                resume=args.resume,
                checkpoint_every=args.checkpoint_every,
            )

    print(f"Done: wrote {out_path}")


if __name__ == "__main__":
    main()
