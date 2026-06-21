#!/usr/bin/env python3
"""
how2sign_prepare_dataset.py - Build an H5 dataset from How2Sign clips (frontal view)

- Uses the provided How2Sign CSVs (tab-separated)
- Works with the pre-segmented green-screen clips (frontal)
- Tokenizes English sentence transcripts (word-level) as targets
- Extracts MediaPipe Holistic features (150 dims/frame)
- Outputs train/val/test splits ready for CTC-style training

Expected layout:
  how-to-sign/
    raw_videos/                # contains *_rgb_front.mp4
    how2sign_train.csv
    how2sign_val.csv
    how2sign_test.csv
    (optionally realigned versions: how2sign_realigned_*.csv)

Example:
  python how2sign_prepare_dataset.py \
    --how2sign-dir ~/Code/how-to-sign \
    --use-realigned \
    --output how2sign_continuous.h5 \
    --max-per-split 500 \
    --max-seq-len 250

Notes:
- How2Sign provides English sentences, not glosses; this is for concept proofing.
- Use the realigned CSVs for better temporal cuts.
"""

import os
import re
import h5py
import json
import argparse
import numpy as np
import pandas as pd
import cv2
import mediapipe as mp
from tqdm import tqdm
from pathlib import Path
from collections import Counter


def clean_text(sentence: str) -> str:
    sentence = sentence.strip().lower()
    sentence = re.sub(r"[^a-z0-9'\s]+", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def tokenize(sentence: str):
    if not sentence:
        return []
    return clean_text(sentence).split()


def build_vocab(df_list, min_freq: int):
    counter = Counter()
    for df in df_list:
        for s in df['SENTENCE'].tolist():
            counter.update(tokenize(str(s)))
    vocab = [w for w, c in counter.most_common() if c >= min_freq]
    print(f"Vocab size (freq>={min_freq}): {len(vocab)}")
    return vocab


def find_video(video_dir: str, sentence_name: str) -> str:
    # Column has e.g., --7E2sU6zP4_10-5-rgb_front
    base = sentence_name
    direct = os.path.join(video_dir, base + '.mp4')
    if os.path.exists(direct):
        return direct
    # fallback glob
    matches = list(Path(video_dir).glob(base + '*.mp4'))
    if matches:
        return str(matches[0])
    return ''


def extract_features(video_path: str, max_frames: int):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return np.zeros((0, 150), dtype=np.float32)

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    feats = []
    frames = 0
    while frames < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = holistic.process(frame_rgb)

        frame_feats = []
        if res.pose_landmarks:
            for lm in res.pose_landmarks.landmark[:25]:
                frame_feats.extend([lm.x, lm.y, lm.z])
        else:
            frame_feats.extend([0.0] * 75)

        hand_points = []
        if res.left_hand_landmarks:
            hand_points.extend(list(res.left_hand_landmarks.landmark[:13]))
        else:
            hand_points.extend([type('obj', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()] * 13)
        if res.right_hand_landmarks:
            hand_points.extend(list(res.right_hand_landmarks.landmark[:12]))
        else:
            hand_points.extend([type('obj', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()] * 12)
        for lm in hand_points:
            frame_feats.extend([lm.x, lm.y, lm.z])

        feats.append(frame_feats[:150])

    cap.release()
    holistic.close()
    return np.array(feats, dtype=np.float32)


def normalize(seq: np.ndarray) -> np.ndarray:
    if seq.size == 0:
        return seq
    mean = seq.mean()
    std = seq.std() + 1e-8
    return (seq - mean) / std


def pad_or_truncate(seq: np.ndarray, target_len: int) -> np.ndarray:
    T = len(seq)
    if T == 0:
        return np.zeros((target_len, 150), dtype=np.float32)
    if T == target_len:
        return seq
    if T < target_len:
        pad = np.repeat(seq[-1][None, :], target_len - T, axis=0)
        return np.concatenate([seq, pad], axis=0)
    return seq[:target_len]


def process_split(
    df,
    video_dir,
    word_to_idx,
    max_seq_len,
    max_per_split=None,
    cache_dir=None,
    split_name=None,
    resume_cache=False,
    return_arrays=True,
):
    if max_per_split:
        df = df.head(max_per_split)

    sequences = []
    labels = []
    label_lengths = []
    seq_lengths = []

    stats = {
        'total': len(df),
        'success': 0,
        'video_missing': 0,
        'feature_fail': 0,
        'empty_tokens': 0,
        'cached_skip': 0,
    }

    cache_split_dir = None
    if cache_dir:
        cache_split_dir = Path(cache_dir) / split_name
        cache_split_dir.mkdir(parents=True, exist_ok=True)

    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc='Extracting')):
        sent_name = str(row['SENTENCE_NAME']).strip()
        sent_text = str(row['SENTENCE'])
        tokens = tokenize(sent_text)
        token_ids = [word_to_idx[t] for t in tokens if t in word_to_idx]
        if not token_ids:
            stats['empty_tokens'] += 1
            continue

        cache_path = None
        if cache_split_dir:
            safe_name = re.sub(r'[^a-zA-Z0-9_-]+', '_', sent_name) or f'sample_{idx}'
            cache_path = cache_split_dir / f"{idx:06d}_{safe_name}.npz"
            if resume_cache and cache_path.exists():
                stats['cached_skip'] += 1
                continue

        vid_path = find_video(video_dir, sent_name)
        if not vid_path:
            stats['video_missing'] += 1
            continue

        feats = extract_features(vid_path, max_frames=max_seq_len)
        if feats.size == 0 or len(feats) < 3:
            stats['feature_fail'] += 1
            continue

        feats = normalize(feats)
        actual_len = len(feats)
        feats = pad_or_truncate(feats, max_seq_len)

        if cache_path:
            np.savez(
                cache_path,
                sequence=feats.astype(np.float32),
                labels=np.array(token_ids, dtype=np.int64),
                label_length=np.int64(len(token_ids)),
                seq_length=np.int64(min(actual_len, max_seq_len)),
                sentence_name=sent_name,
                sentence=sent_text,
            )

        if return_arrays:
            sequences.append(feats)
            labels.append(token_ids)
            label_lengths.append(len(token_ids))
            seq_lengths.append(min(actual_len, max_seq_len))

        stats['success'] += 1

    print(f"  ✓ Success: {stats['success']}")
    print(f"  ✗ Video missing: {stats['video_missing']}")
    print(f"  ✗ Feature fail: {stats['feature_fail']}")
    print(f"  ✗ Empty tokens: {stats['empty_tokens']}")
    if cache_dir:
        print(f"  ↺ Cached skip: {stats['cached_skip']}")

    if not return_arrays:
        return None

    if not sequences:
        return None

    max_lab = max(label_lengths)
    padded_labels = []
    for lab in labels:
        padded = lab + [0] * (max_lab - len(lab))
        padded_labels.append(padded)

    return (
        np.array(sequences, dtype=np.float32),
        np.array(padded_labels, dtype=np.int64),
        np.array(label_lengths, dtype=np.int64),
        np.array(seq_lengths, dtype=np.int64),
    )


def write_vocab_metadata(h5_file, vocab, max_seq_len):
    h5_file.create_dataset('num_classes', data=np.array([len(vocab)], dtype=np.int64))
    h5_file.create_dataset('max_seq_len', data=np.array([max_seq_len], dtype=np.int64))
    dt = h5py.special_dtype(vlen=str)
    gnames = h5_file.create_dataset('gloss_names', (len(vocab),), dtype=dt)
    for i, w in enumerate(vocab):
        gnames[i] = w


def write_h5(path, splits, vocab, max_seq_len):
    with h5py.File(path, 'w') as f:
        for split_name, data in splits.items():
            if data is None:
                continue
            seqs, labs, lab_lens, seq_lens = data
            f.create_dataset(f'{split_name}_sequences', data=seqs)
            f.create_dataset(f'{split_name}_labels', data=labs)
            f.create_dataset(f'{split_name}_label_lengths', data=lab_lens)
            f.create_dataset(f'{split_name}_sequence_lengths', data=seq_lens)

        write_vocab_metadata(f, vocab, max_seq_len)

    print(f"\n✓ Wrote dataset: {path}")


def write_mapping(path, vocab):
    idx_to_gloss = {i: w for i, w in enumerate(vocab)}
    gloss_to_idx = {w: i for i, w in enumerate(vocab)}
    mapping_file = path.replace('.h5', '_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump({
            'idx_to_gloss': idx_to_gloss,
            'gloss_to_idx': gloss_to_idx,
            'num_classes': len(vocab),
            'blank_idx': len(vocab)
        }, f, indent=2)
    print(f"✓ Wrote mapping: {mapping_file}")


def save_cache_meta(cache_dir: str, vocab, max_seq_len: int, args):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    meta = {
        'vocab': vocab,
        'max_seq_len': max_seq_len,
        'min_freq': args.min_freq,
        'use_realigned': args.use_realigned,
        'args': vars(args),
    }
    with open(Path(cache_dir) / 'cache_meta.json', 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"✓ Cache meta saved to {Path(cache_dir) / 'cache_meta.json'}")


def load_cache_meta(cache_dir: str):
    meta_path = Path(cache_dir) / 'cache_meta.json'
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return meta


def append_row(dset, row):
    n = dset.shape[0]
    dset.resize((n + 1,) + dset.shape[1:])
    dset[n] = row


def build_split_from_cache(cache_dir: str, split_name: str, h5_file, max_seq_len: int):
    split_dir = Path(cache_dir) / split_name
    files = sorted(split_dir.glob('*.npz'))
    if not files:
        print(f"No cache files for split {split_name}, skipping")
        return

    # Pass 1: find max label length
    max_lab = 0
    for fp in files:
        with np.load(fp) as d:
            max_lab = max(max_lab, int(d['label_length']))

    seq_ds = h5_file.create_dataset(
        f'{split_name}_sequences',
        shape=(0, max_seq_len, 150),
        maxshape=(None, max_seq_len, 150),
        dtype='float32',
        chunks=(1, max_seq_len, 150),
    )
    lab_ds = h5_file.create_dataset(
        f'{split_name}_labels',
        shape=(0, max_lab),
        maxshape=(None, max_lab),
        dtype='int64',
        chunks=True,
    )
    lab_len_ds = h5_file.create_dataset(
        f'{split_name}_label_lengths',
        shape=(0,),
        maxshape=(None,),
        dtype='int64',
        chunks=True,
    )
    seq_len_ds = h5_file.create_dataset(
        f'{split_name}_sequence_lengths',
        shape=(0,),
        maxshape=(None,),
        dtype='int64',
        chunks=True,
    )

    for fp in files:
        with np.load(fp) as d:
            seq = d['sequence'].astype(np.float32)
            labels = d['labels'].astype(np.int64)
            lab_len = int(d['label_length'])
            seq_len = int(d['seq_length'])

        pad_width = max_lab - len(labels)
        if pad_width > 0:
            labels = np.pad(labels, (0, pad_width), mode='constant')

        append_row(seq_ds, seq)
        append_row(lab_ds, labels)
        append_row(lab_len_ds, np.int64(lab_len))
        append_row(seq_len_ds, np.int64(seq_len))

    print(f"  ✓ Cached split {split_name}: {len(files)} samples, max_label_len={max_lab}")


def write_h5_from_cache(cache_dir: str, output_path: str, vocab, max_seq_len: int):
    with h5py.File(output_path, 'w') as f:
        for split_name in ['train', 'val', 'test']:
            build_split_from_cache(cache_dir, split_name, f, max_seq_len)
        write_vocab_metadata(f, vocab, max_seq_len)
    print(f"\n✓ Built dataset from cache: {output_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--how2sign-dir', help='Path to how-to-sign root (required unless using --from-cache)')
    ap.add_argument('--video-dir', default='raw_videos', help='Relative or absolute path to videos')
    ap.add_argument('--use-realigned', action='store_true', help='Use the realigned CSVs')
    ap.add_argument('--output', default='how2sign_continuous.h5', help='Output H5 file')
    ap.add_argument('--max-per-split', type=int, default=0, help='Limit samples per split (0=all)')
    ap.add_argument('--max-seq-len', type=int, default=250, help='Max frames per sequence')
    ap.add_argument('--min-freq', type=int, default=2, help='Min token frequency for vocab')
    ap.add_argument('--cache-dir', help='Directory to store per-sample cached npz files')
    ap.add_argument('--cache-only', action='store_true', help='Only write caches; skip H5 build')
    ap.add_argument('--from-cache', help='Build H5 from an existing cache directory (skip extraction)')
    ap.add_argument('--resume-cache', action='store_true', help='Skip already-cached samples')
    args = ap.parse_args()

    print('=' * 70)
    print('HOW2SIGN CONTINUOUS DATASET PREPARATION')
    print('=' * 70)

    # Build only from cache (no extraction)
    if args.from_cache:
        meta = load_cache_meta(args.from_cache)
        vocab = meta['vocab']
        max_seq_len = meta['max_seq_len']
        write_h5_from_cache(args.from_cache, args.output, vocab, max_seq_len)
        write_mapping(args.output, vocab)
        print('\nNext: train a CTC model (word-level tokens):')
        print(f"  python train_continuous.py --data {args.output} --epochs 30 --batch-size 8")
        return

    if not args.how2sign_dir:
        raise ValueError('Please provide --how2sign-dir or use --from-cache')

    base = args.how2sign_dir
    vid_dir = args.video_dir
    if not os.path.isabs(vid_dir):
        vid_dir = os.path.join(base, vid_dir)

    if args.use_realigned:
        train_csv = os.path.join(base, 'how2sign_realigned_train.csv')
        val_csv = os.path.join(base, 'how2sign_realigned_val.csv')
        test_csv = os.path.join(base, 'how2sign_realigned_test.csv')
    else:
        train_csv = os.path.join(base, 'how2sign_train.csv')
        val_csv = os.path.join(base, 'how2sign_val.csv')
        test_csv = os.path.join(base, 'how2sign_test.csv')

    # Load CSVs
    train_df = pd.read_csv(train_csv, sep='\t')
    val_df = pd.read_csv(val_csv, sep='\t')
    test_df = pd.read_csv(test_csv, sep='\t')

    # Build vocab from all splits
    vocab = build_vocab([train_df, val_df, test_df], args.min_freq)
    word_to_idx = {w: i for i, w in enumerate(vocab)}

    print(f"Sample vocab: {vocab[:10]}")

    if args.cache_dir:
        save_cache_meta(args.cache_dir, vocab, args.max_seq_len, args)

    # Process splits (cache to disk to survive interruptions)
    return_arrays = args.cache_dir is None  # don't hold everything in RAM when caching
    train_data = process_split(
        train_df,
        vid_dir,
        word_to_idx,
        args.max_seq_len,
        args.max_per_split,
        cache_dir=args.cache_dir,
        split_name='train',
        resume_cache=args.resume_cache,
        return_arrays=return_arrays,
    )
    val_data = process_split(
        val_df,
        vid_dir,
        word_to_idx,
        args.max_seq_len,
        args.max_per_split,
        cache_dir=args.cache_dir,
        split_name='val',
        resume_cache=args.resume_cache,
        return_arrays=return_arrays,
    )
    test_data = process_split(
        test_df,
        vid_dir,
        word_to_idx,
        args.max_seq_len,
        args.max_per_split,
        cache_dir=args.cache_dir,
        split_name='test',
        resume_cache=args.resume_cache,
        return_arrays=return_arrays,
    )

    if args.cache_dir:
        print('\nExtraction cached per-sample.')
        if args.cache_only:
            print('Re-run with --from-cache to assemble the H5 without re-extracting.')
            return
        # Assemble H5 from cached npz files
        write_h5_from_cache(args.cache_dir, args.output, vocab, args.max_seq_len)
        write_mapping(args.output, vocab)
    else:
        splits = {'train': train_data, 'val': val_data, 'test': test_data}
        write_h5(args.output, splits, vocab, args.max_seq_len)
        write_mapping(args.output, vocab)

    print('\nNext: train a CTC model (word-level tokens):')
    print(f"  python train_continuous.py --data {args.output} --epochs 30 --batch-size 8")


if __name__ == '__main__':
    main()
