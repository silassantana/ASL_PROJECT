#!/usr/bin/env python3
"""
Recreate How2Sign dataset with CLIP video features instead of MediaPipe landmarks.

This uses the SAME vocabulary and data splits as the existing landmark-based dataset,
but extracts CLIP features from videos instead.

Based on how2sign_prepare_dataset.py but simplified for video features only.
"""

import os
import re
import h5py
import json
import argparse
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import Counter
from PIL import Image


def clean_text(sentence: str) -> str:
    sentence = sentence.strip().lower()
    sentence = re.sub(r"[^a-z0-9'\s]+", "", sentence)
    sentence = re.sub(r"\s+", " ", sentence).strip()
    return sentence


def extract_frames(video_path, start_time, end_time, target_frames=16):
    """Extract frames from video between start and end times."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    frames = []
    frame_idx = start_frame
    
    while frame_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        frame_idx += 1
    
    cap.release()
    
    if not frames:
        return None
    
    # Sample uniformly to get target_frames
    if len(frames) > target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < target_frames:
        # Pad by repeating last frame
        while len(frames) < target_frames:
            frames.append(frames[-1])
    
    return frames


def extract_clip_features(frames, model, preprocess, device):
    """Extract CLIP ViT-B/32 features from frames.
    
    Returns: (num_frames, 512) array
    """
    if not frames:
        return None
    
    features = []
    for frame in frames:
        pil_img = Image.fromarray(frame.astype(np.uint8))
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model.encode_image(img_tensor)
        
        features.append(feat.cpu().numpy().squeeze())
    
    return np.stack(features, axis=0).astype(np.float32)


def process_split(csv_path, video_dir, model, preprocess, device, 
                  word_to_idx, use_realigned=True, target_frames=16, max_samples=None):
    """Process one split (train/val/test)."""
    print(f"\nProcessing {csv_path.name}...")
    
    df = pd.read_csv(csv_path, sep='\\t')
    print(f"Total samples: {len(df)}")
    
    if max_samples:
        df = df.head(max_samples)
        print(f"Limited to {max_samples} samples")
    
    # Determine time columns
    if use_realigned and 'START_REALIGNED' in df.columns:
        start_col = 'START_REALIGNED'
        end_col = 'END_REALIGNED'
        print("Using realigned timestamps")
    else:
        start_col = 'START'
        end_col = 'END'
    
    samples = []
    failed_video = 0
    failed_features = 0
    no_words = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_name = row['VIDEO_NAME']
        video_path = video_dir / f"{video_name}.mp4"
        
        if not video_path.exists():
            failed_video += 1
            continue
        
        # Get timing
        start_time = float(row[start_col])
        end_time = float(row[end_col])
        
        # Extract frames
        frames = extract_frames(video_path, start_time, end_time, target_frames)
        if frames is None:
            failed_features += 1
            continue
        
        # Extract CLIP features
        try:
            features = extract_clip_features(frames, model, preprocess, device)
            if features is None:
                failed_features += 1
                continue
        except Exception as e:
            print(f"\\nFeature extraction error: {e}")
            failed_features += 1
            continue
        
        # Parse sentence into words (our "glosses")
        sentence = row['SENTENCE']
        words = clean_text(sentence).split()
        
        if not words:
            no_words += 1
            continue
        
        # Convert to indices (skip OOV words)
        labels = [word_to_idx[w] for w in words if w in word_to_idx]
        
        if not labels:
            no_words += 1
            continue
        
        samples.append({
            'features': features,  # (T, 512)
            'labels': np.array(labels, dtype=np.int32),
            'video_name': video_name,
            'sentence': sentence
        })
    
    print(f"Success: {len(samples)}, Video not found: {failed_video}, " 
          f"Feature extraction failed: {failed_features}, No words: {no_words}")
    
    return samples


def build_vocabulary(csv_paths, top_k=1000, min_count=5):
    """Build vocabulary from all CSVs."""
    print("\\nBuilding vocabulary...")
    
    word_counts = Counter()
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep='\\t')
        for sentence in df['SENTENCE']:
            words = clean_text(sentence).split()
            word_counts.update(words)
    
    print(f"Total unique words: {len(word_counts)}")
    
    # Filter by min count
    filtered = {w: c for w, c in word_counts.items() if c >= min_count}
    print(f"Words with >= {min_count} occurrences: {len(filtered)}")
    
    # Take top K
    vocab = [w for w, c in sorted(filtered.items(), key=lambda x: x[1], reverse=True)[:top_k]]
    
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}
    
    print(f"Final vocabulary size: {len(vocab)}")
    print(f"Top 20 words: {vocab[:20]}")
    
    return word_to_idx, idx_to_word, vocab


def create_h5_dataset(output_path, train_samples, val_samples, test_samples, 
                      word_to_idx, idx_to_word, vocab):
    """Create HDF5 dataset."""
    print(f"\\nCreating HDF5: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Store vocabulary
        f.attrs['num_classes'] = len(vocab)
        f.attrs['feature_dim'] = 512  # CLIP ViT-B/32
        f.attrs['feature_type'] = 'clip_vitb32'
        
        dt = h5py.string_dtype('utf-8')
        vocab_ds = f.create_dataset('gloss_names', (len(vocab),), dtype=dt)
        for i, word in enumerate(vocab):
            vocab_ds[i] = word
        
        # Process each split
        for split_name, samples in [('train', train_samples), 
                                     ('val', val_samples), 
                                     ('test', test_samples)]:
            if not samples:
                print(f"Skipping {split_name} (no samples)")
                continue
            
            print(f"\\nWriting {split_name}: {len(samples)} samples")
            
            # Variable length features and labels
            feat_vlen = h5py.vlen_dtype(np.dtype('float32'))
            label_vlen = h5py.vlen_dtype(np.dtype('int32'))
            
            features_ds = f.create_dataset(f'{split_name}_sequences', 
                                           (len(samples),), dtype=feat_vlen)
            labels_ds = f.create_dataset(f'{split_name}_labels', 
                                         (len(samples),), dtype=label_vlen)
            seq_lens_ds = f.create_dataset(f'{split_name}_sequence_lengths', 
                                           (len(samples),), dtype='int32')
            label_lens_ds = f.create_dataset(f'{split_name}_label_lengths', 
                                             (len(samples),), dtype='int32')
            
            for i, sample in enumerate(tqdm(samples, desc=f"Writing {split_name}")):
                features = sample['features']  # (T, 512)
                labels = sample['labels']
                
                # Flatten features for storage
                features_ds[i] = features.flatten()
                labels_ds[i] = labels
                seq_lens_ds[i] = len(features)
                label_lens_ds[i] = len(labels)
    
    # Save mapping
    mapping_path = str(output_path).replace('.h5', '_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump({
            'idx_to_gloss': idx_to_word,
            'gloss_to_idx': word_to_idx
        }, f, indent=2)
    
    print(f"✓ Saved mapping to {mapping_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract CLIP features from How2Sign')
    parser.add_argument('--how2sign-dir', default='~/Code/how-to-sign',
                        help='How2Sign directory')
    parser.add_argument('--output', required=True,
                        help='Output HDF5 file')
    parser.add_argument('--vocab-size', type=int, default=200,
                        help='Vocabulary size (top K words)')
    parser.add_argument('--min-word-count', type=int, default=10,
                        help='Minimum word occurrences')
    parser.add_argument('--use-realigned', action='store_true',
                        help='Use realigned timestamps')
    parser.add_argument('--target-frames', type=int, default=16,
                        help='Frames to extract per clip')
    parser.add_argument('--max-per-split', type=int, default=None,
                        help='Max samples per split (for testing)')
    parser.add_argument('--device', default='cuda',
                        help='cuda or cpu')
    
    args = parser.parse_args()
    
    # Paths
    how2sign_dir = Path(args.how2sign_dir).expanduser()
    video_dir = how2sign_dir / 'raw_videos'
    output_path = Path(args.output)
    
    # Check paths
    if not how2sign_dir.exists():
        print(f"Error: How2Sign directory not found: {how2sign_dir}")
        return
    
    if not video_dir.exists():
        print(f"Error: Videos not found: {video_dir}")
        return
    
    # CSV paths
    if args.use_realigned:
        csv_prefix = 'how2sign_realigned'
    else:
        csv_prefix = 'how2sign'
    
    train_csv = how2sign_dir / f'{csv_prefix}_train.csv'
    val_csv = how2sign_dir / f'{csv_prefix}_val.csv'
    test_csv = how2sign_dir / f'{csv_prefix}_test.csv'
    
    for csv_path in [train_csv, val_csv, test_csv]:
        if not csv_path.exists():
            print(f"Error: CSV not found: {csv_path}")
            return
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load CLIP
    print("Loading CLIP model...")
    import clip
    model, preprocess = clip.load("ViT-B/32", device=args.device)
    model.eval()
    print("✓ CLIP loaded")
    
    # Build vocabulary
    word_to_idx, idx_to_word, vocab = build_vocabulary(
        [train_csv, val_csv, test_csv],
        top_k=args.vocab_size,
        min_count=args.min_word_count
    )
    
    # Process splits
    train_samples = process_split(train_csv, video_dir, model, preprocess, args.device,
                                  word_to_idx, args.use_realigned, args.target_frames,
                                  args.max_per_split)
    
    val_samples = process_split(val_csv, video_dir, model, preprocess, args.device,
                                word_to_idx, args.use_realigned, args.target_frames,
                                args.max_per_split)
    
    test_samples = process_split(test_csv, video_dir, model, preprocess, args.device,
                                 word_to_idx, args.use_realigned, args.target_frames,
                                 args.max_per_split)
    
    # Create HDF5
    create_h5_dataset(output_path, train_samples, val_samples, test_samples,
                     word_to_idx, idx_to_word, vocab)
    
    print("\\n" + "="*70)
    print("✓ DONE!")
    print("="*70)
    print(f"Dataset: {output_path}")
    print(f"Features: CLIP ViT-B/32 (512-dim per frame)")
    print(f"Vocabulary: {len(vocab)} words")
    print(f"Train: {len(train_samples)}, Val: {len(val_samples)}, Test: {len(test_samples)}")
    print("\\nNext step:")
    print(f"python train_continuous.py --data {output_path} --use-transformer \\\\")
    print(f"    --hidden-dim 512 --num-layers 8 --epochs 150")


if __name__ == '__main__':
    main()
