#!/usr/bin/env python3
"""
Extract video features from How2Sign videos using existing gloss vocabulary.

This script:
1. Loads existing gloss vocabulary (from top200/top20 mapping)
2. Reads How2Sign CSV annotations
3. Extracts CLIP video features
4. Creates HDF5 with same format as landmark-based datasets

Much simpler than reprocessing everything from scratch.
"""

import os
import json
import h5py
import numpy as np
import pandas as pd
import torch
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from PIL import Image


def load_vocabulary(mapping_path):
    """Load gloss vocabulary from existing mapping."""
    print(f"Loading vocabulary from {mapping_path}")
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    gloss_to_idx = mapping.get('gloss_to_idx', {})
    idx_to_gloss = mapping.get('idx_to_gloss', {})
    
    # Convert idx_to_gloss keys from string to int if needed
    if idx_to_gloss and isinstance(list(idx_to_gloss.keys())[0], str):
        idx_to_gloss = {int(k): v for k, v in idx_to_gloss.items()}
    
    print(f"Vocabulary size: {len(gloss_to_idx)}")
    print(f"Example glosses: {list(gloss_to_idx.keys())[:10]}")
    
    return gloss_to_idx, idx_to_gloss


def extract_frames_from_video(video_path, start_time, end_time, target_frames=16):
    """Extract frames between start and end times."""
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
    
    # Sample target_frames uniformly
    if len(frames) > target_frames:
        indices = np.linspace(0, len(frames) - 1, target_frames, dtype=int)
        frames = [frames[i] for i in indices]
    elif len(frames) < target_frames:
        # Repeat last frame if needed
        while len(frames) < target_frames:
            frames.append(frames[-1])
    
    return frames


def extract_clip_features_from_frames(frames, model, preprocess, device):
    """Extract CLIP features from list of frames.
    
    Returns: (num_frames, 512) numpy array
    """
    frame_features = []
    
    for frame in frames:
        pil_img = Image.fromarray(frame.astype(np.uint8))
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = model.encode_image(img_tensor)
        
        frame_features.append(features.cpu().numpy().squeeze())
    
    return np.stack(frame_features, axis=0).astype(np.float32)


def process_how2sign_data(
    csv_path,
    video_dir, 
    gloss_to_idx,
    model,
    preprocess,
    device,
    use_realigned=True,
    target_frames=16
):
    """Process How2Sign CSV and extract features."""
    print(f"\nProcessing {csv_path.name}...")
    
    # Read CSV
    df = pd.read_csv(csv_path, sep='\\t')
    print(f"Total samples in CSV: {len(df)}")
    
    # Determine time columns
    if use_realigned and 'START_REALIGNED' in df.columns:
        start_col = 'START_REALIGNED'
        end_col = 'END_REALIGNED'
        print("Using realigned timestamps")
    else:
        start_col = 'START'
        end_col = 'END'
        print("Using original timestamps")
    
    samples = []
    failed = 0
    no_glosses = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        video_name = row['VIDEO_NAME']
        video_path = video_dir / f"{video_name}.mp4"
        
        if not video_path.exists():
            failed += 1
            continue
        
        # Get timing
        start_time = float(row[start_col])
        end_time = float(row[end_col])
        
        # Extract frames
        frames = extract_frames_from_video(video_path, start_time, end_time, target_frames)
        if frames is None:
            failed += 1
            continue
        
        # Extract CLIP features
        try:
            features = extract_clip_features_from_frames(frames, model, preprocess, device)
        except Exception as e:
            print(f"Feature extraction failed: {e}")
            failed += 1
            continue
        
        # For glosses, we need to parse from sentence or use separate gloss file
        # How2Sign CSVs don't have glosses - they have English sentences
        # We'll need to skip this for now or create dummy glosses
        
        # FIXME: How2Sign doesn't have gloss annotations in CSVs!
        # This is the same problem as OpenASL
        
        # For now, create placeholder
        glosses = []  # Would need actual gloss annotations
        
        if not glosses:
            no_glosses += 1
            continue
        
        # Convert glosses to indices
        labels = [gloss_to_idx[g] for g in glosses if g in gloss_to_idx]
        
        if not labels:
            no_glosses += 1
            continue
        
        samples.append({
            'features': features,
            'labels': np.array(labels, dtype=np.int32),
            'video_name': video_name,
            'sentence': row['SENTENCE']
        })
    
    print(f"Processed: {len(samples)}, Failed: {failed}, No glosses: {no_glosses}")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mapping', required=True,
                        help='Gloss mapping JSON (e.g., continuous_gloss_mapping.json)')
    parser.add_argument('--how2sign-dir', default='~/Code/how-to-sign',
                        help='How2Sign directory with CSVs and raw_videos/')
    parser.add_argument('--output', required=True,
                        help='Output HDF5 file')
    parser.add_argument('--use-realigned', action='store_true',
                        help='Use realigned timestamps')
    parser.add_argument('--target-frames', type=int, default=16,
                        help='Number of frames to extract per clip')
    parser.add_argument('--device', default='cuda',
                        help='Device (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Expand paths
    mapping_path = Path(args.mapping).expanduser()
    how2sign_dir = Path(args.how2sign_dir).expanduser()
    video_dir = how2sign_dir / 'raw_videos'
    output_path = Path(args.output)
    
    # Check paths
    if not mapping_path.exists():
        print(f"Error: Mapping file not found: {mapping_path}")
        return
    
    if not how2sign_dir.exists():
        print(f"Error: How2Sign directory not found: {how2sign_dir}")
        return
    
    if not video_dir.exists():
        print(f"Error: Video directory not found: {video_dir}")
        return
    
    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Load vocabulary
    gloss_to_idx, idx_to_gloss = load_vocabulary(mapping_path)
    
    # Load CLIP
    print("\nLoading CLIP model...")
    import clip
    model, preprocess = clip.load("ViT-B/32", device=args.device)
    model.eval()
    print("CLIP loaded")
    
    print("\n" + "="*70)
    print("CRITICAL ISSUE DETECTED")
    print("="*70)
    print("How2Sign CSVs contain English sentences, NOT glosses!")
    print("The CSV columns are: VIDEO_ID, VIDEO_NAME, SENTENCE_ID, START, END, SENTENCE")
    print("")
    print("To extract video features, we need:")
    print("1. The video clips (✓ you have these)")
    print("2. Gloss annotations (✗ not in CSVs)")
    print("")
    print("Your existing how2sign_continuous_top200.h5 HAS glosses.")
    print("But it doesn't store which video each sample came from!")
    print("")
    print("Solutions:")
    print("A. Find How2Sign gloss annotations (separate file)")
    print("B. Use the existing H5 structure but can't map back to videos")
    print("C. Re-extract everything from original How2Sign source")
    print("="*70)
    
    # Check for gloss files
    gloss_files = list(how2sign_dir.glob("*gloss*"))
    if gloss_files:
        print(f"\nFound potential gloss files: {gloss_files}")
    else:
        print("\nNo gloss files found in How2Sign directory")
    
    return


if __name__ == '__main__':
    main()
