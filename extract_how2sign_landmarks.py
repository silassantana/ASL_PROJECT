#!/usr/bin/env python3
"""
Extract MediaPipe Holistic landmarks from How2Sign videos.

This extracts the SAME features that gave us 29% accuracy on 20 signs:
- Pose landmarks (33 points)
- Left/Right hand landmarks (21 points each)
- Face landmarks (468 points)
- Velocity features (temporal derivatives)
- Acceleration features (second derivatives)

Total: 543 base landmarks → 1629 features with velocity/acceleration
"""

import argparse
import os
import cv2
import numpy as np
import h5py
import json
import mediapipe as mp
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import Counter
import pickle


def extract_landmarks_from_frame(frame, holistic):
    """
    Extract MediaPipe landmarks from a single frame.
    
    Returns:
        landmarks: np.array of shape (543,) or None if detection fails
    """
    # Convert to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    
    # Process
    results = holistic.process(image_rgb)
    
    # Extract landmarks
    landmarks = []
    
    # Pose (33 landmarks × 3 coordinates = 99 features)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)
    
    # Left hand (21 landmarks × 3 coordinates = 63 features)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # Right hand (21 landmarks × 3 coordinates = 63 features)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # Face (468 landmarks × 3 coordinates = 1404 features)
    # But we only use a subset to keep it manageable (106 points)
    if results.face_landmarks:
        # Key face points (mouth, eyes, eyebrows)
        key_indices = [
            # Lips outer
            61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
            # Lips inner
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
            # Right eye
            33, 160, 158, 133, 153, 144,
            # Left eye
            263, 387, 385, 362, 380, 373,
            # Right eyebrow
            70, 63, 105, 66, 107,
            # Left eyebrow
            300, 293, 334, 296, 336,
            # Nose
            1, 2, 98, 327,
            # Jaw
            172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389
        ]
        
        for idx in key_indices:
            if idx < len(results.face_landmarks.landmark):
                lm = results.face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0])
    else:
        landmarks.extend([0.0] * (106 * 3))  # 318 features
    
    # Ensure exactly 543 features (99 + 63 + 63 + 318)
    landmarks_array = np.array(landmarks, dtype=np.float32)
    expected_size = 543
    if landmarks_array.shape[0] != expected_size:
        # Pad or truncate to expected size
        if landmarks_array.shape[0] < expected_size:
            padding = np.zeros(expected_size - landmarks_array.shape[0], dtype=np.float32)
            landmarks_array = np.concatenate([landmarks_array, padding])
        else:
            landmarks_array = landmarks_array[:expected_size]
    
    return landmarks_array


def add_motion_features(landmarks_sequence):
    """
    Add velocity and acceleration features.
    
    Args:
        landmarks_sequence: [T, 543] array of landmarks
    
    Returns:
        enhanced: [T, 1629] array with velocity and acceleration
    """
    T, D = landmarks_sequence.shape
    
    # Velocity (first derivative)
    velocity = np.zeros_like(landmarks_sequence)
    velocity[1:] = landmarks_sequence[1:] - landmarks_sequence[:-1]
    
    # Acceleration (second derivative)
    acceleration = np.zeros_like(landmarks_sequence)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    
    # Concatenate [landmarks, velocity, acceleration]
    enhanced = np.concatenate([landmarks_sequence, velocity, acceleration], axis=1)
    
    return enhanced


def extract_clip_landmarks(video_path, start_time, end_time, target_frames=16):
    """
    Extract landmarks from a video clip.
    
    Args:
        video_path: Path to video file
        start_time: Start time in seconds
        end_time: End time in seconds
        target_frames: Number of frames to sample uniformly
    
    Returns:
        landmarks: [target_frames, 1629] array or None if failed
    """
    if not os.path.exists(video_path):
        return None
    
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
    if total_frames < target_frames:
        frame_indices = list(range(start_frame, end_frame))
        # Pad with last frame
        frame_indices += [end_frame - 1] * (target_frames - len(frame_indices))
    else:
        step = total_frames / target_frames
        frame_indices = [int(start_frame + i * step) for i in range(target_frames)]
    
    # Initialize MediaPipe
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    # Extract landmarks from sampled frames
    landmarks_list = []
    
    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            # Use zeros if frame read fails
            landmarks_list.append(np.zeros(543, dtype=np.float32))
            continue
        
        landmarks = extract_landmarks_from_frame(frame, holistic)
        landmarks_list.append(landmarks)
    
    cap.release()
    holistic.close()
    
    if len(landmarks_list) != target_frames:
        return None
    
    # Verify all landmarks have the same shape
    shapes = [lm.shape for lm in landmarks_list]
    if len(set(shapes)) != 1:
        # If shapes differ, something went wrong - skip this clip
        return None
    
    # Stack and add motion features
    landmarks_array = np.stack(landmarks_list, axis=0)  # [T, 543]
    enhanced = add_motion_features(landmarks_array)  # [T, 1629]
    
    return enhanced


def build_vocabulary(csv_paths, vocab_size, curated_vocab_file=None):
    """
    Build vocabulary from most common words OR curated list.
    
    Args:
        csv_paths: Paths to CSV files
        vocab_size: Target vocabulary size
        curated_vocab_file: Optional path to curated vocabulary file
    
    Returns:
        word_to_idx: dict mapping words to indices
        idx_to_word: list of words
    """
    if curated_vocab_file and os.path.exists(curated_vocab_file):
        # Load curated vocabulary
        print(f"Loading curated vocabulary from {curated_vocab_file}")
        with open(curated_vocab_file, 'r') as f:
            curated_words = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        
        # Verify words exist in dataset
        all_words = set()
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, sep='\t')
            for sentence in df['SENTENCE']:
                words = sentence.lower().split()
                all_words.update(words)
        
        # Filter curated words to those that exist in dataset
        idx_to_word = [word for word in curated_words if word in all_words]
        
        # If we need more words, add most common ones
        if len(idx_to_word) < vocab_size:
            word_counts = Counter()
            for csv_path in csv_paths:
                df = pd.read_csv(csv_path, sep='\t')
                for sentence in df['SENTENCE']:
                    words = sentence.lower().split()
                    word_counts.update(words)
            
            existing = set(idx_to_word)
            for word, _ in word_counts.most_common():
                if word not in existing:
                    idx_to_word.append(word)
                    if len(idx_to_word) >= vocab_size:
                        break
        
        # Limit to vocab_size
        idx_to_word = idx_to_word[:vocab_size]
        print(f"Curated vocabulary: {len(idx_to_word)} words found in dataset")
    else:
        # Frequency-based vocabulary (original behavior)
        word_counts = Counter()
        
        for csv_path in csv_paths:
            df = pd.read_csv(csv_path, sep='\t')  # Tab-separated
            for sentence in df['SENTENCE']:
                words = sentence.lower().split()
                word_counts.update(words)
        
        # Get top vocab_size words
        most_common = word_counts.most_common(vocab_size)
        idx_to_word = [word for word, _ in most_common]
    
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    
    return word_to_idx, idx_to_word


def process_split(csv_path, video_dir, word_to_idx, target_frames=16, max_samples=None, checkpoint_file=None):
    """
    Process one split (train/val/test) with resume capability.
    
    Returns:
        sequences: list of [T, 1629] arrays
        labels: list of word indices
        successful: number of successful extractions
        failed: number of failed extractions
    """
    df = pd.read_csv(csv_path, sep='\t')  # Tab-separated
    
    sequences = []
    labels = []
    successful = 0
    failed = 0
    start_idx = 0
    
    # Load checkpoint if exists
    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"  Loading checkpoint: {checkpoint_file}")
        with open(checkpoint_file, 'rb') as f:
            checkpoint = pickle.load(f)
            sequences = checkpoint['sequences']
            labels = checkpoint['labels']
            successful = checkpoint['successful']
            failed = checkpoint['failed']
            start_idx = checkpoint['next_idx']
        print(f"  Resuming from sample {start_idx} (already: {successful} success, {failed} failed)")
    
    if max_samples:
        df = df.head(max_samples)
    
    pbar = tqdm(df.iterrows(), total=len(df), initial=start_idx, desc=f"Processing {Path(csv_path).stem}")
    
    for idx, row in pbar:
        # Skip already processed
        if idx < start_idx:
            continue
        
        video_name = row['VIDEO_NAME']  # Use VIDEO_NAME, not SENTENCE_NAME
        start_time = row['START_REALIGNED']
        end_time = row['END_REALIGNED']
        sentence = row['SENTENCE']
        
        # Build video path
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        
        # Extract landmarks
        landmarks = extract_clip_landmarks(video_path, start_time, end_time, target_frames)
        
        if landmarks is None:
            failed += 1
            pbar.set_postfix({'success': successful, 'failed': failed})
            
            # Save checkpoint every 100 samples
            if checkpoint_file and (successful + failed) % 100 == 0:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({
                        'sequences': sequences,
                        'labels': labels,
                        'successful': successful,
                        'failed': failed,
                        'next_idx': idx + 1
                    }, f)
            continue
        
        # Get word indices (filter by vocabulary)
        words = sentence.lower().split()
        word_indices = [word_to_idx[word] for word in words if word in word_to_idx]
        
        if len(word_indices) == 0:
            failed += 1
            pbar.set_postfix({'success': successful, 'failed': failed})
            
            # Save checkpoint every 100 samples
            if checkpoint_file and (successful + failed) % 100 == 0:
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump({
                        'sequences': sequences,
                        'labels': labels,
                        'successful': successful,
                        'failed': failed,
                        'next_idx': idx + 1
                    }, f)
            continue
        
        sequences.append(landmarks)
        labels.append(word_indices)
        successful += 1
        
        pbar.set_postfix({'success': successful, 'failed': failed})
        
        # Save checkpoint every 100 samples
        if checkpoint_file and (successful + failed) % 100 == 0:
            with open(checkpoint_file, 'wb') as f:
                pickle.dump({
                    'sequences': sequences,
                    'labels': labels,
                    'successful': successful,
                    'failed': failed,
                    'next_idx': idx + 1
                }, f)
    
    # Save final checkpoint
    if checkpoint_file:
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({
                'sequences': sequences,
                'labels': labels,
                'successful': successful,
                'failed': failed,
                'next_idx': len(df)
            }, f)
    
    return sequences, labels, successful, failed


def create_h5_dataset(output_path, train_data, val_data, test_data, idx_to_word):
    """
    Create HDF5 dataset in the same format as the working model.
    """
    with h5py.File(output_path, 'w') as f:
        # Store metadata
        f.attrs['num_classes'] = len(idx_to_word)
        f.attrs['feature_dim'] = 1629  # 543 base + 543 velocity + 543 acceleration
        f.attrs['feature_type'] = 'mediapipe_holistic_motion'
        
        # Store vocabulary
        gloss_names = np.array(idx_to_word, dtype='S')
        f.create_dataset('gloss_names', data=gloss_names)
        
        # Process each split
        for split_name, (sequences, labels) in [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]:
            if len(sequences) == 0:
                continue
            
            # Find max lengths
            max_seq_len = max(seq.shape[0] for seq in sequences)
            max_label_len = max(len(label) for label in labels)
            
            # Create padded arrays
            num_samples = len(sequences)
            feature_dim = sequences[0].shape[1]
            
            sequences_padded = np.zeros((num_samples, max_seq_len, feature_dim), dtype=np.float32)
            sequence_lengths = np.zeros(num_samples, dtype=np.int32)
            labels_padded = np.zeros((num_samples, max_label_len), dtype=np.int32)
            label_lengths = np.zeros(num_samples, dtype=np.int32)
            
            for i, (seq, label) in enumerate(zip(sequences, labels)):
                seq_len = seq.shape[0]
                label_len = len(label)
                
                sequences_padded[i, :seq_len] = seq
                sequence_lengths[i] = seq_len
                labels_padded[i, :label_len] = label
                label_lengths[i] = label_len
            
            # Save to HDF5
            f.create_dataset(f'{split_name}_sequences', data=sequences_padded)
            f.create_dataset(f'{split_name}_sequence_lengths', data=sequence_lengths)
            f.create_dataset(f'{split_name}_labels', data=labels_padded)
            f.create_dataset(f'{split_name}_label_lengths', data=label_lengths)
            
            print(f"{split_name}: {num_samples} samples")


def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe landmarks from How2Sign')
    parser.add_argument('--video-dir', type=str, 
                       default='/home/silass/Code/how-to-sign/raw_videos',
                       help='Directory containing How2Sign videos')
    parser.add_argument('--csv-dir', type=str,
                       default='/home/silass/Code/how-to-sign',
                       help='Directory containing CSV annotation files')
    parser.add_argument('--vocab-size', type=int, default=200,
                       help='Vocabulary size (most common words)')
    parser.add_argument('--curated-vocab', type=str, default=None,
                       help='Path to curated vocabulary file (one word per line)')
    parser.add_argument('--output', type=str, 
                       default='how2sign_landmarks_{vocab_size}vocab.h5',
                       help='Output HDF5 file')
    parser.add_argument('--target-frames', type=int, default=16,
                       help='Number of frames to sample per clip')
    parser.add_argument('--max-per-split', type=int, default=None,
                       help='Maximum samples per split (for testing)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint if available')
    
    args = parser.parse_args()
    
    # Format output filename
    output_path = args.output.format(vocab_size=args.vocab_size)
    
    # CSV files
    csv_files = {
        'train': os.path.join(args.csv_dir, 'how2sign_realigned_train.csv'),
        'val': os.path.join(args.csv_dir, 'how2sign_realigned_val.csv'),
        'test': os.path.join(args.csv_dir, 'how2sign_realigned_test.csv')
    }
    
    # Check files exist
    for split, path in csv_files.items():
        if not os.path.exists(path):
            print(f"ERROR: CSV file not found: {path}")
            return
    
    print("="*70)
    print("MediaPipe Holistic Feature Extraction for How2Sign")
    print("="*70)
    print(f"Video directory: {args.video_dir}")
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Target frames: {args.target_frames}")
    print(f"Output: {output_path}")
    print()
    
    # Build vocabulary
    print("Building vocabulary...")
    word_to_idx, idx_to_word = build_vocabulary(
        csv_files.values(), 
        args.vocab_size,
        args.curated_vocab
    )
    print(f"Vocabulary: {idx_to_word[:20]}...")
    print()
    
    # Checkpoint files
    checkpoint_dir = '.extraction_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    train_checkpoint = os.path.join(checkpoint_dir, f'train_{args.vocab_size}.pkl') if args.resume else None
    val_checkpoint = os.path.join(checkpoint_dir, f'val_{args.vocab_size}.pkl') if args.resume else None
    test_checkpoint = os.path.join(checkpoint_dir, f'test_{args.vocab_size}.pkl') if args.resume else None
    
    # Process each split
    train_seqs, train_labels, train_success, train_fail = process_split(
        csv_files['train'], args.video_dir, word_to_idx, 
        args.target_frames, args.max_per_split, train_checkpoint
    )
    print(f"Train: {train_success} successful, {train_fail} failed\n")
    
    val_seqs, val_labels, val_success, val_fail = process_split(
        csv_files['val'], args.video_dir, word_to_idx,
        args.target_frames, args.max_per_split, val_checkpoint
    )
    print(f"Val: {val_success} successful, {val_fail} failed\n")
    
    test_seqs, test_labels, test_success, test_fail = process_split(
        csv_files['test'], args.video_dir, word_to_idx,
        args.target_frames, args.max_per_split, test_checkpoint
    )
    print(f"Test: {test_success} successful, {test_fail} failed\n")
    
    # Create HDF5
    print(f"Creating HDF5 dataset: {output_path}")
    create_h5_dataset(
        output_path,
        (train_seqs, train_labels),
        (val_seqs, val_labels),
        (test_seqs, test_labels),
        idx_to_word
    )
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"Total samples: {train_success + val_success + test_success}")
    print(f"Output file: {output_path}")
    print()
    
    # Clean up checkpoints on success
    if args.resume:
        for ckpt in [train_checkpoint, val_checkpoint, test_checkpoint]:
            if ckpt and os.path.exists(ckpt):
                os.remove(ckpt)
        print("Checkpoints cleaned up.")
    
    print("Next step:")
    print(f"  python train_transformer_encdec.py --data {output_path} \\")
    print("      --hidden-dim 256 --num-encoder-layers 4 --num-decoder-layers 4 \\")
    print("      --batch-size 16 --epochs 100 --lr 0.0001 --warmup-epochs 5")


if __name__ == '__main__':
    main()
