#!/usr/bin/env python3
"""
Extract MediaPipe features from How2Sign and create trainable dataset.
This allows training the model on MediaPipe features instead of I3D.
"""

import argparse
import os
import re
import cv2
import numpy as np
import h5py
import json
import glob
import shutil
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from collections import Counter
import torch
import pickle


def create_holistic_detector():
    """Create a holistic pose detector using MediaPipe tasks API."""
    try:
        from mediapipe import solutions, Image
        from mediapipe.framework.formats import landmark_pb2
        
        # Try to use old solutions API if available
        class OldStyleHolistic:
            def __init__(self):
                self.holistic = solutions.holistic.Holistic(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True
                )
            
            def process(self, image_rgb):
                return self.holistic.process(image_rgb)
            
            def close(self):
                self.holistic.close()
        
        return OldStyleHolistic()
    except AttributeError:
        # New API - use tasks instead
        print("Using new MediaPipe tasks API...")
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        from mediapipe import Image as MPImage
        import tempfile
        
        class NewStyleHolistic:
            def __init__(self):
                BaseOptions = python.BaseOptions
                
                # Create temporary directory for models
                self.model_dir = tempfile.gettempdir()
                
                try:
                    # Pose landmarker
                    pose_options = vision.PoseLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=self._download_model('pose_landmarker_full.tflite')
                        ),
                        running_mode=vision.RunningMode.IMAGE,
                    )
                    self.pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)
                except:
                    self.pose_landmarker = None
                
                try:
                    # Hand landmarker
                    hand_options = vision.HandLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=self._download_model('hand_landmarker.tflite')
                        ),
                        running_mode=vision.RunningMode.IMAGE,
                    )
                    self.hand_landmarker = vision.HandLandmarker.create_from_options(hand_options)
                except:
                    self.hand_landmarker = None
                
                try:
                    # Face landmarker
                    face_options = vision.FaceLandmarkerOptions(
                        base_options=BaseOptions(
                            model_asset_path=self._download_model('face_landmarker.tflite')
                        ),
                        running_mode=vision.RunningMode.IMAGE,
                    )
                    self.face_landmarker = vision.FaceLandmarker.create_from_options(face_options)
                except:
                    self.face_landmarker = None
            
            def _download_model(self, model_name):
                """Placeholder for model downloading."""
                return None
            
            def process(self, image_rgb):
                """Process image using new tasks API."""
                class Results:
                    pose_landmarks = None
                    left_hand_landmarks = None
                    right_hand_landmarks = None
                    face_landmarks = None
                
                results = Results()
                
                try:
                    from mediapipe import Image as MPImage
                    mp_image = MPImage(image_format=solutions.ImageFormat.SRGB, data=image_rgb)
                    
                    if self.pose_landmarker:
                        pose_result = self.pose_landmarker.detect(mp_image)
                        if pose_result.landmarks:
                            results.pose_landmarks = pose_result.landmarks[0]
                    
                    if self.hand_landmarker:
                        hand_result = self.hand_landmarker.detect(mp_image)
                        if hand_result.landmarks and len(hand_result.landmarks) > 0:
                            results.left_hand_landmarks = hand_result.landmarks[0]
                        if hand_result.landmarks and len(hand_result.landmarks) > 1:
                            results.right_hand_landmarks = hand_result.landmarks[1]
                    
                    if self.face_landmarker:
                        face_result = self.face_landmarker.detect(mp_image)
                        if face_result.landmarks:
                            results.face_landmarks = face_result.landmarks[0]
                except Exception as e:
                    print(f"Warning: Tasks API detection failed: {e}")
                
                return results
            
            def close(self):
                pass
        
        return NewStyleHolistic()


def extract_landmarks_from_frame(frame, holistic):
    """Extract MediaPipe landmarks from a single frame."""
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False
    results = holistic.process(image_rgb)
    
    landmarks = []
    
    # Pose (33 × 3 = 99)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 99)
    
    # Left hand (21 × 3 = 63)
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # Right hand (21 × 3 = 63)
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])
    else:
        landmarks.extend([0.0] * 63)
    
    # Face subset (106 key points × 3 = 318)
    if results.face_landmarks:
        key_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 33, 160, 158, 133, 153, 144, 263, 387, 385, 362, 380, 373, 70, 63, 105, 66, 107, 300, 293, 334, 296, 336, 1, 2, 98, 327, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389]
        for idx in key_indices:
            if idx < len(results.face_landmarks.landmark):
                lm = results.face_landmarks.landmark[idx]
                landmarks.extend([lm.x, lm.y, lm.z])
            else:
                landmarks.extend([0.0, 0.0, 0.0])
    else:
        landmarks.extend([0.0] * 318)
    
    # Ensure exactly 543 features
    landmarks_array = np.array(landmarks, dtype=np.float32)
    if landmarks_array.shape[0] != 543:
        if landmarks_array.shape[0] < 543:
            padding = np.zeros(543 - landmarks_array.shape[0], dtype=np.float32)
            landmarks_array = np.concatenate([landmarks_array, padding])
        else:
            landmarks_array = landmarks_array[:543]
    
    return landmarks_array


def add_motion_features(landmarks_sequence):
    """Add velocity and acceleration features."""
    T, D = landmarks_sequence.shape
    
    velocity = np.zeros_like(landmarks_sequence)
    velocity[1:] = landmarks_sequence[1:] - landmarks_sequence[:-1]
    
    acceleration = np.zeros_like(landmarks_sequence)
    acceleration[1:] = velocity[1:] - velocity[:-1]
    
    enhanced = np.concatenate([landmarks_sequence, velocity, acceleration], axis=1)
    return enhanced


def init_visual_embedder(backbone, device):
    if backbone == "none":
        return None, None, 0
    if backbone == "clip":
        try:
            import open_clip
        except ImportError as exc:
            raise ImportError(
                "open_clip_torch is required for CLIP embeddings. "
                "Install with: pip install open_clip_torch"
            ) from exc

        model, _, preprocess = open_clip.create_model_and_transforms(
            "ViT-B-32", pretrained="openai"
        )
        model = model.to(device)
        model.eval()
        return model, preprocess, 512

    raise ValueError(f"Unsupported visual backbone: {backbone}")


def compute_visual_embedding(frames, model, preprocess, device):
    if model is None or not frames:
        return None
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "Pillow is required for visual embeddings. Install with: pip install pillow"
        ) from exc

    images = [
        preprocess(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        for frame in frames
    ]
    batch = torch.stack(images, dim=0).to(device)

    with torch.no_grad():
        if hasattr(model, "encode_image"):
            feats = model.encode_image(batch)
        else:
            feats = model(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)

    pooled = feats.mean(dim=0).cpu().numpy().astype(np.float32)
    return pooled


def extract_video_clip(
    video_path,
    start_time,
    end_time,
    holistic,
    visual_model=None,
    visual_preprocess=None,
    visual_stride=4,
    visual_max_frames=256,
    device="cpu",
):
    """Extract landmarks and optional visual embedding from video segment."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    if end_frame - start_frame <= 0:
        cap.release()
        return None
    
    landmarks_sequence = []
    visual_frames = []

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        
        if not ret:
            break
        
        landmarks = extract_landmarks_from_frame(frame, holistic)
        landmarks_sequence.append(landmarks)

        if visual_model is not None and (frame_idx - start_frame) % visual_stride == 0:
            if len(visual_frames) < visual_max_frames:
                visual_frames.append(frame)
    
    cap.release()
    
    if len(landmarks_sequence) == 0:
        return None
    
    # Stack and add motion features
    landmarks_array = np.stack(landmarks_sequence, axis=0)  # [T, 543]
    enhanced_features = add_motion_features(landmarks_array)  # [T, 1629]

    # Optional visual embedding (pooled) repeated over time
    if visual_model is not None and visual_frames:
        visual_emb = compute_visual_embedding(
            visual_frames, visual_model, visual_preprocess, device
        )
        if visual_emb is not None:
            repeated = np.repeat(visual_emb[None, :], enhanced_features.shape[0], axis=0)
            enhanced_features = np.concatenate([enhanced_features, repeated], axis=1)
    
    return enhanced_features


def _clean_tokenize(sentence):
    """Lowercase, strip punctuation (keep apostrophes/hyphens), split."""
    s = str(sentence).lower()
    s = re.sub(r"[^a-z0-9'\- ]", ' ', s)
    return s.split()


def build_vocabulary(csv_paths, vocab_size):
    """Build vocabulary from CSVs."""
    word_counts = Counter()
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, sep='\t')
        for sentence in df['SENTENCE']:
            word_counts.update(_clean_tokenize(sentence))
    
    most_common = word_counts.most_common(vocab_size)
    idx_to_word = [word for word, _ in most_common]
    word_to_idx = {word: idx for idx, word in enumerate(idx_to_word)}
    
    return word_to_idx, idx_to_word


def _save_checkpoint_meta(checkpoint_file, successful, failed, next_idx, shard_index):
    """Save lightweight checkpoint metadata using atomic write."""
    import tempfile
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix='.tmp', dir=os.path.dirname(checkpoint_file) or '.')
        with os.fdopen(temp_fd, 'wb') as f:
            pickle.dump({
                'successful': successful,
                'failed': failed,
                'next_idx': next_idx,
                'shard_index': shard_index,
            }, f)
        os.replace(temp_path, checkpoint_file)
    except Exception as e:
        print(f"  Warning: Failed to save checkpoint: {e}")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


def _load_checkpoint_meta(checkpoint_file):
    with open(checkpoint_file, 'rb') as f:
        return pickle.load(f)


def _save_shard(shard_dir, shard_index, sequences, labels):
    """Save a shard of sequences and labels to disk."""
    if not sequences:
        return
    os.makedirs(shard_dir, exist_ok=True)
    shard_path = os.path.join(shard_dir, f"shard_{shard_index:06d}.npz")
    np.savez_compressed(
        shard_path,
        sequences=np.array(sequences, dtype=object),
        labels=np.array(labels, dtype=object),
    )


def _iter_shard_samples(shard_dir):
    """Yield (sequence, label) pairs from shard files without loading all into memory."""
    if not shard_dir or not os.path.exists(shard_dir):
        return
    shard_paths = sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))
    for shard_path in shard_paths:
        data = np.load(shard_path, allow_pickle=True)
        sequences = data["sequences"]
        labels = data["labels"]
        for seq, label in zip(sequences, labels):
            yield seq, label


def _iter_source_samples(source):
    """Yield (sequence, label) from either in-memory tuples or shard-backed source dict."""
    if isinstance(source, tuple):
        sequences, labels = source
        for seq, label in zip(sequences, labels):
            yield seq, label
        return

    shard_dir = source.get("shard_dir")
    for seq, label in _iter_shard_samples(shard_dir):
        yield seq, label


def _normalize_sample(seq, label, expected_feature_dim=None):
    """Convert sample payloads to numeric arrays and validate shape."""
    try:
        seq_arr = np.asarray(seq, dtype=np.float32)
        if seq_arr.ndim != 2:
            return None
        if expected_feature_dim is not None and int(seq_arr.shape[1]) != int(expected_feature_dim):
            return None

        label_arr = np.asarray(label, dtype=np.int32).reshape(-1)
        if label_arr.size == 0:
            return None

        return seq_arr, label_arr
    except (ValueError, TypeError):
        return None


def process_split(
    csv_path,
    video_dir,
    word_to_idx,
    holistic,
    visual_model=None,
    visual_preprocess=None,
    visual_stride=4,
    visual_max_frames=256,
    device="cpu",
    checkpoint_file=None,
    checkpoint_every=200,
):
    """Process split with MediaPipe features."""
    df = pd.read_csv(csv_path, sep='\t')
    
    sequences = []
    labels = []
    successful = 0
    failed = 0
    start_idx = 0
    shard_index = 0
    shard_dir = None

    if checkpoint_file and os.path.exists(checkpoint_file):
        print(f"  Loading checkpoint: {checkpoint_file}")
        try:
            checkpoint = _load_checkpoint_meta(checkpoint_file)
            successful = checkpoint['successful']
            failed = checkpoint['failed']
            start_idx = checkpoint['next_idx']
            shard_index = checkpoint.get('shard_index', 0)
            print(f"  Resuming from row {start_idx} (already: {successful} success, {failed} failed)")
        except (pickle.UnpicklingError, EOFError, KeyError) as e:
            print(f"  ✗ Checkpoint corrupted ({type(e).__name__}), starting from beginning...")
            try:
                os.remove(checkpoint_file)
                print(f"  Deleted corrupted checkpoint")
            except:
                pass
            sequences = []
            labels = []
            successful = 0
            failed = 0
            start_idx = 0
            shard_index = 0

    use_shards = checkpoint_file is not None
    if checkpoint_file:
        shard_dir = f"{os.path.splitext(checkpoint_file)[0]}_shards"

    df_to_process = df.iloc[start_idx:] if start_idx > 0 else df
    
    pbar = tqdm(df_to_process.iterrows(), total=len(df_to_process), desc=f"Processing {Path(csv_path).stem}")
    current_row = start_idx
    
    for _, row in pbar:
        video_name = row['VIDEO_NAME']
        start_time = row['START_REALIGNED']
        end_time = row['END_REALIGNED']
        sentence = row['SENTENCE']
        
        video_path = os.path.join(video_dir, f"{video_name}.mp4")
        
        # Extract landmarks
        features = extract_video_clip(
            video_path,
            start_time,
            end_time,
            holistic,
            visual_model=visual_model,
            visual_preprocess=visual_preprocess,
            visual_stride=visual_stride,
            visual_max_frames=visual_max_frames,
            device=device,
        )
        
        if features is None:
            failed += 1
            pbar.set_postfix({'success': successful, 'failed': failed})
            if checkpoint_file and (successful + failed) % checkpoint_every == 0:
                _save_shard(shard_dir, shard_index, sequences, labels)
                shard_index += 1
                sequences = []
                labels = []
                _save_checkpoint_meta(checkpoint_file, successful, failed, current_row + 1, shard_index)
            current_row += 1
            continue
        
        # Get word indices
        words = _clean_tokenize(sentence)
        word_indices = [word_to_idx[word] for word in words if word in word_to_idx]
        
        if len(word_indices) == 0:
            failed += 1
            pbar.set_postfix({'success': successful, 'failed': failed})
            if checkpoint_file and (successful + failed) % checkpoint_every == 0:
                _save_shard(shard_dir, shard_index, sequences, labels)
                shard_index += 1
                sequences = []
                labels = []
                _save_checkpoint_meta(checkpoint_file, successful, failed, current_row + 1, shard_index)
            current_row += 1
            continue
        
        sequences.append(features)
        labels.append(word_indices)
        successful += 1
        
        pbar.set_postfix({'success': successful, 'failed': failed})

        if checkpoint_file and (successful + failed) % checkpoint_every == 0:
            _save_shard(shard_dir, shard_index, sequences, labels)
            shard_index += 1
            sequences = []
            labels = []
            _save_checkpoint_meta(checkpoint_file, successful, failed, current_row + 1, shard_index)

        current_row += 1

    if checkpoint_file:
        _save_shard(shard_dir, shard_index, sequences, labels)
        shard_index += 1
        sequences = []
        labels = []
        _save_checkpoint_meta(checkpoint_file, successful, failed, start_idx + len(df_to_process), shard_index)

    if use_shards:
        return {
            "mode": "sharded",
            "shard_dir": shard_dir,
            "successful": successful,
            "failed": failed,
        }, successful, failed

    return (sequences, labels), successful, failed


def create_h5_dataset(output_path, train_data, val_data, test_data, idx_to_word, max_seq_len=2048):
    """Create HDF5 dataset with MediaPipe features."""
    with h5py.File(output_path, 'a') as f:
        f.attrs['num_classes'] = len(idx_to_word)

        feature_dim = None
        for source in [train_data, val_data, test_data]:
            for seq, _ in _iter_source_samples(source):
                normalized = _normalize_sample(seq, [0])
                if normalized is None:
                    continue
                feature_dim = int(normalized[0].shape[1])
                break
            if feature_dim is not None:
                break
        if feature_dim is None:
            feature_dim = 1629

        f.attrs['feature_dim'] = feature_dim
        f.attrs['feature_type'] = 'mediapipe_landmarks_with_motion'

        gloss_names = np.array(idx_to_word, dtype='S')
        if 'gloss_names' in f:
            del f['gloss_names']
        f.create_dataset('gloss_names', data=gloss_names)
        
        # Migrate old files: mark splits that are already fully written
        for sname in ['train', 'val', 'test']:
            ckey = f'{sname}_write_complete'
            slk = f'{sname}_sequence_lengths'
            llk = f'{sname}_label_lengths'
            if not f.attrs.get(ckey, False) and slk in f and llk in f:
                sl = f[slk][:]
                ll = f[llk][:]
                if sl.size > 0 and np.all(sl > 0) and np.all(ll > 0):
                    print(f"{sname}: existing data looks complete ({sl.size} samples, all lengths > 0), marking as done")
                    f.attrs[ckey] = True

        for split_name, source in [
            ('train', train_data),
            ('val', val_data),
            ('test', test_data)
        ]:
            num_samples = 0
            split_max_seq = 0
            max_label_len = 0
            dropped_samples = 0

            for seq, label in _iter_source_samples(source):
                normalized = _normalize_sample(seq, label, expected_feature_dim=feature_dim)
                if normalized is None:
                    dropped_samples += 1
                    continue
                seq_arr, label_arr = normalized
                num_samples += 1
                raw_len = int(seq_arr.shape[0])
                capped_len = min(raw_len, max_seq_len) if max_seq_len else raw_len
                split_max_seq = max(split_max_seq, capped_len)
                max_label_len = max(max_label_len, int(label_arr.size))

            if num_samples == 0:
                continue

            print(f"Preparing {split_name} datasets: {num_samples} samples, max_seq={split_max_seq}")
            if dropped_samples > 0:
                print(f"  Warning: dropped {dropped_samples} invalid {split_name} samples during HDF5 write")

            seq_key = f'{split_name}_sequences'
            seq_len_key = f'{split_name}_sequence_lengths'
            labels_key = f'{split_name}_labels'
            label_len_key = f'{split_name}_label_lengths'
            complete_key = f'{split_name}_write_complete'

            if f.attrs.get(complete_key, False):
                print(f"{split_name}: write already verified complete, skipping")
                continue

            for key in [seq_key, seq_len_key, labels_key, label_len_key]:
                if key in f:
                    del f[key]

            chunk_seq = min(split_max_seq, 512)
            sequences_ds = f.create_dataset(
                seq_key,
                shape=(num_samples, split_max_seq, feature_dim),
                dtype=np.float32,
                chunks=(1, chunk_seq, feature_dim),
                compression='gzip',
            )
            sequence_lengths_ds = f.create_dataset(
                seq_len_key,
                shape=(num_samples,),
                dtype=np.int32,
            )
            labels_ds = f.create_dataset(
                labels_key,
                shape=(num_samples, max_label_len),
                dtype=np.int32,
            )
            label_lengths_ds = f.create_dataset(
                label_len_key,
                shape=(num_samples,),
                dtype=np.int32,
            )

            sum_seq_len = 0
            write_iter = _iter_source_samples(source)
            i = 0
            for seq, label in tqdm(write_iter, total=num_samples + dropped_samples, desc=f"Writing {split_name} to HDF5"):
                normalized = _normalize_sample(seq, label, expected_feature_dim=feature_dim)
                if normalized is None:
                    continue
                seq_arr, label_arr = normalized

                seq_len = min(int(seq_arr.shape[0]), split_max_seq)
                seq_arr = seq_arr[:seq_len]
                label_len = int(label_arr.size)
                sequences_ds[i, :seq_len, :] = seq_arr
                labels_ds[i, :label_len] = label_arr
                sequence_lengths_ds[i] = seq_len
                label_lengths_ds[i] = label_len
                sum_seq_len += seq_len
                i += 1

            if i != num_samples:
                raise RuntimeError(
                    f"Unexpected write count mismatch for {split_name}: wrote {i}, expected {num_samples}"
                )

            f.attrs[complete_key] = True
            f.flush()
            print(f"{split_name}: {num_samples} samples, avg seq length: {sum_seq_len / max(num_samples, 1):.0f}")


def main():
    parser = argparse.ArgumentParser(description='Extract MediaPipe features from How2Sign')
    parser.add_argument('--video-dir', type=str,
                       default='/home/silass/Code/how-to-sign/raw_videos',
                       help='Directory containing videos')
    parser.add_argument('--csv-dir', type=str,
                       default='/home/silass/Code/how-to-sign',
                       help='Directory containing CSV files')
    parser.add_argument('--vocab-size', type=int, default=1000,
                       help='Vocabulary size')
    parser.add_argument('--output', type=str,
                       default='how2sign_mediapipe_{vocab_size}vocab.h5',
                       help='Output HDF5 file')
    parser.add_argument('--visual-backbone', type=str, default='none',
                       choices=['none', 'clip'],
                       help='Optional visual embedding backbone')
    parser.add_argument('--visual-stride', type=int, default=4,
                       help='Sample every N frames for visual embeddings')
    parser.add_argument('--visual-max-frames', type=int, default=256,
                       help='Cap number of frames used for visual embeddings')
    parser.add_argument('--max-seq-len', type=int, default=2048,
                       help='Cap sequence length in HDF5 (training default is 512; avoids massive padding from outliers)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from checkpoint files')
    parser.add_argument('--checkpoint-every', type=int, default=200,
                       help='Checkpoint every N samples (success + fail)')
    
    args = parser.parse_args()
    
    output_path = args.output.format(vocab_size=args.vocab_size)
    
    csv_files = {
        'train': os.path.join(args.csv_dir, 'how2sign_realigned_train.csv'),
        'val': os.path.join(args.csv_dir, 'how2sign_realigned_val.csv'),
        'test': os.path.join(args.csv_dir, 'how2sign_realigned_test.csv')
    }
    
    for split, path in csv_files.items():
        if not os.path.exists(path):
            print(f"ERROR: {path} not found")
            return
    
    print("="*70)
    print("MediaPipe Feature Extraction for How2Sign")
    print("="*70)
    print(f"Vocabulary size: {args.vocab_size}")
    print(f"Output: {output_path}")
    print()
    
    # Build vocabulary
    print("Building vocabulary...")
    word_to_idx, idx_to_word = build_vocabulary(csv_files.values(), args.vocab_size)
    print(f"Vocabulary: {idx_to_word[:20]}...")
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize MediaPipe
    print("Initializing MediaPipe holistic detector...")
    holistic = create_holistic_detector()

    visual_model, visual_preprocess, visual_dim = init_visual_embedder(
        args.visual_backbone, device
    )
    if visual_model is not None:
        print(f"Using visual backbone: {args.visual_backbone} (dim={visual_dim})")
    
    # Checkpoints
    checkpoint_dir = '.extraction_checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Clean up corrupted checkpoints
    if args.resume:
        for ckpt_name in [f'mediapipe_train_{args.vocab_size}.pkl', 
                          f'mediapipe_val_{args.vocab_size}.pkl', 
                          f'mediapipe_test_{args.vocab_size}.pkl']:
            ckpt_path = os.path.join(checkpoint_dir, ckpt_name)
            if os.path.exists(ckpt_path):
                try:
                    with open(ckpt_path, 'rb') as f:
                        pickle.load(f)
                    # Checkpoint is valid
                except (pickle.UnpicklingError, EOFError, KeyError):
                    print(f"Found corrupted checkpoint: {ckpt_path}, removing...")
                    os.remove(ckpt_path)
                    shard_dir = f"{os.path.splitext(ckpt_path)[0]}_shards"
                    if os.path.isdir(shard_dir):
                        shutil.rmtree(shard_dir)
    
    train_ckpt = os.path.join(checkpoint_dir, f'mediapipe_train_{args.vocab_size}.pkl') if args.resume else None
    val_ckpt = os.path.join(checkpoint_dir, f'mediapipe_val_{args.vocab_size}.pkl') if args.resume else None
    test_ckpt = os.path.join(checkpoint_dir, f'mediapipe_test_{args.vocab_size}.pkl') if args.resume else None

    # Process splits
    train_data, train_success, train_fail = process_split(
        csv_files['train'],
        args.video_dir,
        word_to_idx,
        holistic,
        visual_model=visual_model,
        visual_preprocess=visual_preprocess,
        visual_stride=args.visual_stride,
        visual_max_frames=args.visual_max_frames,
        device=device,
        checkpoint_file=train_ckpt,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Train: {train_success} successful, {train_fail} failed\n")
    
    val_data, val_success, val_fail = process_split(
        csv_files['val'],
        args.video_dir,
        word_to_idx,
        holistic,
        visual_model=visual_model,
        visual_preprocess=visual_preprocess,
        visual_stride=args.visual_stride,
        visual_max_frames=args.visual_max_frames,
        device=device,
        checkpoint_file=val_ckpt,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Val: {val_success} successful, {val_fail} failed\n")
    
    test_data, test_success, test_fail = process_split(
        csv_files['test'],
        args.video_dir,
        word_to_idx,
        holistic,
        visual_model=visual_model,
        visual_preprocess=visual_preprocess,
        visual_stride=args.visual_stride,
        visual_max_frames=args.visual_max_frames,
        device=device,
        checkpoint_file=test_ckpt,
        checkpoint_every=args.checkpoint_every,
    )
    print(f"Test: {test_success} successful, {test_fail} failed\n")
    
    # Create HDF5
    print(f"Creating HDF5: {output_path}")
    create_h5_dataset(
        output_path,
        train_data,
        val_data,
        test_data,
        idx_to_word,
        max_seq_len=args.max_seq_len,
    )
    
    print("\n" + "="*70)
    print("EXTRACTION COMPLETE!")
    print("="*70)
    print(f"Total: {train_success + val_success + test_success} samples")
    print(f"Output: {output_path}")
    print()

    if args.resume:
        for ckpt in [train_ckpt, val_ckpt, test_ckpt]:
            if ckpt and os.path.exists(ckpt):
                os.remove(ckpt)
            shard_dir = f"{os.path.splitext(ckpt)[0]}_shards" if ckpt else None
            if shard_dir and os.path.isdir(shard_dir):
                shutil.rmtree(shard_dir)
        print("Checkpoints cleaned up.")
    
    print("Next step (train on MediaPipe features):")
    print(f"  python train_transformer_encdec.py --data {output_path} \\")
    print("      --hidden-dim 512 --epochs 150 --batch-size 32 --lr 5e-5")


if __name__ == '__main__':
    main()
