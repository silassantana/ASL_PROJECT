#!/usr/bin/env python3
"""
Inference script using I3D features (matches training data).
"""

import argparse
import cv2
import numpy as np
import torch
import h5py
from pathlib import Path
from tqdm import tqdm

try:
    from torchvision.models.video import r3d_18, R3D_18_Weights
except ImportError:
    print("ERROR: Need torchvision >= 0.13")
    print("Install: pip install torchvision")
    exit(1)

from sign_transformer import SignLanguageTransformer


class I3DFeatureExtractor(torch.nn.Module):
    """Lightweight I3D extractor."""
    def __init__(self):
        super().__init__()
        weights = R3D_18_Weights.KINETICS400_V1
        self.model = r3d_18(weights=weights)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        
    def forward(self, video_clip):
        """
        Extract features from video clip.
        
        Args:
            video_clip: [batch, channels, frames, height, width]
        
        Returns:
            features: [batch, 512]
        """
        with torch.no_grad():
            features = self.model(video_clip)
        return features


def preprocess_clip_for_i3d(frames):
    """
    Preprocess frames for I3D model.
    
    Args:
        frames: [T, H, W, C] numpy array (uint8, BGR from OpenCV)
    
    Returns:
        tensor: [1, C, T, H, W] tensor (float32, normalized)
    """
    # Convert BGR to RGB
    frames = frames[:, :, :, ::-1]
    
    # Convert to float and normalize to [0, 1]
    frames = frames.astype(np.float32) / 255.0
    
    # Convert to tensor: [T, H, W, C] -> [C, T, H, W]
    frames = torch.from_numpy(frames).permute(3, 0, 1, 2)
    
    # Normalize (ImageNet stats)
    mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
    std = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
    frames = (frames - mean) / std
    
    # Add batch dimension: [1, C, T, H, W]
    frames = frames.unsqueeze(0)
    
    return frames


def extract_i3d_features_from_video(video_path, extractor, device, clip_size=16, stride=8, batch_size=4):
    """
    Extract I3D features from entire video with batched processing.
    
    Args:
        video_path: Path to video
        extractor: I3D feature extractor
        device: torch device
        clip_size: Number of frames per clip
        stride: Stride between clips
        batch_size: Process multiple clips at once
    
    Returns:
        features: [num_clips, 512] numpy array
    """
    # Load all frames into memory first (much faster than repeated seeks)
    print("Loading video frames into memory...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Read all frames
    all_frames = []
    for _ in tqdm(range(total_frames), desc="Reading frames"):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        all_frames.append(frame)
    cap.release()
    
    print(f"Loaded {len(all_frames)} frames")
    
    # Extract clips and features in batches
    all_features = []
    clips_to_process = []
    clip_indices = []
    
    print(f"Extracting I3D features (stride={stride})...")
    
    for start_frame in range(0, len(all_frames) - clip_size, stride):
        end_frame = start_frame + clip_size
        
        # Get frames for this clip
        frames = np.stack(all_frames[start_frame:end_frame], axis=0)
        
        # Preprocess
        clip_tensor = preprocess_clip_for_i3d(frames)
        clips_to_process.append(clip_tensor)
        clip_indices.append(start_frame)
        
        # Process batch
        if len(clips_to_process) == batch_size or start_frame + stride >= len(all_frames) - clip_size:
            # Stack clips into batch: [b, 1, C, T, H, W] -> [b, C, T, H, W]
            batch = torch.cat(clips_to_process, dim=0).to(device)
            
            with torch.no_grad():
                batch_features = extractor(batch)
            
            features = batch_features.cpu().numpy()
            all_features.extend(features)
            
            clips_to_process = []
            
            if len(all_features) % 50 == 0:
                print(f"  Processed {len(all_features)} clips...")
    
    if not all_features:
        return None
    
    features_array = np.stack(all_features, axis=0)
    return features_array


def load_model_and_vocab(checkpoint_path, data_path):
    """Load trained model and vocabulary."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    # Load vocabulary
    with h5py.File(data_path, "r") as f:
        gloss_names = f["gloss_names"][:]
        idx_to_gloss = {
            i: name.decode("utf-8") if isinstance(name, bytes) else name
            for i, name in enumerate(gloss_names)
        }
        num_classes = len(idx_to_gloss)
    
    # Create model
    model = SignLanguageTransformer(
        num_classes=num_classes,
        input_features=checkpoint.get("input_features", 512),
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_encoder_layers=checkpoint.get("num_encoder_layers", 4),
        num_decoder_layers=checkpoint.get("num_decoder_layers", 4),
        use_multimodal_fusion=bool(checkpoint.get("use_multimodal_fusion", False)),
        keypoint_dim=int(checkpoint.get("keypoint_dim", 1629)),
        clip_dim=int(checkpoint.get("clip_dim", 512)),
    )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, idx_to_gloss


def predict_video(model, video_features, idx_to_gloss, device="cuda", chunk_size=500, max_glosses=200, temperature=1.0):
    """Predict glosses from I3D features."""
    model = model.to(device)
    
    num_frames = video_features.shape[0]
    
    # If video is short, process normally
    if num_frames <= chunk_size:
        features = torch.FloatTensor(video_features).unsqueeze(0).to(device)
        
        with torch.no_grad():
            memory = model.encode(features)
            predictions = model.generate(memory, max_length=max_glosses)
        
        # Convert to glosses
        glosses = []
        for token_idx in predictions[0]:
            token_idx = token_idx.item()
            if token_idx == model.eos_idx:
                break
            if token_idx >= 3:
                gloss_idx = token_idx - 3
                if gloss_idx in idx_to_gloss:
                    glosses.append(idx_to_gloss[gloss_idx])
        
        return glosses
    
    # For long videos, process in chunks
    print(f"Long video detected ({num_frames} clips), processing in chunks of {chunk_size}...")
    
    all_glosses = []
    stride = chunk_size // 2
    
    for start_idx in range(0, num_frames, stride):
        end_idx = min(start_idx + chunk_size, num_frames)
        chunk = video_features[start_idx:end_idx]
        
        if chunk.shape[0] < 10:
            continue
        
        features = torch.FloatTensor(chunk).unsqueeze(0).to(device)
        
        with torch.no_grad():
            memory = model.encode(features)
            predictions = model.generate(memory, max_length=50)
        
        # Convert to glosses
        chunk_glosses = []
        for token_idx in predictions[0]:
            token_idx = token_idx.item()
            if token_idx == model.eos_idx:
                break
            if token_idx >= 3:
                gloss_idx = token_idx - 3
                if gloss_idx in idx_to_gloss:
                    chunk_glosses.append(idx_to_gloss[gloss_idx])
        
        # Deduplicate consecutive glosses within chunk
        if chunk_glosses:
            deduped = [chunk_glosses[0]]
            for gloss in chunk_glosses[1:]:
                if gloss != deduped[-1]:
                    deduped.append(gloss)
            chunk_glosses = deduped
        
        all_glosses.extend(chunk_glosses)
        
        if len(chunk_glosses) > 20:
            print(f"  Chunk {start_idx}-{end_idx}: {' '.join(chunk_glosses[:20])}... ({len(chunk_glosses)} total)")
        else:
            print(f"  Chunk {start_idx}-{end_idx}: {' '.join(chunk_glosses)}")
    
    # Global deduplication across chunks (remove consecutive duplicates)
    if all_glosses:
        final_glosses = [all_glosses[0]]
        for gloss in all_glosses[1:]:
            if gloss != final_glosses[-1]:
                final_glosses.append(gloss)
        all_glosses = final_glosses
    
    return all_glosses


def main():
    parser = argparse.ArgumentParser(description='Inference with I3D features (matches training)')
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=str, default="how2sign_i3d_200vocab.h5", help="Path to data file (for vocab)")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--clip-size", type=int, default=16, help="Frames per I3D clip")
    parser.add_argument("--stride", type=int, default=16, help="Stride between I3D clips (larger = faster)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for I3D extraction")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunks for long videos")
    parser.add_argument("--max-glosses", type=int, default=300, help="Maximum glosses to generate (increase if predictions are too short)")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("VIDEO INFERENCE - I3D FEATURES (Matches Training Data)")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print()
    
    # Extract I3D features
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    extractor = I3DFeatureExtractor().to(device)
    
    video_features = extract_i3d_features_from_video(
        args.video, extractor, device, args.clip_size, args.stride, args.batch_size
    )
    
    if video_features is None:
        print("ERROR: Failed to extract features from video")
        return
    
    print(f"Video features shape: {video_features.shape}")
    print(f"  - Clips: {video_features.shape[0]}")
    print(f"  - Features per clip: {video_features.shape[1]} (I3D)")
    print()
    
    # Load model
    print("Loading model...")
    model, idx_to_gloss = load_model_and_vocab(args.checkpoint, args.data)
    print(f"Model loaded with {len(idx_to_gloss)} classes")
    print(f"Vocabulary: {list(idx_to_gloss.values())[:20]}...")
    print()
    
    # Predict
    print("Predicting glosses...")
    glosses = predict_video(model, video_features, idx_to_gloss, args.device, args.chunk_size, args.max_glosses)
    
    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"Predicted glosses: {' '.join(glosses)}")
    print(f"Number of glosses: {len(glosses)}")
    print()
    
    # Analysis
    if len(glosses) < 10:
        print("⚠️  WARNING: Predictions are very short!")
        print("   This indicates the model is undertrained and predicts EOS too early.")
        print("   To improve:")
        print("   1. Train for more epochs (current: 50, recommended: 150-200)")
        print("   2. Use larger model (hidden_dim=512, more layers)")
        print("   3. Run inference with --max-glosses higher (current: {})".format(args.max_glosses))
        print()
        print("   Quick train command:")
        print("   python train_transformer_encdec.py --data how2sign_i3d_200vocab.h5 \\")
        print("     --hidden-dim 512 --epochs 200 --batch-size 32 --lr 5e-5 --resume")
        print()


if __name__ == "__main__":
    main()
