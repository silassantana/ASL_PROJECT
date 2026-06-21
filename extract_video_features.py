#!/usr/bin/env python3
"""
Extract pretrained video features from How2Sign using VideoMAE or I3D
These features will have much higher discriminative power than raw landmarks
"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import json
from pathlib import Path
from tqdm import tqdm
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEModel
import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

class VideoFeatureExtractor:
    """Extract features using pretrained VideoMAE"""
    
    def __init__(self, model_name="MCG-NJU/videomae-base"):
        print(f"Loading pretrained model: {model_name}")
        self.processor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEModel.from_pretrained(model_name).to(device)
        self.model.eval()
        
        # VideoMAE outputs 768-dim features (base model)
        self.feature_dim = 768
        
    @torch.no_grad()
    def extract_features(self, video_frames):
        """
        Extract features from video frames
        
        Args:
            video_frames: numpy array [num_frames, H, W, C] in RGB
            
        Returns:
            features: numpy array [num_frames, 768]
        """
        # VideoMAE expects 16-frame clips
        num_frames = len(video_frames)
        clip_size = 16
        
        all_features = []
        
        # Process in overlapping windows
        stride = 8  # 50% overlap
        for start_idx in range(0, num_frames, stride):
            end_idx = start_idx + clip_size
            
            if end_idx > num_frames:
                # Pad last clip
                clip = np.zeros((clip_size, *video_frames.shape[1:]), dtype=np.uint8)
                available = num_frames - start_idx
                clip[:available] = video_frames[start_idx:]
            else:
                clip = video_frames[start_idx:end_idx]
            
            # Preprocess
            inputs = self.processor(list(clip), return_tensors="pt")
            pixel_values = inputs['pixel_values'].to(device)
            
            # Extract features
            outputs = self.model(pixel_values)
            # Take mean over spatial tokens (not CLS token)
            features = outputs.last_hidden_state[:, 1:, :].mean(dim=1)  # [1, 768]
            
            all_features.append(features.cpu().numpy()[0])
        
        # Interpolate to match original frame count
        all_features = np.array(all_features)
        
        if len(all_features) != num_frames:
            # Linearly interpolate to match frame count
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, num_frames-1, len(all_features))
            x_new = np.arange(num_frames)
            
            interpolator = interp1d(x_old, all_features, axis=0, kind='linear', 
                                   fill_value='extrapolate')
            all_features = interpolator(x_new)
        
        return all_features

def load_video_frames(video_path, max_frames=250):
    """Load video frames from file"""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Sample uniformly if video is too long
    if frame_count > max_frames:
        indices = np.linspace(0, frame_count-1, max_frames, dtype=int)
    else:
        indices = range(frame_count)
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    
    cap.release()
    return np.array(frames)

def create_video_feature_dataset(
    video_dir,
    mapping_file,
    output_h5,
    max_frames=250
):
    """
    Create dataset with pretrained video features
    
    Args:
        video_dir: Directory containing How2Sign videos
        mapping_file: JSON file with video->gloss mappings
        output_h5: Output HDF5 file path
        max_frames: Maximum frames per video
    """
    
    print("=" * 80)
    print("VIDEO FEATURE EXTRACTION")
    print("=" * 80)
    
    # Load mappings
    with open(mapping_file, 'r') as f:
        data = json.load(f)
    
    # Initialize feature extractor
    extractor = VideoFeatureExtractor()
    
    print(f"\nFeature dimension: {extractor.feature_dim}")
    print(f"Processing videos from: {video_dir}")
    
    # Process videos
    video_features = []
    video_labels = []
    video_lengths = []
    label_lengths = []
    
    video_dir = Path(video_dir)
    
    for item in tqdm(data['samples'], desc="Extracting features"):
        video_file = video_dir / item['video_file']
        
        if not video_file.exists():
            print(f"Warning: Video not found: {video_file}")
            continue
        
        # Load video
        frames = load_video_frames(video_file, max_frames)
        
        if len(frames) == 0:
            continue
        
        # Extract features
        features = extractor.extract_features(frames)
        
        # Get labels
        labels = item['labels']
        
        video_features.append(features)
        video_labels.append(labels)
        video_lengths.append(len(features))
        label_lengths.append(len(labels))
    
    print(f"\nProcessed {len(video_features)} videos")
    
    # Find max lengths
    max_seq_len = max(video_lengths)
    max_lbl_len = max(label_lengths)
    
    print(f"Max sequence length: {max_seq_len}")
    print(f"Max label length: {max_lbl_len}")
    
    # Create HDF5 dataset
    print(f"\nWriting to {output_h5}...")
    
    with h5py.File(output_h5, 'w') as f:
        n = len(video_features)
        
        # Create datasets
        seqs = f.create_dataset(
            'sequences',
            shape=(n, max_seq_len, extractor.feature_dim),
            dtype='float32',
            compression='gzip'
        )
        lbls = f.create_dataset(
            'labels',
            shape=(n, max_lbl_len),
            dtype='int32'
        )
        seq_lens = f.create_dataset(
            'sequence_lengths',
            shape=(n,),
            dtype='int32',
            data=video_lengths
        )
        lbl_lens = f.create_dataset(
            'label_lengths',
            shape=(n,),
            dtype='int32',
            data=label_lengths
        )
        
        # Fill data
        for i, (features, labels) in enumerate(zip(video_features, video_labels)):
            seq_len = len(features)
            lbl_len = len(labels)
            
            # Pad sequences
            padded_seq = np.zeros((max_seq_len, extractor.feature_dim), dtype='float32')
            padded_seq[:seq_len] = features
            seqs[i] = padded_seq
            
            # Pad labels
            padded_lbl = np.zeros(max_lbl_len, dtype='int32')
            padded_lbl[:lbl_len] = labels
            lbls[i] = padded_lbl
    
    print(f"✓ Dataset saved!")

def convert_landmark_dataset_to_video_features(
    input_h5,
    output_h5,
    video_dir=None
):
    """
    Alternative: If we don't have raw videos, simulate better features
    by using a pretrained image model on reconstructed pose visualizations
    """
    print("=" * 80)
    print("ALTERNATIVE: SIMULATED VIDEO FEATURES")
    print("=" * 80)
    print("\nNote: This uses pose visualization + pretrained CNN")
    print("Real video features would be better, but this is a fallback\n")
    
    # Use ResNet50 as feature extractor
    from torchvision import models, transforms
    
    resnet = models.resnet50(pretrained=True)
    # Remove final FC layer
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("Loading landmark dataset...")
    with h5py.File(input_h5, 'r') as f_in:
        # Get metadata
        num_classes = int(f_in['num_classes'][()].item())
        
        print(f"Creating enhanced features...")
        
        with h5py.File(output_h5, 'w') as f_out:
            f_out.create_dataset('num_classes', data=num_classes)
            
            for split in ['train', 'val']:
                sequences = f_in[f'{split}_sequences']
                labels = f_in[f'{split}_labels'][:]
                label_lengths = f_in[f'{split}_label_lengths'][:]
                sequence_lengths = f_in[f'{split}_sequence_lengths'][:]
                
                n = len(sequences)
                max_len = sequences.shape[1]
                
                # Create output datasets (2048-dim ResNet features)
                out_seqs = f_out.create_dataset(
                    f'{split}_sequences',
                    shape=(n, max_len, 2048),
                    dtype='float32',
                    compression='gzip'
                )
                
                print(f"\nProcessing {split} set ({n} samples)...")
                
                for i in tqdm(range(n)):
                    seq = sequences[i]
                    seq_len = int(sequence_lengths[i])
                    
                    # Extract features per frame
                    frame_features = []
                    
                    for t in range(seq_len):
                        # Reshape landmarks to 2D image-like structure
                        landmarks = seq[t].reshape(-1, 3)  # [150 landmarks, 3 coords]
                        
                        # Create simple pose visualization (150x3 -> 224x224 image)
                        img = create_pose_image(landmarks)
                        
                        # Extract ResNet features
                        with torch.no_grad():
                            img_tensor = transform(img).unsqueeze(0).to(device)
                            features = resnet(img_tensor).squeeze().cpu().numpy()
                        
                        frame_features.append(features)
                    
                    # Pad to max length
                    padded = np.zeros((max_len, 2048), dtype='float32')
                    padded[:seq_len] = np.array(frame_features)
                    out_seqs[i] = padded
                
                # Copy other datasets unchanged
                f_out.create_dataset(f'{split}_labels', data=labels)
                f_out.create_dataset(f'{split}_label_lengths', data=label_lengths)
                f_out.create_dataset(f'{split}_sequence_lengths', data=sequence_lengths)
    
    print(f"\n✓ Enhanced dataset saved: {output_h5}")
    print(f"  Feature dimension: 450 → 2048 (ResNet50)")

def create_pose_image(landmarks, img_size=224):
    """Create a simple visualization of pose landmarks"""
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    
    # Normalize landmarks to image coordinates
    if landmarks.shape[1] >= 2:
        x = landmarks[:, 0]
        y = landmarks[:, 1]
        
        # Scale to image size
        x_norm = ((x - x.min()) / (x.max() - x.min() + 1e-8) * (img_size - 20) + 10).astype(int)
        y_norm = ((y - y.min()) / (y.max() - y.min() + 1e-8) * (img_size - 20) + 10).astype(int)
        
        # Draw points
        for xi, yi in zip(x_norm, y_norm):
            cv2.circle(img, (xi, yi), 2, (255, 255, 255), -1)
        
        # Draw connections (simple skeleton)
        # Left hand (0-20), right hand (21-41), body/face (42+)
        for i in range(len(landmarks) - 1):
            if i == 20 or i == 41:  # Skip between hand/body boundaries
                continue
            cv2.line(img, (x_norm[i], y_norm[i]), (x_norm[i+1], y_norm[i+1]), 
                    (128, 128, 128), 1)
    
    return img

if __name__ == "__main__":
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract pretrained video features")
    parser.add_argument('--input', type=str, required=True,
                       help='Input H5 file with landmark features')
    parser.add_argument('--output', type=str, required=True,
                       help='Output H5 file for video features')
    parser.add_argument('--video-dir', type=str, default=None,
                       help='Directory with raw video files (if available)')
    
    args = parser.parse_args()
    
    try:
        # Try installing required packages
        import scipy
    except ImportError:
        print("Installing scipy...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "scipy"])
    
    if args.video_dir and Path(args.video_dir).exists():
        print("Using real video features (VideoMAE)")
        # Would need video files and metadata
        print("ERROR: Video-based extraction not yet implemented.")
        print("Using pose visualization method instead...\n")
        convert_landmark_dataset_to_video_features(args.input, args.output)
    else:
        print("Using pose visualization + ResNet50")
        convert_landmark_dataset_to_video_features(args.input, args.output)
