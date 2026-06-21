#!/usr/bin/env python3
"""
inference_isolated_temporal.py - Inference for isolated sign classification
"""

import torch
import torch.nn.functional as F
import cv2
import numpy as np
import mediapipe as mp
import json
import argparse

from train_isolated_temporal import TemporalSignClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class IsolatedSignRecognizer:
    """Recognizer for isolated signs using temporal model"""
    
    def __init__(self, checkpoint_path):
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location=device)
        self.num_classes = checkpoint['num_classes']
        self.idx_to_gloss = checkpoint['idx_to_gloss']
        
        self.model = TemporalSignClassifier(self.num_classes).to(device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        print(f"Loaded model with {self.num_classes} classes")
        print(f"Model accuracy: {checkpoint['val_acc']:.2%}")
        
        # MediaPipe
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    
    def extract_frame_features(self, frame):
        """Extract 150 features from a frame"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.holistic.process(frame_rgb)
        
        features = []
        
        # Pose (25 * 3 = 75)
        if results.pose_landmarks:
            for lm in results.pose_landmarks.landmark[:25]:
                features.extend([lm.x, lm.y, lm.z])
        else:
            features.extend([0.0] * 75)
        
        # Hands (25 * 3 = 75)
        hand_points = []
        if results.left_hand_landmarks:
            hand_points.extend(list(results.left_hand_landmarks.landmark[:13]))
        else:
            hand_points.extend([type('obj', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()] * 13)
        
        if results.right_hand_landmarks:
            hand_points.extend(list(results.right_hand_landmarks.landmark[:12]))
        else:
            hand_points.extend([type('obj', (), {'x': 0.0, 'y': 0.0, 'z': 0.0})()] * 12)
        
        for lm in hand_points:
            features.extend([lm.x, lm.y, lm.z])
        
        return np.array(features[:150], dtype=np.float32)
    
    def extract_video_features(self, video_path, max_frames=200):
        """Extract features from entire video"""
        cap = cv2.VideoCapture(video_path)
        features = []
        
        print(f"\nExtracting features from {video_path}...")
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_count >= max_frames):
                break
            
            frame_features = self.extract_frame_features(frame)
            features.append(frame_features)
            frame_count += 1
            
            if frame_count % 50 == 0:
                print(f"  Processed {frame_count} frames...")
        
        cap.release()
        print(f"  Extracted {len(features)} frames")
        
        return np.array(features, dtype=np.float32)
    
    def predict(self, video_path, top_k=5):
        """
        Predict sign from video
        
        Returns:
            predictions: List of (gloss, confidence) tuples
        """
        # Extract features
        features = self.extract_video_features(video_path)
        
        if len(features) == 0:
            return []
        
        # Normalize
        features = torch.FloatTensor(features)
        mean = features.mean()
        std = features.std() + 1e-8
        features = (features - mean) / std
        
        # Add batch dimension
        features = features.unsqueeze(0).to(device)
        seq_length = torch.tensor([len(features[0])], dtype=torch.long).to(device)
        
        # Predict
        with torch.no_grad():
            logits = self.model(features, seq_length)
            probs = F.softmax(logits, dim=-1)
        
        # Get top-k predictions
        topk_probs, topk_indices = probs[0].topk(top_k)
        
        predictions = []
        for prob, idx in zip(topk_probs, topk_indices):
            gloss = self.idx_to_gloss.get(idx.item(), f"UNK_{idx.item()}")
            predictions.append((gloss, prob.item()))
        
        return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('video', help='Input video file')
    parser.add_argument('--checkpoint', default='isolated_temporal_best.pt',
                        help='Model checkpoint')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Show top-k predictions')
    args = parser.parse_args()
    
    print("=" * 70)
    print("ISOLATED SIGN RECOGNITION")
    print("=" * 70)
    
    # Create recognizer
    recognizer = IsolatedSignRecognizer(args.checkpoint)
    
    # Predict
    predictions = recognizer.predict(args.video, top_k=args.top_k)
    
    print("\n" + "=" * 70)
    print("PREDICTIONS")
    print("=" * 70)
    
    if predictions:
        print(f"\n🏆 Top prediction: {predictions[0][0]} ({predictions[0][1]:.1%} confidence)\n")
        
        print(f"Top {len(predictions)} predictions:")
        for i, (gloss, conf) in enumerate(predictions, 1):
            bar_length = int(conf * 40)
            bar = "█" * bar_length + "░" * (40 - bar_length)
            print(f"  {i}. {gloss:25s} {bar} {conf:6.1%}")
    else:
        print("\n❌ No predictions (could not extract features)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
