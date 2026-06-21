#!/usr/bin/env python3
"""
inference_improved.py - Better smoothing and confidence handling
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import mediapipe as mp
import argparse
import time
import sys
from collections import deque, Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Simple model matching training
class SimpleModel(torch.nn.Module):
    def __init__(self, num_classes, input_dim=150):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv1d(input_dim, 128, 3, padding="same")
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.conv2 = torch.nn.Conv1d(128, 256, 3, padding="same")
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.gru = torch.nn.GRU(
            256, 128, 2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.fc = torch.nn.Linear(256, num_classes + 1)

    def forward(self, x):
        B, T, C = x.shape
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        logits = self.fc(x)
        return logits


#!/usr/bin/env python3
"""
inference_improved.py - Better smoothing and confidence handling
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import mediapipe as mp
import argparse
import time
import sys
from collections import deque, Counter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Simple model matching training
class SimpleModel(torch.nn.Module):
    def __init__(self, num_classes, input_dim=150):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = torch.nn.Conv1d(input_dim, 128, 3, padding="same")
        self.bn1 = torch.nn.BatchNorm1d(128)
        self.conv2 = torch.nn.Conv1d(128, 256, 3, padding="same")
        self.bn2 = torch.nn.BatchNorm1d(256)
        self.gru = torch.nn.GRU(
            256, 128, 2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.fc = torch.nn.Linear(256, num_classes + 1)

    def forward(self, x):
        B, T, C = x.shape
        x = x.transpose(1, 2)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        logits = self.fc(x)
        return logits


class SimpleGreedyDecoder:
    def __init__(self, idx_to_gloss, blank_idx):
        self.idx_to_gloss = idx_to_gloss
        self.blank_idx = blank_idx
        self.output_sequence = []  # Fixed: Changed from self.output
        self.last_pred = None

    def update(self, frame_idx, pred_idx, confidence):
        if pred_idx != self.blank_idx and pred_idx != self.last_pred:
            gloss = self.idx_to_gloss.get(pred_idx, f"UNK_{pred_idx}")
            self.output_sequence.append(f"{gloss}({confidence:.2f})")
            print(f"Frame {frame_idx}: {gloss} ({confidence:.3f})")
            self.last_pred = pred_idx

    def finalize(self):
        result = " ".join(self.output_sequence)
        self.output_sequence = []
        self.last_pred = None
        return result


# In inference_improved.py, update to match the new data format:


def extract_landmarks_realistic(frame, holistic):
    """Extract landmarks to match realistic_patterns_v2.h5 format"""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(frame_rgb)

    landmarks = []

    # We need 75 points × 2 coordinates = 150 features

    # 1. Pose landmarks (25 points)
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark[:25]:  # First 25 pose points
            landmarks.extend([lm.x, lm.y])  # Only x, y (no z)
    else:
        landmarks.extend([0.0] * 50)  # 25 × 2 = 50

    # 2. Hand landmarks (50 points total = 25 per hand)
    # Left hand
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark[
            :25
        ]:  # All 21 hand points + 4 padding
            landmarks.extend([lm.x, lm.y])
    else:
        landmarks.extend([0.0] * 50)  # 25 × 2 = 50

    # Right hand
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark[:25]:
            landmarks.extend([lm.x, lm.y])
    else:
        landmarks.extend([0.0] * 50)  # 25 × 2 = 50

    # Should have exactly 150 features: (25 pose + 25 left + 25 right) × 2
    assert len(landmarks) == 150, f"Expected 150 features, got {len(landmarks)}"

    return np.array(landmarks[:150], dtype=np.float32).reshape(
        75, 2
    )  # Reshape to match HDF5


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", help="Input video")
    parser.add_argument("--checkpoint", default="strong_fixed.pt")
    parser.add_argument("--output", default="asl_output.txt")
    parser.add_argument("--no-display", action="store_true")
    args = parser.parse_args()

    # Load model

    try:
        # First try with weights_only=True (safer)
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    except:
        # Fall back to weights_only=False if needed
        checkpoint = torch.load(
            args.checkpoint, map_location=device, weights_only=False
        )

    model = SimpleModel(checkpoint["num_classes"], input_dim=150).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    model.eval()

    idx_to_gloss = checkpoint["idx_to_gloss"]
    blank_idx = checkpoint["blank_idx"]

    print(f"Model loaded with {len(idx_to_gloss)} glosses")
    print(f"Top 10 glosses: {list(idx_to_gloss.values())[:10]}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Initialize MediaPipe
    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize decoder
    decoder = SimpleGreedyDecoder(idx_to_gloss, blank_idx)

    # Buffer for sequence (30 frames like training)
    sequence_buffer = []
    sequence_length = 30

    frame_count = 0
    start_time = time.time()

    print(f"\nProcessing video...")
    print(f"FPS: {fps}, Total frames: {total_frames}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process every 2nd frame
            if frame_count % 2 == 0:
                landmarks = extract_landmarks_realistic(frame, holistic)
                sequence_buffer.append(landmarks)

                # Keep buffer size
                if len(sequence_buffer) > sequence_length:
                    sequence_buffer.pop(0)

                # Make prediction when buffer is full
                if len(sequence_buffer) == sequence_length:
                    # Prepare input
                    seq_array = np.array(sequence_buffer)

                    # Normalize like training
                    seq_mean = seq_array.mean()
                    seq_std = seq_array.std() + 1e-8
                    seq_norm = (seq_array - seq_mean) / seq_std

                    # Reshape to [1, 30, 150]
                    input_tensor = (
                        torch.from_numpy(seq_norm).float().to(device).unsqueeze(0)
                    )

                    # Predict
                    with torch.no_grad():
                        logits = model(input_tensor)
                        probs = F.softmax(logits, dim=-1)

                        # Get prediction for last timestep
                        last_probs = probs[0, -1, :]
                        pred_idx = last_probs.argmax().item()
                        confidence = last_probs.max().item()

                        # Update decoder
                        decoder.update(frame_count, pred_idx, confidence)

            # Display (optional)
            if not args.no_display and frame_count % 10 == 0:
                # Simple display
                display_frame = cv2.resize(frame, (640, 360))
                cv2.putText(
                    display_frame,
                    f"Frame: {frame_count}/{total_frames}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                current_output = " ".join(decoder.output_sequence[-3:])  # Last 3 words
                cv2.putText(
                    display_frame,
                    f"Output: {current_output}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow("ASL Recognition", display_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_count += 1

            # Progress
            if frame_count % 100 == 0:
                elapsed = time.time() - start_time
                print(
                    f"  Processed {frame_count}/{total_frames} ({frame_count / elapsed:.1f} fps)"
                )

    except KeyboardInterrupt:
        print("\nInterrupted")

    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()

    # Get final result
    final_phrase = decoder.finalize()

    print(f"\n{'=' * 60}")
    print(f"FINAL PHRASE:")
    print(f"{final_phrase}")
    print(f"{'=' * 60}")

    # Save output
    with open(args.output, "w") as f:
        f.write(final_phrase)
    print(f"Saved to: {args.output}")

    return final_phrase


if __name__ == "__main__":
    main()
