#!/usr/bin/env python3
"""
Inference script for encoder-decoder transformer on video.
"""

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import h5py
from sign_transformer import SignLanguageTransformer


def create_holistic_detector():
    """Create a holistic pose detector using MediaPipe tasks API."""
    try:
        import mediapipe as mp
        from mediapipe import solutions
        
        # Try to use old solutions API if available
        class OldStyleHolistic:
            def __init__(self):
                self.holistic = mp.solutions.holistic.Holistic(
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
                import mediapipe as mp
                from mediapipe import solutions
                
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

    # Pose (33 landmarks × 3 = 99)
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

    # Face (106 key points × 3 = 318)
    if results.face_landmarks:
        key_indices = [
            61,
            185,
            40,
            39,
            37,
            0,
            267,
            269,
            270,
            409,
            291,
            78,
            191,
            80,
            81,
            82,
            13,
            312,
            311,
            310,
            415,
            308,
            33,
            160,
            158,
            133,
            153,
            144,
            263,
            387,
            385,
            362,
            380,
            373,
            70,
            63,
            105,
            66,
            107,
            300,
            293,
            334,
            296,
            336,
            1,
            2,
            98,
            327,
            172,
            136,
            150,
            149,
            176,
            148,
            152,
            377,
            400,
            378,
            379,
            365,
            397,
            288,
            361,
            323,
            454,
            356,
            389,
        ]
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


def extract_video_features(
    video_path,
    max_frames=None,
    start_sec=None,
    end_sec=None,
    visual_model=None,
    visual_preprocess=None,
    visual_stride=4,
    visual_max_frames=256,
    device="cpu",
):
    """
    Extract landmarks from entire video.

    Returns:
        features: [T, 1629] array of landmarks with motion
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    if start_sec is not None:
        start_frame = max(0, int(start_sec * fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        start_frame = 0

    if end_sec is not None:
        end_frame = max(start_frame + 1, int(end_sec * fps))
    else:
        end_frame = None

    holistic = create_holistic_detector()

    landmarks_list = []
    visual_frames = []
    frame_count = 0

    print(f"Extracting landmarks from video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        abs_frame = start_frame + frame_count
        if end_frame is not None and abs_frame >= end_frame:
            break

        if max_frames and frame_count >= max_frames:
            break

        landmarks = extract_landmarks_from_frame(frame, holistic)
        landmarks_list.append(landmarks)

        if visual_model is not None and frame_count % visual_stride == 0:
            if len(visual_frames) < visual_max_frames:
                visual_frames.append(frame)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"  Processed {frame_count} frames...")

    cap.release()
    holistic.close()

    if len(landmarks_list) == 0:
        raise ValueError("No landmarks extracted from video")

    print(f"Extracted {len(landmarks_list)} frames")

    # Stack and add motion
    landmarks_array = np.stack(landmarks_list, axis=0)  # [T, 543]
    enhanced = add_motion_features(landmarks_array)  # [T, 1629]

    # Optional visual embedding (pooled) repeated over time
    if visual_model is not None and visual_frames:
        visual_emb = compute_visual_embedding(
            visual_frames, visual_model, visual_preprocess, device
        )
        if visual_emb is not None:
            repeated = np.repeat(visual_emb[None, :], enhanced.shape[0], axis=0)
            enhanced = np.concatenate([enhanced, repeated], axis=1)

    return enhanced


def load_model_and_vocab(checkpoint_path, data_path):
    """Load trained model and vocabulary."""
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Load vocabulary from data file
    with h5py.File(data_path, "r") as f:
        gloss_names = f["gloss_names"][:]
        idx_to_gloss = {
            i: name.decode("utf-8") if isinstance(name, bytes) else name
            for i, name in enumerate(gloss_names)
        }
        num_classes = len(idx_to_gloss)

    expected_input_dim = checkpoint.get("input_features", 512)
    use_channel_attention = checkpoint.get("use_channel_attention", False)
    attention_reduction = checkpoint.get("attention_reduction", 8)
    use_multimodal_fusion = bool(checkpoint.get("use_multimodal_fusion", False))
    keypoint_dim = int(checkpoint.get("keypoint_dim", 1629))
    clip_dim = int(checkpoint.get("clip_dim", 512))

    # Create model
    model = SignLanguageTransformer(
        num_classes=num_classes,
        input_features=expected_input_dim,
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_encoder_layers=checkpoint.get("num_encoder_layers", 4),
        num_decoder_layers=checkpoint.get("num_decoder_layers", 4),
        use_channel_attention=use_channel_attention,
        attention_reduction=attention_reduction,
        use_multimodal_fusion=use_multimodal_fusion,
        keypoint_dim=keypoint_dim,
        clip_dim=clip_dim,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, idx_to_gloss, expected_input_dim


def _beam_search_single(
    model,
    memory,
    max_length=200,
    beam_size=4,
    length_penalty=0.7,
    repetition_penalty=1.2,
    memory_key_padding_mask=None,
):
    """Beam search for one encoded sample. Returns token ids without SOS."""
    device = memory.device
    sos, eos, pad = model.sos_idx, model.eos_idx, model.pad_idx

    beams = [([sos], 0.0, False)]  # (tokens, sum_logprob, finished)

    def norm_score(sum_logprob, tok_len):
        lp = ((5.0 + tok_len) / 6.0) ** max(length_penalty, 0.0)
        return sum_logprob / lp

    for _ in range(max_length):
        all_candidates = []
        all_finished = True

        for toks, score, finished in beams:
            if finished:
                all_candidates.append((toks, score, True))
                continue

            all_finished = False
            tgt = torch.tensor([toks], dtype=torch.long, device=device)
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            logits = model.decode_step(
                memory,
                tgt,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            next_logits = logits[:, -1, :].squeeze(0)

            if repetition_penalty != 1.0:
                prev = torch.tensor(sorted(set(toks)), dtype=torch.long, device=device)
                pos = next_logits[prev] > 0
                next_logits[prev[pos]] /= repetition_penalty
                next_logits[prev[~pos]] *= repetition_penalty

            log_probs = F.log_softmax(next_logits, dim=-1)
            topk_logp, topk_idx = torch.topk(log_probs, k=min(beam_size, log_probs.numel()))

            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                ntoks = toks + [int(idx)]
                nscore = score + float(lp)
                nfinished = (int(idx) == eos)
                all_candidates.append((ntoks, nscore, nfinished))

        if all_finished:
            break

        all_candidates.sort(key=lambda x: norm_score(x[1], len(x[0])), reverse=True)
        beams = all_candidates[:beam_size]

    best = max(beams, key=lambda x: norm_score(x[1], len(x[0])))[0]
    out = []
    for tok in best[1:]:  # drop SOS
        if tok == eos:
            break
        if tok != pad:
            out.append(tok)
    return out


def predict_video(
    model,
    video_features,
    idx_to_gloss,
    device="cuda",
    chunk_size=500,
    max_glosses=200,
    decode="greedy",
    beam_size=4,
    length_penalty=0.7,
    repetition_penalty=1.2,
    chunk_overlap=0.5,
):
    """
    Predict glosses from video features with chunking for long videos.

    Args:
        chunk_size: Process video in chunks of this many frames
    Returns:
        glosses: list of predicted gloss strings
    """
    model = model.to(device)

    num_frames = video_features.shape[0]

    # If video is short, process normally
    if num_frames <= chunk_size:
        features = torch.FloatTensor(video_features).unsqueeze(0).to(device)

        with torch.no_grad():
            memory = model.encode(features)
            if decode == "beam":
                pred_tokens = _beam_search_single(
                    model,
                    memory,
                    max_length=max_glosses,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                )
                predictions = torch.tensor([pred_tokens], dtype=torch.long, device=memory.device)
            else:
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

    # For long videos, process in chunks with overlap.
    print(
        f"Long video detected ({num_frames} frames), processing in chunks of {chunk_size}..."
    )

    def _merge_with_overlap(base, nxt, max_overlap=20):
        if not base:
            return list(nxt)
        max_k = min(max_overlap, len(base), len(nxt))
        overlap = 0
        for k in range(max_k, 0, -1):
            if base[-k:] == nxt[:k]:
                overlap = k
                break
        return base + nxt[overlap:]

    all_glosses = []
    overlap = float(np.clip(chunk_overlap, 0.0, 0.95))
    stride = max(1, int(chunk_size * (1.0 - overlap)))

    for start_idx in range(0, num_frames, stride):
        end_idx = min(start_idx + chunk_size, num_frames)
        chunk = video_features[start_idx:end_idx]

        # Skip if chunk too small
        if chunk.shape[0] < 10:
            continue

        features = torch.FloatTensor(chunk).unsqueeze(0).to(device)

        # Keep chunk generation length proportional to chunk duration.
        chunk_max_glosses = min(max_glosses, max(12, int(chunk.shape[0] / 18) + 8))

        with torch.no_grad():
            memory = model.encode(features)
            if decode == "beam":
                pred_tokens = _beam_search_single(
                    model,
                    memory,
                    max_length=chunk_max_glosses,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                )
                predictions = torch.tensor([pred_tokens], dtype=torch.long, device=memory.device)
            else:
                predictions = model.generate(memory, max_length=chunk_max_glosses)

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

        all_glosses = _merge_with_overlap(all_glosses, chunk_glosses)

        print(f"  Chunk {start_idx}-{end_idx}: {' '.join(chunk_glosses)}")

        # Break if we've reached the end
        if end_idx >= num_frames:
            break
    
    # Deduplicate consecutive repeated sequences
    if len(all_glosses) > 0:
        deduped = []
        i = 0
        while i < len(all_glosses):
            deduped.append(all_glosses[i])
            # Skip consecutive duplicates
            j = i + 1
            while j < len(all_glosses) and all_glosses[j] == all_glosses[i]:
                j += 1
            i = j
        all_glosses = deduped

    return all_glosses


def main():
    parser = argparse.ArgumentParser(description="Infer glosses from video")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints_encdec/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data",
        type=str,
        default="how2sign_mediapipe_clip_1000vocab.h5",
        help="Path to training data (for vocabulary)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Maximum frames to process"
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Optional start time (seconds) for inference window",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Optional end time (seconds) for inference window",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Process long videos in chunks of this many frames",
    )
    parser.add_argument(
        "--max-glosses",
        type=int,
        default=300,
        help="Maximum glosses to generate",
    )
    parser.add_argument(
        "--visual-backbone",
        type=str,
        default="none",
        choices=["none", "clip"],
        help="Optional visual embedding backbone",
    )
    parser.add_argument(
        "--visual-stride",
        type=int,
        default=4,
        help="Sample every N frames for visual embeddings",
    )
    parser.add_argument(
        "--visual-max-frames",
        type=int,
        default=256,
        help="Cap number of frames used for visual embeddings",
    )
    parser.add_argument(
        "--decode",
        type=str,
        default="greedy",
        choices=["greedy", "beam"],
        help="Decoding strategy",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=4,
        help="Beam size when --decode beam",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.7,
        help="Length penalty when --decode beam",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.2,
        help="Repetition penalty when --decode beam (1.0 disables)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=float,
        default=0.5,
        help="Chunk overlap fraction for long videos (0.0-0.95)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("VIDEO INFERENCE - ENCODER-DECODER TRANSFORMER")
    print("=" * 70)
    print(f"Video: {args.video}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")
    print(f"Decoding: {args.decode}")
    if args.decode == "beam":
        print(
            f"Beam size: {args.beam_size}, Length penalty: {args.length_penalty}, "
            f"Repetition penalty: {args.repetition_penalty}"
        )
    if args.start_sec is not None or args.end_sec is not None:
        print(f"Window: start={args.start_sec}, end={args.end_sec}")
    print()

    # Extract features
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    visual_model, visual_preprocess, visual_dim = init_visual_embedder(
        args.visual_backbone, device
    )
    if visual_model is not None:
        print(f"Using visual backbone: {args.visual_backbone} (dim={visual_dim})")

    video_features = extract_video_features(
        args.video,
        args.max_frames,
        start_sec=args.start_sec,
        end_sec=args.end_sec,
        visual_model=visual_model,
        visual_preprocess=visual_preprocess,
        visual_stride=args.visual_stride,
        visual_max_frames=args.visual_max_frames,
        device=device,
    )
    print(f"Video features shape: {video_features.shape}")
    print()

    # Load model
    print("Loading model...")
    model, idx_to_gloss, expected_input_dim = load_model_and_vocab(args.checkpoint, args.data)

    if video_features.shape[1] != expected_input_dim:
        raise ValueError(
            f"Feature mismatch: video features are {video_features.shape[1]}D, "
            f"but the checkpoint expects {expected_input_dim}D.\n"
            "Train a checkpoint on MediaPipe features (1629D) and try again."
        )
    
    print(f"Model loaded with {len(idx_to_gloss)} classes")
    print(f"Vocabulary: {list(idx_to_gloss.values())[:20]}...")
    print()

    # Apply global normalization (must match training)
    stats_path = args.data + ".train_stats.npz"
    import os
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        g_mean, g_std = stats['mean'], stats['std']
        # Truncate/pad stats to match feature dim (handles CLIP concat)
        feat_dim = video_features.shape[1]
        if g_mean.shape[0] < feat_dim:
            g_mean = np.concatenate([g_mean, np.zeros(feat_dim - g_mean.shape[0], dtype=np.float32)])
            g_std = np.concatenate([g_std, np.ones(feat_dim - g_std.shape[0], dtype=np.float32)])
        elif g_mean.shape[0] > feat_dim:
            g_mean = g_mean[:feat_dim]
            g_std = g_std[:feat_dim]
        video_features = np.nan_to_num(video_features, nan=0.0, posinf=0.0, neginf=0.0)
        video_features = (video_features - g_mean) / g_std
        video_features = np.clip(video_features, -10.0, 10.0)
        print("Applied global normalization from training stats.")
    else:
        print(f"WARNING: No training stats found at {stats_path} — running without normalization.")
        print("  Results may be poor. Re-run training to generate the stats cache.")
    print()

    # Predict
    print("Predicting glosses...")
    glosses = predict_video(
        model,
        video_features,
        idx_to_gloss,
        args.device,
        args.chunk_size,
        args.max_glosses,
        decode=args.decode,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        chunk_overlap=args.chunk_overlap,
    )

    print("=" * 70)
    print("PREDICTION RESULT")
    print("=" * 70)
    print(f"Predicted glosses: {' '.join(glosses)}")
    print(f"Number of glosses: {len(glosses)}")
    print()    

if __name__ == "__main__":
    main()
