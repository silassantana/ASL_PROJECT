# ASL→English Translation Project Brief
> For Claude Code — full context from the research/investigation session

## The Mission

Build a **free, real-time ASL-to-English translation tool** for the deaf and hard-of-hearing community. A deaf person signs; a hearing person who knows no sign language reads English. No intermediary. No barrier.

- Free product. Open source. Community-first.
- Target: continuous, natural signing — not isolated signs, not fingerspelling only
- Languages: ASL first, then Libras and others
- Platform: real-time webcam input → English text output

---

## What We Tried and Why It Failed

This section is critical. Do not repeat these approaches without understanding why they failed.

### Attempt 1: Direct video→T5 translation (`train_asl2text_t5.py`)

**What**: CLIP ViT-B/32 frame features → custom visual encoder → T5-small decoder → English text  
**Dataset**: `how2sign_clip_text_realigned.h5` (31K clips, 512-D CLIP features, English sentences)  
**Result**: BLEU-4 ≈ 0.03–0.48 after 30 epochs. Decoder output: "if you're going to go for a..." regardless of input.

**Why it failed** — diagnosed via ablation:
1. **Zero-ablation test**: zeroing out visual features gave *identical* BLEU/WER as real features → decoder completely ignored encoder output
2. **Cross-attention entropy diagnostic**: entropy ratio = 0.984 (1.0 = perfectly uniform) → attention was near-uniform across all source frames, not grounding on anything
3. **CLIP retrieval probe**: mean-pooled CLIP frame features produced chance-level retrieval (diagonal similarity gap = 0.0010) → CLIP per-frame embeddings carry no usable sign language signal
4. **Root cause**: CLIP was trained on static image-text pairs. A person mid-sign looks like "person on camera" in every frame. Mean-pooling 142 frames of "person gesturing" → generic "person on camera" embedding. The decoder learned English n-gram statistics and ignored the visual pathway entirely.

### Attempt 2: Gloss recognizer (`train_transformer_encdec.py`)

**What**: MediaPipe + CLIP features (2141-D) → Transformer encoder-decoder → gloss sequence  
**Dataset**: `how2sign_mediapipe_clip_1000vocab.h5` (mediapipe holistic landmarks + motion + CLIP, 1000-gloss vocab)  
**Best checkpoint**: `checkpoints_clean_v2/best_model.pt` — BLEU 2.78, WER 99.91%

**Why it failed** — diagnosed via `diagnose_recognizer.py`:
- Predictions: "so what you want to do is a little bit of the ball..." for every input
- Only 70/1000 vocab tokens ever predicted (all high-frequency function words)
- Cross-attention entropy ratio: 0.984 — **identical failure mode to the T5 translator**
- Decoder learned ASL-gloss-register language model. Encoder ignored.

### Attempt 3: CTC auxiliary loss (`--ctc-weight 0.3` and `0.6`)

**What**: Added `nn.CTCLoss` on top of encoder memory (`sign_transformer.py` + `train_transformer_encdec.py`)  
**Hypothesis**: CTC forces encoder→gloss alignment independently of decoder LM, breaking the "ignore encoder" pathology  
**Result**: Best BLEU 2.52 (w=0.3) / 2.63 (w=0.6) after 30 epochs, early stopping. Slight improvement but same collapse pattern by epoch 20+.

**Why it partially worked but wasn't enough**: CTC changed early-epoch behavior (more diverse outputs epochs 4-12) but the decoder's target-side LM eventually reasserted itself. CTC weight 0.3-0.6 wasn't strong enough to maintain encoder grounding throughout training.

**Key insight from CTC run**: The architecture and training recipe are structurally sound. The bottleneck is **the visual representation**, not the seq2seq framework.

### Root Cause (confirmed by multiple independent diagnostics)

**Mean-pooled CLIP frame features carry no temporally-discriminative information for sign language.**

Signs are defined by:
- Hand SHAPE (what configuration the hand is in)
- Hand POSITION (where relative to body)
- MOTION TRAJECTORY (how hands move through space)
- FACIAL GRAMMAR (eyebrow raises, mouth morphemes)
- BOTH HANDS simultaneously

CLIP per-frame embeddings encode "what does this photo look like" — not motion, not trajectory, not the relationship between successive frames. Averaging across frames destroys all temporal information.

---

## The Right Architecture (What to Build Next)

### Core Insight
**Sign language is motion, not snapshots.** The right input representation is a TIME SERIES of body landmarks, not averaged image features.

### Data: YouTube-ASL (11,095 videos, 984 hours)

Google Research / NeurIPS 2023 paper. CC BY 4.0.  
**Already being extracted** by `scrape_youtube_asl.py` running locally.

Each processed video saved as `~/youtube_asl_poses/<video_id>.npz`:
```python
data = np.load('video_id.npz', allow_pickle=True)
poses    = data['poses']    # [T, 1629] float32 — pose trajectory
caption  = data['caption']  # str — full English caption
segments = data['segments'] # JSON str — [{start, end, text}, ...] time-aligned
```

Feature format: **[T, 1629] float32**
- T = number of frames (at 15fps, stride-downsampled from source)
- 1629 = 543 landmarks × 3 streams (position + velocity + acceleration)
  - 543 landmarks = 33 pose + 21 left hand + 21 right hand + 67 face keypoints × 3 (x, y, z)
  - Velocity: frame-to-frame difference
  - Acceleration: second-order difference
- **This is the same format as `how2sign_mediapipe_clip_1000vocab.h5`** (feature_dim=2141 includes CLIP concat, but the 1629-D landmark+motion portion is identical)

### Architecture Plan: Temporal Pose Encoder

```
Input: [T, 1629] landmark trajectory (at 15fps)
  ↓
Local temporal convolution (captures hand arc within a sign, ~8-16 frame window)
  ↓
Positional encoding (temporal position)
  ↓
Transformer encoder (captures how signs flow into each other, long-range dependencies)
  ↓
[T', hidden_dim] continuous representation
  ↓
[Pretraining: masked pose modeling — predict masked frames from context]
[Fine-tuning: seq2seq decoder → English text + CTC auxiliary loss]
```

Key differences from failed approach:
1. **No CLIP features** — pure landmark trajectories
2. **No mean-pooling** — full temporal sequence preserved
3. **Temporal convolutions first** — capture local motion patterns before global attention
4. **Self-supervised pretraining** on unlabeled pose sequences before supervised fine-tuning

### Masked Pose Modeling (Self-Supervised Pretraining)

Inspired by BERT/MAE. No labels needed for this phase.
- Mask 15-30% of frame windows randomly
- Predict masked landmark positions from surrounding context
- Forces encoder to learn: "given how hands were moving before and after, what was happening during?"
- This is how the encoder learns motion patterns, coarticulation, signing rhythms

After pretraining, encoder has learned **what natural signing motion looks like** before ever seeing a label.

### Fine-tuning on YouTube-ASL (Supervised)

With a pretrained temporal encoder:
- Attach seq2seq decoder (same architecture as existing `train_transformer_encdec.py`)
- Add CTC auxiliary loss (already implemented in `sign_transformer.py`)
- Fine-tune on `(poses, caption)` pairs from YouTube-ASL
- 984 hours >> 80 hours (How2Sign) — 12× more data with better representation

---

## Existing Codebase

All files in `~/Code/ASL-Project/`:

| File | Status | Description |
|------|--------|-------------|
| `sign_transformer.py` | ✅ Modified | SignLanguageTransformer with CTC head + `ctc_logits()` + `return_memory` |
| `train_transformer_encdec.py` | ✅ Modified | Training loop with `--ctc-weight` flag, separate CE+CTC loss tracking |
| `train_asl2text_t5.py` | ✅ Modified | T5 translator with `--ablate-visual` diagnostic flag |
| `modal_train.py` | ✅ Modified | Modal entrypoints: `train_ctc_smoke`, `train_translator_ablation` |
| `infer_video_encdec.py` | ✅ Unchanged | Real-time inference from webcam/video file |
| `prepare_how2sign_text_h5.py` | ✅ Unchanged | CLIP feature extraction pipeline |
| `scrape_youtube_asl.py` | ✅ New | YouTube-ASL pose+caption extractor (RUNNING) |
| `diagnose_recognizer.py` | ✅ New | Cross-attention entropy diagnostic tool |
| `mediapipe_collapse_probe.py` | ✅ New | Feature collapse diagnostic |

### Key Checkpoints (on Modal volume `asl-training-data`)

| Dir | BLEU | Notes |
|-----|------|-------|
| `checkpoints_clean_v2/` | 2.78 | OLD architecture, single-stream, incompatible with modified sign_transformer.py |
| `checkpoints_ctc_smoke_w0.3/` | 2.52 | New dual-stream + CTC 0.3, 30 epochs |
| `checkpoints_ctc_smoke_w0.6/` | 2.63 | New dual-stream + CTC 0.6, 27 epochs (early stop) |

⚠️ **Checkpoint compatibility note**: `sign_transformer.py` was refactored from single-stream to dual-stream multimodal fusion. `checkpoints_clean_v2/best_model.pt` will NOT load into the current `sign_transformer.py`. Use `use_multimodal_fusion=False` and `input_features=2141` treating it as a single linear projection if you need to load old checkpoints.

---

## Data Available

### On Modal Volume (`asl-training-data` at `/data/`):
- `how2sign_mediapipe_clip_1000vocab.h5` — 25.4GB, 2141-D features, 1000-gloss vocab, NO English text
- `how2sign_clip_text_realigned.h5` — 8.9GB, 512-D CLIP features, English sentences (31K train)
- Various checkpoint directories (see above)

### Being Generated Locally (`~/youtube_asl_poses/`):
- 11,095 videos from YouTube-ASL (NeurIPS 2023, Google Research)
- [T, 1629] pose trajectories + English captions + time-aligned segments
- ~9GB total projected, ~4.2GB at 47% complete as of last check
- **This is the foundation dataset for the next training run**

---

## Diagnostics Reference

These tools exist and work. Use them before making architectural decisions.

### `diagnose_recognizer.py`
```bash
python diagnose_recognizer.py \
  --checkpoint checkpoints_clean_v2/best_model.py \
  --data val_slice.h5  # extracted by extract_val_slice.py
```
Reports: prediction length distribution, vocab coverage, cross-attention entropy ratio.
**Key metric**: entropy ratio near 1.0 = encoder ignored. Near 0.0 = encoder used.

### `train_asl2text_t5.py --ablate-visual zero`
Zero out visual features during training/eval. If metrics identical to real features → decoder ignores encoder.

### `clip_retrieval_probe.py`
Tests if mean-pooled CLIP features carry semantic signal. Run on Modal.

---

## Infrastructure

- **Training**: Modal Labs (`modal run modal_train.py::<function>`)
- **GPU**: A100-40GB on Modal
- **Local**: Intel HD 620 GPU (for MediaPipe extraction only, CPU-bound anyway)
- **Storage**: Modal Volume `asl-training-data`, local `/home` (228GB free)
- **Environment**: Python 3.12, PyTorch, MediaPipe, transformers, yt-dlp

---

## Immediate Next Steps (when scraper finishes)

1. **Build `train_temporal_encoder.py`** — masked pose modeling pretraining on YouTube-ASL poses
   - Input: `[T, 1629]` landmark trajectories
   - Architecture: 1D temporal conv → positional encoding → transformer encoder
   - Objective: predict masked frame windows from context
   - No labels needed — self-supervised

2. **Build H5 dataset from YouTube-ASL npz files** — for efficient Modal training
   - Consolidate individual `.npz` files into a single H5
   - Structure: `train_sequences [N, T, 1629]`, `train_texts [N]` (from captions)
   - Reuse `prepare_how2sign_text_h5.py` pattern

3. **Fine-tune on YouTube-ASL** — seq2seq with CTC, pretrained encoder
   - Plug pretrained temporal encoder into `sign_transformer.py`
   - Fine-tune with `--ctc-weight 0.5` on `(poses, captions)` pairs

4. **Real-time inference** — extend `infer_video_encdec.py`
   - Webcam → MediaPipe → temporal encoder → beam search → English text display

---

## Philosophy

> "We start big and go bigger."

This is not a research project optimizing BLEU scores. This is infrastructure for human communication across a barrier that affects millions of people.

Every architectural decision should be evaluated against: **does this get us closer to a deaf person being understood by a hearing person in real time?**

BLEU scores are a proxy. Working software is the goal.

The deaf community doesn't need a perfect model. They need something working in their hands today that gets better over time — built *with* them, not just *for* them.
