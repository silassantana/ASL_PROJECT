# ASL → English Real-Time Translator

A free, open-source system for continuous American Sign Language to English translation — built for the deaf and hard-of-hearing community. A deaf person signs; a hearing person who knows no sign language reads English. No barrier.

**Status:** Paused. Architecture validated; not actively maintained — contributions welcome.

---

## What This Is

This project builds a real-time translation pipeline:

```
Webcam → MediaPipe landmarks → Temporal Pose Encoder → English text
```

The target is **continuous, natural signing** — not isolated signs, not fingerspelling only. The model processes landmark trajectories through time rather than image features, because sign language is motion, not snapshots.

---

## Architecture

### Input Representation

Raw frames are processed by MediaPipe Holistic into body landmark trajectories:

```
[T, 1629] float32 per video clip
  ├── 543 landmarks × 3 channels (x, y, z)
  │     ├── 33 pose keypoints
  │     ├── 21 left hand keypoints
  │     ├── 21 right hand keypoints
  │     └── 67 face keypoints
  └── 3 temporal streams: position · velocity · acceleration
```

### Temporal Pose Encoder

```
[T, 1629] landmark trajectory
     ↓
1D temporal convolution   — captures hand shape and arc within a sign (~8–16 frame window)
     ↓
Positional encoding
     ↓
Transformer encoder       — captures coarticulation and long-range sign dependencies
     ↓
[T', hidden_dim] continuous representation
     ↓
Self-supervised pretraining: masked pose modeling (predict masked frames from context)
Fine-tuning:               seq2seq decoder → English text  +  CTC auxiliary loss
```

### Why Landmark Trajectories

Signs are defined by hand shape, position, motion trajectory, and facial grammar — all temporal. Frame-level image embeddings (e.g. CLIP) encode "what does this photo look like," not motion. Averaging across frames destroys all temporal information.

Landmark trajectories preserve exactly what matters: how hands move through space over time.

---

## Dataset

**YouTube-ASL** (Google Research / NeurIPS 2023, CC BY 4.0) — 11,095 videos, ~984 hours of continuous signing with English captions.

Each processed clip is saved as `~/youtube_asl_poses/<video_id>.npz`:

```python
data = np.load('video_id.npz', allow_pickle=True)
poses    = data['poses']    # [T, 1629] float32 — landmark trajectory at 15fps
caption  = data['caption']  # str — English caption
segments = data['segments'] # JSON — [{start, end, text}] time-aligned segments
```

Prior experiments used How2Sign (80h) and the Microsoft ASL Citizen Dataset. YouTube-ASL is ~12× larger and paired with English captions rather than gloss annotations, making it the right foundation for direct sign→text translation.

---

## What We Learned (Failed Approaches)

### CLIP features → T5 decoder (BLEU-4 ≈ 0.03)

Zero-ablation test: zeroing out visual features produced **identical** BLEU scores. Cross-attention entropy ratio = 0.984 (1.0 = perfectly uniform). The decoder learned English n-gram statistics and ignored the visual pathway entirely.

Root cause: CLIP is trained on static image-text pairs. A person mid-sign looks like "person on camera" in every frame. Mean-pooling 142 frames → generic embedding with no sign-discriminative signal.

### MediaPipe + CLIP → Transformer gloss recognizer (BLEU 2.78, WER 99.91%)

Same collapse: only 70/1000 vocab tokens ever predicted, all high-frequency function words. Adding CTC auxiliary loss (weight 0.3–0.6) improved early-epoch diversity but the decoder's language model reasserted itself by epoch 20.

Key insight from the CTC run: the seq2seq framework and training recipe are sound. The bottleneck was the visual representation, not the architecture.

---

## Project Structure

| File | Description |
|------|-------------|
| `sign_transformer.py` | SignLanguageTransformer with CTC head and dual-stream fusion |
| `train_transformer_encdec.py` | Training loop with `--ctc-weight` flag |
| `train_asl2text_t5.py` | T5 translator (prior approach, includes `--ablate-visual` diagnostic) |
| `scrape_youtube_asl.py` | YouTube-ASL pose + caption extractor |
| `infer_video_encdec.py` | Real-time inference from webcam or video file |
| `diagnose_recognizer.py` | Cross-attention entropy diagnostic |
| `modal_train.py` | Modal Labs training entrypoints |

---

## Roadmap

- [ ] `train_temporal_encoder.py` — masked pose modeling pretraining on YouTube-ASL
- [ ] H5 dataset consolidation from YouTube-ASL `.npz` files
- [ ] Supervised fine-tuning: temporal encoder → seq2seq + CTC on `(poses, captions)`
- [ ] Real-time inference: webcam → MediaPipe → encoder → beam search → text display
- [ ] Libras (Brazilian Sign Language) support

---

## Philosophy

This is not a research project optimizing BLEU scores. It is infrastructure for human communication across a barrier that affects millions of people.

Every architectural decision is evaluated against one question: **does this get us closer to a deaf person being understood by a hearing person in real time?**

The deaf community doesn't need a perfect model. They need something working in their hands today that gets better over time — built *with* them, not just *for* them.

---

## Contributing

Open to contributors, especially those with lived experience in the deaf and hard-of-hearing community. Issues and pull requests welcome.

## License

Apache 2.0
