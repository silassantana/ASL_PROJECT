# ASL → English Translation

Continuous American Sign Language → English translation from MediaPipe
landmark trajectories. Pre-alpha research project, revived and actively
developed as of 2026-11.

[![CI](https://img.shields.io/github/actions/workflow/status/silassantana/ASL_PROJECT/ci.yml?branch=main&label=CI)](https://github.com/silassantana/ASL_PROJECT/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

The goal is a free, open-source translator for continuous signing: a deaf
person signs, and a hearing person who knows no sign language reads English.
Signs are motion, so the pipeline uses landmark trajectories, not image
embeddings.

There is no trained landmark → text model, no reproducible benchmark, and no
working webcam translator yet. What exists is a real data pipeline, a
masked-pose pretraining encoder, a diagnostic seq2seq trainer, and an honest
record of what failed. The code is still a flat collection of research scripts
(31 root-level Python files, ~14.2k LOC) from a solo project; the revival adds
packaging, tests, and docs around the parts that work.

## Status and what works today

| Component | State | Notes |
|---|---|---|
| YouTube-ASL → pose `.npz` (`scrape_youtube_asl.py`) | Working | Downloads videos with `yt-dlp`, extracts MediaPipe trajectories + captions. Long-running, network-dependent, YouTube rate limits apply |
| H5 dataset builders (`build_youtube_asl_h5.py`, `build_segment_h5.py`) | Working | Video-level and segment-level H5; segment splits are made at the video level to avoid signer leakage |
| Masked-pose pretraining (`train_temporal_encoder.py`) | Working | Self-supervised masked-pose modeling; saves `best_encoder_weights.pt`, but nothing loads it yet (roadmap) |
| Seq2seq landmark → text trainer (`train_seq2seq_diagnostic.py`) | Working (diagnostic only) | Short run whose main output is a cross-attention entropy ratio. Not a usable translator |
| CPU smoke test (`scripts/smoke_train.py`) | Working (new) | Deterministic toy training run on synthetic poses; no data or GPU needed |
| Landmark → text inference (`scripts/infer_landmarks.py`) | Experimental (new) | CLI exists; no real checkpoint exists yet |
| Webcam demo (`scripts/webcam_demo.py`) | Experimental (new) | CPU webcam loop, experimental; no trained model behind it |
| Pretrained encoder wired into fine-tuning | Roadmap | The encoder is saved but not loaded by any training script |
| Reproducible benchmark | None yet | See [RESULTS.md](RESULTS.md) |
| Libras support | Roadmap | See [ROADMAP.md](ROADMAP.md) |

## Data representation

MediaPipe Holistic gives 142 landmarks: 33 pose, 21 left hand, 21 right hand,
and 67 face points. At 3 coordinates each that is **426 real values per frame**.
The scraper zero-pads each frame to 543 values, then stacks three streams —
position, velocity, acceleration — into a `[T, 1629]` float32 tensor.

Because the padding is 117 values per stream, **351 of the 1,629 input
dimensions are always zero**, and more are zero when hands or face are not
detected. Older docs called this "543 landmarks"; trimming it to the 426 real
coordinates is [roadmap milestone 3](ROADMAP.md).

## Architecture

```text
YouTube-ASL video
  └─ MediaPipe Holistic (scrape_youtube_asl.py)
       └─ 142 landmarks × 3 coords = 426 real values per frame
            └─ padded to 543 values, stacked as position + velocity + acceleration
                 └─ [T, 1629] pose trajectory
                      │
                      ├─ masked-pose pretraining (train_temporal_encoder.py)
                      │    1D temporal conv → positional encoding → Transformer encoder
                      │    → reconstruct masked frames (self-supervised)
                      │
                      └─ supervised seq2seq (train_seq2seq_diagnostic.py)
                           1D temporal conv → positional encoding → Transformer encoder
                           → Transformer decoder → English text
```

The encoder in the fine-tuning path is currently trained from scratch; loading
the pretrained weights is [roadmap milestone 1](ROADMAP.md). A CTC auxiliary
head exists only in the legacy gloss pipeline (`sign_transformer.py`), not in
the landmark → text path.

## Quickstart

Runs on CPU; CI covers Python 3.10–3.12. No dataset, checkpoint, or GPU is needed for tests or the smoke run.

```bash
git clone https://github.com/silassantana/ASL_PROJECT.git
cd ASL_PROJECT
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest -q
make smoke
```

`make smoke` is a deterministic toy training run on synthetic poses. It writes
`runs/smoke/metrics.json` and `runs/smoke/model.pt` in under a minute and
checks that the model path executes; its numbers are a plumbing check, not a
translation result (see [RESULTS.md](RESULTS.md)).

Run the experimental landmark → text CLI against the smoke checkpoint:

```bash
python scripts/infer_landmarks.py --checkpoint runs/smoke/model.pt --npz path/to/poses.npz
```

The smoke checkpoint is trained on synthetic toy data, so its output is
meaningless; no real checkpoint exists yet.

The experimental webcam demo needs the MediaPipe extra:

```bash
pip install -e ".[mediapipe]"
python scripts/webcam_demo.py --checkpoint runs/smoke/model.pt --camera 0
```

Real-data commands are in [RESULTS.md](RESULTS.md); dataset terms and
acquisition in [THIRD_PARTY.md](THIRD_PARTY.md). No data is included.

## Project structure

| Path | What it is |
|---|---|
| `scrape_youtube_asl.py` | YouTube-ASL video → per-video `.npz` pose trajectory + caption + segments |
| `build_youtube_asl_h5.py` | Video-level H5 (`poses`, `captions`, `segments`, `index`) for pretraining |
| `build_segment_h5.py` | Segment-level H5 with train/val split and normalization stats for seq2seq |
| `train_temporal_encoder.py` | Masked-pose self-supervised pretraining (`TemporalPoseEncoder`) |
| `train_seq2seq_diagnostic.py` | Small seq2seq landmark → text diagnostic trainer + attention-entropy metric |
| `sign_transformer.py`, `train_transformer_encdec.py` | Legacy How2Sign/ASL Citizen gloss pipeline — failed; kept for reference |
| `diagnose_recognizer.py`, `clip_retrieval_probe.py`, `mediapipe_collapse_probe.py` | Diagnostics used to show why the old pipelines failed |
| `scripts/` | CPU smoke test, inference CLI, experimental webcam demo |
| `tests/`, `.github/workflows/ci.yml` | CPU tests and CI |
| `docs/` | Detailed docs, including [FAILED_APPROACHES.md](docs/FAILED_APPROACHES.md) |

The other ~21 root-level scripts are abandoned How2Sign/ASL Citizen
experiments. They are kept for provenance but are not part of the current
pipeline.

## What we learned from the failed approaches

Two supervised approaches were tried before the revival: mean-pooled CLIP frame
features → T5, and MediaPipe + CLIP → a gloss-recognizer Transformer. Both
produced fluent-looking output that ignored the input. The evidence:

- zeroing the visual features left BLEU/WER unchanged;
- cross-attention entropy ratio was 0.984 (1.0 = uniform attention, encoder
  ignored);
- mean-pooled CLIP features retrieved the paired sentence at chance level.

All reported numbers from those runs are **historical and not reproducible**
from this repository: the data and checkpoints were not retained, and no
held-out test split was ever evaluated. The detailed post-mortem and the
diagnostics methodology are in
[docs/FAILED_APPROACHES.md](docs/FAILED_APPROACHES.md).

## Datasets and attribution

The code is MIT-licensed; the datasets are not covered by that license and are
not redistributed here.

- **YouTube-ASL** — Google Research, CC BY 4.0. ~11k videos / 984 hours.
  The repo ships the official video-ID list; videos are downloaded with
  `yt-dlp` and stay subject to YouTube's terms.
- **How2Sign** — CC BY-NC 4.0, research use only; used pre-revival.
- **ASL Citizen** — community-sourced isolated-sign dataset, research terms;
  used pre-revival.

See [THIRD_PARTY.md](THIRD_PARTY.md) for attribution, dataset terms, and
instructions for obtaining each dataset. Large third-party metadata CSVs were
removed from git in the revival; no videos, pose files, H5 files, or
checkpoints are included.

## Roadmap

[ROADMAP.md](ROADMAP.md) is the source of truth for what is next. The near-term
order is: wire the pretrained encoder into fine-tuning, produce a real
landmark → text checkpoint, trim the 1,629-dim input to the 426 real
coordinates, then publish a small reproducible benchmark.

## Contributing

Contributions are welcome, especially from Deaf and hard-of-hearing
contributors and sign-language researchers. See
[CONTRIBUTING.md](CONTRIBUTING.md) for setup, tests, and where to start.

## License

MIT, Copyright © 2026 Silas Santana. See [LICENSE](LICENSE); the license covers
the code, not the datasets — see [THIRD_PARTY.md](THIRD_PARTY.md).
