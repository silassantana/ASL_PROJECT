# Results

## No reproducible benchmark yet.

This is deliberate, not modesty. There is currently **no trained
landmark → text model and no benchmark number that a stranger can reproduce
from this repository**. The only reproducible number today is the CPU smoke
test below, which is a plumbing check on synthetic data — not a translation
result.

A reviewer should treat every other number in this repository as historical.

## Why there is no benchmark

- **No data is included.** YouTube-ASL has to be re-scraped from YouTube (hours
  to days, rate-limited, some videos gone); How2Sign requires accepting its
  research-only terms; ASL Citizen requires requesting the dataset from
  Microsoft. See [THIRD_PARTY.md](THIRD_PARTY.md).
- **No checkpoints were retained.** Pre-revival checkpoints lived on a private
  Modal volume (`asl-training-data`) and the feature H5 files on a private
  Hugging Face dataset. Neither is available to anyone else.
- **No test split was ever evaluated.** The old training code loads only
  `train` and `val` splits. All reported numbers were validation numbers.
- **Metrics were printed, not saved.** There are no JSON/CSV logs, run
  directories, W&B/MLflow records, or seeds tied to any reported number.
- **The historical metrics disagree.** 2.36, 2.52, 2.63, and 2.78 all appear as
  "best BLEU" in different pre-revival docs, with no run IDs or configs tying
  them together.

So the honest statement is: no successful measured result exists here that can
be independently checked. The value in the repository is the pipeline, the
diagnostics, and the failure analysis in
[docs/FAILED_APPROACHES.md](docs/FAILED_APPROACHES.md).

## Historical results (not reproducible)

These were reported in the old README and pre-revival notes. They are kept for
context only. None has a retained artifact, and all were computed on a
validation split.

| Attempt | Pipeline | Reported metric (historical) | Reproducible |
|---|---|---|---|
| 1 | Mean-pooled CLIP frame features → T5-small decoder | BLEU-4 ≈ 0.03–0.48 after 30 epochs | No |
| 2 | MediaPipe + CLIP → Transformer gloss recognizer | BLEU 2.78, WER 99.91% | No |
| 3 | Attempt 2 + CTC auxiliary loss, weight 0.3 | BLEU 2.52, 30 epochs | No |
| 4 | Attempt 2 + CTC auxiliary loss, weight 0.6 | BLEU 2.63, 27 epochs (early stop) | No |
| — | "Modal best" noted in a pre-revival migration note (file since removed) | BLEU 2.36 | No; conflicts with the rows above |

Caveats for the table:

- BLEU came from an in-house, no-smoothing implementation
  (`train_transformer_encdec.py:26-58`) that returns `0.0` when any n-gram count
  is zero; WER was over gloss tokens, not English text
  (`train_transformer_encdec.py:61-79`). Neither was validated against
  `sacrebleu` or `jiwer`.
- No held-out test split was used, and no metric log was kept.
- A "29% accuracy on 20 signs" note in `extract_how2sign_landmarks.py` refers to
  a different isolated-sign experiment. It also has no retained artifact.

Do not compare these numbers to published sign-language translation work. They
are not comparable: the split, the metric implementation, and the data differ,
and none of it is verifiable.

## Reproducible today: the CPU smoke test

The revival adds `scripts/smoke_train.py`: a deterministic, CPU-only run on
synthetic landmark tensors. It checks that the model path (encode → decode →
loss → backward) executes and that metrics can be written. It does not measure
translation quality.

```bash
pip install -e ".[dev]"
make smoke
# equivalent to:
python scripts/smoke_train.py
```

It runs in under a minute without a GPU or any dataset, and writes:

- `runs/smoke/metrics.json` — `seed`, `steps`, `batch_size`, `device`,
  `wall_s`, `loss_start`, `loss_end`, `toy_token_accuracy`
- `runs/smoke/model.pt` — the toy checkpoint used by the inference CLI

Actual output, from `python scripts/smoke_train.py` (defaults:
`--steps 40 --batch-size 8 --out runs/smoke --seed 42`) on a CPU-only machine
running Python 3.12 / torch CPU in November 2026:

```json
{
  "seed": 42,
  "steps": 40,
  "batch_size": 8,
  "device": "cpu",
  "wall_s": 4.411,
  "loss_start": 1.584378072619438,
  "loss_end": 9.294285371197475e-06,
  "toy_token_accuracy": 1.0
}
```

The run takes roughly 5–15 seconds wall-clock on this machine depending on
load; the `wall_s` field covers the training loop only and is the one
non-deterministic value. Everything else is deterministic for a fixed seed:
repeated runs reproduced `loss_start`, `loss_end`, and `toy_token_accuracy`
exactly.

Interpretation: the loss falls to ~0 and toy token accuracy reaches 1.0, so the
encode → decode → loss → backward path is wired and learning. The toy corpus is
small and learnable by design, so saturation is expected; it proves plumbing,
not ASL ability. The absolute values are meaningless for ASL — the inputs and
targets are synthetic tensors, not real signing data.

## How to reproduce a full run (data required)

These commands exercise the real pipeline. They require YouTube-ASL data and
(except for the scraper and H5 builders) a GPU to finish in reasonable time.
They produce a **diagnostic** result (cross-attention entropy), not a benchmark.

```bash
# 1. Acquire YouTube-ASL poses + captions (long-running; see THIRD_PARTY.md)
python scrape_youtube_asl.py \
  --ids youtube_asl_video_ids.txt \
  --output-dir ~/youtube_asl_poses \
  --resume

# 2. Build the video-level H5 for self-supervised pretraining
python build_youtube_asl_h5.py \
  --npz-dir ~/youtube_asl_poses \
  --output youtube_asl_poses.h5

# 3. Masked-pose pretraining (saves best_encoder_weights.pt)
python train_temporal_encoder.py \
  --data youtube_asl_poses.h5 \
  --save-dir checkpoints_temporal \
  --epochs 50

# 4. Build the segment-level H5 for seq2seq (video-level train/val split)
python build_segment_h5.py \
  --npz-dir ~/youtube_asl_poses \
  --output youtube_asl_segments.h5

# 5. Diagnostic seq2seq run: prints BLEU-1 and cross-attention entropy per epoch
python train_seq2seq_diagnostic.py \
  --data youtube_asl_segments.h5 \
  --save-dir checkpoints_seq2seq_diag \
  --epochs 10
```

Notes:

- Step 5 is explicitly a **diagnostic** (`train_seq2seq_diagnostic.py:1-20`):
  it is meant to answer "is the encoder contributing signal?", not to produce a
  translator. A cross-attention entropy ratio near 1.0 means the decoder is
  ignoring the encoder; below 0.8 is the signal the project is looking for.
- It writes `best_checkpoint.pt` and `latest_checkpoint.pt` but only prints
  metrics — no JSON/CSV is produced. That is part of why this is not a
  benchmark.
- The pretrained encoder from step 3 is **not** loaded in step 5. Wiring it in
  is [roadmap milestone 1](ROADMAP.md).
- Hardcoded personal paths in some legacy scripts do not affect this path; the
  five scripts above take their inputs via CLI flags.

## What would make a result reproducible

A result becomes citable here when all of the following exist:

1. A fixed public split, checked in or published, with a documented license.
2. A deterministic training command (seed, config, hardware noted) that anyone
   can run.
3. A published checkpoint, or a model small enough to retrain cheaply.
4. An evaluation script that writes `metrics.json` (BLEU via `sacrebleu`, WER
   via `jiwer` or a documented equivalent) over a held-out test split.
5. The command and output added to this file, plus a note on what the number
   does and does not show.

That is roadmap milestone 4. Until then, no BLEU/WER value in this repository
should be presented as a current result.
