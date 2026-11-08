# Roadmap

**Status: revived 2026-11, pre-alpha, active development.** This file is the
source of truth for what is next. Each milestone is deliberately checkable:
it states what "done" means, not just an aspiration. [RESULTS.md](RESULTS.md)
records what is actually reproducible today.

## Where the project is

The data path is real: YouTube-ASL videos → MediaPipe landmark trajectories →
HDF5 datasets → masked-pose pretraining → a small diagnostic seq2seq trainer.
What is missing is the join between those pieces and any evidence that
landmark trajectories translate to English. Specifically:

- the pretrained temporal encoder is saved but **never loaded** by a training
  script;
- there is **no trained landmark → text model**;
- there is **no reproducible benchmark** (no retained data, checkpoints, or
  metric logs — see [RESULTS.md](RESULTS.md)).

## Milestones

### 1. Wire the pretrained temporal encoder into fine-tuning

`train_temporal_encoder.py` saves `best_encoder_weights.pt`, but
`train_seq2seq_diagnostic.py` constructs a fresh encoder and trains from
scratch. Load the pretrained encoder in the seq2seq trainer, with options to
freeze or unfreeze it and to report the source of the weights.

**Done when:** a fine-tuning run logs which encoder checkpoint it loaded; the
load path is covered by a test; the exact command is documented; and any claim
of improvement is backed by a saved metric artifact, not a printed line.

**Status:** not started. This is the highest-value next step.

### 2. Landmark → text inference path + a checkpoint

An inference CLI that loads a checkpoint and decodes English text from a
landmark `.npz` file, plus an experimental webcam wrapper. The revival adds
`scripts/infer_landmarks.py` and `scripts/webcam_demo.py` as scaffolding; they
need a real checkpoint and hardened input handling before they mean anything.

**Done when:** one documented command takes a checkpoint + `.npz` and prints
text; a checkpoint (even a small one trained on public data) is available with
its exact provenance; `RESULTS.md` lists the command, the checkpoint, and the
output; the webcam demo reports measured latency and its failure modes.

**Status:** experimental scaffolding landed in this revival
(`scripts/infer_landmarks.py`, `scripts/webcam_demo.py`); a real checkpoint is
still missing.

### 3. Trim the input from 1,629 dims to the real 426 coordinates

The current `.npz`/H5 representation is 142 landmarks × 3 coordinates = 426
real values per frame, padded to 543 values per stream, then stacked with
velocity and acceleration into `[T, 1629]`. That leaves **351 always-zero
dimensions** (117 per stream × 3), plus more zeros whenever hands or face are
not detected. Rebuild the representation around the 426 real coordinates,
re-derive motion features from them, and update the scraper, H5 builders,
normalization statistics, tests, and docs together.

**Done when:** a test asserts that no always-zero padding reaches the model; H5
shapes and stats reflect the trimmed representation; the README architecture
section matches the code; existing scraped data can be migrated or clearly
documented as legacy.

**Status:** not started. Recommended now — it makes every later training run
cheaper and removes a source of confusion in the docs.

### 4. A small reproducible benchmark on a public split

Fix a split, a seed, a checkpoint, and an evaluation script that writes
machine-readable metrics (`sacrebleu` for BLEU, `jiwer` for WER or an
explicitly documented in-house equivalent). Keep it small enough to run on one
CPU/consumer GPU and to be repeated by a reviewer.

**Done when:** `RESULTS.md` contains numbers produced by a listed command from
this repository; the split file and checkpoint are available; the metrics are
saved as JSON/CSV and committed or published; the README links to the result;
and the comparison to published sign-language translation work is stated
honestly (data size, split, and metric differences included).

**Status:** not started. Until this exists, no number in this project should be
called a benchmark.

### 5. Webcam demo hardening

The revival's `scripts/webcam_demo.py` is an experimental wrapper around the
inference path. Harden it: frame buffering, landmark smoothing, model stride,
missing-hand/face handling, CPU latency measurement, and a clear on-screen
"no trained model / experimental" state.

**Done when:** the demo runs from the documented command on a normal laptop;
measured latency is recorded; errors (no camera, no checkpoint, hands out of
frame) are handled or explained; and the README describes it as experimental
without implying a working translator.

**Status:** experimental, being added in this revival.

### 6. Libras support

Brazilian Sign Language support: choose a dataset and its terms, keep the
landmark extraction language-agnostic, and add a language/config switch plus an
evaluation plan. This is a data and evaluation problem first, a model problem
second.

**Done when:** the chosen Libras dataset and its terms are documented in
[THIRD_PARTY.md](THIRD_PARTY.md); the pipeline extracts landmarks from at least
one Libras example; and the roadmap lists a concrete evaluation plan. No
accuracy claim until milestone 4 exists for that language.

**Status:** not started.

## Done in this revival (2026-11)

- Documentation rewritten for accuracy: `README.md`, `RESULTS.md`,
  `ROADMAP.md`, `docs/FAILED_APPROACHES.md`, `CONTRIBUTING.md`,
  `THIRD_PARTY.md`.
- `LICENSE` replaced with MIT © 2026 Silas Santana; the Microsoft-copied
  licensing/scaffolding removed.
- Dataset attribution added, and the large third-party ASL Citizen metadata
  CSVs removed from version control.
- Packaging added (`pyproject.toml`, dev extras), a CPU test suite, CI, and a
  CPU smoke-test path that needs no dataset, GPU, or private infrastructure.
- The incorrect "543 landmarks" claim corrected to the real 142-landmark /
  426-coordinate representation.
- Experimental inference CLI and webcam demo scaffolding added
  (`scripts/`), explicitly not yet backed by a trained model.

## Explicit non-goals / do not repeat

- Do not re-run the mean-pooled CLIP → T5 or gloss-recognizer experiments
  without reading [docs/FAILED_APPROACHES.md](docs/FAILED_APPROACHES.md) first.
- Do not report BLEU/WER without a retained artifact, a named split, and the
  command that produced it.
- Do not describe webcam or real-time translation as working until a trained
  landmark → text model exists and milestone 4 has numbers.

## Proposing changes to the roadmap

Open an issue (or a PR) with the milestone you want to change, why, and what
"done" should mean. Keep milestones small enough to finish and checkable from
the outside.
