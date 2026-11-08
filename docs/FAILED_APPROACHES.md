# Failed approaches: post-mortem

This is the detailed record behind the short "what we learned" section in the
[README](../README.md). It covers the supervised pipelines that were tried
before the revival, why they failed, and the diagnostics used to establish that.

> **All numbers on this page are historical and not reproducible from this
> repository.** They were measured in runs whose checkpoints, datasets, and logs
> were not retained: the training data lived on a private Modal volume, the H5
> files are not in git, and metrics were printed to stdout instead of being
> saved. Treat everything below as notes from the author's failure analysis,
> not as benchmarks. See [RESULTS.md](../RESULTS.md) for what can be reproduced
> today.

## Context

The project began as continuous ASL → English translation on How2Sign and
ASL Citizen data. Two feature representations were tried, and a third variant
was added to rescue the second. All three failed in the same way: the decoder
ignored the visual encoder and produced fluent-looking text or gloss sequences
that did not depend on the input.

The diagnostics used to prove that are described in the last section; they are
the reusable part of this work.

## Attempt 1 — mean-pooled CLIP frame features → T5 decoder

**Code:** `train_asl2text_t5.py` (CLIP ViT-B/32 frame features → custom visual
encoder → T5-small decoder → English text).

**Data (historical):** How2Sign with pre-extracted CLIP features, roughly 31K
training clips of 512-D CLIP vectors and English sentences.

**What was reported:** BLEU-4 ≈ 0.03–0.48 after 30 epochs on the validation
split. The decoder emitted generic English regardless of the input, for example
sentences beginning "if you're going to go for a...". No output from that run
was retained.

**Diagnosis:**

1. **Zero-ablation.** With `--ablate-visual zero` (visual features replaced by
   zeros before the encoder) the reported BLEU and WER were identical to the
   real-feature run. The visual pathway had no measurable influence.
2. **Cross-attention entropy ratio.** 0.984, where 1.0 means perfectly uniform
   attention (the decoder spreads attention evenly over all source frames
   instead of grounding on anything).
3. **CLIP retrieval probe** (`clip_retrieval_probe.py`). Mean-pooled CLIP frame
   embeddings were compared against CLIP text embeddings of the paired
   sentences. Retrieval was near chance; the mean diagonal-minus-off-diagonal
   similarity gap was about 0.0010.

**Interpretation:** CLIP embeddings of single frames encode "a person in front
of a camera", which is nearly constant across a signing clip. Mean-pooling
removes what little temporal variation remained, and the decoder learned
English n-gram statistics from the target side instead.

## Attempt 2 — MediaPipe + CLIP features → gloss recognizer

**Code:** `sign_transformer.py` (encoder–decoder Transformer with optional
channel attention and cross-modal fusion) and `train_transformer_encdec.py`
(training loop).

**Data (historical):** `how2sign_mediapipe_clip_1000vocab.h5` — 2141-D
features (1629 landmark + motion dims concatenated with 512 CLIP dims), a
1000-gloss vocabulary, How2Sign videos.

**What was reported:** BLEU 2.78 and WER 99.91%, from
`checkpoints_clean_v2/best_model.pt`. Every input decoded to roughly
"so what you want to do is a little bit of the ball...", and only 70 of the
1000 vocabulary tokens were ever predicted, all high-frequency function words.

**Diagnosis:** `diagnose_recognizer.py` ran greedy decoding on a validation
slice, captured the decoder's cross-attention, and reported the same
near-uniform attention as Attempt 1 (entropy ratio 0.984), plus prediction
length and token-diversity statistics.

**Note on checkpoints:** the model was refactored from single-stream to
dual-stream later. The old checkpoint does not load into the current
`sign_transformer.py` without `use_multimodal_fusion=False` and
`input_features=2141`. The checkpoint is on the author's private Modal volume
(`asl-training-data`), not in this repository.

## Attempt 3 — CTC auxiliary loss

**Code:** a `ctc_logits()` head in `sign_transformer.py` and the
`--ctc-weight` flag in `train_transformer_encdec.py`. An `nn.CTCLoss` over the
encoder memory was added alongside the cross-entropy decoder loss, to force an
alignment between encoder frames and gloss tokens independent of the decoder's
language model.

**What was reported:** BLEU 2.52 with CTC weight 0.3 (30 epochs) and 2.63 with
weight 0.6 (27 epochs, early stop). Early epochs produced more diverse outputs;
the decoder's target-side language model reasserted itself by around epoch 20,
and the same collapse pattern returned.

**The numbers do not reconcile.** A pre-revival migration note (since removed)
calls 2.36 the "Modal best", while other pre-revival notes report 2.78, 2.52,
and 2.63. There are no run IDs, configs, or artifacts tying any of these
numbers to a command. Treat the whole set as approximate historical notes.

**Interpretation:** the architecture and training recipe were serviceable; the
bottleneck was the representation. CTC changed early-epoch behaviour but was not
strong enough to keep the encoder grounded for the whole run.

## What the reported metrics actually measured

- **BLEU** came from an in-house implementation in `train_transformer_encdec.py:26-58`
  with no smoothing: if any n-gram count is zero, the function returns exactly
  `0.0`. It was never checked against `sacrebleu`.
- **WER** was an in-house Levenshtein implementation over gloss tokens, not
  over English text (`train_transformer_encdec.py:61-79`), and was never
  checked against `jiwer`.
- **No held-out test split was ever evaluated.** The training/eval code loads
  only `train` and `val` splits; the `test_*` datasets written by the extraction
  scripts were never read by a training or evaluation script.
- **Nothing was persisted.** Metrics were printed at `train_transformer_encdec.py:1262`
  and the only saved artifact was `best_model.pt`; there are no JSON/CSV metric
  logs, run directories, seeds tied to reported numbers, or W&B/MLflow records.

This is why the old README's "BLEU 2.78 / WER 99.91%" should not be read as a
current result: it is a remembered validation number, not a measurement a
reviewer can reproduce.

## Diagnostics methodology

These tools are the genuinely reusable part of the pre-revival work. They need
artifacts that no longer exist in the repository, so the *method* is sound but
the historical outputs are not verifiable.

| Diagnostic | File | What it measures | Historical observation |
|---|---|---|---|
| Zero-ablation | `train_asl2text_t5.py` (`--ablate-visual zero\|shuffle`) | Metrics when visual features are zeroed or batch-shuffled before the encoder | Identical BLEU/WER with real and zeroed features |
| Cross-attention entropy ratio | `diagnose_recognizer.py`; `train_seq2seq_diagnostic.py::measure_cross_attn_entropy` | `mean_entropy / log(src_len)` over decoder cross-attention. 1.0 = uniform (encoder ignored); near 0 = peaked | 0.984 for both failed pipelines |
| CLIP retrieval probe | `clip_retrieval_probe.py` (Modal-only) | Top-1/top-5 retrieval between mean-pooled CLIP frame features and CLIP text embeddings of paired sentences | Near chance; diagonal–off-diagonal gap ≈ 0.0010 |
| Representation-collapse probe | `mediapipe_collapse_probe.py` | Spread of pairwise cosine similarities for CLIP features vs landmark+motion features | CLIP features were tightly collapsed; landmark+motion features varied more (historical observation, no saved output) |

**Caveats when reusing them:** the zero-ablation path needs an H5 with CLIP
features and a checkpoint; the entropy diagnostics need a trained checkpoint
(`checkpoints_clean_v2/best_model.pt`, absent); `clip_retrieval_probe.py` is
bound to the author's Modal volume; the collapse probe needs How2Sign videos
plus CSVs that are not redistributed. None of the historical outputs were
saved, and none of the reported numbers has a recorded seed or command.

## What the failures did and did not establish

**Established (as working hypotheses, with the evidence now lost):**

- Mean-pooled static image embeddings are the wrong input representation for
  continuous signing; the decoder can and will ignore them.
- A zero-ablation control and a cross-attention entropy ratio are the right
  first checks when a multimodal decoder appears to be fluent but input-blind.
- Target-side language modelling can mask a dead encoder, and CTC alone did not
  fix it.

**Not established:**

- That landmark trajectories solve the problem. The pivot to landmarks has not
  yet been evaluated end-to-end; see [RESULTS.md](../RESULTS.md).
- Any BLEU/WER number comparable to published sign-language translation work.
- Real-time feasibility. Pre-revival "webcam inference" existed only as a
  design goal; the experimental demo in this revival has no trained model.

**What changed because of this:** the project moved to YouTube-ASL landmark
trajectories, video-level train/val splits, masked-pose self-supervised
pretraining (`train_temporal_encoder.py`), and a short diagnostic seq2seq
trainer (`train_seq2seq_diagnostic.py`) whose main output is the same attention
entropy metric. See the [README](../README.md) and [ROADMAP](../ROADMAP.md).

If you rebuild any of these pipelines, run the zero-ablation check first: if the
metrics are unchanged when the visual input is zeroed, stop and fix the
representation before tuning the model.
