#!/usr/bin/env python3
"""
Modal training script for ASL Encoder-Decoder Transformer.

Setup (one-time):
  1. Create a private HF dataset repo and note your username:
       https://huggingface.co/new-dataset
  2. Set HF_REPO below to "your-username/your-repo-name".
  3. Upload the HDF5 from your local machine:
       modal run modal_train.py::upload_to_hf
  4. Store your HF token as a Modal secret (name: huggingface):
       modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxx

Workflow:
    # Start / resume legacy custom training loop:
  modal run modal_train.py::train

    # Start / resume the current local recipe on Modal (recommended):
    modal run modal_train.py::train_clean

    # Start / resume max-speed recipe on Modal (throughput-first):
    modal run modal_train.py::train_clean_fast

    # Resume with a stability-focused recipe (to try escaping BLEU plateau):
    modal run modal_train.py::train_clean_recover

  # Resume from best checkpoint with LR warm restarts (for escaping deep plateau):
  modal run modal_train.py::train_warmrestart
    modal run modal_train.py::upload_checkpoints --local-dir checkpoints_new_clean --remote-dir checkpoints_new_clean

  # Download latest checkpoints to local ./checkpoints_encdec/:
    modal run modal_train.py::download_checkpoints --remote-dir checkpoints_new_clean --local-dir checkpoints_new_clean_modal

  # Show what's stored in the volume:
  modal run modal_train.py::list_volume
"""

import os
import sys
import time
import signal

import modal

# ---------------------------------------------------------------------------
# Modal app and persistent volume
# ---------------------------------------------------------------------------

app = modal.App("asl-transformer-training")

# Everything persists here between runs: the HDF5 file, checkpoints, stats cache.
volume = modal.Volume.from_name("asl-training-data", create_if_missing=True)
VOLUME_PATH = "/data"  # mount point inside container

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "torchaudio==2.3.1",
        index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "h5py==3.11.0",
        "numpy==1.26.4",
        "tqdm==4.66.4",
        "transformers==4.41.2",
        "huggingface_hub[hf_transfer]==0.23.2",
    )
    # Enable fast C-accelerated HF transfers (up to 500 MB/s inside datacenter)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    # Bake source files into the image so they're available without extra uploads.
    # If you change sign_transformer.py locally, Modal will rebuild this layer.
    .add_local_file("sign_transformer.py", remote_path="/root/sign_transformer.py")
    .add_local_file(
        "train_transformer_encdec.py", remote_path="/root/train_transformer_encdec.py"
    )
    .add_local_file("train_asl2text_t5.py", remote_path="/root/train_asl2text_t5.py")
    .add_local_file(
        "train_temporal_encoder.py", remote_path="/root/train_temporal_encoder.py"
    )
    .add_local_file(
        "train_seq2seq_diagnostic.py", remote_path="/root/train_seq2seq_diagnostic.py"
    )
)

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

TRAINING_CONFIG = dict(
    hidden_dim=512,
    num_encoder_layers=3,
    num_decoder_layers=3,
    num_heads=8,
    dropout=0.1,
    use_channel_attention=True,
    attention_reduction=8,
    batch_size=32,  # A100 40 GB can handle this with hidden=512
    eval_batch_size=64,
    epochs=300,  # will be split across many 1-hour runs
    lr=5e-5,
    warmup_epochs=5,
    grad_accum_steps=1,  # no need to accumulate on A100 with batch=32
    max_seq_len=512,
    num_workers=0,
    h5_cache_mb=512,
    amp=True,
    checkpoint_every=10,  # save numbered checkpoint every N epochs
)

H5_FILENAME = "how2sign_mediapipe_clip_1000vocab.h5"
TRANSLATOR_H5_FILENAME = "how2sign_clip_text_realigned.h5"  # CLIP-only (abandoned — features are noise)
MEDIAPIPE_TEXT_H5_FILENAME = "how2sign_mediapipe_clip_1000vocab.h5"  # mediapipe+motion+CLIP + *_texts

# ---- SET THIS before running ------------------------------------------------
# Your Hugging Face dataset repo, e.g. "silassantanajr/asl-how2sign-features"
HF_REPO = "silassantana/asl-how2sign-features"
# -----------------------------------------------------------------------------

# Abort a new epoch if less than this many seconds remain in the 1-hour window.
# This gives the current epoch time to finish and write its checkpoint.
EPOCH_GUARD_SECONDS = 300  # 5 minutes

# Hard Modal timeout (seconds).
MODAL_TIMEOUT = 21600


# Recommended "clean track" args mirrored from local runs.
TRAIN_CLEAN_ARGS = [
    "--hidden-dim",
    "512",
    "--num-encoder-layers",
    "3",
    "--num-decoder-layers",
    "3",
    "--batch-size",
    "8",
    "--grad-accum-steps",
    "4",
    "--lr",
    "2e-5",
    "--use-channel-attention",
    "--early-stop-patience",
    "10",
    "--early-stop-min-delta",
    "0.05",
    "--report-probe-samples",
    "128",
    "--report-beam-size",
    "4",
    "--report-length-penalty",
    "0.7",
    "--resume",
]


# Throughput-first args for A100 (sacrifices diagnostics for speed).
TRAIN_CLEAN_FAST_ARGS = [
    "--hidden-dim",
    "512",
    "--num-encoder-layers",
    "3",
    "--num-decoder-layers",
    "3",
    "--batch-size",
    "32",
    "--eval-batch-size",
    "64",
    "--grad-accum-steps",
    "1",
    "--lr",
    "2e-5",
    "--num-workers",
    "8",
    "--amp",
    "--use-channel-attention",
    "--early-stop-patience",
    "10",
    "--early-stop-min-delta",
    "0.05",
    "--report-probe-samples",
    "0",
    "--resume",
]


# Smoke test for the new CTC-auxiliary-loss training path. Fresh checkpoint
# dir (new ctc_head param means old checkpoints won't load), short run,
# no --resume so it always starts clean.
TRAIN_CTC_SMOKE_ARGS = [
    "--hidden-dim",
    "512",
    "--num-encoder-layers",
    "3",
    "--num-decoder-layers",
    "3",
    "--batch-size",
    "16",
    "--eval-batch-size",
    "32",
    "--grad-accum-steps",
    "2",
    "--lr",
    "2e-5",
    "--num-workers",
    "8",
    "--amp",
    "--use-channel-attention",
    "--ctc-weight",
    "0.3",
    "--epochs",
    "3",
    "--report-probe-samples",
    "0",
    "--resume",
]


# Resume recipe aimed at breaking plateaus without dropping throughput too far.
TRAIN_CLEAN_RECOVER_ARGS = [
    "--hidden-dim",
    "512",
    "--num-encoder-layers",
    "3",
    "--num-decoder-layers",
    "3",
    "--batch-size",
    "16",
    "--eval-batch-size",
    "32",
    "--grad-accum-steps",
    "2",
    "--lr",
    "1e-5",
    "--num-workers",
    "8",
    "--use-channel-attention",
    "--use-class-weights",
    "--class-weight-power",
    "0.5",
    "--class-weight-min",
    "0.2",
    "--class-weight-max",
    "5.0",
    "--early-stop-patience",
    "15",
    "--early-stop-min-delta",
    "0.005",
    "--report-probe-samples",
    "64",
    "--report-beam-size",
    "4",
    "--report-length-penalty",
    "0.7",
    "--eval-beam-size",
    "2",
    "--eval-length-penalty",
    "0.7",
    "--resume",
]


# Strict ablation B (multimodal fusion ON) for remote A100 runs.
TRAIN_ABLATION_B_ARGS = [
    "--use-multimodal-fusion",
    "--hidden-dim",
    "512",
    "--num-encoder-layers",
    "3",
    "--num-decoder-layers",
    "3",
    "--num-heads",
    "8",
    "--batch-size",
    "8",
    "--eval-batch-size",
    "8",
    "--grad-accum-steps",
    "4",
    "--max-seq-len",
    "256",
    "--num-workers",
    "8",
    "--epochs",
    "30",
    "--lr",
    "5e-5",
    "--lr-restart-every",
    "8",
    "--warmup-epochs",
    "3",
    "--label-smoothing",
    "0.05",
    "--use-channel-attention",
    "--use-class-weights",
    "--class-weight-power",
    "0.5",
    "--eval-beam-size",
    "4",
    "--eval-length-penalty",
    "1.0",
    "--eval-repetition-penalty",
    "1.2",
    "--report-probe-samples",
    "256",
    "--report-beam-size",
    "5",
    "--report-length-penalty",
    "1.2",
    "--report-repetition-penalty",
    "1.35",
    "--seed",
    "42",
]


# Visual-to-T5 translator recipes.
TRANSLATOR_SMOKE_ARGS = [
    "--t5-model",
    "t5-small",
    "--max-seq-len",
    "256",
    "--max-text-len",
    "64",
    "--max-new-tokens",
    "40",
    "--min-new-tokens",
    "4",
    "--max-new-tokens-per-frame",
    "0.06",
    "--num-beams",
    "4",
    "--decode-repetition-penalty",
    "1.35",
    "--decode-length-penalty",
    "0.8",
    "--decode-no-repeat-ngram",
    "4",
    "--enc-layers",
    "3",
    "--enc-heads",
    "8",
    "--batch-size",
    "8",
    "--eval-batch-size",
    "8",
    "--epochs",
    "8",
    "--lr",
    "5e-5",
    "--warmup-steps",
    "1000",
    "--grad-accum-steps",
    "2",
    "--num-workers",
    "8",
    "--unfreeze-decoder-last-n",
    "1",
    "--smoke-fail-epoch",
    "2",
    "--smoke-min-bleu",
    "2.0",
    "--smoke-max-wer",
    "110",
    "--seed",
    "42",
]

TRANSLATOR_FULL_ARGS = [
    "--t5-model",
    "t5-small",
    "--max-seq-len",
    "256",
    "--max-text-len",
    "64",
    "--max-new-tokens",
    "40",
    "--min-new-tokens",
    "4",
    "--max-new-tokens-per-frame",
    "0.06",
    "--num-beams",
    "4",
    "--decode-repetition-penalty",
    "1.35",
    "--decode-length-penalty",
    "0.8",
    "--decode-no-repeat-ngram",
    "4",
    "--enc-layers",
    "3",
    "--enc-heads",
    "8",
    "--batch-size",
    "8",
    "--eval-batch-size",
    "8",
    "--epochs",
    "30",
    "--lr",
    "5e-5",
    "--warmup-steps",
    "1000",
    "--grad-accum-steps",
    "2",
    "--num-workers",
    "8",
    "--unfreeze-decoder-last-n",
    "1",
    "--disable-smoke-gate",
    "--seed",
    "42",
    "--resume",
]

# Args for the mediapipe-feature translator.
# Key differences from TRANSLATOR_SMOKE_ARGS:
#   - max-seq-len 512 (mediapipe seqs are up to 2048 frames; 512 covers most utterances)
#   - enc-layers 4 (one extra layer for the richer 2141-D input)
#   - unfreeze-decoder-last-n 6 (ablation showed this matters for cross-attn to engage)
MEDIAPIPE_TRANSLATOR_SMOKE_ARGS = [
    "--t5-model", "t5-small",
    "--max-seq-len", "512",
    "--max-text-len", "64",
    "--max-new-tokens", "40",
    "--min-new-tokens", "4",
    "--max-new-tokens-per-frame", "0.06",
    "--num-beams", "4",
    "--decode-repetition-penalty", "1.35",
    "--decode-length-penalty", "0.8",
    "--decode-no-repeat-ngram", "4",
    "--enc-layers", "4",
    "--enc-heads", "8",
    "--batch-size", "8",
    "--eval-batch-size", "8",
    "--epochs", "8",
    "--lr", "5e-5",
    "--warmup-steps", "1000",
    "--grad-accum-steps", "2",
    "--num-workers", "8",
    "--unfreeze-decoder-last-n", "6",
    "--smoke-fail-epoch", "2",
    "--smoke-min-bleu", "2.0",
    "--smoke-max-wer", "110",
    "--seed", "42",
]

MEDIAPIPE_TRANSLATOR_FULL_ARGS = [
    "--t5-model", "t5-small",
    "--max-seq-len", "512",
    "--max-text-len", "64",
    "--max-new-tokens", "40",
    "--min-new-tokens", "4",
    "--max-new-tokens-per-frame", "0.06",
    "--num-beams", "4",
    "--decode-repetition-penalty", "1.35",
    "--decode-length-penalty", "0.8",
    "--decode-no-repeat-ngram", "4",
    "--enc-layers", "4",
    "--enc-heads", "8",
    "--batch-size", "8",
    "--eval-batch-size", "8",
    "--epochs", "50",
    "--lr", "5e-5",
    "--warmup-steps", "1000",
    "--grad-accum-steps", "2",
    "--num-workers", "8",
    "--unfreeze-decoder-last-n", "6",
    "--disable-smoke-gate",
    "--seed", "42",
    "--resume",
]

# ---------------------------------------------------------------------------
# Helper: RAM-backed dataset (replaces lazy HDF5 reads on A100)
# ---------------------------------------------------------------------------

_TRAIN_SOURCE = None  # module-level cache (lives for duration of container)
_VAL_SOURCE = None


def _ensure_data(h5_path: str, filename: str = H5_FILENAME) -> None:
    """
    Download the HDF5 from HuggingFace Hub into the volume if it isn't there yet.
    Runs inside the Modal container — datacenter-to-datacenter, very fast (~5 min).
    """
    if os.path.exists(h5_path):
        return

    import huggingface_hub

    token = os.environ.get("HF_TOKEN")
    print(
        f"HDF5 not found at {h5_path} — downloading {filename} from {HF_REPO} …",
        flush=True,
    )
    huggingface_hub.hf_hub_download(
        repo_id=HF_REPO,
        filename=filename,
        repo_type="dataset",
        local_dir=VOLUME_PATH,
        token=token,
    )
    volume.commit()
    print("HDF5 downloaded and committed to volume.", flush=True)


def _load_split_to_ram(h5_path, split, max_seq_len, global_mean, global_std):
    """
    Load an entire HDF5 split into RAM as numpy arrays.
    Returns (sequences_list, labels_list, idx_to_gloss, num_classes, feature_dim).
    sequences_list[i] is a float16 array already normalised and clipped.
    """
    import h5py
    import numpy as np

    print(f"[{split}] Loading split into RAM …", flush=True)
    t0 = time.time()

    with h5py.File(h5_path, "r") as f:
        raw_seqs = f[f"{split}_sequences"]  # (N, T_max, D)
        seq_lens = f[f"{split}_sequence_lengths"][:].astype(np.int32)
        all_labels = f[f"{split}_labels"][:]
        label_lens = f[f"{split}_label_lengths"][:].astype(np.int32)

        num_classes = int(f.attrs.get("num_classes", 0))
        feature_dim = int(raw_seqs.shape[2])

        if "gloss_names" in f:
            names = f["gloss_names"][:]
            idx_to_gloss = {
                i: (n.decode("utf-8") if isinstance(n, bytes) else n)
                for i, n in enumerate(names)
            }
        else:
            idx_to_gloss = {i: f"class_{i}" for i in range(num_classes)}

        # Clamp lengths to safe bounds
        max_seq_width = int(raw_seqs.shape[1])
        max_label_width = int(all_labels.shape[1])
        seq_lens = np.clip(seq_lens, 0, max_seq_width)
        label_lens = np.clip(label_lens, 0, max_label_width)
        if max_seq_len:
            seq_lens = np.minimum(seq_lens, max_seq_len)

        valid = (seq_lens > 0) & (label_lens > 0)
        valid_idx = np.flatnonzero(valid)
        print(
            f"[{split}] {len(valid_idx)}/{len(seq_lens)} valid samples — reading …",
            flush=True,
        )

        sequences = []
        labels = []
        for raw_i in valid_idx:
            sl = int(seq_lens[raw_i])
            ll = int(label_lens[raw_i])
            seq = raw_seqs[int(raw_i), :sl, :].astype(np.float32)
            seq = np.nan_to_num(seq, nan=0.0, posinf=0.0, neginf=0.0)
            if global_mean is not None:
                seq = (seq - global_mean) / global_std
            seq = np.clip(seq, -10.0, 10.0).astype(np.float16)
            sequences.append(seq)
            labels.append(all_labels[int(raw_i), :ll].astype(np.int32))

    elapsed = time.time() - t0
    print(
        f"[{split}] Loaded {len(sequences)} samples into RAM in {elapsed:.1f}s",
        flush=True,
    )
    return sequences, labels, idx_to_gloss, num_classes, feature_dim


# ---------------------------------------------------------------------------
# Helper: RAM dataset + dataloader (no HDF5 I/O during training)
# ---------------------------------------------------------------------------


def _build_ram_dataset_and_loader(
    sequences, labels, batch_size, shuffle, num_workers, collate_fn
):
    import torch
    from torch.utils.data import Dataset, DataLoader

    class RAMDataset(Dataset):
        def __init__(self, seqs, labs):
            self.seqs = seqs
            self.labs = labs

        def __len__(self):
            return len(self.seqs)

        def __getitem__(self, idx):
            import torch

            seq = torch.from_numpy(self.seqs[idx]).float()
            lab = torch.from_numpy(self.labs[idx])
            return seq, lab, len(self.seqs[idx]), len(self.labs[idx])

    ds = RAMDataset(sequences, labels)

    # Bucket sampler: sort by length within buckets to minimise padding waste
    lengths = [len(s) for s in sequences]
    lengths_arr = __import__("numpy").array(lengths, dtype="int32")

    from torch.utils.data import Sampler
    import numpy as np

    class BucketBatchSampler(Sampler):
        def __init__(self, lens, bs, shuffle=True, bucket_mult=50):
            self.lens = lens
            self.bs = bs
            self.shuffle = shuffle
            self.bucket_size = max(bs, bs * bucket_mult)

        def __len__(self):
            return (len(self.lens) + self.bs - 1) // self.bs

        def __iter__(self):
            indices = np.arange(len(self.lens))
            if self.shuffle:
                np.random.shuffle(indices)
            batches = []
            for start in range(0, len(indices), self.bucket_size):
                bucket = indices[start : start + self.bucket_size]
                order = np.argsort(self.lens[bucket], kind="stable")
                bucket = bucket[order]
                for bs in range(0, len(bucket), self.bs):
                    b = bucket[bs : bs + self.bs]
                    if len(b) > 0:
                        batches.append(b.tolist())
            if self.shuffle:
                np.random.shuffle(batches)
            for b in batches:
                yield b

    sampler = BucketBatchSampler(lengths_arr, batch_size, shuffle=shuffle)

    loader = DataLoader(
        ds,
        batch_sampler=sampler,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    return ds, loader


# ---------------------------------------------------------------------------
# Graceful shutdown helper
# ---------------------------------------------------------------------------


class _TimeoutFlag:
    """Set by SIGALRM so the training loop can finish the current epoch first."""

    def __init__(self):
        self.fired = False

    def arm(self, seconds):
        signal.signal(signal.SIGALRM, self._handler)
        signal.alarm(seconds)
        print(f"[timeout] Armed: will request graceful stop in {seconds}s", flush=True)

    def _handler(self, signum, frame):
        self.fired = True
        print(
            "[timeout] Graceful-stop flag set — will stop after this epoch's checkpoint",
            flush=True,
        )


class SoftTimeout(Exception):
    """Raised when we hit our internal soft time budget before Modal hard timeout."""

    pass


# ---------------------------------------------------------------------------
# compute_global_stats (reused from training script, runs inside container)
# ---------------------------------------------------------------------------


def _compute_global_stats_ram(sequences, feature_dim, sample_cap=2000):
    """Compute mean/std from RAM-loaded sequences."""
    import numpy as np

    rng = np.random.RandomState(42)
    indices = rng.choice(
        len(sequences), size=min(len(sequences), sample_cap), replace=False
    )
    running_sum = np.zeros(feature_dim, dtype=np.float64)
    running_sq = np.zeros(feature_dim, dtype=np.float64)
    total_frames = 0
    for i in indices:
        row = sequences[i].astype(np.float64)
        running_sum += row.sum(axis=0)
        running_sq += (row**2).sum(axis=0)
        total_frames += row.shape[0]
    mean = (running_sum / total_frames).astype("float32")
    var = (running_sq / total_frames) - (mean.astype("float64") ** 2)
    std = (
        __import__("numpy")
        .sqrt(__import__("numpy").maximum(var, 1e-8))
        .astype("float32")
    )
    return mean, std


def _compute_global_stats_h5(h5_path, split, max_seq_len=None, sample_cap=2000):
    """Compute mean/std directly from HDF5 without loading the whole split into RAM."""
    import h5py
    import numpy as np

    print(f"[{split}] Computing global feature statistics from HDF5 …", flush=True)
    with h5py.File(h5_path, "r") as f:
        ds = f[f"{split}_sequences"]
        lens = f[f"{split}_sequence_lengths"][:].astype(np.int32)
        n = len(ds)
        feature_dim = int(ds.shape[2])

        rng = np.random.RandomState(42)
        indices = rng.choice(n, size=min(n, sample_cap), replace=False)
        indices.sort()

        running_sum = np.zeros(feature_dim, dtype=np.float64)
        running_sq = np.zeros(feature_dim, dtype=np.float64)
        total_frames = 0

        for idx in indices:
            seq_len = int(min(lens[idx], max_seq_len or lens[idx]))
            if seq_len <= 0:
                continue
            row = ds[int(idx), :seq_len, :].astype(np.float64)
            row = np.nan_to_num(row, nan=0.0, posinf=0.0, neginf=0.0)
            running_sum += row.sum(axis=0)
            running_sq += (row**2).sum(axis=0)
            total_frames += seq_len

    mean = (running_sum / total_frames).astype(np.float32)
    var = (running_sq / total_frames) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    print(
        f"[{split}] Stats computed over {total_frames} frames from {len(indices)} samples.",
        flush=True,
    )
    return mean, std


# ---------------------------------------------------------------------------
# The actual training function (runs on Modal A100)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=131072,  # 128 GB RAM headroom for RAM-cached dataset + model + workers
)
def train():
    import torch
    import torch.nn as nn
    import numpy as np
    from functools import partial
    from tqdm import tqdm

    # sign_transformer.py is baked into the image at /root/
    sys.path.insert(0, "/root")
    from sign_transformer import SignLanguageTransformer, prepare_targets

    # Download HDF5 from HuggingFace if not already in the volume
    h5_path = os.path.join(VOLUME_PATH, H5_FILENAME)
    _ensure_data(h5_path)

    cfg = TRAINING_CONFIG
    save_dir = os.path.join(VOLUME_PATH, "checkpoints_encdec")
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    # -----------------------------------------------------------------------
    # 1. Global normalisation stats
    # -----------------------------------------------------------------------
    stats_cache = os.path.join(VOLUME_PATH, f"{H5_FILENAME}.train_stats.npz")
    if os.path.exists(stats_cache):
        d = np.load(stats_cache)
        global_mean, global_std = d["mean"], d["std"]
        print("Loaded normalisation stats from cache.", flush=True)
    else:
        print("Computing normalisation stats …", flush=True)
        global_mean, global_std = _compute_global_stats_h5(
            h5_path, "train", max_seq_len=cfg["max_seq_len"]
        )
        np.savez(stats_cache, mean=global_mean, std=global_std)
        volume.commit()
        print("Stats saved.", flush=True)

    # -----------------------------------------------------------------------
    # 2. Load all data into RAM
    # -----------------------------------------------------------------------
    train_seqs, train_labs, idx_to_gloss, num_classes, feature_dim = _load_split_to_ram(
        h5_path, "train", cfg["max_seq_len"], global_mean, global_std
    )
    val_seqs, val_labs, _, _, _ = _load_split_to_ram(
        h5_path, "val", cfg["max_seq_len"], global_mean, global_std
    )

    # -----------------------------------------------------------------------
    # 3. Collate fn (copied from train_transformer_encdec.py)
    # -----------------------------------------------------------------------
    def collate_fn(batch):
        sequences, labels, seq_lengths, label_lengths = zip(*batch)
        effective = [min(sl, cfg["max_seq_len"]) for sl in seq_lengths]
        max_sl = max(effective)
        feature_d = sequences[0].size(1)
        bs = len(sequences)

        padded_seqs = torch.zeros(bs, max_sl, feature_d)
        for i, seq in enumerate(sequences):
            padded_seqs[i, : effective[i]] = seq[: effective[i]]

        src_mask = torch.ones(bs, max_sl, dtype=torch.bool)
        for i, sl in enumerate(effective):
            src_mask[i, :sl] = False

        safe_ll = [
            max(0, min(int(label_lengths[i]), int(labels[i].numel())))
            for i in range(bs)
        ]
        max_ll = max(safe_ll) if safe_ll else 0
        padded_labels = torch.zeros(bs, max_ll, dtype=torch.long)
        for i, label in enumerate(labels):
            ll = safe_ll[i]
            if ll > 0:
                padded_labels[i, :ll] = label[:ll]

        return (
            padded_seqs,
            padded_labels,
            torch.LongTensor(effective),
            torch.LongTensor(safe_ll),
            src_mask,
        )

    _, train_loader = _build_ram_dataset_and_loader(
        train_seqs,
        train_labs,
        cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        collate_fn=collate_fn,
    )
    _, val_loader = _build_ram_dataset_and_loader(
        val_seqs,
        val_labs,
        cfg["eval_batch_size"],
        shuffle=False,
        num_workers=max(1, cfg["num_workers"] // 2),
        collate_fn=collate_fn,
    )

    # -----------------------------------------------------------------------
    # 4. Model, optimiser, scheduler
    # -----------------------------------------------------------------------
    model = SignLanguageTransformer(
        num_classes=num_classes,
        input_features=feature_dim,
        hidden_dim=cfg["hidden_dim"],
        num_encoder_layers=cfg["num_encoder_layers"],
        num_decoder_layers=cfg["num_decoder_layers"],
        num_heads=cfg["num_heads"],
        dropout=cfg["dropout"],
        use_channel_attention=cfg["use_channel_attention"],
        attention_reduction=cfg["attention_reduction"],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["amp"])

    def lr_lambda(epoch):
        if epoch < cfg["warmup_epochs"]:
            return (epoch + 1) / cfg["warmup_epochs"]
        progress = (epoch - cfg["warmup_epochs"]) / max(
            1, cfg["epochs"] - cfg["warmup_epochs"]
        )
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # -----------------------------------------------------------------------
    # 5. Resume from latest checkpoint if present
    # -----------------------------------------------------------------------
    best_bleu = 0.0
    start_epoch = 1
    latest_ckpt = os.path.join(save_dir, "latest_checkpoint.pt")
    if os.path.exists(latest_ckpt):
        print(f"Resuming from {latest_ckpt}", flush=True)
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_bleu = ckpt.get("best_bleu", 0.0)
        for _ in range(ckpt["epoch"]):
            scheduler.step()
        print(
            f"Resumed from epoch {start_epoch - 1}, best BLEU: {best_bleu:.2f}",
            flush=True,
        )

    # -----------------------------------------------------------------------
    # 6. Graceful-shutdown timer
    #    Fire EPOCH_GUARD_SECONDS before the hard Modal timeout so the current
    #    epoch has time to finish and flush a checkpoint.
    # -----------------------------------------------------------------------
    run_start = time.monotonic()
    soft_budget_s = max(60, MODAL_TIMEOUT - EPOCH_GUARD_SECONDS)

    def _remaining_s():
        return soft_budget_s - (time.monotonic() - run_start)

    # -----------------------------------------------------------------------
    # 7. Training loop (imported helpers from train_transformer_encdec.py)
    # -----------------------------------------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.05)

    def _train_one_epoch(epoch):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_tokens = 0
        amp_enabled = cfg["amp"]
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]", file=sys.stdout)
        for step, (seqs, labs, seq_lens, lab_lens, src_mask) in enumerate(pbar):
            # Periodically bail out before hard timeout so we can checkpoint cleanly.
            if (step % 20 == 0) and _remaining_s() <= 0:
                raise SoftTimeout()

            seqs = seqs.to(device, non_blocking=True)
            labs = labs.to(device, non_blocking=True)
            lab_lens = lab_lens.to(device, non_blocking=True)
            src_mask = src_mask.to(device, non_blocking=True)

            dec_in, dec_tgt, _ = prepare_targets(
                labs, lab_lens, sos_idx=model.sos_idx, eos_idx=model.eos_idx, pad_idx=0
            )

            with torch.amp.autocast("cuda", enabled=amp_enabled):
                logits = model(seqs, dec_in, src_key_padding_mask=src_mask)

            bs2, ml, vs = logits.shape
            loss = criterion(logits.reshape(-1, vs), dec_tgt.reshape(-1))

            if not torch.isfinite(loss):
                optimizer.zero_grad(set_to_none=True)
                amp_enabled = False
                continue

            loss_acc = loss / max(1, cfg["grad_accum_steps"])
            if scaler is not None and amp_enabled:
                scaler.scale(loss_acc).backward()
            else:
                loss_acc.backward()

            should_step = (step + 1) % max(1, cfg["grad_accum_steps"]) == 0 or (
                step + 1 == len(train_loader)
            )
            if should_step:
                if scaler is not None and amp_enabled:
                    scaler.unscale_(optimizer)
                    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    if not torch.isfinite(gn):
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            mask = dec_tgt != 0
            total_correct += ((logits.argmax(-1) == dec_tgt) & mask).sum().item()
            total_tokens += mask.sum().item()
            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        return total_loss / len(train_loader), 100 * total_correct / max(
            total_tokens, 1
        )

    def _evaluate(epoch):
        from collections import Counter
        import math

        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for step, (seqs, labs, seq_lens, lab_lens, src_mask) in enumerate(
                tqdm(val_loader, desc=f"Epoch {epoch} [val]", file=sys.stdout)
            ):
                if (step % 10 == 0) and _remaining_s() <= 0:
                    raise SoftTimeout()

                seqs = seqs.to(device)
                src_mask = src_mask.to(device)
                memory = model.encode(seqs, src_key_padding_mask=src_mask)
                preds = model.generate(
                    memory, max_length=30, memory_key_padding_mask=src_mask
                )

                for i in range(preds.size(0)):
                    pred_g = []
                    for tok in preds[i]:
                        tok = tok.item()
                        if tok == model.eos_idx:
                            break
                        if tok >= 3 and (tok - 3) in idx_to_gloss:
                            pred_g.append(idx_to_gloss[tok - 3])
                    all_preds.append(pred_g)

                    tgt_g = []
                    L = lab_lens[i].item()
                    for gi in labs[i, :L]:
                        gi = gi.item()
                        if gi in idx_to_gloss:
                            tgt_g.append(idx_to_gloss[gi])
                    all_targets.append(tgt_g)

        # BLEU-4
        clip_c = Counter()
        tot_c = Counter()
        ref_len = hyp_len = 0
        for ref, hyp in zip(all_targets, all_preds):
            ref_len += len(ref)
            hyp_len += len(hyp)
            for n in range(1, 5):
                rc = Counter(tuple(ref[i : i + n]) for i in range(len(ref) - n + 1))
                hc = Counter(tuple(hyp[i : i + n]) for i in range(len(hyp) - n + 1))
                for ng, cnt in hc.items():
                    clip_c[n] += min(cnt, rc.get(ng, 0))
                    tot_c[n] += cnt

        bleu = 0.0
        if hyp_len > 0 and all(tot_c[n] > 0 and clip_c[n] > 0 for n in range(1, 5)):
            log_b = sum((1.0 / 4) * math.log(clip_c[n] / tot_c[n]) for n in range(1, 5))
            bp = min(1.0, math.exp(1 - ref_len / hyp_len))
            bleu = bp * math.exp(log_b) * 100.0

        # WER
        total_ed = total_rl = 0
        for ref, hyp in zip(all_targets, all_preds):
            r, h = len(ref), len(hyp)
            d = [[0] * (h + 1) for _ in range(r + 1)]
            for ii in range(r + 1):
                d[ii][0] = ii
            for jj in range(h + 1):
                d[0][jj] = jj
            for ii in range(1, r + 1):
                for jj in range(1, h + 1):
                    cost = 0 if ref[ii - 1] == hyp[jj - 1] else 1
                    d[ii][jj] = min(
                        d[ii - 1][jj] + 1, d[ii][jj - 1] + 1, d[ii - 1][jj - 1] + cost
                    )
            total_ed += d[r][h]
            total_rl += r
        wer = 100.0 * total_ed / max(total_rl, 1)

        # Diagnostics
        avg_p = sum(len(p) for p in all_preds) / max(len(all_preds), 1)
        avg_t = sum(len(t) for t in all_targets) / max(len(all_targets), 1)
        print(f"  Avg pred len: {avg_p:.1f}, Avg target len: {avg_t:.1f}", flush=True)
        print("  Examples:", flush=True)
        for i in range(min(3, len(all_preds))):
            print(f"    tgt : {' '.join(all_targets[i])}", flush=True)
            print(f"    pred: {' '.join(all_preds[i])}", flush=True)

        return bleu, wer

    def _save(epoch, bleu, wer, is_best=False, is_periodic=False):
        payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "bleu": bleu,
            "wer": wer,
            "best_bleu": best_bleu,
            "use_channel_attention": cfg["use_channel_attention"],
            "attention_reduction": cfg["attention_reduction"],
            "hidden_dim": cfg["hidden_dim"],
            "num_encoder_layers": cfg["num_encoder_layers"],
            "num_decoder_layers": cfg["num_decoder_layers"],
            "num_classes": num_classes,
            "input_features": feature_dim,
        }
        torch.save(payload, os.path.join(save_dir, "latest_checkpoint.pt"))
        if is_best:
            torch.save(payload, os.path.join(save_dir, "best_model.pt"))
        if is_periodic:
            torch.save(payload, os.path.join(save_dir, f"checkpoint_epoch{epoch}.pt"))
        volume.commit()  # flush to persistent storage

    # -----------------------------------------------------------------------
    # 8. Main loop
    # -----------------------------------------------------------------------
    print("=" * 70, flush=True)
    print(f"Training from epoch {start_epoch} to {cfg['epochs']}", flush=True)
    print(
        f"  batch={cfg['batch_size']}, hidden={cfg['hidden_dim']}, AMP={cfg['amp']}",
        flush=True,
    )
    print(
        f"  soft timeout budget: {soft_budget_s}s (hard={MODAL_TIMEOUT}s)", flush=True
    )
    print("=" * 70, flush=True)

    last_epoch_seconds = None
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        # Avoid starting an epoch we likely cannot finish before hard timeout.
        remaining = _remaining_s()
        needed = (
            (last_epoch_seconds * 1.15) if last_epoch_seconds is not None else 180.0
        )
        if remaining <= max(30.0, needed):
            print(
                f"[timeout] Not enough budget for another epoch (remaining={remaining:.0f}s, need~{needed:.0f}s).",
                flush=True,
            )
            break

        epoch_t0 = time.monotonic()
        try:
            train_loss, train_acc = _train_one_epoch(epoch)
            bleu, wer = _evaluate(epoch)
        except SoftTimeout:
            print(
                f"[timeout] Soft timeout reached during epoch {epoch}; saving safe resume checkpoint.",
                flush=True,
            )
            # Save resume point as previous completed epoch to avoid skipping unfinished work.
            resume_epoch = max(start_epoch - 1, epoch - 1)
            payload = {
                "epoch": resume_epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "bleu": 0.0,
                "wer": 0.0,
                "best_bleu": best_bleu,
                "use_channel_attention": cfg["use_channel_attention"],
                "attention_reduction": cfg["attention_reduction"],
                "hidden_dim": cfg["hidden_dim"],
                "num_encoder_layers": cfg["num_encoder_layers"],
                "num_decoder_layers": cfg["num_decoder_layers"],
                "num_classes": num_classes,
                "input_features": feature_dim,
                "incomplete_epoch": epoch,
            }
            torch.save(payload, os.path.join(save_dir, "latest_checkpoint.pt"))
            volume.commit()
            break

        last_epoch_seconds = time.monotonic() - epoch_t0

        scheduler.step()

        is_best = bleu > best_bleu
        if is_best:
            best_bleu = bleu
        is_periodic = epoch % cfg["checkpoint_every"] == 0

        # Always save latest; save best + periodic when appropriate
        _save(epoch, bleu, wer, is_best=is_best, is_periodic=is_periodic)

        lr_now = scheduler.get_last_lr()[0]
        marker = " ★ best" if is_best else ""
        marker += " 💾 checkpoint" if is_periodic else ""
        print(
            f"\nEpoch {epoch}/{cfg['epochs']}  "
            f"loss={train_loss:.4f}  acc={train_acc:.1f}%  "
            f"BLEU={bleu:.2f}  WER={wer:.2f}%  "
            f"lr={lr_now:.2e}{marker}",
            flush=True,
        )

        if _remaining_s() <= 0:
            print(
                f"[timeout] Soft budget exhausted after epoch {epoch}; checkpoint already saved.",
                flush=True,
            )
            break

    print(f"\nRun complete. Best BLEU-4: {best_bleu:.2f}", flush=True)


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_clean():
    """Run train_transformer_encdec.py on Modal with the current clean recipe."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, H5_FILENAME)
    _ensure_data(h5_path)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_new_clean")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_transformer_encdec.py",
        "--data",
        h5_path,
        "--save-dir",
        save_dir,
        *TRAIN_CLEAN_ARGS,
    ]

    print("Running clean training on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    # Stream logs directly to caller terminal.
    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_clean failed with exit code {proc.returncode}")


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_clean_fast():
    """Run train_transformer_encdec.py with throughput-first args on Modal."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, H5_FILENAME)
    _ensure_data(h5_path)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_new_clean")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_transformer_encdec.py",
        "--data",
        h5_path,
        "--save-dir",
        save_dir,
        *TRAIN_CLEAN_FAST_ARGS,
    ]

    print("Running FAST clean training on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_clean_fast failed with exit code {proc.returncode}")


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_ctc_smoke(ctc_weight: float = 0.3, epochs: int = 3):
    """Smoke test for the new auxiliary-CTC-loss recognizer training path.

    New ctc_head param means old checkpoints (e.g. checkpoints_clean_v2)
    can't be resumed -- this always starts fresh in a new save dir.
    """
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, H5_FILENAME)
    _ensure_data(h5_path)

    save_dir = os.path.join(VOLUME_PATH, f"checkpoints_ctc_smoke_w{ctc_weight}")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_transformer_encdec.py",
        "--data",
        h5_path,
        "--save-dir",
        save_dir,
        *TRAIN_CTC_SMOKE_ARGS,
        "--ctc-weight",
        str(ctc_weight),
        "--epochs",
        str(epochs),
    ]

    print("Running CTC smoke training on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_ctc_smoke failed with exit code {proc.returncode}")


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_clean_recover():
    """Resume from existing clean checkpoints with a stability-focused recipe."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, H5_FILENAME)
    _ensure_data(h5_path)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_new_clean")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_transformer_encdec.py",
        "--data",
        h5_path,
        "--save-dir",
        save_dir,
        *TRAIN_CLEAN_RECOVER_ARGS,
    ]

    print("Running RECOVER clean training on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(
            f"train_clean_recover failed with exit code {proc.returncode}"
        )


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_warmrestart():
    """
    Resume from best_model.pt with cosine warm restarts to escape the BLEU plateau.
    Copies best_model.pt -> latest_checkpoint.pt before starting so the run
    resumes from the best known state rather than the last (possibly worse) state.
    """
    import subprocess
    import shutil

    h5_path = os.path.join(VOLUME_PATH, H5_FILENAME)
    _ensure_data(h5_path)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_new_clean")
    os.makedirs(save_dir, exist_ok=True)

    # Seed the resume from the best checkpoint, not the last one.
    best_ckpt = os.path.join(save_dir, "best_model.pt")
    latest_ckpt = os.path.join(save_dir, "latest_checkpoint.pt")
    if os.path.exists(best_ckpt):
        shutil.copy2(best_ckpt, latest_ckpt)
        print(f"Seeded latest_checkpoint.pt from best_model.pt", flush=True)

    cmd = [
        sys.executable,
        "/root/train_transformer_encdec.py",
        "--data",
        h5_path,
        "--save-dir",
        save_dir,
        "--hidden-dim",
        "512",
        "--num-encoder-layers",
        "3",
        "--num-decoder-layers",
        "3",
        "--batch-size",
        "16",
        "--eval-batch-size",
        "32",
        "--grad-accum-steps",
        "2",
        "--lr",
        "5e-5",  # higher LR — the restart will cycle down from here
        "--lr-restart-every",
        "8",
        "--num-workers",
        "8",
        "--use-channel-attention",
        "--use-class-weights",
        "--class-weight-power",
        "0.5",
        "--label-smoothing",
        "0.1",
        "--seed",
        "42",
        "--early-stop-patience",
        "20",
        "--early-stop-min-delta",
        "0.005",
        "--report-probe-samples",
        "64",
        "--report-beam-size",
        "4",
        "--report-length-penalty",
        "0.7",
        "--eval-beam-size",
        "2",
        "--eval-length-penalty",
        "0.7",
        "--resume",
    ]

    print("Running WARM-RESTART training on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_warmrestart failed with exit code {proc.returncode}")


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_ablation_b():
    """Run strict Ablation-B (multimodal fusion ON) on Modal."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, H5_FILENAME)
    _ensure_data(h5_path)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_ablation_B_multimodal")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_transformer_encdec.py",
        "--data",
        h5_path,
        "--save-dir",
        save_dir,
        *TRAIN_ABLATION_B_ARGS,
    ]

    print("Running ABLATION-B (multimodal fusion ON) on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_ablation_b failed with exit code {proc.returncode}")


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_translator_smoke():
    """Run fail-fast smoke test for visual-to-T5 translator."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, TRANSLATOR_H5_FILENAME)
    _ensure_data(h5_path, filename=TRANSLATOR_H5_FILENAME)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_asl2text_t5_grounded_smoke_v1")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_asl2text_t5.py",
        "--data",
        h5_path,
        "--save-dir",
        save_dir,
        *TRANSLATOR_SMOKE_ARGS,
    ]

    print("Running translator smoke test on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(
            f"train_translator_smoke failed with exit code {proc.returncode}"
        )


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_translator_full():
    """Run visual-to-T5 translator training on Modal (resume enabled)."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, TRANSLATOR_H5_FILENAME)
    _ensure_data(h5_path, filename=TRANSLATOR_H5_FILENAME)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_asl2text_t5_grounded_v1")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_asl2text_t5.py",
        "--data",
        h5_path,
        "--save-dir",
        save_dir,
        *TRANSLATOR_FULL_ARGS,
    ]

    print("Running full translator training on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(
            f"train_translator_full failed with exit code {proc.returncode}"
        )


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_translator_ablation(
    ablate_visual: str = "none", unfreeze_decoder_last_n: int = 1, epochs: int = 1
):
    """Run 1-epoch visual ablation / cross-attention capacity diagnostic for the translator."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, TRANSLATOR_H5_FILENAME)
    _ensure_data(h5_path, filename=TRANSLATOR_H5_FILENAME)

    save_dir = os.path.join(
        VOLUME_PATH,
        f"checkpoints_asl2text_t5_ablate_{ablate_visual}_unfreeze{unfreeze_decoder_last_n}",
    )
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_asl2text_t5.py",
        "--data",
        h5_path,
        "--save-dir",
        save_dir,
        *TRANSLATOR_SMOKE_ARGS,
        "--epochs",
        str(epochs),
        "--disable-smoke-gate",
        "--ablate-visual",
        ablate_visual,
        "--unfreeze-decoder-last-n",
        str(unfreeze_decoder_last_n),
    ]

    print("Running translator ablation on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(
            f"train_translator_ablation failed with exit code {proc.returncode}"
        )


# ---------------------------------------------------------------------------
# Ablation: landmarks-only (drop CLIP) vs full 2141-dim baseline
# Same hyperparams as the 2.78 BLEU-4 run, only --input-feature-dim changes.
# ---------------------------------------------------------------------------

LANDMARKS_ONLY_ARGS = [
    "--hidden-dim", "512",
    "--num-encoder-layers", "3",
    "--num-decoder-layers", "3",
    "--batch-size", "8",
    "--grad-accum-steps", "4",
    "--lr", "2e-5",
    "--use-channel-attention",
    "--early-stop-patience", "10",
    "--early-stop-min-delta", "0.05",
    "--report-probe-samples", "128",
    "--report-beam-size", "4",
    "--report-length-penalty", "0.7",
    "--input-feature-dim", "1629",   # landmarks only, no CLIP
]


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=131072,
)
def train_landmarks_only():
    """
    Ablation: How2Sign with MediaPipe landmarks only (first 1629 dims),
    dropping the 512-dim CLIP features. Same architecture and hyperparams
    as the 2.78 BLEU-4 baseline. Compare directly.

    Run:
      modal run modal_train.py::train_landmarks_only
    """
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, H5_FILENAME)
    _ensure_data(h5_path)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_landmarks_only")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_transformer_encdec.py",
        "--data",     h5_path,
        "--save-dir", save_dir,
        *LANDMARKS_ONLY_ARGS,
    ]

    print("Running landmarks-only ablation on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_landmarks_only failed with exit code {proc.returncode}")


# ---------------------------------------------------------------------------
# Phase 1: Temporal encoder pretraining (masked pose modeling)
# ---------------------------------------------------------------------------

TEMPORAL_H5_FILENAME = "youtube_asl_poses.h5"

TEMPORAL_PRETRAIN_ARGS = [
    "--hidden-dim", "256",
    "--num-layers", "4",
    "--num-heads",  "8",
    "--conv-kernel", "9",
    "--dropout",    "0.1",
    "--mask-ratio", "0.20",
    "--max-frames", "4096",
    "--epochs",     "100",
    "--batch-size", "16",
    "--grad-accum-steps", "2",
    "--lr",         "3e-4",
    "--warmup-epochs", "5",
    "--val-split",  "0.15",
    "--num-workers", "4",
    "--amp",
    "--resume",
]


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_temporal():
    """
    Phase 1: self-supervised masked pose modeling on YouTube-ASL poses.
    Saves encoder weights to /data/checkpoints_temporal/best_encoder_weights.pt
    for loading at fine-tune time.

    Upload the H5 first:
      modal volume put asl-training-data youtube_asl_poses.h5 youtube_asl_poses.h5

    Then run:
      modal run modal_train.py::train_temporal
    """
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, TEMPORAL_H5_FILENAME)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"{h5_path} not found in volume. Upload it first:\n"
            f"  modal volume put asl-training-data youtube_asl_poses.h5 youtube_asl_poses.h5"
        )

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_temporal")
    os.makedirs(save_dir, exist_ok=True)

    stats_cache = os.path.join(save_dir, "train_stats.npz")

    cmd = [
        sys.executable,
        "/root/train_temporal_encoder.py",
        "--data",       h5_path,
        "--save-dir",   save_dir,
        "--stats-cache", stats_cache,
        *TEMPORAL_PRETRAIN_ARGS,
    ]

    print("Running temporal encoder pretraining on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_temporal failed with exit code {proc.returncode}")


# ---------------------------------------------------------------------------
# Diagnostic: seq2seq cross-attention entropy test
# ---------------------------------------------------------------------------

SEGMENT_H5_FILENAME = "youtube_asl_segments.h5"

SEQ2SEQ_DIAG_ARGS = [
    "--hidden-dim", "256",
    "--enc-layers", "4",
    "--dec-layers", "3",
    "--num-heads",  "8",
    "--max-vocab",  "4000",
    "--max-src",    "600",
    "--max-tgt",    "48",
    "--epochs",     "10",
    "--batch-size", "32",
    "--grad-accum", "2",
    "--lr",         "1e-4",
    "--warmup-epochs", "2",
    "--num-workers", "4",
    "--amp",
]


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_seq2seq_diagnostic():
    """
    10-epoch diagnostic run to measure cross-attention entropy on segment-level data.
    Answers: does the temporal encoder contribute signal?

    Upload the segment H5 first:
      modal volume put asl-training-data youtube_asl_segments.h5 youtube_asl_segments.h5

    Then run:
      modal run modal_train.py::train_seq2seq_diagnostic
    """
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, SEGMENT_H5_FILENAME)
    if not os.path.exists(h5_path):
        raise FileNotFoundError(
            f"{h5_path} not found in volume. Upload it first:\n"
            f"  modal volume put asl-training-data youtube_asl_segments.h5 youtube_asl_segments.h5"
        )

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_seq2seq_diag")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_seq2seq_diagnostic.py",
        "--data",     h5_path,
        "--save-dir", save_dir,
        *SEQ2SEQ_DIAG_ARGS,
    ]

    print("Running seq2seq diagnostic on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_seq2seq_diagnostic failed with exit code {proc.returncode}")


# ---------------------------------------------------------------------------
# Utility: upload HDF5 to Hugging Face Hub from your local machine
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def upload_to_hf():
    """
    Upload the local HDF5 to your HuggingFace dataset repo.
    Uses hf_transfer for maximum speed (~300-500 MB/s on a good connection).

    Prerequisites:
      pip install huggingface_hub[hf_transfer]
      huggingface-cli login          # or set HF_TOKEN env var

    Run with:  modal run modal_train.py::upload_to_hf
    """
    if HF_REPO == "YOUR_HF_USERNAME/asl-how2sign-features":
        print("ERROR: Set HF_REPO at the top of modal_train.py before uploading.")
        return

    if not os.path.exists(H5_FILENAME):
        print(f"ERROR: {H5_FILENAME} not found in current directory.")
        return

    import subprocess

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    # Sanity-check login and show the active account.
    who = subprocess.run(
        ["hf", "auth", "whoami"],
        text=True,
        capture_output=True,
    )
    if who.returncode != 0:
        print("ERROR: Hugging Face CLI is not authenticated. Run: hf auth login")
        print(who.stderr.strip())
        return
    print(who.stdout.strip())

    # Create repo if needed (ignore failures if it already exists).
    create = subprocess.run(
        ["hf", "repo", "create", HF_REPO, "--repo-type", "dataset", "--private"],
        text=True,
        capture_output=True,
        env=os.environ.copy(),
    )
    if create.returncode == 0:
        print(f"Created dataset repo: {HF_REPO}")
    else:
        msg = (create.stderr or create.stdout or "").lower()
        if "already exists" in msg or "409" in msg:
            print(f"Dataset repo already exists: {HF_REPO}")
        else:
            print("\nERROR: Failed to create dataset repo.")
            print((create.stderr or create.stdout).strip())
            return

    print(
        f"Uploading {H5_FILENAME} ({os.path.getsize(H5_FILENAME) / 1e9:.1f} GB) to {HF_REPO} …"
    )
    print("This may take a while; if interrupted, rerun the same command.")

    upload = subprocess.run(
        [
            "hf",
            "upload",
            HF_REPO,
            H5_FILENAME,
            H5_FILENAME,
            "--repo-type",
            "dataset",
        ],
        env=os.environ.copy(),
    )
    if upload.returncode != 0:
        print("\nERROR: hf upload failed. Re-run the same command to continue/retry.")
        return

    print(f"Done. File available at: https://huggingface.co/datasets/{HF_REPO}")
    print("\nNext steps:")
    print("  1. modal secret create huggingface HF_TOKEN=hf_xxxxxxxxxxxx")
    print("  2. modal run modal_train.py::train")


@app.local_entrypoint()
def upload_file_to_hf(local_file: str, remote_name: str = ""):
    """
    Upload an arbitrary local file to the configured Hugging Face dataset repo.

    Example:
      modal run modal_train.py::upload_file_to_hf \
        --local-file how2sign_clip_text_realigned.h5 \
        --remote-name how2sign_clip_text_realigned.h5
    """
    import subprocess

    if HF_REPO == "YOUR_HF_USERNAME/asl-how2sign-features":
        print("ERROR: Set HF_REPO at the top of modal_train.py before uploading.")
        return

    if not os.path.exists(local_file):
        print(f"ERROR: {local_file} not found in current directory.")
        return

    target_name = remote_name.strip() if remote_name else os.path.basename(local_file)

    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    print(f"Uploading {local_file} -> {HF_REPO}/{target_name} …")

    upload = subprocess.run(
        [
            "hf",
            "upload",
            HF_REPO,
            local_file,
            target_name,
            "--repo-type",
            "dataset",
        ],
        env=os.environ.copy(),
    )

    if upload.returncode != 0:
        print("\nERROR: hf upload failed. Re-run the same command to continue/retry.")
        return

    print(f"Done. File available at: https://huggingface.co/datasets/{HF_REPO}")


# ---------------------------------------------------------------------------
# Utility: download checkpoints back to local machine
# ---------------------------------------------------------------------------


@app.local_entrypoint()
def upload_checkpoints(
    local_dir: str = "checkpoints_new_clean", remote_dir: str = "checkpoints_new_clean"
):
    """
    Upload local checkpoint directory into Modal volume.

    Example:
      modal run modal_train.py::upload_checkpoints --local-dir checkpoints_new_clean --remote-dir checkpoints_new_clean
    """
    import subprocess

    if not os.path.isdir(local_dir):
        print(f"ERROR: Local directory not found: {local_dir}")
        return

    print(f"Uploading {local_dir} -> volume:{remote_dir} …")
    subprocess.run(
        [
            "modal",
            "volume",
            "put",
            "asl-training-data",
            local_dir,
            remote_dir,
        ],
        check=True,
    )
    print("Upload complete.")


@app.local_entrypoint()
def download_checkpoints(
    remote_dir: str = "checkpoints_new_clean",
    local_dir: str = "checkpoints_new_clean_modal",
):
    """
    Download checkpoints from the Modal volume to a local folder.

    Example:
      modal run modal_train.py::download_checkpoints --remote-dir checkpoints_new_clean --local-dir checkpoints_new_clean_modal
    """
    import subprocess

    os.makedirs(local_dir, exist_ok=True)
    print(f"Downloading volume:{remote_dir} -> {local_dir} …")
    subprocess.run(
        [
            "modal",
            "volume",
            "get",
            "asl-training-data",
            remote_dir,
            local_dir,
        ],
        check=True,
    )
    print(f"Done. Files saved to ./{local_dir}/")


# ---------------------------------------------------------------------------
# Utility: list volume contents
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Encoder-decoder targeting English text (the real task)
# Same proven architecture as the 2.78-BLEU gloss run, now with *_texts targets
# and a 4K word-level vocabulary instead of 1K gloss tokens.
# ---------------------------------------------------------------------------

ENCDEC_TEXT_ARGS = [
    "--target-type", "text",
    "--text-vocab-size", "4000",
    "--hidden-dim", "512",
    "--num-encoder-layers", "3",
    "--num-decoder-layers", "3",
    "--num-heads", "8",
    "--batch-size", "32",
    "--eval-batch-size", "64",
    "--grad-accum-steps", "1",
    "--lr", "2e-5",
    "--warmup-epochs", "5",
    "--max-seq-len", "512",
    "--num-workers", "8",
    "--amp",
    "--use-channel-attention",
    "--attention-reduction", "8",
    "--label-smoothing", "0.05",
    "--eval-beam-size", "4",
    "--eval-length-penalty", "0.7",
    "--report-probe-samples", "128",
    "--report-beam-size", "4",
    "--report-length-penalty", "0.7",
    "--early-stop-patience", "20",
    "--early-stop-min-delta", "0.1",
    "--epochs", "150",
    "--resume",
]


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_encdec_text():
    """Train pose→English text encoder-decoder on How2Sign mediapipe features."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, MEDIAPIPE_TEXT_H5_FILENAME)
    _ensure_data(h5_path, filename=MEDIAPIPE_TEXT_H5_FILENAME)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_encdec_text_v1")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_transformer_encdec.py",
        "--data", h5_path,
        "--save-dir", save_dir,
        *ENCDEC_TEXT_ARGS,
    ]

    print("Running pose→text encoder-decoder training on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(f"train_encdec_text failed with exit code {proc.returncode}")


@app.local_entrypoint()
def list_volume():
    """Show what's stored in the Modal volume."""
    import subprocess

    subprocess.run(["modal", "volume", "ls", "asl-training-data"], check=True)


# ---------------------------------------------------------------------------
# Mediapipe-feature translator (the right features — CLIP features are noise)
# ---------------------------------------------------------------------------


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_translator_mediapipe_smoke():
    """Smoke test: pose→text translator using 2141-D mediapipe+motion+CLIP features."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, MEDIAPIPE_TEXT_H5_FILENAME)
    _ensure_data(h5_path, filename=MEDIAPIPE_TEXT_H5_FILENAME)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_asl2text_mediapipe_smoke_v1")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_asl2text_t5.py",
        "--data", h5_path,
        "--save-dir", save_dir,
        *MEDIAPIPE_TRANSLATOR_SMOKE_ARGS,
    ]

    print("Running mediapipe translator smoke test on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(
            f"train_translator_mediapipe_smoke failed with exit code {proc.returncode}"
        )


@app.function(
    image=image,
    gpu="a100-40gb",
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface")],
    timeout=MODAL_TIMEOUT,
    memory=65536,
)
def train_translator_mediapipe_full():
    """Full training: pose→text translator using 2141-D mediapipe+motion+CLIP features."""
    import subprocess

    h5_path = os.path.join(VOLUME_PATH, MEDIAPIPE_TEXT_H5_FILENAME)
    _ensure_data(h5_path, filename=MEDIAPIPE_TEXT_H5_FILENAME)

    save_dir = os.path.join(VOLUME_PATH, "checkpoints_asl2text_mediapipe_v1")
    os.makedirs(save_dir, exist_ok=True)

    cmd = [
        sys.executable,
        "/root/train_asl2text_t5.py",
        "--data", h5_path,
        "--save-dir", save_dir,
        *MEDIAPIPE_TRANSLATOR_FULL_ARGS,
    ]

    print("Running mediapipe translator full training on Modal:", flush=True)
    print(" ".join(cmd), flush=True)

    proc = subprocess.run(cmd, cwd="/root", check=False)
    volume.commit()

    if proc.returncode != 0:
        raise RuntimeError(
            f"train_translator_mediapipe_full failed with exit code {proc.returncode}"
        )
