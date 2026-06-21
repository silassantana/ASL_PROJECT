#!/usr/bin/env python3
"""
Phase 1: Temporal pose encoder pretraining via masked pose modeling.

Self-supervised BERT-style training on raw [T, 1629] pose trajectories.
Masks ~20% of frames in contiguous spans, learns to reconstruct from context.
No labels required — runs on any pose .h5 file.

After pretraining, the encoder weights are saved separately and loaded for
Phase 2 seq2seq fine-tuning.

Usage (local / debug):
  python train_temporal_encoder.py \
    --data youtube_asl_poses.h5 \
    --save-dir checkpoints_temporal \
    --epochs 50 \
    --hidden-dim 256

On Modal:
  modal run modal_train.py::train_temporal
"""

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TemporalPoseEncoder(nn.Module):
    """
    Local-conv + Transformer encoder for [T, input_dim] pose sequences.
    Used for both masked pretraining and (later) seq2seq fine-tuning.
    """

    def __init__(
        self,
        input_dim: int = 1629,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        conv_kernel: int = 8,
        max_len: int = 8192,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Depthwise-separable 1D conv captures local hand arcs (within a sign).
        # padding='same' keeps sequence length unchanged.
        self.local_conv = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=conv_kernel,
                      padding="same", groups=hidden_dim),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
        )

        self.register_buffer("pe", self._sinusoidal_pe(max_len, hidden_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN is more stable
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)

        # Learnable mask token (replaces masked frames before encoding)
        self.mask_token = nn.Parameter(torch.zeros(hidden_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Reconstruction head — only used during pretraining
        self.mask_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim),
        )

    @staticmethod
    def _sinusoidal_pe(max_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0)  # [1, max_len, d_model]

    def _embed(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, input_dim] → [B, T, hidden_dim] after proj + conv + pos-enc."""
        B, T, _ = x.shape
        h = self.input_proj(x)                       # [B, T, hidden_dim]
        h = h.transpose(1, 2)                        # [B, hidden_dim, T]
        h = self.local_conv(h)
        h = h.transpose(1, 2)                        # [B, T, hidden_dim]
        h = h + self.pe[:, :T, :]
        return h

    def encode(self, x: torch.Tensor, src_key_padding_mask=None) -> torch.Tensor:
        """Clean encode without masking — used at fine-tune / inference time."""
        h = self._embed(x)
        return self.encoder(h, src_key_padding_mask=src_key_padding_mask)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor,
                src_key_padding_mask=None):
        """
        Masked pretraining forward pass.

        Args:
            x:                [B, T, input_dim]  — normalised pose sequences
            frame_mask:       [B, T]  bool — True = masked (to predict)
            src_key_padding_mask: [B, T] bool — True = padding (to ignore)

        Returns:
            pred:   [B, T, input_dim]  — reconstruction (loss on masked frames)
        """
        B, T, _ = x.shape
        h = self.input_proj(x)                       # [B, T, hidden_dim]

        # Replace masked frames with the learnable mask token
        h = torch.where(
            frame_mask.unsqueeze(-1),
            self.mask_token.view(1, 1, -1).expand(B, T, -1),
            h,
        )

        h = h.transpose(1, 2)
        h = self.local_conv(h)
        h = h.transpose(1, 2)
        h = h + self.pe[:, :T, :]

        h = self.encoder(h, src_key_padding_mask=src_key_padding_mask)
        return self.mask_head(h)                     # [B, T, input_dim]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PoseH5Dataset(Dataset):
    """
    Reads variable-length [T, 1629] pose sequences from the YouTube-ASL H5.
    Returns (poses, video_id).
    """

    def __init__(self, h5_path: str, video_ids: list[str],
                 max_frames: int = 4096,
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None):
        self.h5_path = h5_path
        self.video_ids = video_ids
        self.max_frames = max_frames
        self.mean = mean
        self.std = std
        self._h5 = None  # opened lazily per worker

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, idx):
        self._open()
        vid = self.video_ids[idx]
        poses = self._h5["poses"][vid][:self.max_frames].astype(np.float32)

        if self.mean is not None:
            poses = (poses - self.mean) / self.std
            poses = np.clip(poses, -10.0, 10.0)

        return torch.from_numpy(poses), vid


# ---------------------------------------------------------------------------
# Sampler: bucket by sequence length to reduce padding waste
# ---------------------------------------------------------------------------

class BucketBatchSampler(Sampler):
    def __init__(self, lengths, batch_size, shuffle=True, bucket_mult=50):
        self.lengths = np.array(lengths)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(batch_size, batch_size * bucket_mult)

    def __len__(self):
        return math.ceil(len(self.lengths) / self.batch_size)

    def __iter__(self):
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            np.random.shuffle(indices)
        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start: start + self.bucket_size]
            order = np.argsort(self.lengths[bucket], kind="stable")
            bucket = bucket[order]
            for b in range(0, len(bucket), self.batch_size):
                chunk = bucket[b: b + self.batch_size]
                if len(chunk) > 0:
                    batches.append(chunk.tolist())
        if self.shuffle:
            np.random.shuffle(batches)
        for b in batches:
            yield b


def collate_poses(batch):
    """Pad variable-length pose sequences to the max length in the batch."""
    seqs, vids = zip(*batch)
    lengths = [s.shape[0] for s in seqs]
    T_max = max(lengths)
    D = seqs[0].shape[1]
    B = len(seqs)

    padded = torch.zeros(B, T_max, D)
    pad_mask = torch.ones(B, T_max, dtype=torch.bool)  # True = padding

    for i, (seq, L) in enumerate(zip(seqs, lengths)):
        padded[i, :L] = seq
        pad_mask[i, :L] = False

    return padded, torch.tensor(lengths), pad_mask, list(vids)


# ---------------------------------------------------------------------------
# Span masking
# ---------------------------------------------------------------------------

def make_span_mask(lengths: torch.Tensor, mask_ratio: float = 0.20,
                   span_len_min: int = 3, span_len_max: int = 6) -> torch.Tensor:
    """
    Generates a boolean frame mask [B, T_max] where True = masked.
    Uses random contiguous spans (SpanBERT-style) so the model can't
    trivially interpolate individual missing frames.
    """
    B = lengths.shape[0]
    T_max = int(lengths.max().item())
    mask = torch.zeros(B, T_max, dtype=torch.bool)

    for i in range(B):
        T = int(lengths[i].item())
        target = max(1, int(T * mask_ratio))
        masked = 0
        attempts = 0
        while masked < target and attempts < 200:
            span = np.random.randint(span_len_min, span_len_max + 1)
            start = np.random.randint(0, max(1, T - span + 1))
            end = min(start + span, T)
            mask[i, start:end] = True
            masked = int(mask[i, :T].sum().item())
            attempts += 1

    return mask


# ---------------------------------------------------------------------------
# Global statistics
# ---------------------------------------------------------------------------

def compute_stats(h5_path: str, video_ids: list[str],
                  sample_cap: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(42)
    sample = rng.choice(len(video_ids), size=min(len(video_ids), sample_cap),
                        replace=False)

    with h5py.File(h5_path, "r") as f:
        first = f["poses"][video_ids[0]][:]
        D = first.shape[1]
        running_sum = np.zeros(D, dtype=np.float64)
        running_sq = np.zeros(D, dtype=np.float64)
        total = 0
        for idx in tqdm(sample, desc="Computing stats", unit="video"):
            poses = f["poses"][video_ids[idx]][:].astype(np.float64)
            running_sum += poses.sum(axis=0)
            running_sq += (poses ** 2).sum(axis=0)
            total += poses.shape[0]

    mean = (running_sum / total).astype(np.float32)
    var = (running_sq / total) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, scaler, device,
                    mask_ratio, amp_enabled, grad_accum_steps):
    model.train()
    total_loss = 0.0
    total_masked = 0
    optimizer.zero_grad(set_to_none=True)

    for step, (poses, lengths, pad_mask, _) in enumerate(
        tqdm(loader, desc="  train", unit="batch", leave=False)
    ):
        poses = poses.to(device, non_blocking=True)
        pad_mask = pad_mask.to(device, non_blocking=True)

        frame_mask = make_span_mask(lengths, mask_ratio=mask_ratio).to(device)
        # Don't mask padding frames
        frame_mask = frame_mask & ~pad_mask

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred = model(poses, frame_mask, src_key_padding_mask=pad_mask)
            # MSE only on masked, non-padding positions
            active = frame_mask & ~pad_mask
            if active.sum() == 0:
                continue
            loss = ((pred[active] - poses[active]) ** 2).mean()

        loss_acc = loss / grad_accum_steps
        if amp_enabled:
            scaler.scale(loss_acc).backward()
        else:
            loss_acc.backward()

        if (step + 1) % grad_accum_steps == 0 or (step + 1 == len(loader)):
            if amp_enabled:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()
        total_masked += int(active.sum().item())

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, mask_ratio, amp_enabled):
    model.eval()
    total_loss = 0.0

    for poses, lengths, pad_mask, _ in tqdm(loader, desc="  val", unit="batch", leave=False):
        poses = poses.to(device)
        pad_mask = pad_mask.to(device)
        frame_mask = make_span_mask(lengths, mask_ratio=mask_ratio).to(device)
        frame_mask = frame_mask & ~pad_mask

        with torch.amp.autocast("cuda", enabled=amp_enabled):
            pred = model(poses, frame_mask, src_key_padding_mask=pad_mask)
            active = frame_mask & ~pad_mask
            if active.sum() == 0:
                continue
            loss = ((pred[active] - poses[active]) ** 2).mean()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path: str, epoch: int, model: nn.Module,
                    optimizer, scheduler, best_val_loss: float, args):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "best_val_loss": best_val_loss,
        "args": vars(args),
    }, path)


def save_encoder_weights(path: str, model: nn.Module):
    """Save only the encoder (no mask_head) for loading at fine-tune time."""
    state = {k: v for k, v in model.state_dict().items()
             if not k.startswith("mask_head")}
    torch.save(state, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",        required=True, help="Path to youtube_asl_poses.h5")
    parser.add_argument("--save-dir",    default="checkpoints_temporal")
    parser.add_argument("--hidden-dim",  type=int, default=256)
    parser.add_argument("--num-layers",  type=int, default=4)
    parser.add_argument("--num-heads",   type=int, default=8)
    parser.add_argument("--conv-kernel", type=int, default=9)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--mask-ratio",  type=float, default=0.20)
    parser.add_argument("--max-frames",  type=int, default=4096)
    parser.add_argument("--epochs",      type=int, default=100)
    parser.add_argument("--batch-size",  type=int, default=8)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--val-split",   type=float, default=0.15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp",         action="store_true")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--resume",      action="store_true")
    parser.add_argument("--stats-cache", default="", help="Path to .npz stats cache")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    # -----------------------------------------------------------------------
    # 1. Load video index and split train/val
    # -----------------------------------------------------------------------
    with h5py.File(args.data, "r") as f:
        raw_ids = [v.decode("utf-8") if isinstance(v, bytes) else v
                   for v in f["index"][:]]

    rng = np.random.RandomState(args.seed)
    rng.shuffle(raw_ids)
    n_val = max(1, int(len(raw_ids) * args.val_split))
    val_ids   = raw_ids[:n_val]
    train_ids = raw_ids[n_val:]
    print(f"Dataset: {len(train_ids)} train / {len(val_ids)} val", flush=True)

    # -----------------------------------------------------------------------
    # 2. Normalization stats (computed from training set)
    # -----------------------------------------------------------------------
    stats_cache = args.stats_cache or str(save_dir / "train_stats.npz")
    if os.path.exists(stats_cache):
        d = np.load(stats_cache)
        mean, std = d["mean"], d["std"]
        print("Loaded normalisation stats from cache.", flush=True)
    else:
        print("Computing normalisation stats from training set …", flush=True)
        mean, std = compute_stats(args.data, train_ids)
        np.savez(stats_cache, mean=mean, std=std)
        print(f"Stats saved to {stats_cache}", flush=True)

    # -----------------------------------------------------------------------
    # 3. Datasets and loaders
    # -----------------------------------------------------------------------
    mk_ds = lambda ids: PoseH5Dataset(
        args.data, ids, max_frames=args.max_frames, mean=mean, std=std
    )
    train_ds = mk_ds(train_ids)
    val_ds   = mk_ds(val_ids)

    def get_lengths(ds):
        with h5py.File(args.data, "r") as f:
            return [min(f["poses"][vid].shape[0], args.max_frames) for vid in ds.video_ids]

    train_lens = get_lengths(train_ds)
    val_lens   = get_lengths(val_ds)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=BucketBatchSampler(train_lens, args.batch_size, shuffle=True),
        collate_fn=collate_poses,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=BucketBatchSampler(val_lens, args.batch_size, shuffle=False),
        collate_fn=collate_poses,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------------------------------------------------------
    # 4. Model, optimizer, scheduler
    # -----------------------------------------------------------------------
    model = TemporalPoseEncoder(
        input_dim=1629,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        conv_kernel=args.conv_kernel,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        progress = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # -----------------------------------------------------------------------
    # 5. Resume
    # -----------------------------------------------------------------------
    best_val_loss = float("inf")
    start_epoch = 1
    latest_ckpt = save_dir / "latest_checkpoint.pt"

    if args.resume and latest_ckpt.exists():
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        print(f"Resumed from epoch {start_epoch - 1}, best val MSE: {best_val_loss:.6f}", flush=True)

    # -----------------------------------------------------------------------
    # 6. Training loop
    # -----------------------------------------------------------------------
    print("=" * 60, flush=True)
    print(f"Masked pose pretraining  |  epochs {start_epoch}–{args.epochs}", flush=True)
    print(f"  hidden={args.hidden_dim}  layers={args.num_layers}  heads={args.num_heads}", flush=True)
    print(f"  mask_ratio={args.mask_ratio}  batch={args.batch_size}×{args.grad_accum_steps}", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.monotonic()

        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device,
            args.mask_ratio, args.amp, args.grad_accum_steps,
        )
        val_loss = evaluate(model, val_loader, device, args.mask_ratio, args.amp)

        scheduler.step()
        elapsed = time.monotonic() - t0
        lr_now = scheduler.get_last_lr()[0]

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(str(latest_ckpt), epoch, model, optimizer, scheduler,
                        best_val_loss, args)
        if is_best:
            save_checkpoint(str(save_dir / "best_checkpoint.pt"), epoch, model,
                            optimizer, scheduler, best_val_loss, args)
            save_encoder_weights(str(save_dir / "best_encoder_weights.pt"), model)

        marker = " ★" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"train={train_loss:.6f}  val={val_loss:.6f}  "
            f"lr={lr_now:.2e}  {elapsed:.0f}s{marker}",
            flush=True,
        )

    print(f"\nDone. Best val MSE: {best_val_loss:.6f}", flush=True)
    print(f"Encoder weights: {save_dir / 'best_encoder_weights.pt'}", flush=True)


if __name__ == "__main__":
    main()
