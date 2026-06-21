#!/usr/bin/env python3
"""
Diagnostic seq2seq training on YouTube-ASL segment-level pose→text pairs.

Goal: determine whether temporal landmark features carry signal for ASL→English
translation by measuring cross-attention entropy after a short training run.

Key metric: cross-attention entropy ratio
  1.0 = decoder ignores encoder (uniform attention = encoder useless)
  <0.8 = encoder contributing signal
  <0.6 = strong encoder usage (what we want)

Our previous CLIP-based attempts scored 0.984 — encoder fully ignored.

Architecture:
  TemporalPoseEncoder (from train_temporal_encoder.py)
  → cross-attention Transformer decoder
  → word-level vocabulary

Usage:
  python train_seq2seq_diagnostic.py \
    --data youtube_asl_segments.h5 \
    --save-dir checkpoints_seq2seq_diag \
    --epochs 10

On Modal:
  modal run modal_train.py::train_seq2seq_diagnostic
"""

import argparse
import math
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Sampler
from tqdm import tqdm

# Reuse the encoder from train_temporal_encoder.py
sys.path.insert(0, str(Path(__file__).parent))
from train_temporal_encoder import TemporalPoseEncoder, BucketBatchSampler


# ---------------------------------------------------------------------------
# Vocabulary (word-level, built from training texts)
# ---------------------------------------------------------------------------

PAD, SOS, EOS, UNK = 0, 1, 2, 3
SPECIAL = ["<pad>", "<sos>", "<eos>", "<unk>"]


def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9' ]", " ", text.lower()).split()


def build_vocab(texts: list[str], max_vocab: int = 4000) -> dict[str, int]:
    counts = Counter(tok for t in texts for tok in tokenize(t))
    vocab = {s: i for i, s in enumerate(SPECIAL)}
    for word, _ in counts.most_common(max_vocab - len(SPECIAL)):
        vocab[word] = len(vocab)
    return vocab


def encode(text: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(t, UNK) for t in tokenize(text)]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SegmentDataset(Dataset):
    def __init__(self, h5_path: str, split: str, vocab: dict[str, int],
                 max_src_frames: int = 600, max_tgt_tokens: int = 64,
                 mean=None, std=None):
        self.h5_path = h5_path
        self.split = split
        self.vocab = vocab
        self.max_src = max_src_frames
        self.max_tgt = max_tgt_tokens
        self.mean = mean
        self.std = std
        self._h5 = None

        with h5py.File(h5_path, "r") as f:
            self.n = int(f[f"{split}/n"][()])

    def _open(self):
        if self._h5 is None:
            self._h5 = h5py.File(self.h5_path, "r", swmr=True)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        self._open()
        clip = self._h5[f"{self.split}/clips/{idx}"][:self.max_src].astype(np.float32)
        text = self._h5[f"{self.split}/texts/{idx}"][()].decode("utf-8")

        if self.mean is not None:
            clip = np.clip((clip - self.mean) / self.std, -10.0, 10.0)

        tokens = encode(text, self.vocab)[:self.max_tgt]
        return (
            torch.from_numpy(clip),
            torch.tensor(tokens, dtype=torch.long),
            text,
        )


def collate_fn(batch):
    clips, token_seqs, texts = zip(*batch)
    src_lens = [c.shape[0] for c in clips]
    tgt_lens = [len(t) for t in token_seqs]
    T_max = max(src_lens)
    L_max = max(tgt_lens)
    D = clips[0].shape[1]
    B = len(clips)

    src = torch.zeros(B, T_max, D)
    src_pad = torch.ones(B, T_max, dtype=torch.bool)   # True = padding
    for i, (c, sl) in enumerate(zip(clips, src_lens)):
        src[i, :sl] = c
        src_pad[i, :sl] = False

    tgt = torch.zeros(B, L_max, dtype=torch.long)
    for i, (t, ll) in enumerate(zip(token_seqs, tgt_lens)):
        tgt[i, :ll] = t

    return src, tgt, torch.tensor(src_lens), torch.tensor(tgt_lens), src_pad, list(texts)


# ---------------------------------------------------------------------------
# Model: encoder + decoder
# ---------------------------------------------------------------------------

class ASLSeq2Seq(nn.Module):
    def __init__(self, vocab_size: int, input_dim: int = 1629,
                 hidden_dim: int = 256, enc_layers: int = 4, dec_layers: int = 3,
                 num_heads: int = 8, dropout: float = 0.1, max_len: int = 8192):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.encoder = TemporalPoseEncoder(
            input_dim=input_dim, hidden_dim=hidden_dim,
            num_layers=enc_layers, num_heads=num_heads,
            dropout=dropout, max_len=max_len,
        )
        # Remove the pretraining head — not used in seq2seq
        del self.encoder.mask_head

        self.tgt_embed = nn.Embedding(vocab_size, hidden_dim, padding_idx=PAD)
        self.tgt_pos   = nn.Embedding(512, hidden_dim)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)
        self.out_proj = nn.Linear(hidden_dim, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.tgt_embed.weight, std=0.02)
        nn.init.normal_(self.tgt_pos.weight, std=0.02)
        nn.init.zeros_(self.out_proj.bias)

    def encode(self, src, src_key_padding_mask=None):
        return self.encoder.encode(src, src_key_padding_mask=src_key_padding_mask)

    def decode(self, memory, tgt, memory_key_padding_mask=None):
        B, L = tgt.shape
        pos = torch.arange(L, device=tgt.device).unsqueeze(0)
        tgt_emb = self.tgt_embed(tgt) + self.tgt_pos(pos)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=tgt.device)
        tgt_pad = (tgt == PAD)
        out = self.decoder(
            tgt_emb, memory,
            tgt_mask=causal,
            tgt_key_padding_mask=tgt_pad,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.out_proj(out)

    def forward(self, src, tgt, src_key_padding_mask=None):
        memory = self.encode(src, src_key_padding_mask)
        return self.decode(memory, tgt, memory_key_padding_mask=src_key_padding_mask)

    @torch.no_grad()
    def greedy_decode(self, src, src_key_padding_mask=None, max_len: int = 40):
        B = src.size(0)
        memory = self.encode(src, src_key_padding_mask)
        ys = torch.full((B, 1), SOS, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)
        for _ in range(max_len):
            logits = self.decode(memory, ys, memory_key_padding_mask=src_key_padding_mask)
            next_tok = logits[:, -1, :].argmax(-1)
            ys = torch.cat([ys, next_tok.unsqueeze(1)], dim=1)
            done |= (next_tok == EOS)
            if done.all():
                break
        return ys[:, 1:]   # strip SOS


# ---------------------------------------------------------------------------
# Cross-attention entropy diagnostic
# ---------------------------------------------------------------------------

@torch.no_grad()
def measure_cross_attn_entropy(model: ASLSeq2Seq, loader: DataLoader,
                                device, n_batches: int = 20) -> float:
    """
    Hooks into the first decoder layer's cross-attention to measure how
    peaked (vs uniform) the attention distribution is.

    Entropy ratio = mean_entropy / log(src_len)
      → 1.0: uniform (encoder ignored)
      → 0.0: perfectly peaked (one encoder position)
    """
    model.eval()
    entropies = []
    attn_weights = []

    def hook(module, input, output):
        # output[1] is attn_weight [B, tgt_len, src_len] when need_weights=True
        if output[1] is not None:
            attn_weights.append(output[1].detach().cpu())

    # Hook the first decoder layer's multihead cross-attention
    first_dec = model.decoder.layers[0]
    handle = first_dec.multihead_attn.register_forward_hook(hook)

    batches_done = 0
    for src, tgt, src_lens, tgt_lens, src_pad, _ in loader:
        if batches_done >= n_batches:
            break
        src = src.to(device)
        tgt = tgt.to(device)
        src_pad = src_pad.to(device)

        memory = model.encode(src, src_key_padding_mask=src_pad)
        dec_in = tgt[:, :-1]
        if dec_in.shape[1] == 0:
            continue

        attn_weights.clear()
        _ = model.decode(memory, dec_in, memory_key_padding_mask=src_pad)

        if attn_weights:
            w = attn_weights[0]   # [B, tgt_len, src_len]
            # Entropy per query position
            w = w.clamp(min=1e-9)
            ent = -(w * w.log()).sum(-1)   # [B, tgt_len]
            # Normalise by log(src_len) per sample
            for b in range(w.shape[0]):
                sl = int(src_lens[b].item())
                max_ent = math.log(max(sl, 2))
                # Only non-padding query positions
                tl = int(tgt_lens[b].item()) - 1
                if tl > 0:
                    ratio = (ent[b, :tl] / max_ent).mean().item()
                    entropies.append(ratio)

        batches_done += 1

    handle.remove()
    return float(np.mean(entropies)) if entropies else 1.0


# ---------------------------------------------------------------------------
# BLEU-1 (fast, word-level)
# ---------------------------------------------------------------------------

def bleu1(preds: list[str], refs: list[str]) -> float:
    clip, tot = 0, 0
    for p, r in zip(preds, refs):
        p_tok = tokenize(p)
        r_tok = set(tokenize(r))
        for w in p_tok:
            tot += 1
            if w in r_tok:
                clip += 1
    return 100.0 * clip / max(tot, 1)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(model, loader, optimizer, scaler, device, amp, grad_accum):
    model.train()
    criterion = nn.CrossEntropyLoss(ignore_index=PAD, label_smoothing=0.1)
    total_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    for step, (src, tgt, src_lens, tgt_lens, src_pad, _) in enumerate(
        tqdm(loader, desc="  train", leave=False, unit="batch")
    ):
        src     = src.to(device, non_blocking=True)
        tgt     = tgt.to(device, non_blocking=True)
        src_pad = src_pad.to(device, non_blocking=True)

        dec_in  = tgt[:, :-1]
        dec_tgt = tgt[:, 1:]
        if dec_in.shape[1] == 0:
            continue

        with torch.amp.autocast("cuda", enabled=amp):
            logits = model(src, dec_in, src_key_padding_mask=src_pad)
            B, L, V = logits.shape
            loss = criterion(logits.reshape(-1, V), dec_tgt.reshape(-1))

        (scaler.scale(loss / grad_accum) if amp else (loss / grad_accum)).backward()

        if (step + 1) % grad_accum == 0 or (step + 1 == len(loader)):
            if amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader, device, vocab_inv, amp, n_examples: int = 5):
    model.eval()
    preds_all, refs_all = [], []

    for src, tgt, src_lens, tgt_lens, src_pad, texts in tqdm(
        loader, desc="  val", leave=False, unit="batch"
    ):
        src     = src.to(device)
        src_pad = src_pad.to(device)
        out = model.greedy_decode(src, src_key_padding_mask=src_pad)

        for i in range(src.size(0)):
            tokens = []
            for tok in out[i].tolist():
                if tok == EOS:
                    break
                if tok not in (PAD, SOS, UNK):
                    tokens.append(vocab_inv.get(tok, ""))
            preds_all.append(" ".join(tokens))
            refs_all.append(texts[i])

    b1 = bleu1(preds_all, refs_all)

    print(f"  Examples (pred → ref):")
    for p, r in zip(preds_all[:n_examples], refs_all[:n_examples]):
        print(f"    PRED: {p[:80]}")
        print(f"    REF:  {r[:80]}")
        print()

    return b1, preds_all, refs_all


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",       required=True)
    parser.add_argument("--save-dir",   default="checkpoints_seq2seq_diag")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--enc-layers", type=int, default=4)
    parser.add_argument("--dec-layers", type=int, default=3)
    parser.add_argument("--num-heads",  type=int, default=8)
    parser.add_argument("--max-vocab",  type=int, default=4000)
    parser.add_argument("--max-src",    type=int, default=600,
                        help="Max source frames (~40s at 15fps)")
    parser.add_argument("--max-tgt",    type=int, default=48)
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=2)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp",        action="store_true")
    parser.add_argument("--resume",     action="store_true")
    parser.add_argument("--seed",       type=int, default=42)
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
    # 1. Load stats and build vocabulary from training texts
    # -----------------------------------------------------------------------
    with h5py.File(args.data, "r") as f:
        mean = f["stats/mean"][:]
        std  = f["stats/std"][:]
        n_train = int(f["train/n"][()])
        train_texts = [f[f"train/texts/{i}"][()].decode("utf-8")
                       for i in range(n_train)]

    vocab = build_vocab(train_texts, max_vocab=args.max_vocab)
    vocab_inv = {v: k for k, v in vocab.items()}
    print(f"Vocabulary size: {len(vocab)}", flush=True)

    # -----------------------------------------------------------------------
    # 2. Datasets and loaders
    # -----------------------------------------------------------------------
    mk_ds = lambda split: SegmentDataset(
        args.data, split, vocab,
        max_src_frames=args.max_src, max_tgt_tokens=args.max_tgt,
        mean=mean, std=std,
    )
    train_ds = mk_ds("train")
    val_ds   = mk_ds("val")

    with h5py.File(args.data, "r") as f:
        train_lens = [min(f[f"train/clips/{i}"].shape[0], args.max_src)
                      for i in range(len(train_ds))]
        val_lens   = [min(f[f"val/clips/{i}"].shape[0], args.max_src)
                      for i in range(len(val_ds))]

    train_loader = DataLoader(
        train_ds,
        batch_sampler=BucketBatchSampler(train_lens, args.batch_size, shuffle=True),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=BucketBatchSampler(val_lens, args.batch_size, shuffle=False),
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------------------------------------------------------
    # 3. Model
    # -----------------------------------------------------------------------
    model = ASLSeq2Seq(
        vocab_size=len(vocab), input_dim=1629,
        hidden_dim=args.hidden_dim, enc_layers=args.enc_layers,
        dec_layers=args.dec_layers, num_heads=args.num_heads,
        dropout=args.dropout,
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    def lr_lambda(epoch):
        if epoch < args.warmup_epochs:
            return (epoch + 1) / args.warmup_epochs
        t = (epoch - args.warmup_epochs) / max(1, args.epochs - args.warmup_epochs)
        return 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # -----------------------------------------------------------------------
    # 4. Resume
    # -----------------------------------------------------------------------
    start_epoch = 1
    best_b1 = 0.0
    ckpt_path = save_dir / "latest_checkpoint.pt"
    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_b1 = ckpt.get("best_b1", 0.0)
        print(f"Resumed from epoch {start_epoch-1}", flush=True)

    # -----------------------------------------------------------------------
    # 5. Training loop
    # -----------------------------------------------------------------------
    print("=" * 60, flush=True)
    print(f"Seq2seq diagnostic  |  {len(train_ds)} train / {len(val_ds)} val segments", flush=True)
    print(f"  hidden={args.hidden_dim}  enc={args.enc_layers}  dec={args.dec_layers}", flush=True)
    print("=" * 60, flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.monotonic()
        loss = train_epoch(model, train_loader, optimizer, scaler,
                           device, args.amp, args.grad_accum)
        b1, _, _ = evaluate(model, val_loader, device, vocab_inv, args.amp)

        # Cross-attention entropy (every epoch — this is the key diagnostic)
        entropy = measure_cross_attn_entropy(model, val_loader, device, n_batches=30)

        scheduler.step()
        elapsed = time.monotonic() - t0

        is_best = b1 > best_b1
        if is_best:
            best_b1 = b1

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_b1": best_b1,
            "vocab": vocab,
        }, ckpt_path)
        if is_best:
            torch.save(torch.load(ckpt_path, weights_only=False),
                       save_dir / "best_checkpoint.pt")

        marker = " ★" if is_best else ""
        print(
            f"Epoch {epoch:3d}/{args.epochs}  "
            f"loss={loss:.4f}  BLEU-1={b1:.1f}  "
            f"xattn_entropy={entropy:.4f}  "
            f"{elapsed:.0f}s{marker}",
            flush=True,
        )

        # Early signal check
        if epoch == 3:
            if entropy > 0.95:
                print("\n[WARNING] Cross-attention entropy still near 1.0 after 3 epochs.", flush=True)
                print("  Encoder may not be contributing signal. Consider reviewing data quality.", flush=True)
            else:
                print(f"\n[SIGNAL] Encoder contributing — entropy={entropy:.4f} at epoch 3.", flush=True)

    print(f"\nFinal cross-attention entropy: {entropy:.4f}", flush=True)
    print(f"Best BLEU-1: {best_b1:.1f}", flush=True)
    if entropy < 0.80:
        print("VERDICT: Encoder is useful. Continue to full training.", flush=True)
    elif entropy < 0.92:
        print("VERDICT: Weak encoder signal. May improve with more data/pretraining.", flush=True)
    else:
        print("VERDICT: Encoder ignored. Architecture or data issue.", flush=True)


if __name__ == "__main__":
    main()
