#!/usr/bin/env python3
"""
Train a visual-to-text ASL translator using a pretrained T5 decoder.

This script reads the existing How2Sign H5 features and converts label indices
back into sentence text using `gloss_names`. It then trains a visual encoder
that feeds into a pretrained T5 model.
"""

import argparse
import os
import random
from dataclasses import dataclass

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput


@dataclass
class Batch:
    visual_feats: torch.Tensor
    visual_mask: torch.Tensor
    text_ids: torch.Tensor
    text_mask: torch.Tensor
    raw_texts: list


def compute_bleu(references, hypotheses, max_n=4):
    """Simple corpus BLEU-4 implementation without extra dependencies."""
    from collections import Counter
    import math

    clipped_counts = Counter()
    total_counts = Counter()
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_toks = ref.split()
        hyp_toks = hyp.split()
        ref_len += len(ref_toks)
        hyp_len += len(hyp_toks)

        for n in range(1, max_n + 1):
            ref_ngrams = Counter(tuple(ref_toks[i:i + n]) for i in range(max(0, len(ref_toks) - n + 1)))
            hyp_ngrams = Counter(tuple(hyp_toks[i:i + n]) for i in range(max(0, len(hyp_toks) - n + 1)))
            for ng, cnt in hyp_ngrams.items():
                clipped_counts[n] += min(cnt, ref_ngrams.get(ng, 0))
                total_counts[n] += cnt

    if hyp_len == 0:
        return 0.0

    log_bleu = 0.0
    for n in range(1, max_n + 1):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_bleu += (1.0 / max_n) * math.log(clipped_counts[n] / total_counts[n])

    bp = min(1.0, math.exp(1 - ref_len / hyp_len)) if hyp_len > 0 else 0.0
    return bp * math.exp(log_bleu) * 100.0


def compute_wer(references, hypotheses):
    """Corpus WER (%)."""
    total_edits = 0
    total_ref_len = 0

    for ref, hyp in zip(references, hypotheses):
        r = ref.split()
        h = hyp.split()
        m, n = len(r), len(h)
        d = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            d[i][0] = i
        for j in range(n + 1):
            d[0][j] = j

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                cost = 0 if r[i - 1] == h[j - 1] else 1
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + cost,
                )
        total_edits += d[m][n]
        total_ref_len += m

    if total_ref_len == 0:
        return 0.0
    return 100.0 * total_edits / total_ref_len


class H5VisualTextDataset(Dataset):
    def __init__(self, h5_path, split="train", max_seq_len=256):
        self.h5_path = h5_path
        self.split = split
        self.max_seq_len = max_seq_len
        self.h5 = None
        self.seqs = None

        with h5py.File(h5_path, "r") as f:
            self.feature_dim = int(f.attrs["feature_dim"])
            self.seq_lengths = f[f"{split}_sequence_lengths"][:].astype(np.int32)
            self.num_samples = int(f[f"{split}_sequences"].shape[0])
            self.has_text_targets = (f"{split}_texts" in f)

            if self.has_text_targets:
                texts = f[f"{split}_texts"][:]
                self.text_targets = [
                    (x.decode("utf-8") if isinstance(x, bytes) else str(x)).strip()
                    for x in texts
                ]
                self.labels = None
                self.label_lengths = None
                self.id_to_token = None
            else:
                self.labels = f[f"{split}_labels"][:]
                self.label_lengths = f[f"{split}_label_lengths"][:].astype(np.int32)
                names = f["gloss_names"][:]
                self.id_to_token = [n.decode("utf-8") if isinstance(n, bytes) else str(n) for n in names]
                self.text_targets = None

        self.seq_lengths = np.clip(self.seq_lengths, 0, self.max_seq_len)
        if self.has_text_targets:
            valid_text = np.array([len(t) > 0 for t in self.text_targets], dtype=bool)
            valid = (self.seq_lengths > 0) & valid_text
        else:
            valid = (self.seq_lengths > 0) & (self.label_lengths > 0)
        self.valid_idx = np.flatnonzero(valid).astype(np.int64)

    def _ensure_open(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")
            self.seqs = self.h5[f"{self.split}_sequences"]

    def __len__(self):
        return len(self.valid_idx)

    def _labels_to_text(self, label_row, label_len):
        toks = []
        for tid in label_row[:label_len]:
            tid = int(tid)
            if 0 <= tid < len(self.id_to_token):
                toks.append(self.id_to_token[tid])
        text = " ".join(toks).strip()
        return " ".join(text.split())

    def __getitem__(self, idx):
        self._ensure_open()
        ridx = int(self.valid_idx[idx])
        slen = int(self.seq_lengths[ridx])

        feats = self.seqs[ridx, :slen, :].astype(np.float32)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        if self.has_text_targets:
            text = self.text_targets[ridx]
        else:
            llen = int(self.label_lengths[ridx])
            text = self._labels_to_text(self.labels[ridx], llen)

        return torch.from_numpy(feats), text


class VisualEncoder(nn.Module):
    def __init__(self, input_dim, d_model, num_layers=3, num_heads=8, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def forward(self, x, key_padding_mask):
        x = self.proj(x)
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class VisualT5(nn.Module):
    def __init__(
        self,
        input_dim,
        t5_model_name,
        num_layers=3,
        num_heads=8,
        dropout=0.1,
        freeze_t5=True,
        unfreeze_decoder_last_n=1,
        ablate_visual="none",
    ):
        super().__init__()
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self._apply_freeze(freeze_t5=freeze_t5, unfreeze_decoder_last_n=unfreeze_decoder_last_n)
        d_model = self.t5.config.d_model
        self.visual_encoder = VisualEncoder(
            input_dim=input_dim,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.ablate_visual = ablate_visual

    def _apply_ablation(self, visual_feats):
        if self.ablate_visual == "zero":
            return torch.zeros_like(visual_feats)
        elif self.ablate_visual == "shuffle":
            perm = torch.randperm(visual_feats.size(0), device=visual_feats.device)
            return visual_feats[perm]
        return visual_feats

    def _apply_freeze(self, freeze_t5=True, unfreeze_decoder_last_n=1):
        if not freeze_t5:
            return

        # Freeze the full pretrained T5 stack by default.
        for p in self.t5.parameters():
            p.requires_grad = False

        # Keep output head + final decoder norm trainable for domain adaptation.
        for p in self.t5.lm_head.parameters():
            p.requires_grad = True
        for p in self.t5.decoder.final_layer_norm.parameters():
            p.requires_grad = True

        # Optionally unfreeze only the last few decoder blocks.
        n = max(0, int(unfreeze_decoder_last_n))
        if n > 0:
            blocks = self.t5.decoder.block
            for i in range(max(0, len(blocks) - n), len(blocks)):
                for p in blocks[i].parameters():
                    p.requires_grad = True

    def forward(self, visual_feats, visual_mask, labels=None):
        visual_feats = self._apply_ablation(visual_feats)
        enc_states = self.visual_encoder(visual_feats, key_padding_mask=~visual_mask.bool())
        enc_out = BaseModelOutput(last_hidden_state=enc_states)
        return self.t5(
            encoder_outputs=enc_out,
            attention_mask=visual_mask,
            labels=labels,
            return_dict=True,
        )

    @torch.no_grad()
    def generate(
        self,
        visual_feats,
        visual_mask,
        max_new_tokens=48,
        num_beams=4,
        min_new_tokens=2,
        max_new_tokens_per_frame=0.08,
        repetition_penalty=1.25,
        length_penalty=0.9,
        no_repeat_ngram_size=4,
    ):
        enc_states = self.visual_encoder(self._apply_ablation(visual_feats), key_padding_mask=~visual_mask.bool())
        enc_out = BaseModelOutput(last_hidden_state=enc_states)
        src_lens = visual_mask.sum(dim=1).float()
        dyn_cap = int(torch.clamp((src_lens.max() * max_new_tokens_per_frame), min=min_new_tokens, max=max_new_tokens).item())
        dyn_cap = max(min_new_tokens, dyn_cap)

        return self.t5.generate(
            encoder_outputs=enc_out,
            attention_mask=visual_mask,
            max_new_tokens=dyn_cap,
            min_new_tokens=min_new_tokens,
            num_beams=num_beams,
            do_sample=False,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            early_stopping=True,
        )


def build_collate(tokenizer, max_text_len=64):
    def _collate(items):
        feats, texts = zip(*items)
        bsz = len(feats)
        max_len = max(x.size(0) for x in feats)
        fdim = feats[0].size(1)

        visual = torch.zeros(bsz, max_len, fdim, dtype=torch.float32)
        vmask = torch.zeros(bsz, max_len, dtype=torch.long)
        for i, x in enumerate(feats):
            l = x.size(0)
            visual[i, :l] = x
            vmask[i, :l] = 1

        tok = tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=max_text_len,
            return_tensors="pt",
        )
        labels = tok["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        return Batch(
            visual_feats=visual,
            visual_mask=vmask,
            text_ids=labels,
            text_mask=tok["attention_mask"],
            raw_texts=list(texts),
        )

    return _collate


def evaluate(
    model,
    loader,
    tokenizer,
    device,
    max_new_tokens=48,
    num_beams=4,
    min_new_tokens=2,
    max_new_tokens_per_frame=0.08,
    repetition_penalty=1.25,
    length_penalty=0.9,
    no_repeat_ngram_size=4,
):
    model.eval()
    refs = []
    hyps = []

    for batch in tqdm(loader, desc="Evaluating"):
        visual = batch.visual_feats.to(device)
        vmask = batch.visual_mask.to(device)

        pred_ids = model.generate(
            visual_feats=visual,
            visual_mask=vmask,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            min_new_tokens=min_new_tokens,
            max_new_tokens_per_frame=max_new_tokens_per_frame,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        pred_txt = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        refs.extend([" ".join(x.split()) for x in batch.raw_texts])
        hyps.extend([" ".join(x.split()) for x in pred_txt])

    bleu = compute_bleu(refs, hyps)
    wer = compute_wer(refs, hyps)
    return bleu, wer, refs, hyps


def main():
    parser = argparse.ArgumentParser(description="Train ASL visual-to-text model (T5 decoder)")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--save-dir", type=str, default="checkpoints_asl2text_t5")
    parser.add_argument("--t5-model", type=str, default="t5-small")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--max-text-len", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--min-new-tokens", type=int, default=2)
    parser.add_argument("--max-new-tokens-per-frame", type=float, default=0.08)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--decode-repetition-penalty", type=float, default=1.25)
    parser.add_argument("--decode-length-penalty", type=float, default=0.9)
    parser.add_argument("--decode-no-repeat-ngram", type=int, default=4)
    parser.add_argument("--enc-layers", type=int, default=3)
    parser.add_argument("--enc-heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--train-full-t5", action="store_true",
                        help="Train all T5 weights (default freezes most T5 params)")
    parser.add_argument("--unfreeze-decoder-last-n", type=int, default=1,
                        help="When T5 is frozen, unfreeze only the last N decoder blocks")
    parser.add_argument("--ablate-visual", type=str, default="none",
                        choices=["none", "zero", "shuffle"],
                        help="Diagnostic: zero out or batch-shuffle visual features before encoding")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-smoke-gate", action="store_true",
                        help="Disable early-failure gate")
    parser.add_argument("--smoke-fail-epoch", type=int, default=2,
                        help="Apply failure gate up to this epoch (inclusive)")
    parser.add_argument("--smoke-min-bleu", type=float, default=2.0,
                        help="Minimum BLEU required to pass smoke gate")
    parser.add_argument("--smoke-max-wer", type=float, default=110.0,
                        help="Maximum WER allowed to pass smoke gate")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(args.t5_model)

    train_ds = H5VisualTextDataset(args.data, split="train", max_seq_len=args.max_seq_len)
    val_ds = H5VisualTextDataset(args.data, split="val", max_seq_len=args.max_seq_len)
    print(f"Target source: {'split_texts' if train_ds.has_text_targets else 'labels/gloss_names'}")

    collate = build_collate(tokenizer, max_text_len=args.max_text_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=max(0, args.num_workers // 2),
        collate_fn=collate,
        pin_memory=(device.type == "cuda"),
    )

    model = VisualT5(
        input_dim=train_ds.feature_dim,
        t5_model_name=args.t5_model,
        num_layers=args.enc_layers,
        num_heads=args.enc_heads,
        dropout=args.dropout,
        freeze_t5=(not args.train_full_t5),
        unfreeze_decoder_last_n=args.unfreeze_decoder_last_n,
        ablate_visual=args.ablate_visual,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable_params / max(1, total_params)
    print(f"Parameters: {total_params:,} | Trainable: {trainable_params:,} ({pct:.2f}%)")
    print(f"Visual ablation: {args.ablate_visual}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    total_steps = max(1, (len(train_loader) * args.epochs) // max(1, args.grad_accum_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: min(1.0, float(step + 1) / float(max(1, args.warmup_steps))),
    )

    best_bleu = 0.0
    start_epoch = 1

    latest_path = os.path.join(args.save_dir, "latest_checkpoint.pt")
    if args.resume and os.path.exists(latest_path):
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        best_bleu = float(ckpt.get("best_bleu", 0.0))
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        print(f"Resumed from epoch {start_epoch - 1} with best BLEU {best_bleu:.2f}")

    step_count = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
        for i, batch in enumerate(pbar):
            visual = batch.visual_feats.to(device, non_blocking=True)
            vmask = batch.visual_mask.to(device, non_blocking=True)
            labels = batch.text_ids.to(device, non_blocking=True)

            out = model(visual_feats=visual, visual_mask=vmask, labels=labels)
            loss = out.loss / max(1, args.grad_accum_steps)
            loss.backward()

            if (i + 1) % max(1, args.grad_accum_steps) == 0 or (i + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step_count += 1

            total_loss += float(out.loss.item())
            pbar.set_postfix({"loss": f"{out.loss.item():.4f}"})

        avg_loss = total_loss / max(1, len(train_loader))

        bleu, wer, refs, hyps = evaluate(
            model,
            val_loader,
            tokenizer,
            device,
            max_new_tokens=args.max_new_tokens,
            num_beams=args.num_beams,
            min_new_tokens=args.min_new_tokens,
            max_new_tokens_per_frame=args.max_new_tokens_per_frame,
            repetition_penalty=args.decode_repetition_penalty,
            length_penalty=args.decode_length_penalty,
            no_repeat_ngram_size=args.decode_no_repeat_ngram,
        )

        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train loss: {avg_loss:.4f}")
        print(f"  Val BLEU-4: {bleu:.2f} | WER: {wer:.2f}%")
        if refs:
            for k in range(min(3, len(refs))):
                print(f"  REF: {refs[k]}")
                print(f"  HYP: {hyps[k]}")

        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_bleu": max(best_bleu, bleu),
            "t5_model": args.t5_model,
            "feature_dim": train_ds.feature_dim,
            "max_seq_len": args.max_seq_len,
            "train_full_t5": args.train_full_t5,
            "unfreeze_decoder_last_n": args.unfreeze_decoder_last_n,
        }
        torch.save(ckpt, latest_path)

        if bleu > best_bleu:
            best_bleu = bleu
            torch.save(ckpt, os.path.join(args.save_dir, "best_model.pt"))
            print(f"  Saved new best checkpoint (BLEU {bleu:.2f})")

        if (
            (not args.disable_smoke_gate)
            and epoch <= args.smoke_fail_epoch
            and (bleu < args.smoke_min_bleu or wer > args.smoke_max_wer)
        ):
            print("\nSmoke gate triggered. Stopping early due to low-signal run:")
            print(f"  Epoch {epoch} | BLEU {bleu:.2f} (min {args.smoke_min_bleu:.2f})")
            print(f"  Epoch {epoch} | WER {wer:.2f}% (max {args.smoke_max_wer:.2f}%)")
            break

    print("\nTraining finished")
    print(f"Best BLEU-4: {best_bleu:.2f}")


if __name__ == "__main__":
    main()
