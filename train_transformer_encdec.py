#!/usr/bin/env python3
"""
Training script for Encoder-Decoder Transformer (NO CTC!).

This uses standard cross-entropy loss with teacher forcing.
Much more stable than CTC for continuous sign recognition.
"""

import argparse
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler, Subset
import numpy as np
from tqdm import tqdm
import json
import os
import random
from functools import partial
from collections import Counter

from sign_transformer import SignLanguageTransformer, prepare_targets


def compute_bleu(references, hypotheses, max_n=4):
    """Corpus BLEU-4 with brevity penalty (no external deps)."""
    clipped_counts = Counter()
    total_counts = Counter()
    ref_len = 0
    hyp_len = 0

    for ref, hyp in zip(references, hypotheses):
        ref_len += len(ref)
        hyp_len += len(hyp)
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(
                tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)
            )
            hyp_ngrams = Counter(
                tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1)
            )
            for ng, cnt in hyp_ngrams.items():
                clipped_counts[n] += min(cnt, ref_ngrams.get(ng, 0))
                total_counts[n] += cnt

    if hyp_len == 0:
        return 0.0

    import math
    log_bleu = 0.0
    for n in range(1, max_n + 1):
        if total_counts[n] == 0 or clipped_counts[n] == 0:
            return 0.0
        log_bleu += (1.0 / max_n) * math.log(clipped_counts[n] / total_counts[n])

    bp = min(1.0, math.exp(1 - ref_len / hyp_len)) if hyp_len > 0 else 0.0
    return bp * math.exp(log_bleu) * 100.0


def compute_wer(references, hypotheses):
    """Corpus-level Word Error Rate (edit distance)."""
    total_edits = 0
    total_ref_len = 0
    for ref, hyp in zip(references, hypotheses):
        # Standard DP edit distance
        r, h = len(ref), len(hyp)
        d = [[0] * (h + 1) for _ in range(r + 1)]
        for i in range(r + 1):
            d[i][0] = i
        for j in range(h + 1):
            d[0][j] = j
        for i in range(1, r + 1):
            for j in range(1, h + 1):
                cost = 0 if ref[i - 1] == hyp[j - 1] else 1
                d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + cost)
        total_edits += d[r][h]
        total_ref_len += r
    return (100.0 * total_edits / total_ref_len) if total_ref_len > 0 else 0.0


def _pieces_to_words(tokens):
    """Convert SentencePiece/BPE tokens to whitespace word tokens for eval readability."""
    if not tokens:
        return tokens
    # If this is not SentencePiece-style text, leave untouched.
    if not any('▁' in t for t in tokens):
        return tokens

    pieces = [t for t in tokens if t and t not in ('<unk>', '<s>', '</s>')]
    text = ''.join(pieces).replace('▁', ' ').strip()
    if not text:
        return []
    return text.split()


def _compute_global_stats(h5_path, split, max_seq_len=None, sample_cap=2000):
    """Compute global (mean, std) per feature across the split.

    Reads up to *sample_cap* randomly-chosen samples to keep the scan fast.
    Returns (mean, std) each of shape [feature_dim].
    """
    stats_cache = f"{h5_path}.{split}_stats.npz"
    if os.path.exists(stats_cache):
        d = np.load(stats_cache)
        return d['mean'], d['std']

    print(f"[{split}] Computing global feature statistics (cached to {stats_cache}) …")
    with h5py.File(h5_path, 'r') as f:
        ds = f[f'{split}_sequences']
        lens = f[f'{split}_sequence_lengths'][:].astype(np.int32)
        n = len(ds)
        feature_dim = ds.shape[2] if ds.ndim == 3 else int(f.attrs.get('feature_dim', 512))

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
            running_sq += (row ** 2).sum(axis=0)
            total_frames += seq_len

    mean = (running_sum / total_frames).astype(np.float32)
    var = (running_sq / total_frames) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)

    np.savez(stats_cache, mean=mean, std=std)
    print(f"[{split}] Stats computed over {total_frames} frames from {len(indices)} samples.")
    return mean, std


def build_text_vocab(h5_path, vocab_size=4000):
    """Build word-level vocabulary from train_texts in an H5 file.

    Returns (word_to_idx, idx_to_word) where word indices start at 0 and
    get shifted by +3 inside prepare_targets (PAD=0, SOS=1, EOS=2).
    """
    with h5py.File(h5_path, 'r') as f:
        if 'train_texts' not in f:
            raise ValueError(
                "H5 file has no train_texts dataset. Run add_texts_local.py first."
            )
        texts = [
            (t.decode('utf-8') if isinstance(t, bytes) else str(t)).strip()
            for t in f['train_texts'][:]
        ]

    counts = Counter()
    for text in texts:
        counts.update(text.lower().split())

    idx_to_word = [w for w, _ in counts.most_common(vocab_size)]
    word_to_idx = {w: i for i, w in enumerate(idx_to_word)}
    return word_to_idx, idx_to_word


class H5Dataset(Dataset):
    """Lazy HDF5 dataset with metadata in RAM and bounded sample reads."""
    def __init__(self, h5_path, split='train', max_seq_len=None, h5_cache_mb=256,
                 global_mean=None, global_std=None, input_feature_dim=None,
                 target_type='gloss', text_vocab=None):
        self.h5_path = h5_path
        self.split = split
        self.max_seq_len = max_seq_len
        self.h5_cache_bytes = int(h5_cache_mb * 1024 * 1024)
        self.h5_f = None
        self.seqs_ds = None
        self.global_mean = global_mean
        self.global_std = global_std
        self.input_feature_dim = input_feature_dim  # if set, slice first N dims
        self.target_type = target_type

        with h5py.File(h5_path, 'r') as f:
            self.length = len(f[f'{split}_sequences'])
            self._total_raw = self.length

            # Get actual feature_dim from the data itself
            sample_seq = f[f'{split}_sequences'][0]
            if len(sample_seq.shape) == 2:
                raw_feature_dim = sample_seq.shape[1]
            else:
                raw_feature_dim = int(f.attrs.get('feature_dim', 512))
            self.feature_dim = input_feature_dim if input_feature_dim else raw_feature_dim

            max_seq_width = int(f[f'{split}_sequences'].shape[1])
            self._seq_lengths = f[f'{split}_sequence_lengths'][:].astype(np.int32)
            if self.max_seq_len is not None:
                self._effective_seq_lengths = np.minimum(self._seq_lengths, self.max_seq_len)
            else:
                self._effective_seq_lengths = self._seq_lengths.copy()
            self._seq_lengths = np.clip(self._seq_lengths, 0, max_seq_width)
            self._effective_seq_lengths = np.clip(self._effective_seq_lengths, 0, max_seq_width)

            self._is_flat = (len(f[f'{split}_sequences'].shape) == 2)

            if target_type == 'text':
                # Text mode: tokenize *_texts with the provided word vocabulary.
                if text_vocab is None:
                    raise ValueError("text_vocab must be provided when target_type='text'")
                word_to_idx, idx_to_word = text_vocab
                self.num_classes = len(idx_to_word)
                self.idx_to_gloss = {i: w for i, w in enumerate(idx_to_word)}

                texts_key = f'{split}_texts'
                if texts_key not in f:
                    raise ValueError(
                        f"H5 file has no {texts_key}. Run add_texts_local.py first."
                    )
                raw_texts = [
                    (t.decode('utf-8') if isinstance(t, bytes) else str(t)).strip()
                    for t in f[texts_key][:]
                ]
                self._labels = []
                self._label_lengths = np.zeros(self.length, dtype=np.int32)
                for i, text in enumerate(raw_texts):
                    tokens = np.array(
                        [word_to_idx[w] for w in text.lower().split() if w in word_to_idx],
                        dtype=np.int32,
                    )
                    self._labels.append(tokens)
                    self._label_lengths[i] = len(tokens)
            else:
                # Gloss mode: read integer label indices from *_labels.
                if 'num_classes' in f.attrs:
                    self.num_classes = int(f.attrs['num_classes'])
                elif 'num_classes' in f:
                    self.num_classes = int(f['num_classes'][()])
                else:
                    raise ValueError("Cannot find num_classes in HDF5 file")

                if 'gloss_names' in f:
                    gloss_names = f['gloss_names'][:]
                    self.idx_to_gloss = {
                        i: name.decode('utf-8') if isinstance(name, bytes) else name
                        for i, name in enumerate(gloss_names)
                    }
                elif 'gloss_to_idx' in f.attrs:
                    gloss_mapping = json.loads(f.attrs['gloss_to_idx'])
                    self.idx_to_gloss = {v: k for k, v in gloss_mapping.items()}
                elif 'gloss_to_idx' in f:
                    gloss_mapping = json.loads(f['gloss_to_idx'][()])
                    self.idx_to_gloss = {v: k for k, v in gloss_mapping.items()}
                else:
                    self.idx_to_gloss = {i: f'class_{i}' for i in range(self.num_classes)}

                all_labels = f[f'{split}_labels'][:]
                max_label_width = int(all_labels.shape[1]) if all_labels.ndim == 2 else 0
                self._label_lengths = f[f'{split}_label_lengths'][:].astype(np.int32)

                recovered = 0
                invalid_len = (self._label_lengths < 0) | (self._label_lengths > max_label_width)
                if np.any(invalid_len):
                    bad_idx = np.flatnonzero(invalid_len)
                    for i in bad_idx:
                        row = all_labels[i]
                        nz = np.flatnonzero(row != 0)
                        guessed = int(nz[-1] + 1) if nz.size > 0 else 0
                        self._label_lengths[i] = guessed
                        recovered += int(guessed > 0)

                self._label_lengths = np.clip(self._label_lengths, 0, max_label_width)
                self._labels = [
                    all_labels[i, :int(self._label_lengths[i])].astype(np.int32)
                    for i in range(self.length)
                ]
                if recovered > 0:
                    print(f"[{split}] Recovered label lengths for {recovered} samples.", flush=True)

            # Keep only samples that have at least 1 frame and 1 label after sanitization.
            valid = (self._effective_seq_lengths > 0) & (self._label_lengths > 0)
            self._valid_indices = np.flatnonzero(valid).astype(np.int64)
            self.length = int(self._valid_indices.size)
            dropped = int(self._total_raw - self.length)
            if dropped > 0:
                print(
                    f"[{split}] Dropping {dropped}/{self._total_raw} invalid samples "
                    f"(empty/corrupted lengths).",
                    flush=True,
                )

    def _ensure_open(self):
        if self.h5_f is None:
            self.h5_f = h5py.File(
                self.h5_path,
                'r',
                rdcc_nbytes=self.h5_cache_bytes,
                rdcc_nslots=1_000_003,
            )
            self.seqs_ds = self.h5_f[f'{self.split}_sequences']

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        self._ensure_open()

        real_idx = int(self._valid_indices[idx])

        seq_len = int(self._effective_seq_lengths[real_idx])
        if self._is_flat:
            raw = self.seqs_ds[real_idx, :seq_len * self.feature_dim]
            sequence = raw.reshape(seq_len, self.feature_dim).astype(np.float32)
        else:
            sequence = self.seqs_ds[real_idx, :seq_len, :self.feature_dim].astype(np.float32)

        # Normalize using global dataset statistics (preserves cross-sample scale).
        sequence = np.nan_to_num(sequence, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if self.global_mean is not None:
            sequence = (sequence - self.global_mean) / self.global_std
        sequence = np.clip(sequence, -10.0, 10.0)

        return (
            torch.from_numpy(sequence),
            torch.from_numpy(self._labels[real_idx]),
            seq_len,
            int(self._label_lengths[real_idx]),
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state['h5_f'] = None
        state['seqs_ds'] = None
        return state

    def __del__(self):
        if getattr(self, 'h5_f', None) is not None:
            try:
                self.h5_f.close()
            except Exception:
                pass


class BucketBatchSampler(Sampler):
    """Shuffle globally, sort locally by sequence length, then batch."""

    def __init__(self, lengths, batch_size, shuffle=True, bucket_multiplier=50):
        self.lengths = np.asarray(lengths, dtype=np.int32)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = max(batch_size, batch_size * bucket_multiplier)

    def __len__(self):
        return (len(self.lengths) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        indices = np.arange(len(self.lengths))
        if self.shuffle:
            np.random.shuffle(indices)

        batches = []
        for start in range(0, len(indices), self.bucket_size):
            bucket = indices[start:start + self.bucket_size]
            order = np.argsort(self.lengths[bucket], kind='stable')
            bucket = bucket[order]
            for batch_start in range(0, len(bucket), self.batch_size):
                batch = bucket[batch_start:batch_start + self.batch_size]
                if len(batch) > 0:
                    batches.append(batch.tolist())

        if self.shuffle:
            np.random.shuffle(batches)

        for batch in batches:
            yield batch


def collate_fn(batch, max_seq_len=None):
    """Collate with padding. Returns src_key_padding_mask for the encoder."""
    sequences, labels, seq_lengths, label_lengths = zip(*batch)

    effective_seq_lengths = []
    trimmed_sequences = []
    for seq, seq_len in zip(sequences, seq_lengths):
        if max_seq_len is not None:
            trimmed_len = min(seq_len, max_seq_len)
            trimmed_sequences.append(seq[:trimmed_len])
            effective_seq_lengths.append(trimmed_len)
        else:
            trimmed_sequences.append(seq)
            effective_seq_lengths.append(seq_len)
    
    # Pad sequences
    max_seq_len_batch = max(effective_seq_lengths)
    feature_dim = trimmed_sequences[0].size(1)
    batch_size = len(sequences)
    
    padded_seqs = torch.zeros(batch_size, max_seq_len_batch, feature_dim)
    for i, seq in enumerate(trimmed_sequences):
        padded_seqs[i, :effective_seq_lengths[i]] = seq

    # Build src_key_padding_mask: True where padded
    src_key_padding_mask = torch.ones(batch_size, max_seq_len_batch, dtype=torch.bool)
    for i, slen in enumerate(effective_seq_lengths):
        src_key_padding_mask[i, :slen] = False
    
    # Pad labels
    safe_label_lengths = [max(0, min(int(label_lengths[i]), int(labels[i].numel()))) for i in range(len(labels))]
    max_label_len = max(safe_label_lengths) if safe_label_lengths else 0
    padded_labels = torch.zeros(batch_size, max_label_len, dtype=torch.long)
    for i, label in enumerate(labels):
        ll = safe_label_lengths[i]
        if ll > 0:
            padded_labels[i, :ll] = label[:ll]
    
    return (
        padded_seqs,
        padded_labels,
        torch.LongTensor(effective_seq_lengths),
        torch.LongTensor(safe_label_lengths),
        src_key_padding_mask,
    )


def train_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler=None,
    grad_accum_steps=1,
    use_amp=False,
    pad_idx=0,
    class_weights=None,
    label_smoothing=0.05,
    ctc_weight=0.0,
):
    """Train one epoch."""
    model.train()
    total_loss = 0.0
    total_ce_loss = 0.0
    total_ctc_loss = 0.0
    total_correct = 0
    total_tokens = 0
    
    criterion = nn.CrossEntropyLoss(
        ignore_index=pad_idx,
        label_smoothing=label_smoothing,
        weight=class_weights,
    )
    ctc_criterion = nn.CTCLoss(blank=0, zero_infinity=True) if ctc_weight > 0 else None
    amp_enabled = use_amp
    
    pbar = tqdm(loader, desc='Training')
    optimizer.zero_grad(set_to_none=True)
    for step, (sequences, labels, seq_lengths, label_lengths, src_padding_mask) in enumerate(pbar):
        sequences = sequences.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        seq_lengths = seq_lengths.to(device, non_blocking=True)
        label_lengths = label_lengths.to(device, non_blocking=True)
        src_padding_mask = src_padding_mask.to(device, non_blocking=True)
        
        # Prepare decoder targets
        decoder_input, decoder_target, target_lengths = prepare_targets(
            labels, label_lengths, 
            sos_idx=model.sos_idx, 
            eos_idx=model.eos_idx,
            pad_idx=pad_idx
        )
        
        # Forward pass (optionally mixed precision)
        with torch.amp.autocast(device_type='cuda', enabled=amp_enabled):
            logits, memory = model(
                sequences, decoder_input, src_key_padding_mask=src_padding_mask, return_memory=True
            )  # logits: [batch, max_len+1, vocab_size], memory: [batch, src_len, hidden_dim]
        
        # Compute seq2seq cross-entropy loss
        batch_size, max_len, vocab_size = logits.shape
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = decoder_target.reshape(-1)

        ce_loss = criterion(logits_flat, targets_flat)

        # Compute CTC auxiliary loss (operates on encoder memory directly,
        # independent of the decoder's target-side language model).
        ctc_loss = None
        if ctc_criterion is not None:
            with torch.amp.autocast(device_type='cuda', enabled=False):
                ctc_log_probs = model.ctc_logits(memory.float())  # [src_len, batch, num_classes+1]
                # CTC targets: raw 0-indexed gloss ids shifted by +1 (0 reserved for blank).
                ctc_targets = (labels + 1).clamp(min=1)
                # input_lengths/target_lengths must be on CPU for nn.CTCLoss.
                ctc_input_lengths = seq_lengths.clamp(max=memory.size(1)).cpu()
                ctc_target_lengths = label_lengths.clamp(min=0).cpu()
                ctc_loss = ctc_criterion(
                    ctc_log_probs, ctc_targets, ctc_input_lengths, ctc_target_lengths
                )

        if ctc_loss is not None:
            loss = ce_loss + ctc_weight * ctc_loss
        else:
            loss = ce_loss

        if not torch.isfinite(loss):
            optimizer.zero_grad(set_to_none=True)
            if amp_enabled:
                amp_enabled = False
                pbar.write('Encountered non-finite AMP loss; falling back to full precision.')
            pbar.set_postfix({'loss': 'non-finite', 'acc': 'skip'})
            continue
        loss_for_backward = loss / grad_accum_steps
        
        # Backward + optimizer step (with gradient accumulation)
        if scaler is not None and amp_enabled:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = ((step + 1) % grad_accum_steps == 0) or (step + 1 == len(loader))
        if should_step:
            if scaler is not None and amp_enabled:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    amp_enabled = False
                    pbar.write('Encountered non-finite AMP gradients; falling back to full precision.')
                    continue
                scaler.step(optimizer)
                scaler.update()
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if not torch.isfinite(grad_norm):
                    optimizer.zero_grad(set_to_none=True)
                    continue
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        
        # Compute accuracy (on non-padding tokens)
        predictions = logits.argmax(dim=-1)
        mask = decoder_target != pad_idx
        correct = ((predictions == decoder_target) & mask).sum().item()
        total_correct += correct
        total_tokens += mask.sum().item()
        
        total_loss += loss.item()
        total_ce_loss += ce_loss.item()
        if ctc_loss is not None:
            total_ctc_loss += ctc_loss.item()
        
        postfix = {
            'loss': f'{loss.item():.4f}',
            'ce': f'{ce_loss.item():.4f}',
            'acc': f'{100*correct/max(mask.sum().item(), 1):.1f}%'
        }
        if ctc_loss is not None:
            postfix['ctc'] = f'{ctc_loss.item():.4f}'
        pbar.set_postfix(postfix)
    
    avg_loss = total_loss / len(loader)
    avg_ce_loss = total_ce_loss / len(loader)
    avg_ctc_loss = total_ctc_loss / len(loader) if ctc_weight > 0 else 0.0
    avg_acc = 100 * total_correct / total_tokens if total_tokens > 0 else 0
    
    return avg_loss, avg_acc, avg_ce_loss, avg_ctc_loss


def evaluate(model, loader, device, idx_to_gloss, pad_idx=0, desc='Evaluating', show_examples=True):
    """Evaluate model with BLEU-4 and WER."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for sequences, labels, seq_lengths, label_lengths, src_padding_mask in tqdm(loader, desc=desc):
            sequences = sequences.to(device)
            src_padding_mask = src_padding_mask.to(device)
            
            # Encode with padding mask
            memory = model.encode(sequences, src_key_padding_mask=src_padding_mask)
            
            # Generate predictions (pass memory_key_padding_mask so decoder cross-attn ignores pads)
            predictions = model.generate(memory, max_length=30, memory_key_padding_mask=src_padding_mask)
            
            # Convert to glosses
            for i in range(predictions.size(0)):
                pred_glosses = []
                for token_idx in predictions[i]:
                    token_idx = token_idx.item()
                    if token_idx == model.eos_idx:
                        break
                    if token_idx >= 3:  # Skip PAD, SOS, EOS
                        gloss_idx = token_idx - 3
                        if gloss_idx in idx_to_gloss:
                            pred_glosses.append(idx_to_gloss[gloss_idx])
                
                # Ground truth
                target_glosses = []
                L = label_lengths[i].item()
                for gloss_idx in labels[i, :L]:
                    gloss_idx = gloss_idx.item()
                    if gloss_idx in idx_to_gloss:
                        target_glosses.append(idx_to_gloss[gloss_idx])

                all_predictions.append(_pieces_to_words(pred_glosses))
                all_targets.append(_pieces_to_words(target_glosses))
    
    # Compute BLEU-4 and WER
    bleu = compute_bleu(all_targets, all_predictions)
    wer = compute_wer(all_targets, all_predictions)

    # Diagnostics: avg prediction length and EOS rate
    pred_lens = [len(p) for p in all_predictions]
    tgt_lens = [len(t) for t in all_targets]
    avg_pred = sum(pred_lens) / max(len(pred_lens), 1)
    avg_tgt = sum(tgt_lens) / max(len(tgt_lens), 1)
    empty_preds = sum(1 for p in all_predictions if len(p) == 0)
    print(f"\n  Avg pred len: {avg_pred:.1f}, Avg target len: {avg_tgt:.1f}, Empty preds: {empty_preds}/{len(all_predictions)}")

    # Show examples
    if show_examples:
        print("\nExample predictions:")
        for i in range(min(5, len(all_predictions))):
            print(f"  Target: {' '.join(all_targets[i])}")
            print(f"  Prediction: {' '.join(all_predictions[i])}")
    
    return bleu, wer


def _beam_search_single(
    model,
    memory,
    memory_key_padding_mask,
    max_length=30,
    beam_size=4,
    length_penalty=0.7,
    repetition_penalty=1.2,
    token_log_prior=None,
    lm_alpha=0.0,
):
    """Beam search for a single encoded sample. Returns token ids without SOS."""
    device = memory.device
    sos, eos, pad = model.sos_idx, model.eos_idx, model.pad_idx

    # (tokens, sum_logprob, finished)
    beams = [([sos], 0.0, False)]

    def norm_score(sum_logprob, tok_len):
        # GNMT-style length penalty to avoid over-short outputs.
        lp = ((5.0 + tok_len) / 6.0) ** max(length_penalty, 0.0)
        return sum_logprob / lp

    for _ in range(max_length):
        all_candidates = []
        all_finished = True

        for toks, score, finished in beams:
            if finished:
                all_candidates.append((toks, score, True))
                continue

            all_finished = False
            tgt = torch.tensor([toks], dtype=torch.long, device=device)
            tgt_mask = model.generate_square_subsequent_mask(tgt.size(1)).to(device)
            logits = model.decode_step(
                memory,
                tgt,
                tgt_mask=tgt_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            next_logits = logits[:, -1, :].squeeze(0)

            # Optional shallow-fusion style unigram LM bias.
            if token_log_prior is not None and lm_alpha != 0.0:
                next_logits = next_logits + (lm_alpha * token_log_prior.to(device))

            # Repetition penalty to reduce degenerate loops.
            if repetition_penalty != 1.0:
                prev = torch.tensor(sorted(set(toks)), dtype=torch.long, device=device)
                pos = next_logits[prev] > 0
                next_logits[prev[pos]] /= repetition_penalty
                next_logits[prev[~pos]] *= repetition_penalty

            log_probs = F.log_softmax(next_logits, dim=-1)
            topk_logp, topk_idx = torch.topk(log_probs, k=min(beam_size, log_probs.numel()))

            for lp, idx in zip(topk_logp.tolist(), topk_idx.tolist()):
                ntoks = toks + [int(idx)]
                nscore = score + float(lp)
                nfinished = (int(idx) == eos)
                all_candidates.append((ntoks, nscore, nfinished))

        if all_finished:
            break

        all_candidates.sort(key=lambda x: norm_score(x[1], len(x[0])), reverse=True)
        beams = all_candidates[:beam_size]

    best = max(beams, key=lambda x: norm_score(x[1], len(x[0])))[0]

    # Remove SOS and truncate at EOS.
    out = []
    for tok in best[1:]:
        if tok == eos:
            break
        if tok != pad:
            out.append(tok)
    return out


def evaluate_with_beam(
    model,
    loader,
    device,
    idx_to_gloss,
    pad_idx=0,
    beam_size=4,
    length_penalty=0.7,
    repetition_penalty=1.2,
    token_log_prior=None,
    lm_alpha=0.0,
    desc='Evaluating (beam)',
    show_examples=True,
):
    """Evaluate model with beam search decoding."""
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, labels, seq_lengths, label_lengths, src_padding_mask in tqdm(loader, desc=desc):
            sequences = sequences.to(device)
            src_padding_mask = src_padding_mask.to(device)

            memory = model.encode(sequences, src_key_padding_mask=src_padding_mask)

            for i in range(sequences.size(0)):
                mem_i = memory[i:i+1]
                mask_i = src_padding_mask[i:i+1]
                pred_tokens = _beam_search_single(
                    model,
                    mem_i,
                    memory_key_padding_mask=mask_i,
                    max_length=30,
                    beam_size=beam_size,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    token_log_prior=token_log_prior,
                    lm_alpha=lm_alpha,
                )

                pred_glosses = []
                for token_idx in pred_tokens:
                    if token_idx >= 3:
                        gloss_idx = token_idx - 3
                        if gloss_idx in idx_to_gloss:
                            pred_glosses.append(idx_to_gloss[gloss_idx])

                target_glosses = []
                L = label_lengths[i].item()
                for gloss_idx in labels[i, :L]:
                    gloss_idx = gloss_idx.item()
                    if gloss_idx in idx_to_gloss:
                        target_glosses.append(idx_to_gloss[gloss_idx])

                all_predictions.append(_pieces_to_words(pred_glosses))
                all_targets.append(_pieces_to_words(target_glosses))

    bleu = compute_bleu(all_targets, all_predictions)
    wer = compute_wer(all_targets, all_predictions)

    pred_lens = [len(p) for p in all_predictions]
    tgt_lens = [len(t) for t in all_targets]
    avg_pred = sum(pred_lens) / max(len(pred_lens), 1)
    avg_tgt = sum(tgt_lens) / max(len(tgt_lens), 1)
    empty_preds = sum(1 for p in all_predictions if len(p) == 0)
    print(f"\n  Avg pred len: {avg_pred:.1f}, Avg target len: {avg_tgt:.1f}, Empty preds: {empty_preds}/{len(all_predictions)}")

    if show_examples:
        print("\nExample predictions:")
        for i in range(min(5, len(all_predictions))):
            print(f"  Target: {' '.join(all_targets[i])}")
            print(f"  Prediction: {' '.join(all_predictions[i])}")

    return bleu, wer


def _build_class_weights(train_dataset, num_classes, power=0.5, min_w=0.2, max_w=5.0):
    """Build inverse-frequency weights over decoder vocabulary (PAD/SOS/EOS + glosses)."""
    vocab_size = num_classes + 3
    counts = np.ones(vocab_size, dtype=np.float64)  # smoothing to avoid inf weights

    # Raw labels are gloss indices in [0, num_classes-1]. Decoder targets shift by +3.
    for arr in train_dataset._labels:
        if arr.size == 0:
            continue
        shifted = arr.astype(np.int64) + 3
        np.add.at(counts, shifted, 1)

    # Keep special tokens neutral-ish.
    counts[0] = counts[3:].mean()  # PAD is ignored anyway.
    counts[1] = counts[3:].mean()
    counts[2] = counts[3:].mean()

    weights = 1.0 / np.power(counts, max(power, 1e-6))
    weights = weights / np.mean(weights[3:])
    weights = np.clip(weights, min_w, max_w).astype(np.float32)
    return torch.from_numpy(weights)


def _build_token_log_prior(train_dataset, num_classes, smoothing=1.0):
    """Build log unigram prior over decoder vocabulary (PAD/SOS/EOS + labels)."""
    vocab_size = num_classes + 3
    counts = np.full(vocab_size, float(smoothing), dtype=np.float64)

    for arr in train_dataset._labels:
        if arr.size == 0:
            continue
        shifted = arr.astype(np.int64) + 3
        np.add.at(counts, shifted, 1.0)

    # Keep special tokens available but not dominant.
    mean_count = counts[3:].mean() if vocab_size > 3 else counts.mean()
    counts[0] = mean_count
    counts[1] = mean_count
    counts[2] = mean_count

    probs = counts / counts.sum()
    return torch.from_numpy(np.log(np.maximum(probs, 1e-12)).astype(np.float32))


def main():
    parser = argparse.ArgumentParser(description='Train Encoder-Decoder Transformer')
    parser.add_argument('--data', type=str, required=True, help='Path to HDF5 data file')
    parser.add_argument('--input-feature-dim', type=int, default=None,
                        help='Use only first N feature dims (e.g. 1629 for landmarks-only, dropping CLIP)')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-encoder-layers', type=int, default=3, help='Number of encoder layers')
    parser.add_argument('--num-decoder-layers', type=int, default=3, help='Number of decoder layers')
    parser.add_argument('--num-heads', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--use-channel-attention', action='store_true', help='Enable channel attention for keypoint features')
    parser.add_argument('--attention-reduction', type=int, default=8, help='Reduction ratio for channel attention')
    parser.add_argument('--use-multimodal-fusion', action='store_true',
                        help='Force multimodal fusion encoder (keypoint+CLIP cross-attention)')
    parser.add_argument('--disable-multimodal-fusion', action='store_true',
                        help='Force single-stream encoder even when fused 2141-D features are provided')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--eval-batch-size', type=int, default=None, help='Validation batch size (default: max(1, batch_size//2))')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='Warmup epochs')
    parser.add_argument('--grad-accum-steps', type=int, default=4, help='Gradient accumulation steps')
    parser.add_argument('--max-seq-len', type=int, default=512, help='Cap on input frame length per sample')
    parser.add_argument('--num-workers', type=int, default=2, help='DataLoader worker count (0 = main process)')
    parser.add_argument('--h5-cache-mb', type=int, default=128, help='Per-worker HDF5 raw chunk cache in MB')
    parser.add_argument('--amp', action='store_true', help='Enable automatic mixed precision on CUDA')
    parser.add_argument('--label-smoothing', type=float, default=0.05, help='Label smoothing for cross-entropy loss (default: 0.05)')
    parser.add_argument('--ctc-weight', type=float, default=0.0,
                         help='Weight for auxiliary CTC loss on encoder memory (0 = disabled). '
                              'Common values: 0.2-0.5. Forces encoder output to be directly '
                              'predictive of gloss sequence via monotonic alignment, independent '
                              'of the decoder target-side language model.')
    parser.add_argument('--lr-restart-every', type=int, default=0,
                        help='Cosine annealing warm restart period in epochs (0 = disabled, uses lambda schedule)')
    parser.add_argument('--eval-beam-size', type=int, default=1,
                        help='Beam size for validation decoding (1 = greedy)')
    parser.add_argument('--eval-length-penalty', type=float, default=0.7,
                        help='Length penalty for beam search validation decoding')
    parser.add_argument('--eval-repetition-penalty', '--eval-repetiotion-penalty',
                        dest='eval_repetition_penalty', type=float, default=1.2,
                        help='Repetition penalty for beam search validation decoding (1.0 disables)')
    parser.add_argument('--eval-lm-alpha', type=float, default=0.0,
                        help='Unigram LM bias strength for validation beam search (0 disables)')
    parser.add_argument('--report-probe-samples', type=int, default=256,
                        help='Fixed validation subset size for per-epoch greedy/beam probe reports (0 disables)')
    parser.add_argument('--report-beam-size', type=int, default=4,
                        help='Beam size used in per-epoch probe report (only when report-probe-samples > 0)')
    parser.add_argument('--report-length-penalty', type=float, default=0.7,
                        help='Length penalty used in per-epoch beam probe report')
    parser.add_argument('--report-repetition-penalty', '--report-repetiotion-penalty',
                        dest='report_repetition_penalty', type=float, default=1.2,
                        help='Repetition penalty used in per-epoch beam probe report (1.0 disables)')
    parser.add_argument('--report-lm-alpha', type=float, default=0.0,
                        help='Unigram LM bias strength for probe beam report (0 disables)')
    parser.add_argument('--use-class-weights', action='store_true',
                        help='Use inverse-frequency class weights in CE loss')
    parser.add_argument('--class-weight-power', type=float, default=0.5,
                        help='Power for inverse-frequency class weights (higher = stronger reweighting)')
    parser.add_argument('--class-weight-min', type=float, default=0.2,
                        help='Minimum class weight clamp')
    parser.add_argument('--class-weight-max', type=float, default=5.0,
                        help='Maximum class weight clamp')
    parser.add_argument('--early-stop-patience', type=int, default=10,
                        help='Stop if BLEU does not improve for N consecutive epochs (0 disables)')
    parser.add_argument('--early-stop-min-delta', type=float, default=0.0,
                        help='Minimum BLEU improvement to reset early-stop patience')
    parser.add_argument('--save-dir', type=str, default='checkpoints_encdec', help='Save directory')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint')
    parser.add_argument('--resume-model-only', action='store_true',
                        help='Resume only model weights from latest checkpoint (keep fresh optimizer/scheduler from current CLI args)')
    parser.add_argument('--reset-early-stop', action='store_true',
                        help='Reset early-stop counter to 0 when resuming')
    parser.add_argument('--target-type', choices=['gloss', 'text'], default='gloss',
                        help="'gloss' uses *_labels; 'text' uses *_texts with a word-level vocab")
    parser.add_argument('--text-vocab-size', type=int, default=4000,
                        help='Word vocabulary size when --target-type text')
    parser.add_argument('--max-train-samples', type=int, default=None,
                        help='Cap training set to first N samples (for quick local tests)')
    parser.add_argument('--seed', type=int, default=42, help='Global random seed for reproducibility')
    parser.add_argument('--deterministic', action='store_true',
                        help='Enable deterministic CUDA ops (slower but more reproducible)')
    
    args = parser.parse_args()

    # Reproducibility controls.
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = (not args.deterministic)
        if args.deterministic:
            torch.backends.cudnn.deterministic = True
            # warn_only avoids hard failures on ops without deterministic kernels.
            torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Compute global feature statistics once (cached to disk).
    # Val is normalized with TRAIN stats to avoid distribution shift.
    train_mean, train_std = _compute_global_stats(args.data, 'train', max_seq_len=args.max_seq_len)
    if args.input_feature_dim:
        train_mean = train_mean[:args.input_feature_dim]
        train_std  = train_std[:args.input_feature_dim]

    # Build text vocabulary if needed (from train_texts in the H5).
    text_vocab = None
    if args.target_type == 'text':
        print(f"Building text vocabulary (vocab_size={args.text_vocab_size}) …")
        word_to_idx, idx_to_word = build_text_vocab(args.data, vocab_size=args.text_vocab_size)
        text_vocab = (word_to_idx, idx_to_word)
        vocab_path = os.path.join(args.save_dir, 'text_vocab.json')
        with open(vocab_path, 'w') as vf:
            json.dump(idx_to_word, vf)
        print(f"  Vocabulary: {len(idx_to_word)} words, saved to {vocab_path}")
        print(f"  Sample: {idx_to_word[:10]}")

    # Load dataset
    train_dataset = H5Dataset(
        args.data,
        'train',
        max_seq_len=args.max_seq_len,
        h5_cache_mb=args.h5_cache_mb,
        global_mean=train_mean,
        global_std=train_std,
        input_feature_dim=args.input_feature_dim,
        target_type=args.target_type,
        text_vocab=text_vocab,
    )
    val_dataset = H5Dataset(
        args.data,
        'val',
        max_seq_len=args.max_seq_len,
        h5_cache_mb=args.h5_cache_mb,
        global_mean=train_mean,
        global_std=train_std,
        input_feature_dim=args.input_feature_dim,
        target_type=args.target_type,
        text_vocab=text_vocab,
    )

    if args.max_train_samples is not None and args.max_train_samples < len(train_dataset):
        rng = np.random.RandomState(args.seed)
        chosen = np.sort(rng.choice(len(train_dataset), size=args.max_train_samples, replace=False))
        train_dataset = Subset(train_dataset, chosen.tolist())
        print(f"Capped training set to {args.max_train_samples} samples.")

    num_classes = val_dataset.num_classes
    input_features = val_dataset.feature_dim
    idx_to_gloss = val_dataset.idx_to_gloss

    print(f"Target type: {args.target_type}")
    print(f"Classes: {num_classes}")
    print(f"Input features: {input_features}")
    print(f"Sample tokens: {[idx_to_gloss[i] for i in range(min(10, num_classes))]}")
    
    eval_batch_size = args.eval_batch_size if args.eval_batch_size is not None else max(1, args.batch_size // 2)
    collate = partial(collate_fn, max_seq_len=None)

    _nw = args.num_workers
    _pin = device.type == 'cuda'

    def _seed_worker(worker_id):
        # Keep worker RNG streams deterministic across launches.
        worker_seed = (args.seed + worker_id) % (2 ** 32)
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    # BucketBatchSampler needs per-sample sequence lengths.
    # When train_dataset is a Subset, unwrap to the underlying H5Dataset.
    _base_ds = train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset
    _subset_indices = train_dataset.indices if isinstance(train_dataset, Subset) else None
    _all_eff_lens = _base_ds._effective_seq_lengths[_base_ds._valid_indices]
    _bucket_lens = _all_eff_lens[_subset_indices] if _subset_indices is not None else _all_eff_lens

    train_batch_sampler = BucketBatchSampler(
        _bucket_lens,
        batch_size=args.batch_size,
        shuffle=True,
    )
    train_loader_kwargs = {
        'collate_fn': collate,
        'num_workers': _nw,
        'pin_memory': _pin,
        'persistent_workers': (_nw > 0),
        'batch_sampler': train_batch_sampler,
        'worker_init_fn': _seed_worker if _nw > 0 else None,
    }
    if _nw > 0:
        train_loader_kwargs['prefetch_factor'] = 4

    train_loader = DataLoader(
        train_dataset,
        **train_loader_kwargs,
    )

    val_workers = (_nw // 2) if _nw > 1 else _nw
    val_loader_kwargs = {
        'batch_size': eval_batch_size,
        'shuffle': False,
        'collate_fn': collate,
        'num_workers': val_workers,
        'pin_memory': _pin,
        'persistent_workers': (val_workers > 0),
        'worker_init_fn': _seed_worker if val_workers > 0 else None,
    }
    if val_workers > 0:
        val_loader_kwargs['prefetch_factor'] = 4

    val_loader = DataLoader(
        val_dataset,
        **val_loader_kwargs,
    )

    # Fixed validation probe subset for apples-to-apples greedy vs beam tracking.
    probe_loader = None
    if args.report_probe_samples > 0:
        n_probe = min(int(args.report_probe_samples), len(val_dataset))
        rng = np.random.RandomState(42)
        probe_idx = np.sort(rng.choice(len(val_dataset), size=n_probe, replace=False))
        probe_subset = Subset(val_dataset, probe_idx.tolist())
        probe_loader = DataLoader(
            probe_subset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=val_workers,
            pin_memory=_pin,
            persistent_workers=(val_workers > 0),
            worker_init_fn=_seed_worker if val_workers > 0 else None,
            prefetch_factor=4 if val_workers > 0 else None,
        )
        print(f"Probe subset: {n_probe} fixed validation samples (seed=42)")

    print(f"Seed: {args.seed} | Deterministic: {args.deterministic}")

    resume_path = os.path.join(args.save_dir, 'latest_checkpoint.pt')
    resume_ckpt = None
    if args.resume and os.path.exists(resume_path):
        resume_ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)

    use_multimodal_fusion = None
    if args.use_multimodal_fusion and args.disable_multimodal_fusion:
        raise ValueError("Choose only one: --use-multimodal-fusion or --disable-multimodal-fusion")
    if args.use_multimodal_fusion:
        use_multimodal_fusion = True
    elif args.disable_multimodal_fusion:
        use_multimodal_fusion = False
    elif resume_ckpt is not None:
        use_multimodal_fusion = bool(resume_ckpt.get('use_multimodal_fusion', False))
    
    # Create model
    model = SignLanguageTransformer(
        num_classes=num_classes,
        input_features=input_features,
        hidden_dim=args.hidden_dim,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_channel_attention=args.use_channel_attention,
        attention_reduction=args.attention_reduction,
        use_multimodal_fusion=use_multimodal_fusion,
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")
    print(f"Multimodal fusion: {bool(model.use_multimodal_fusion)}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda', enabled=(args.amp and device.type == 'cuda'))
    
    # Learning rate scheduler with warmup
    if args.lr_restart_every > 0:
        # Cosine annealing with warm restarts — good for escaping plateaus on resume
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.lr_restart_every, T_mult=1, eta_min=1e-7
        )
    else:
        def lr_lambda(epoch):
            if epoch < args.warmup_epochs:
                return (epoch + 1) / args.warmup_epochs
            else:
                # Cosine decay
                progress = (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs)
                return 0.5 * (1 + np.cos(np.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    class_weights = None
    if args.use_class_weights:
        class_weights = _build_class_weights(
            train_dataset,
            num_classes=num_classes,
            power=args.class_weight_power,
            min_w=args.class_weight_min,
            max_w=args.class_weight_max,
        ).to(device)
        wmin = float(class_weights[3:].min().item())
        wmax = float(class_weights[3:].max().item())
        print(f"Using class weights: min={wmin:.3f}, max={wmax:.3f}, power={args.class_weight_power}")

    token_log_prior = None
    if (args.eval_lm_alpha != 0.0) or (args.report_lm_alpha != 0.0):
        token_log_prior = _build_token_log_prior(train_dataset, num_classes=num_classes)
        print(
            f"Using beam LM bias: eval_alpha={args.eval_lm_alpha}, "
            f"report_alpha={args.report_lm_alpha}"
        )
    
    # Training loop
    best_bleu = 0.0
    start_epoch = 1
    epochs_without_improvement = 0
    
    # Resume from checkpoint if requested
    if args.resume:
        if resume_ckpt is not None:
            print(f"\nResuming from checkpoint: {resume_path}")
            model.load_state_dict(resume_ckpt['model_state_dict'])
            start_epoch = resume_ckpt['epoch'] + 1
            best_bleu = resume_ckpt.get('best_bleu', resume_ckpt.get('best_gloss_recall', 0.0))
            if args.reset_early_stop:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement = int(resume_ckpt.get('epochs_without_improvement', 0))

            if args.resume_model_only:
                print("Resuming model weights only (fresh optimizer/scheduler from current args).")
            else:
                optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])

                # Restore scheduler state without stepping-before-optimizer warning.
                if 'scheduler_state_dict' in resume_ckpt:
                    scheduler.load_state_dict(resume_ckpt['scheduler_state_dict'])
                else:
                    scheduler.last_epoch = int(resume_ckpt['epoch'])
                    scheduler._last_lr = [group['lr'] for group in optimizer.param_groups]
            
            print(f"Resuming from epoch {start_epoch}, best BLEU: {best_bleu:.2f}\n")
        else:
            print(f"No checkpoint found at {resume_path}, starting from scratch\n")
    
    print("\n" + "="*70)
    print("ENCODER-DECODER TRANSFORMER TRAINING")
    print("="*70)
    if args.max_seq_len is not None:
        print(f"Input frame cap: {args.max_seq_len}")
    print(f"Train batch: {args.batch_size}, Eval batch: {eval_batch_size}, Grad accum: {args.grad_accum_steps}, AMP: {args.amp and device.type == 'cuda'}")
    if args.eval_beam_size > 1:
        print(
            f"Validation decoding: beam={args.eval_beam_size}, "
            f"length_penalty={args.eval_length_penalty}, "
            f"repetition_penalty={args.eval_repetition_penalty}, "
            f"lm_alpha={args.eval_lm_alpha}"
        )
    else:
        print("Validation decoding: greedy")
    
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc, train_ce_loss, train_ctc_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler=scaler,
            grad_accum_steps=max(1, args.grad_accum_steps),
            use_amp=(args.amp and device.type == 'cuda'),
            class_weights=class_weights,
            label_smoothing=args.label_smoothing,
            ctc_weight=args.ctc_weight,
        )
        if args.ctc_weight > 0:
            print(f"  Train loss: {train_loss:.4f} (CE: {train_ce_loss:.4f}, CTC: {train_ctc_loss:.4f})")
        if args.eval_beam_size > 1:
            bleu, wer = evaluate_with_beam(
                model,
                val_loader,
                device,
                idx_to_gloss,
                beam_size=args.eval_beam_size,
                length_penalty=args.eval_length_penalty,
                repetition_penalty=args.eval_repetition_penalty,
                token_log_prior=token_log_prior,
                lm_alpha=args.eval_lm_alpha,
                desc='Evaluating (beam)',
                show_examples=True,
            )
        else:
            bleu, wer = evaluate(model, val_loader, device, idx_to_gloss, desc='Evaluating', show_examples=True)

        if probe_loader is not None:
            print("\nProbe report (fixed subset):")
            p_greedy_bleu, p_greedy_wer = evaluate(
                model,
                probe_loader,
                device,
                idx_to_gloss,
                desc='Probe greedy',
                show_examples=False,
            )
            p_beam_bleu, p_beam_wer = evaluate_with_beam(
                model,
                probe_loader,
                device,
                idx_to_gloss,
                beam_size=max(1, args.report_beam_size),
                length_penalty=args.report_length_penalty,
                repetition_penalty=args.report_repetition_penalty,
                token_log_prior=token_log_prior,
                lm_alpha=args.report_lm_alpha,
                desc=f'Probe beam-{max(1, args.report_beam_size)}',
                show_examples=False,
            )
            print(
                f"  Greedy BLEU/WER: {p_greedy_bleu:.2f}/{p_greedy_wer:.2f}% | "
                f"Beam BLEU/WER: {p_beam_bleu:.2f}/{p_beam_wer:.2f}%"
            )
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val BLEU-4: {bleu:.2f}, WER: {wer:.2f}%")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        scheduler.step()
        
        # Save best model (by BLEU-4, higher is better)
        improved = (bleu - best_bleu) > args.early_stop_min_delta
        if improved:
            best_bleu = bleu
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bleu': bleu,
                'wer': wer,
                'num_classes': num_classes,
                'input_features': input_features,
                'hidden_dim': args.hidden_dim,
                'num_encoder_layers': args.num_encoder_layers,
                'num_decoder_layers': args.num_decoder_layers,
                'use_channel_attention': args.use_channel_attention,
                'attention_reduction': args.attention_reduction,
                'use_multimodal_fusion': bool(model.use_multimodal_fusion),
                'keypoint_dim': int(getattr(model, 'keypoint_dim', 0)),
                'clip_dim': int(getattr(model, 'clip_dim', 0)),
                'best_bleu': best_bleu,
                'scheduler_state_dict': scheduler.state_dict(),
            }, os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  ✓ Saved best model (BLEU-4={bleu:.2f})")
        else:
            epochs_without_improvement += 1
        
        # Save latest checkpoint (for resuming)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'bleu': bleu,
            'wer': wer,
            'best_bleu': best_bleu,
            'epochs_without_improvement': epochs_without_improvement,
            'scheduler_state_dict': scheduler.state_dict(),
            'use_channel_attention': args.use_channel_attention,
            'attention_reduction': args.attention_reduction,
            'use_multimodal_fusion': bool(model.use_multimodal_fusion),
            'keypoint_dim': int(getattr(model, 'keypoint_dim', 0)),
            'clip_dim': int(getattr(model, 'clip_dim', 0)),
        }, os.path.join(args.save_dir, 'latest_checkpoint.pt'))
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'bleu': bleu,
                'wer': wer,
                'best_bleu': best_bleu,
                'epochs_without_improvement': epochs_without_improvement,
                'scheduler_state_dict': scheduler.state_dict(),
                'use_channel_attention': args.use_channel_attention,
                'attention_reduction': args.attention_reduction,
                'use_multimodal_fusion': bool(model.use_multimodal_fusion),
                'keypoint_dim': int(getattr(model, 'keypoint_dim', 0)),
                'clip_dim': int(getattr(model, 'clip_dim', 0)),
            }, os.path.join(args.save_dir, f'checkpoint_epoch{epoch}.pt'))

        if args.early_stop_patience > 0:
            print(f"  Early-stop counter: {epochs_without_improvement}/{args.early_stop_patience}")
            if epochs_without_improvement >= args.early_stop_patience:
                print(
                    f"\nEarly stopping: no BLEU improvement > {args.early_stop_min_delta:.4f} "
                    f"for {args.early_stop_patience} consecutive epochs."
                )
                break
    
    print("\n" + "="*70)
    print(f"Training complete! Best BLEU-4: {best_bleu:.2f}")
    print("="*70)


if __name__ == '__main__':
    main()
