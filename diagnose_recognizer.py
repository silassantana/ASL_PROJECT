#!/usr/bin/env python3
"""
Diagnose the gloss recognizer's failure mode on a small val slice.

Loads checkpoints_clean_v2/best_model.pt and runs greedy generation
(no beam search, no repetition penalty -- raw model behavior) over a
val slice, reporting:
  - prediction length distribution (empty / near-empty predictions?)
  - token diversity (does it predict the same few glosses repeatedly?)
  - per-sample target vs prediction for the first N examples

Usage:
  python diagnose_recognizer.py \
    --checkpoint checkpoints_clean_v2/best_model.pt \
    --data val_slice.h5
"""

import argparse
import json
from collections import Counter

import h5py
import numpy as np
import torch

from sign_transformer import SignLanguageTransformer


class CrossAttentionCapture:
    """Temporarily wraps each decoder layer's cross-attention (multihead_attn)
    to force need_weights=True and capture the resulting attention weights.

    Cross-attention weights: [batch, tgt_len, src_len] (averaged over heads).
    One entry per decoder layer, captured on every decode_step call.
    """

    def __init__(self, model):
        self.model = model
        self.captured = []  # list of lists: captured[layer_idx] -> list of [batch, tgt_len, src_len] tensors
        self._originals = []

    def __enter__(self):
        layers = self.model.decoder.layers
        self.captured = [[] for _ in layers]
        for i, layer in enumerate(layers):
            mha = layer.multihead_attn
            orig_forward = mha.forward
            self._originals.append((mha, orig_forward))

            def make_patched(orig_fn, idx):
                def patched(query, key, value, **kwargs):
                    kwargs["need_weights"] = True
                    kwargs["average_attn_weights"] = True
                    out, weights = orig_fn(query, key, value, **kwargs)
                    if weights is not None:
                        self.captured[idx].append(weights.detach().cpu())
                    return out, weights
                return patched

            mha.forward = make_patched(orig_forward, i)
        return self

    def __exit__(self, *exc):
        for mha, orig_forward in self._originals:
            mha.forward = orig_forward
        return False


def load_model_and_vocab(checkpoint_path, data_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    with h5py.File(data_path, "r") as f:
        gloss_names = f["gloss_names"][:]
        idx_to_gloss = {
            i: (name.decode("utf-8") if isinstance(name, bytes) else name)
            for i, name in enumerate(gloss_names)
        }
        num_classes = len(idx_to_gloss)

    expected_input_dim = checkpoint.get("input_features", 512)
    use_channel_attention = checkpoint.get("use_channel_attention", False)
    attention_reduction = checkpoint.get("attention_reduction", 8)
    use_multimodal_fusion = bool(checkpoint.get("use_multimodal_fusion", False))
    keypoint_dim = int(checkpoint.get("keypoint_dim", 1629))
    clip_dim = int(checkpoint.get("clip_dim", 512))

    model = SignLanguageTransformer(
        num_classes=num_classes,
        input_features=expected_input_dim,
        hidden_dim=checkpoint.get("hidden_dim", 256),
        num_encoder_layers=checkpoint.get("num_encoder_layers", 4),
        num_decoder_layers=checkpoint.get("num_decoder_layers", 4),
        use_channel_attention=use_channel_attention,
        attention_reduction=attention_reduction,
        use_multimodal_fusion=use_multimodal_fusion,
        keypoint_dim=keypoint_dim,
        clip_dim=clip_dim,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    meta = {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_state_dict", "scheduler_state_dict")}
    return model, idx_to_gloss, expected_input_dim, meta


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default="checkpoints_clean_v2/best_model.pt")
    parser.add_argument("--data", type=str, default="val_slice.h5")
    parser.add_argument("--n-examples", type=int, default=10, help="How many per-sample examples to print")
    parser.add_argument("--max-length", type=int, default=30)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)

    print("Loading model...")
    model, idx_to_gloss, expected_input_dim, meta = load_model_and_vocab(args.checkpoint, args.data)
    model = model.to(device)
    print("Checkpoint meta:", json.dumps({k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in meta.items()}, indent=2))
    print(f"use_multimodal_fusion (effective): {model.use_multimodal_fusion}")
    print(f"input_features: {expected_input_dim}, keypoint_dim: {model.keypoint_dim}, clip_dim: {model.clip_dim}")
    print()

    with h5py.File(args.data, "r") as f:
        sequences = f["val_sequences"][:]
        seq_lengths = f["val_sequence_lengths"][:]
        labels = f["val_labels"][:]
        label_lengths = f["val_label_lengths"][:]

    n = sequences.shape[0]
    print(f"Loaded {n} val samples. Feature dim: {sequences.shape[2]}")

    # Apply normalization stats if available (matches training).
    stats_path = args.data + ".train_stats.npz"
    import os
    if os.path.exists(stats_path):
        stats = np.load(stats_path)
        g_mean, g_std = stats["mean"], stats["std"]
        feat_dim = sequences.shape[2]
        if g_mean.shape[0] < feat_dim:
            g_mean = np.concatenate([g_mean, np.zeros(feat_dim - g_mean.shape[0], dtype=np.float32)])
            g_std = np.concatenate([g_std, np.ones(feat_dim - g_std.shape[0], dtype=np.float32)])
        elif g_mean.shape[0] > feat_dim:
            g_mean = g_mean[:feat_dim]
            g_std = g_std[:feat_dim]
        sequences = np.nan_to_num(sequences, nan=0.0, posinf=0.0, neginf=0.0)
        sequences = (sequences - g_mean) / g_std
        sequences = np.clip(sequences, -10.0, 10.0)
        print("Applied global normalization from training stats.")
    else:
        print("WARNING: no stats file found, running without normalization (may degrade results).")
    print()

    all_pred_lens = []
    all_target_lens = []
    pred_token_counter = Counter()
    target_token_counter = Counter()
    empty_preds = 0
    examples = []
    attn_entropy_ratios = []  # entropy of cross-attn dist / entropy of uniform dist, per decode step
    attn_max_weights = []     # max attention weight per decode step (peakiness)

    with torch.no_grad():
        for i in range(n):
            slen = int(seq_lengths[i])
            feats = torch.from_numpy(sequences[i, :slen, :]).unsqueeze(0).float().to(device)
            src_padding_mask = torch.zeros(1, slen, dtype=torch.bool, device=device)

            memory = model.encode(feats, src_key_padding_mask=src_padding_mask)

            with CrossAttentionCapture(model) as cap:
                predictions = model.generate(memory, max_length=args.max_length, memory_key_padding_mask=src_padding_mask)

            # cap.captured[layer_idx] is a list of [1, tgt_len_so_far, src_len] tensors,
            # one per autoregressive step (tgt_len grows by 1 each step).
            # Use the LAST layer's attention at each step, looking at the attention
            # distribution for the most recently generated position (last row).
            last_layer = cap.captured[-1]
            uniform_entropy = np.log(slen)
            for step_weights in last_layer:
                # step_weights: [1, tgt_len, src_len] -> take last query position
                w = step_weights[0, -1, :].numpy()  # [src_len]
                w = np.clip(w, 1e-12, None)
                w = w / w.sum()
                ent = -(w * np.log(w)).sum()
                attn_entropy_ratios.append(ent / max(uniform_entropy, 1e-8))
                attn_max_weights.append(w.max())

            pred_glosses = []
            for tok in predictions[0]:
                tok = tok.item()
                if tok == model.eos_idx:
                    break
                if tok >= 3:
                    g = tok - 3
                    if g in idx_to_gloss:
                        pred_glosses.append(idx_to_gloss[g])
                        pred_token_counter[idx_to_gloss[g]] += 1

            L = int(label_lengths[i])
            target_glosses = []
            for g in labels[i, :L]:
                g = int(g)
                if g in idx_to_gloss:
                    target_glosses.append(idx_to_gloss[g])
                    target_token_counter[idx_to_gloss[g]] += 1

            all_pred_lens.append(len(pred_glosses))
            all_target_lens.append(len(target_glosses))
            if len(pred_glosses) == 0:
                empty_preds += 1

            if i < args.n_examples:
                examples.append((target_glosses, pred_glosses))

    pred_lens = np.array(all_pred_lens)
    target_lens = np.array(all_target_lens)

    print("=" * 60)
    print("RECOGNIZER DIAGNOSTIC")
    print("=" * 60)
    print(f"N samples: {n}")
    print(f"Prediction length: mean={pred_lens.mean():.2f}, std={pred_lens.std():.2f}, "
          f"min={pred_lens.min()}, max={pred_lens.max()}")
    print(f"Target length:     mean={target_lens.mean():.2f}, std={target_lens.std():.2f}, "
          f"min={target_lens.min()}, max={target_lens.max()}")
    print(f"Empty predictions: {empty_preds}/{n} ({100*empty_preds/n:.1f}%)")
    print()
    print(f"Unique gloss tokens predicted: {len(pred_token_counter)} / {len(idx_to_gloss)} vocab")
    print(f"Unique gloss tokens in targets: {len(target_token_counter)} / {len(idx_to_gloss)} vocab")
    print()

    entropy_ratios = np.array(attn_entropy_ratios)
    max_weights = np.array(attn_max_weights)
    print("=" * 60)
    print("CROSS-ATTENTION (last decoder layer, last query position per step)")
    print("=" * 60)
    print(f"  Decode steps analyzed: {len(entropy_ratios)}")
    print(f"  Entropy ratio (1.0 = uniform/no grounding, 0.0 = fully peaked):")
    print(f"    mean={entropy_ratios.mean():.4f}, std={entropy_ratios.std():.4f}, "
          f"min={entropy_ratios.min():.4f}, max={entropy_ratios.max():.4f}")
    print(f"  Max attention weight per step (1/src_len = uniform):")
    print(f"    mean={max_weights.mean():.4f}, std={max_weights.std():.4f}, "
          f"min={max_weights.min():.4f}, max={max_weights.max():.4f}")
    print(f"  (for reference, uniform max weight for typical src_len~{int(np.median(seq_lengths))} "
          f"would be ~{1.0/np.median(seq_lengths):.4f})")
    print()
    print("Top 10 most-predicted glosses:")
    for tok, cnt in pred_token_counter.most_common(10):
        print(f"  {cnt:5d}  {tok}")
    print()
    print("Top 10 most-common target glosses (this slice):")
    for tok, cnt in target_token_counter.most_common(10):
        print(f"  {cnt:5d}  {tok}")
    print()

    print(f"Per-sample examples (first {args.n_examples}):")
    for i, (tgt, pred) in enumerate(examples):
        print(f"\n  [{i}]")
        print(f"    TARGET ({len(tgt)}): {' '.join(tgt)}")
        print(f"    PRED   ({len(pred)}): {' '.join(pred)}")


if __name__ == "__main__":
    main()
