#!/usr/bin/env python3
"""
train_continuous.py - Train continuous sign recognition model with CTC
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader
import argparse
from tqdm import tqdm
import time
import json
import os
from pathlib import Path
from glob import glob

from continuous_sign_model import ContinuousSignModel, CTCDecoder
from transformer_ctc_model import TransformerCTCModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class ContinuousSignDataset(Dataset):
    """Dataset for continuous sign recognition with CTC"""

    def __init__(self, h5_file, split="train", max_samples=None):
        self.h5_f = h5py.File(h5_file, "r", libver="latest", swmr=True)

        # Load data
        self.sequences = self.h5_f[f"{split}_sequences"]
        self.labels = self.h5_f[f"{split}_labels"]
        self.label_lengths = self.h5_f[f"{split}_label_lengths"][:]
        self.sequence_lengths = self.h5_f[f"{split}_sequence_lengths"][:]

        self.num_samples = len(self.sequences)
        if max_samples:
            self.num_samples = min(self.num_samples, max_samples)

        # Try to load num_classes from attributes first, then fall back to dataset
        if "num_classes" in self.h5_f.attrs:
            self.num_classes = int(self.h5_f.attrs["num_classes"])
        else:
            self.num_classes = int(self.h5_f["num_classes"][()].item())

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Load sequence
        seq = self.sequences[idx]
        seq_length = self.sequence_lengths[idx]

        # Handle different shapes
        if len(seq.shape) == 3:
            # Shape: [T, J, F_dims] - need to flatten
            T, J, F_dims = seq.shape
            seq = seq.reshape(T, J * F_dims)
        elif len(seq.shape) == 2:
            # Shape: [T, features] - already correct
            pass
        elif len(seq.shape) == 1:
            # Flattened: need to reshape using stored sequence length
            # Get feature dimension from HDF5 attributes if available
            if "feature_dim" in self.h5_f.attrs:
                feature_dim = int(self.h5_f.attrs["feature_dim"])
                seq = seq.reshape(seq_length, feature_dim)
            else:
                # Fall back: assume it's landmarks (150 dims) or features
                # Try to infer from total length
                total_len = len(seq)
                # Common feature dims: 150 (landmarks), 450 (enhanced), 512 (CLIP), 1024 (I3D)
                for dim in [512, 1024, 768, 450, 150]:
                    if total_len % dim == 0:
                        seq = seq.reshape(-1, dim)
                        break
                else:
                    raise ValueError(f"Cannot infer feature dimension from shape: {seq.shape}")
        else:
            raise ValueError(f"Unexpected sequence shape: {seq.shape}")

        # Normalize
        seq_mean = seq.mean()
        seq_std = seq.std() + 1e-8
        seq = (seq - seq_mean) / seq_std

        # Load labels (sequence of glosses)
        labels = self.labels[idx]
        label_length = self.label_lengths[idx]

        return (
            torch.from_numpy(seq).float(),
            torch.tensor(labels[:label_length], dtype=torch.long),
            torch.tensor(seq_length, dtype=torch.long),
            torch.tensor(label_length, dtype=torch.long),
        )


def collate_fn(batch):
    """Custom collate function to handle variable-length sequences"""
    sequences, labels, seq_lengths, label_lengths = zip(*batch)

    # Sequences are already padded in H5
    sequences = torch.stack(sequences, dim=0)
    seq_lengths = torch.stack(seq_lengths, dim=0)
    label_lengths = torch.stack(label_lengths, dim=0)

    # Pad labels to max length in batch
    max_label_len = max(len(l) for l in labels)
    labels_padded = torch.zeros(len(labels), max_label_len, dtype=torch.long)
    for i, label in enumerate(labels):
        labels_padded[i, : len(label)] = label

    return sequences, labels_padded, seq_lengths, label_lengths


def _make_uniform_frame_targets(labels, label_lengths, seq_lengths, max_time):
    """Create per-frame targets by uniformly segmenting frames across labels.

    Returns a LongTensor [batch, max_time] with -100 for frames beyond true seq length.
    """
    batch_size = len(labels)
    targets = torch.full((batch_size, max_time), -100, dtype=torch.long)
    for i in range(batch_size):
        L = int(label_lengths[i].item())
        T = int(seq_lengths[i].item())
        if L <= 0 or T <= 0:
            continue
        # Segment edges from 0..T into L equal parts
        seg_edges = np.linspace(0, T, L + 1)
        seg_edges = np.floor(seg_edges).astype(int)
        seg_edges[-1] = T
        for s in range(L):
            start = int(seg_edges[s])
            end = int(seg_edges[s + 1])
            if end > start:
                targets[i, start:end] = labels[i][s].item()
        # Frames beyond T remain -100 (ignored)
    return targets


def train_epoch(
    model,
    loader,
    criterion,
    optimizer,
    epoch,
    blank_idx,
    device,
    blank_logit_offset=0.0,
    blank_penalty_scale=0.1,
    blank_penalty_margin=0.5,
    pretrain_ce_epochs=0,
    ce_label_smoothing=0.0,
):
    """Train one epoch with CTC loss and explicit blank suppression"""
    model.train()
    total_loss = 0
    total_batches = 0
    total_blank_pct = 0

    pbar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
    for batch_idx, (sequences, labels, seq_lengths, label_lengths) in enumerate(pbar):
        # Move to device
        sequences = sequences.to(device)
        seq_lengths = seq_lengths.to(device)
        label_lengths = label_lengths.to(device)

        # Flatten labels for CTC (it expects 1D tensor)
        labels_flat = []
        for i in range(len(labels)):
            label_len = int(label_lengths[i].item())
            labels_flat.extend(labels[i][:label_len].tolist())
        labels_flat = torch.tensor(labels_flat, dtype=torch.long, device=device)

        # Forward pass
        logits = model(sequences, seq_lengths)  # [batch, time, num_classes+1]

        # CRITICAL: Explicitly suppress blank during transition from CE to CTC
        # Subtract from blank logit to make it less likely during critical epochs
        if epoch >= pretrain_ce_epochs and epoch < pretrain_ce_epochs + 5:
            logits = logits.clone()
            # Strong suppression: -2.0 for epochs 3-5, -1.0 for epochs 6-7
            suppression = 2.0 if epoch < pretrain_ce_epochs + 2 else 1.0
            logits[:, :, blank_idx] = logits[:, :, blank_idx] - suppression

        # Lower blank logit to fight collapse toward blank predictions
        if blank_logit_offset > 0:
            logits = logits.clone()
            logits[:, :, blank_idx] = logits[:, :, blank_idx] - blank_logit_offset

        # Monitor blank predictions
        with torch.no_grad():
            preds = logits.argmax(dim=-1)  # [batch, time]
            blank_count = (preds == blank_idx).sum().item()
            total_preds = preds.numel()
            blank_pct = 100.0 * blank_count / total_preds
            total_blank_pct += blank_pct

        # Choose loss: CE pretraining for early epochs or CTC
        if epoch < pretrain_ce_epochs:
            max_time = sequences.shape[1]
            frame_targets = _make_uniform_frame_targets(
                labels, label_lengths, seq_lengths, max_time
            ).to(device)
            # Use non-blank classes for CE
            ce_logits = logits[:, :, :blank_idx]  # [batch, time, num_classes]
            ce_loss_fn = nn.CrossEntropyLoss(
                ignore_index=-100, label_smoothing=ce_label_smoothing
            )
            loss = ce_loss_fn(ce_logits.reshape(-1, blank_idx), frame_targets.reshape(-1))
        else:
            # Prepare for CTC loss
            log_probs = F.log_softmax(logits, dim=-1)
            log_probs = log_probs.transpose(0, 1)  # [time, batch, num_classes+1]
            # Compute CTC loss
            loss = criterion(log_probs, labels_flat, seq_lengths, label_lengths)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  WARNING: Loss is {loss.item()} at batch {batch_idx}")
            print(f"   Skipping this batch...")
            continue

        # Entropy encouragement during CE pretraining to promote exploration
        if epoch < pretrain_ce_epochs:
            probs = F.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = loss - 0.1 * entropy
        
        # CRITICAL: Gradual transition from CE to CTC - prevent blank collapse
        # After CE pretraining, aggressively suppress blanks for several epochs
        if epoch >= pretrain_ce_epochs and epoch < pretrain_ce_epochs + 5:
            probs = F.softmax(logits, dim=-1)
            blank_probs = probs[:, :, blank_idx].mean()
            # Strong penalty for blank dominance during transition
            blank_dominance_penalty = torch.relu(blank_probs - 0.3) * 5.0
            loss = loss + blank_dominance_penalty
            
            # Also encourage entropy to prevent collapse
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = loss - 0.05 * entropy

        # Add blank regularization: penalize model for being too confident about blanks
        # If blank logits are much higher than other classes, penalize
        if blank_penalty_scale > 0:
            blank_logits = logits[:, :, blank_idx]  # [batch, time]
            other_logits = logits[:, :, :blank_idx]  # [batch, time, num_classes]
            max_other = torch.max(other_logits, dim=-1)[0]  # [batch, time]
            # Penalize if blank is much higher than max of other classes
            blank_penalty = torch.relu(blank_logits - max_other - blank_penalty_margin).mean()
            loss = loss + blank_penalty_scale * blank_penalty

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_batches += 1

        # Update progress bar
        pbar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "avg_loss": f"{total_loss / total_batches:.4f}",
                "blank%": f"{blank_pct:.1f}",
            }
        )

    avg_blank_pct = total_blank_pct / total_batches
    return total_loss / total_batches, avg_blank_pct


def validate(model, loader, decoder, idx_to_gloss, blank_idx, device):
    """
    Validate with sequence accuracy

    Metrics:
    - Exact Match Accuracy: % of sequences predicted exactly correct
    - Gloss Accuracy: % of individual glosses predicted correctly (ignoring order)
    - Blank Percentage: % of timesteps predicted as blank
    """
    model.eval()
    exact_matches = 0
    total_correct_glosses = 0
    total_glosses = 0
    total_samples = 0
    total_blanks = 0
    total_timesteps = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for sequences, labels, seq_lengths, label_lengths in tqdm(
            loader, desc="Validating", leave=False
        ):
            sequences = sequences.to(device)

            # Forward pass
            logits = model(sequences, seq_lengths)

            # Monitor blank predictions
            preds = logits.argmax(dim=-1)
            total_blanks += (preds == blank_idx).sum().item()
            total_timesteps += preds.numel()

            # Decode each sequence
            for i in range(len(sequences)):
                # Decode prediction
                pred_indices = decoder.greedy_decode(logits[i])
                true_indices = labels[i][: label_lengths[i]].tolist()

                # Exact match
                if pred_indices == true_indices:
                    exact_matches += 1

                # Gloss-level accuracy (how many glosses are correct, ignoring order)
                pred_set = set(pred_indices)
                true_set = set(true_indices)
                total_correct_glosses += len(pred_set & true_set)
                total_glosses += len(true_set)

                total_samples += 1

                # Store for analysis
                pred_glosses = [
                    idx_to_gloss.get(idx, f"UNK_{idx}") for idx in pred_indices
                ]
                true_glosses = [
                    idx_to_gloss.get(idx, f"UNK_{idx}") for idx in true_indices
                ]
                all_predictions.append(pred_glosses)
                all_targets.append(true_glosses)

    exact_acc = exact_matches / total_samples if total_samples > 0 else 0
    gloss_acc = total_correct_glosses / total_glosses if total_glosses > 0 else 0
    blank_pct = 100.0 * total_blanks / total_timesteps if total_timesteps > 0 else 0

    return exact_acc, gloss_acc, blank_pct, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="continuous_asl_dataset.h5", help="Data file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0005,
        help="Learning rate (reduced to prevent collapse)",
    )
    parser.add_argument("--hidden-dim", type=int, default=512, help="Hidden dimension")
    parser.add_argument(
        "--num-layers", type=int, default=2, help="Number of GRU layers"
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints_continuous",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use class-balanced sampling based on inverse gloss frequency",
    )
    parser.add_argument(
        "--blank-logit-offset",
        type=float,
        default=0.0,
        help="Subtract this value from the blank logit to reduce blank dominance (0 = no offset)",
    )
    parser.add_argument(
        "--blank-penalty-scale",
        type=float,
        default=0.0,
        help="Weight for the blank margin penalty term (0 = disabled)",
    )
    parser.add_argument(
        "--blank-penalty-margin",
        type=float,
        default=0.5,
        help="Margin before the blank penalty activates (blank - max_other - margin)",
    )
    parser.add_argument(
        "--pretrain-ce-epochs",
        type=int,
        default=3,
        help="Number of initial epochs to pretrain with uniform frame-level CrossEntropy",
    )
    parser.add_argument(
        "--ce-label-smoothing",
        type=float,
        default=0.1,
        help="Label smoothing factor for CE pretraining",
    )
    parser.add_argument(
        "--use-transformer",
        action="store_true",
        help="Use Transformer architecture instead of GRU",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=8,
        help="Number of attention heads for Transformer",
    )
    args = parser.parse_args()

    print(f"\n{'=' * 70}")
    print("CONTINUOUS SIGN RECOGNITION TRAINING")
    print(f"{'=' * 70}\n")

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Load gloss mapping - auto-detect from data filename
    mapping_file = args.data.replace(".h5", "_mapping.json")
    if not os.path.exists(mapping_file):
        mapping_file = "continuous_gloss_mapping.json"  # fallback

    print(f"Loading mapping from: {mapping_file}")
    with open(mapping_file, "r") as f:
        mapping = json.load(f)
        idx_to_gloss = {int(k): v for k, v in mapping["idx_to_gloss"].items()}
        num_classes = mapping.get("num_classes", len(idx_to_gloss))
        blank_idx = mapping.get("blank_idx", num_classes)  # Blank is after all classes

    print(f"Number of classes: {num_classes}")
    print(f"Blank index: {blank_idx}")
    print(f"Sample glosses: {list(idx_to_gloss.items())[:5]}")

    # Load datasets
    print(f"\nLoading {args.data}...")
    train_ds = ContinuousSignDataset(args.data, "train")
    val_ds = ContinuousSignDataset(args.data, "val")

    print(f"Training samples: {len(train_ds):,}")
    print(f"Validation samples: {len(val_ds):,}")

    # DataLoaders
    if args.balanced_sampling:
        # Compute per-sample weights based on inverse gloss frequency
        print("\nComputing balanced sampling weights...")
        gloss_counts = np.zeros(num_classes, dtype=np.int64)
        for i in range(len(train_ds.labels)):
            lbl_len = int(train_ds.label_lengths[i])
            if lbl_len == 0:
                continue
            lbls = train_ds.labels[i][:lbl_len]
            for idx in lbls:
                idx_int = int(idx)
                if 0 <= idx_int < num_classes:
                    gloss_counts[idx_int] += 1

        # Avoid div by zero
        gloss_counts = np.maximum(gloss_counts, 1)
        inv_freq = 1.0 / gloss_counts

        # Per-sample weight = mean inverse frequency of its labels
        sample_weights = []
        for i in range(len(train_ds.labels)):
            lbl_len = int(train_ds.label_lengths[i])
            if lbl_len == 0:
                sample_weights.append(1.0)
                continue
            lbls = train_ds.labels[i][:lbl_len]
            vals = [inv_freq[int(idx)] for idx in lbls if 0 <= int(idx) < num_classes]
            if len(vals) == 0:
                sample_weights.append(1.0)
            else:
                sample_weights.append(float(np.mean(vals)))

        # Normalize weights
        sample_weights = np.asarray(sample_weights, dtype=np.float64)
        sample_weights = sample_weights / (sample_weights.mean() + 1e-8)

        from torch.utils.data import WeightedRandomSampler

        sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.double),
            num_samples=len(train_ds),
            replacement=True,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            sampler=sampler,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )
    else:
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=2,
            pin_memory=torch.cuda.is_available(),
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    # Model
    print(f"\nCreating model...")
    # Auto-detect input feature dimension from dataset
    sample_seq = train_ds[0][0]  # Get first sequence
    if len(sample_seq.shape) == 2:
        input_features = sample_seq.shape[1]
    else:
        input_features = sample_seq.shape[0]
    print(f"Detected input feature dimension: {input_features}")
    print(f"Architecture: {'Transformer' if args.use_transformer else 'GRU'}")
    
    if args.use_transformer:
        model = TransformerCTCModel(
            num_classes=num_classes,
            input_features=input_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            num_heads=args.num_heads,
        ).to(device)
    else:
        model = ContinuousSignModel(
            num_classes=num_classes,
            input_features=input_features,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
        ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Loss and optimizer
    criterion = nn.CTCLoss(blank=blank_idx, reduction="mean", zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3, verbose=True
    )

    # Decoder
    decoder = CTCDecoder(idx_to_gloss, blank_idx)

    # Resume from checkpoint if requested
    start_epoch = 0
    best_exact_acc = 0
    best_gloss_acc = 0
    history = {"train_loss": [], "exact_acc": [], "gloss_acc": [], "lr": []}

    checkpoint_pattern = os.path.join(args.checkpoint_dir, "epoch_*.pt")
    checkpoints = sorted(glob(checkpoint_pattern))

    if args.resume and checkpoints:
        latest_ckpt = checkpoints[-1]
        print(f"\nResuming from: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_exact_acc = checkpoint.get("best_exact_acc", 0)
        best_gloss_acc = checkpoint.get("best_gloss_acc", 0)
        history = checkpoint.get("history", history)
        print(f"Resuming from epoch {start_epoch}, best exact_acc: {best_exact_acc:.2%}")
    elif args.resume:
        print(f"\nNo checkpoint found in {args.checkpoint_dir}, starting fresh")

    # Training loop
    print(f"\n{'=' * 70}")
    print("TRAINING")
    print(f"{'=' * 70}\n")

    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_start = time.time()

        # Train
        train_loss, train_blank_pct = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            epoch,
            blank_idx,
            device,
            blank_logit_offset=args.blank_logit_offset,
            blank_penalty_scale=args.blank_penalty_scale,
            blank_penalty_margin=args.blank_penalty_margin,
            pretrain_ce_epochs=args.pretrain_ce_epochs,
            ce_label_smoothing=args.ce_label_smoothing,
        )

        # Validate
        exact_acc, gloss_acc, val_blank_pct, predictions, targets = validate(
            model, val_loader, decoder, idx_to_gloss, blank_idx, device
        )

        # Update scheduler (monitor gloss accuracy for smoother signal)
        scheduler.step(gloss_acc)

        # Update history
        history["train_loss"].append(train_loss)
        history["exact_acc"].append(exact_acc)
        history["gloss_acc"].append(gloss_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        epoch_time = time.time() - epoch_start

        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Train Blank%: {train_blank_pct:.1f}% (Val: {val_blank_pct:.1f}%)")
        print(f"  Exact Match Acc: {exact_acc:.2%}")
        print(f"  Gloss Acc: {gloss_acc:.2%}")
        print(f"  Time: {epoch_time:.1f}s, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save periodic checkpoint (resume capability)
        periodic_ckpt = os.path.join(args.checkpoint_dir, f"epoch_{epoch:03d}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "exact_acc": exact_acc,
                "gloss_acc": gloss_acc,
                "best_exact_acc": best_exact_acc,
                "best_gloss_acc": best_gloss_acc,
                "num_classes": num_classes,
                "blank_idx": blank_idx,
                "idx_to_gloss": idx_to_gloss,
                "history": history,
            },
            periodic_ckpt,
        )

        # Save best model
        if exact_acc > best_exact_acc:
            best_exact_acc = exact_acc
            best_gloss_acc = gloss_acc

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "exact_acc": exact_acc,
                    "gloss_acc": gloss_acc,
                    "num_classes": num_classes,
                    "blank_idx": blank_idx,
                    "idx_to_gloss": idx_to_gloss,
                    "history": history,
                },
                "continuous_best_model.pt",
            )

            print(
                f"  🏆 New best model! Exact: {exact_acc:.2%}, Gloss: {gloss_acc:.2%}"
            )

            # Show some predictions
            print(f"\n  Sample predictions:")
            for i in range(min(3, len(predictions))):
                print(f"    True: {' → '.join(targets[i])}")
                print(f"    Pred: {' → '.join(predictions[i])}")
                print()

    print(f"\n{'=' * 70}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 70}")
    print(f"Best exact match accuracy: {best_exact_acc:.2%}")
    print(f"Best gloss accuracy: {best_gloss_acc:.2%}")

    print(f"\n✓ Model saved to continuous_best_model.pt")
    print(f"\nNext steps:")
    print(f"  1. Test inference: python continuous_inference.py your_video.mp4")
    print(f"  2. Train on real data once you have it")
    print(f"  3. Add language model for phrase correction")


if __name__ == "__main__":
    main()
