#!/usr/bin/env python3
"""
train_fixed_simple.py - Fixed training script for our new data format
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset, DataLoader
import argparse
import time
import json
from tqdm import tqdm
import warnings
from pathlib import Path
import os

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")


class SimpleDataset(Dataset):
    def __init__(self, h5_file, split="train", max_samples=None):
        self.h5_file = h5_file
        self.split = split

        # Open HDF5 file
        self.h5_f = h5py.File(h5_file, "r")

        # Get datasets
        self.sequences = self.h5_f[f"{split}_sequences"]
        self.labels = self.h5_f[f"{split}_labels"]
        self.label_lengths = self.h5_f[f"{split}_label_lengths"]

        self.num_samples = self.sequences.shape[0]
        if max_samples is not None:
            self.num_samples = min(self.num_samples, max_samples)

        self.num_classes = int(self.h5_f["num_classes"][()].item())
        self.seq_length = self.sequences.shape[1]

    def __del__(self):
        if hasattr(self, "h5_f"):
            self.h5_f.close()

    def __getitem__(self, idx):
        # Get sequence [T, 75, 2]
        seq = self.sequences[idx]

        # Reshape to [T, 150]
        T, J, F = seq.shape
        seq_reshaped = seq.reshape(T, J * F)

        # Normalize
        seq_mean = seq_reshaped.mean()
        seq_std = seq_reshaped.std() + 1e-8
        seq_normalized = (seq_reshaped - seq_mean) / seq_std

        # Get labels
        labels = self.labels[idx]
        llen = self.label_lengths[idx]
        labels = labels[:llen]

        return (
            torch.from_numpy(seq_normalized).float(),
            torch.tensor(labels, dtype=torch.long),
            llen,
        )

    def __len__(self):
        return self.num_samples


# Keep your existing SimpleModel class
class SimpleModel(nn.Module):
    def __init__(self, num_classes, input_dim=150):
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv1d(input_dim, 128, 3, padding="same")
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, 3, padding="same")
        self.bn2 = nn.BatchNorm1d(256)
        self.gru = nn.GRU(
            256, 128, 2, batch_first=True, bidirectional=True, dropout=0.3
        )
        self.fc = nn.Linear(256, num_classes + 1)

    def forward(self, x):
        B, T, C = x.shape
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.transpose(1, 2)
        x, _ = self.gru(x)
        logits = self.fc(x)
        return logits


def collate_fn(batch):
    xs, ys, llens = zip(*batch)
    x_batch = torch.stack(xs)

    max_llen = max(llens)
    y_padded = torch.zeros(len(ys), max_llen, dtype=torch.long)
    for i, y in enumerate(ys):
        y_padded[i, : len(y)] = y

    T = x_batch.shape[1]
    ilens = torch.full((len(xs),), T, dtype=torch.long)

    return x_batch, y_padded, ilens, torch.tensor(llens)


def decode_ctc(logits, blank_idx):
    """Simple CTC greedy decoding"""
    probs = F.softmax(logits, dim=-1)
    predictions = probs.argmax(dim=-1)  # [B, T]

    decoded = []
    for batch_idx in range(predictions.shape[0]):
        seq = []
        prev = -1
        for t in range(predictions.shape[1]):
            current = predictions[batch_idx, t].item()
            if current != blank_idx and current != prev:
                seq.append(current)
            prev = current
        decoded.append(seq)

    return decoded


def train_simple(args):
    print(f"\nLoading data from {args.data}")

    # Create datasets
    train_ds = SimpleDataset(args.data, "train", max_samples=args.train_samples)
    val_ds = SimpleDataset(args.data, "val", max_samples=args.val_samples)

    print(f"\nDataset info:")
    print(f"  Training samples: {len(train_ds)}")
    print(f"  Validation samples: {len(val_ds)}")
    print(f"  Classes: {train_ds.num_classes}")
    print(f"  Sequence length: {train_ds.seq_length}")

    # Create gloss mapping
    idx_to_gloss = {i: f"GLOSS_{i}" for i in range(train_ds.num_classes)}

    # Model
    model = SimpleModel(train_ds.num_classes).to(device)
    blank_idx = train_ds.num_classes

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # Simpler
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
    )

    # Training
    print(f"\n{'=' * 60}")
    print("TRAINING STARTING")
    print(f"{'=' * 60}")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0
        batches = 0

        pbar = tqdm(train_loader, desc="Training", leave=False)
        for x, y, ilens, llens in pbar:
            x = x.to(device)
            y = y.to(device)
            ilens = ilens.to(device)
            llens = llens.to(device)

            # Forward
            logits = model(x)

            # CTC Loss
            loss = F.ctc_loss(
                F.log_softmax(logits, dim=-1).transpose(0, 1),
                y,
                ilens,
                llens,
                blank=blank_idx,
                reduction="mean",
                zero_infinity=True,
            )

            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        avg_loss = train_loss / batches if batches > 0 else 0

        # Validation
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y, ilens, llens in val_loader:
                x = x.to(device)
                y = y.to(device)
                ilens = ilens.to(device)
                llens = llens.to(device)

                # Forward
                logits = model(x)

                # Decode
                decoded = decode_ctc(logits, blank_idx)

                # Compare
                for batch_idx in range(len(decoded)):
                    pred_seq = decoded[batch_idx]
                    true_seq = y[batch_idx, : llens[batch_idx]].cpu().tolist()

                    if len(pred_seq) > 0 and len(true_seq) > 0:
                        if pred_seq[0] == true_seq[0]:
                            correct += 1
                    total += 1

        accuracy = correct / total if total > 0 else 0
        epoch_time = time.time() - epoch_start

        print(f"  Loss: {avg_loss:.3f}")
        print(f"  Accuracy: {accuracy:.1%} ({correct}/{total})")
        print(f"  Time: {epoch_time:.1f}s")

        # Save checkpoint
        checkpoint_path = f"checkpoint_simple_epoch_{epoch + 1}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "accuracy": accuracy,
                "loss": avg_loss,
                "num_classes": train_ds.num_classes,
                "idx_to_gloss": idx_to_gloss,
                "blank_idx": blank_idx,
            },
            checkpoint_path,
        )
        print(f"  Saved checkpoint to: {checkpoint_path}")

    # Save final model
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": train_ds.num_classes,
            "idx_to_gloss": idx_to_gloss,
            "blank_idx": blank_idx,
        },
        "simple_model_final.pt",
    )

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"\nModel saved to: simple_model_final.pt")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=str, default="simple_patterns.h5", help="HDF5 data file"
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--train-samples", type=int, default=10000, help="Max training samples"
    )
    parser.add_argument(
        "--val-samples", type=int, default=2000, help="Max validation samples"
    )

    args = parser.parse_args()

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Train
    train_simple(args)
