# train_frame_classifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np


class FrameClassifierDataset(Dataset):
    """Dataset for frame-wise classification"""

    def __init__(self, npz_path: str, split: str = "train", top_k: int = 20):
        data = np.load(npz_path, allow_pickle=True)

        sequences = data[f"{split}_sequences"]
        labels = data[f"{split}_labels"]
        label_lengths = data[f"{split}_label_lengths"]

        # Find most common labels (same as before)
        all_labels = []
        for split_name in ["train", "val"]:
            split_labels = data[f"{split_name}_labels"]
            split_lengths = data[f"{split_name}_label_lengths"]
            for seq_labels, seq_len in zip(split_labels, split_lengths):
                all_labels.extend(seq_labels[:seq_len])

        from collections import Counter

        label_counts = Counter(all_labels)
        top_classes = [label for label, _ in label_counts.most_common(top_k)]

        self.label_map = {orig: new for new, orig in enumerate(top_classes)}
        self.num_classes = top_k

        # Create frame-level dataset
        self.frames = []
        self.frame_labels = []

        for seq, seq_labels, seq_len in zip(sequences, labels, label_lengths):
            if seq_len == 0:
                continue

            # Use the most common label in this sequence
            from collections import Counter

            seq_label_counts = Counter(seq_labels[:seq_len])
            most_common = seq_label_counts.most_common(1)[0][0]

            if most_common in self.label_map:
                label_idx = self.label_map[most_common]

                # Use every 4th frame to avoid too much redundancy
                for t in range(0, len(seq), 4):
                    self.frames.append(seq[t])
                    self.frame_labels.append(label_idx)

        print(f"{split}: {len(self.frames)} frames, {top_k} classes")

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = torch.from_numpy(self.frames[idx]).float().view(-1)  # (316,)
        label = torch.tensor(self.frame_labels[idx], dtype=torch.long)
        return frame, label


class FrameClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(79 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def train_frame_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_ds = FrameClassifierDataset("ctc_sequences_multi_100.npz", "train", top_k=20)
    val_ds = FrameClassifierDataset("ctc_sequences_multi_100.npz", "val", top_k=20)

    print(f"\nTraining frames: {len(train_ds)}")
    print(f"Validation frames: {len(val_ds)}")

    # Model
    model = FrameClassifier(train_ds.num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    print("\nTraining frame classifier...")

    best_acc = 0
    for epoch in range(20):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            outputs = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y).sum().item()
            total += y.size(0)

        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = outputs.max(1)
                val_correct += predicted.eq(y).sum().item()
                val_total += y.size(0)

        val_acc = 100.0 * val_correct / val_total

        print(
            f"Epoch {epoch + 1:2d}: loss={total_loss / len(train_loader):.3f}, "
            f"train_acc={train_acc:.1f}%, val_acc={val_acc:.1f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_classes": train_ds.num_classes,
                    "label_map": train_ds.label_map,
                    "val_acc": val_acc,
                },
                "frame_classifier.pt",
            )
            print(f"  ✓ Saved best model ({val_acc:.1f}%)")

    print(f"\nBest validation accuracy: {best_acc:.1f}%")

    # Use this as initialization for CTC
    return model


if __name__ == "__main__":
    train_frame_classifier()
