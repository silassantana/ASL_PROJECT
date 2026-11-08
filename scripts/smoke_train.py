#!/usr/bin/env python3
"""CPU smoke-train for the ASL landmark -> text seq2seq model.

This is a *pipeline* check, not a research run: it builds a fully synthetic,
learnable toy corpus in memory, trains the real ``ASLSeq2Seq`` model from
``train_seq2seq_diagnostic.py`` for a handful of Adam steps on CPU, and writes
artifacts that other tooling (``scripts/infer_landmarks.py``, tests, docs) can
consume.  No dataset, checkpoint, GPU, or network access is required.

Why it can learn
----------------
Target "sentences" are a deterministic function of the pose sequence mean:
each sample is generated from a class prototype, the label class is recovered
from the *actual* pose mean, and the sentence is ``filler_a, class_word,
filler_b, filler_b, ...``.  So the loss genuinely decreases -- the task is not
random-label noise.  Batches are class-balanced to keep the loss curve stable.

Contract (do not change without updating tests/docs/infer script)
-----------------------------------------------------------------
CLI::

    python scripts/smoke_train.py \\
        --steps 40 --batch-size 8 --out runs/smoke --seed 42 \\
        --device cpu --hidden-dim 64 --vocab-size 64

Outputs:

* ``{out}/metrics.json`` -- exactly these keys:
  ``seed, steps, batch_size, device, wall_s, loss_start, loss_end,
  toy_token_accuracy``
* ``{out}/model.pt`` -- ``torch.save`` of::

      {
        "model_state": model.state_dict(),
        "vocab": vocab,                      # dict[str, int]
        "config": {
            "hidden_dim": int, "enc_layers": int, "dec_layers": int,
            "num_heads": int, "input_features": 1629, "max_tgt": int,
        },
        "source": "smoke_train",
      }

``loss_start`` is the mean training loss of the first ``max(1, steps // 4)``
steps; ``loss_end`` is the mean of the last ``max(1, steps // 4)`` steps.
``toy_token_accuracy`` is teacher-forced token accuracy on a held-out synthetic
batch after training (excluding ``<pad>`` positions).

The run is deterministic for a fixed ``--seed`` (Python/NumPy/torch RNGs are
seeded and data order is fixed) and exits 0.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reuse the real model / collate / token pipeline -- never fork them.
from train_seq2seq_diagnostic import (  # noqa: E402
    PAD,
    SPECIAL,
    ASLSeq2Seq,
    collate_fn,
)

POSE_DIM = 1629          # 142 landmarks * 3 for pos + vel + acc (real model input)
MIN_POSE_FRAMES = 16
MAX_POSE_FRAMES = 48
POSE_NOISE = 0.10        # per-coordinate noise; mean stays class-separable
PROTOTYPE_SPAN = (-3.0, 3.0)


# ---------------------------------------------------------------------------
# Synthetic, learnable toy corpus
# ---------------------------------------------------------------------------

def build_toy_vocab(vocab_size: int) -> dict[str, int]:
    """Vocabulary with the real special tokens plus toy word tokens."""
    if vocab_size < 8:
        raise SystemExit("--vocab-size must be >= 8 (4 specials + 1 class + 2 fillers)")
    vocab = {s: i for i, s in enumerate(SPECIAL)}
    for i in range(vocab_size - len(SPECIAL)):
        vocab[f"toy{i}"] = len(vocab)
    return vocab


def _make_prototypes(n_classes: int, rng: np.random.RandomState) -> np.ndarray:
    """Class prototypes whose first coordinate identifies the class.

    The first coordinate of a pose sequence mean is a deterministic function of
    the class (``PROTOTYPE_SPAN`` divided into ``n_classes`` buckets), so the
    synthetic target sentence is a deterministic function of the pose mean.
    """
    protos = rng.randn(n_classes, POSE_DIM).astype(np.float32) * 0.05
    if n_classes == 1:
        protos[0, 0] = 0.0
    else:
        lo, hi = PROTOTYPE_SPAN
        protos[:, 0] = np.linspace(lo + (hi - lo) / (2 * n_classes),
                                   hi - (hi - lo) / (2 * n_classes),
                                   n_classes).astype(np.float32)
    return protos


def _class_from_pose_mean(pose: np.ndarray, n_classes: int) -> int:
    """Deterministic classification rule: quantize the pose mean's 1st coord."""
    lo, hi = PROTOTYPE_SPAN
    m0 = float(pose.mean(axis=0)[0])
    k = int(math.floor((m0 - lo) / (hi - lo) * n_classes))
    return max(0, min(n_classes - 1, k))


class ToyCorpus:
    """Deterministic in-memory generator of synthetic segment samples."""

    def __init__(self, seed: int, vocab: dict[str, int], max_tgt: int = 8,
                 n_classes: int = 8):
        self.vocab = vocab
        self.max_tgt = max_tgt
        self.n_classes = n_classes
        self.rng = np.random.RandomState(seed)

        n_words = len(vocab) - len(SPECIAL)
        if n_words < n_classes + 2:
            n_classes = max(1, n_words - 2)
            self.n_classes = n_classes
        first_word = len(SPECIAL)
        self.class_ids = [first_word + k for k in range(self.n_classes)]
        # Fillers live in the tail of the vocab so they never equal a class word.
        self.filler_ids = [first_word + n_words - 2, first_word + n_words - 1]
        self.prototypes = _make_prototypes(self.n_classes, self.rng)

    def _balanced_classes(self, batch_size: int) -> np.ndarray:
        """Class-balanced sample list so per-batch loss stays low-variance."""
        reps = max(1, int(round(batch_size / self.n_classes)))
        ks = np.tile(np.arange(self.n_classes), reps)[:batch_size]
        if len(ks) < batch_size:  # remainder -> round-robin
            extra = (np.arange(batch_size - len(ks)) % self.n_classes)
            ks = np.concatenate([ks, extra])
        self.rng.shuffle(ks)
        return ks

    def make_batch(self, batch_size: int):
        """Return (src, tgt, src_lens, tgt_lens, src_pad, texts) via collate_fn."""
        samples = []
        for k in self._balanced_classes(batch_size):
            T = int(self.rng.randint(MIN_POSE_FRAMES, MAX_POSE_FRAMES + 1))
            pose = (self.prototypes[k][None, :]
                    + POSE_NOISE * self.rng.randn(T, POSE_DIM)).astype(np.float32)
            # Target sentence is derived from the *actual* pose mean.
            cls = _class_from_pose_mean(pose, self.n_classes)
            L = int(self.rng.randint(3, self.max_tgt + 1))
            tokens = [self.filler_ids[0], self.class_ids[cls]]
            tokens += [self.filler_ids[1]] * (L - 2)
            text = " ".join(f"toy{t}" for t in tokens)
            samples.append(
                (torch.from_numpy(pose), torch.tensor(tokens, dtype=torch.long), text)
            )
        return collate_fn(samples)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _token_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Teacher-forced accuracy over non-pad target tokens."""
    pred = logits.argmax(dim=-1)
    mask = targets != PAD
    n = int(mask.sum().item())
    if n == 0:
        return 0.0
    return float((pred[mask] == targets[mask]).float().mean().item())


def _resolve_device(name: str) -> torch.device:
    dev = torch.device(name)
    if dev.type == "cuda" and not torch.cuda.is_available():
        print(f"[smoke] requested device {name!r} but CUDA is unavailable; using cpu",
              flush=True)
        return torch.device("cpu")
    return dev


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="CPU smoke-train the ASL landmark->text seq2seq model on "
                    "synthetic learnable data.",
    )
    p.add_argument("--steps", type=int, default=40,
                   help="Number of Adam steps (default: 40)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--out", type=str, default="runs/smoke",
                   help="Output directory for metrics.json and model.pt")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--vocab-size", type=int, default=64,
                   help="Total vocabulary size incl. 4 special tokens")
    # Extra knobs kept explicit so the saved config is sufficient for inference.
    p.add_argument("--enc-layers", type=int, default=2)
    p.add_argument("--dec-layers", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--max-tgt", type=int, default=8,
                   help="Maximum synthetic target sentence length")
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--log-every", type=int, default=0,
                   help="Print a table row every N steps (0 = auto)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.steps < 1:
        raise SystemExit("--steps must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    # ---- determinism -------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    device = _resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    vocab = build_toy_vocab(args.vocab_size)
    corpus = ToyCorpus(args.seed, vocab, max_tgt=args.max_tgt)

    # ---- model (the real class; do not fork) ------------------------------
    model = ASLSeq2Seq(
        vocab_size=len(vocab),
        input_dim=POSE_DIM,
        hidden_dim=args.hidden_dim,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        # scripts/infer_landmarks.py rebuilds the model with max_len=512, so the
        # saved "encoder.pe" buffer must be 512 rows for strict load_state_dict.
        max_len=512,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[smoke] device={device}  vocab={len(vocab)}  "
          f"classes={corpus.n_classes}  params={n_params:,}", flush=True)
    print(f"[smoke] steps={args.steps}  batch={args.batch_size}  "
          f"hidden={args.hidden_dim}  lr={args.lr}", flush=True)

    log_every = args.log_every if args.log_every > 0 else max(1, args.steps // 8)
    print(f"{'step':>5} {'loss':>9} {'tok_acc':>8} {'wall_s':>8}", flush=True)

    losses: list[float] = []
    accs: list[float] = []
    t_train = time.monotonic()
    model.train()
    for step in range(1, args.steps + 1):
        src, tgt, _src_lens, _tgt_lens, src_pad, _texts = corpus.make_batch(args.batch_size)
        src = src.to(device)
        tgt = tgt.to(device)
        src_pad = src_pad.to(device)

        dec_in, dec_tgt = tgt[:, :-1], tgt[:, 1:]
        logits = model(src, dec_in, src_key_padding_mask=src_pad)
        loss = criterion(logits.reshape(-1, logits.size(-1)), dec_tgt.reshape(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(float(loss.item()))
        accs.append(_token_accuracy(logits.detach(), dec_tgt))
        if step % log_every == 0 or step == 1 or step == args.steps:
            print(f"{step:>5} {losses[-1]:>9.4f} {accs[-1]:>8.3f} "
                  f"{time.monotonic() - t_train:>8.2f}", flush=True)

    # ---- held-out toy accuracy + metrics ----------------------------------
    model.eval()
    with torch.no_grad():
        eval_src, eval_tgt, _l1, _l2, eval_pad, _t = corpus.make_batch(args.batch_size)
        eval_logits = model(
            eval_src.to(device),
            eval_tgt[:, :-1].to(device),
            src_key_padding_mask=eval_pad.to(device),
        )
        toy_acc = _token_accuracy(eval_logits.cpu(), eval_tgt[:, 1:])

    window = max(1, args.steps // 4)
    loss_start = float(np.mean(losses[:window]))
    loss_end = float(np.mean(losses[-window:]))
    wall_s = round(time.monotonic() - t0, 3)

    metrics = {
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": str(device),
        "wall_s": wall_s,
        "loss_start": loss_start,
        "loss_end": loss_end,
        "toy_token_accuracy": float(toy_acc),
    }
    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    # ---- checkpoint: exact schema consumed by scripts/infer_landmarks.py --
    ckpt = {
        "model_state": model.state_dict(),
        "vocab": vocab,
        "config": {
            "hidden_dim": args.hidden_dim,
            "enc_layers": args.enc_layers,
            "dec_layers": args.dec_layers,
            "num_heads": args.num_heads,
            "input_features": POSE_DIM,
            "max_tgt": args.max_tgt,
        },
        "source": "smoke_train",
    }
    torch.save(ckpt, out_dir / "model.pt")

    print(f"[smoke] loss_start={loss_start:.4f}  loss_end={loss_end:.4f}  "
          f"toy_token_accuracy={toy_acc:.3f}", flush=True)
    print(f"[smoke] wrote {metrics_path} and {out_dir / 'model.pt'} "
          f"({wall_s:.2f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
