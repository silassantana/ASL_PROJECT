#!/usr/bin/env python3
"""Landmark trajectory -> English inference CLI.

Loads a checkpoint that follows the contract used by ``scripts/smoke_train.py``
(and can be produced by any training script that adopts the same schema),
rebuilds the real ``ASLSeq2Seq`` model from ``train_seq2seq_diagnostic.py``,
and greedily decodes one sentence from a MediaPipe pose trajectory.  The model
class is imported, never forked, so inference runs exactly the architecture the
checkpoint was trained with.

Checkpoint contract
-------------------
``torch.load(checkpoint, map_location="cpu")`` returns::

    {
        "model_state": state_dict,
        "vocab": {"<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3, "word": 4, ...},
        "config": {
            "hidden_dim": int, "enc_layers": int, "dec_layers": int,
            "num_heads": int, "input_features": int, "max_tgt": int,
        },
        "source": "smoke_train",          # informational
    }

Input contract (``--npz``)
--------------------------
The ``.npz`` must contain a ``poses`` array:

* ``[T, 1629]`` -- already stacked position + velocity + acceleration, exactly
  what ``scrape_youtube_asl.py`` writes.  Used as-is.
* ``[T, 543]``  -- pre-motion landmarks.  Expanded with the same 3-line
  pos/vel/acc stacking helper (imported from ``scrape_youtube_asl.py`` when
  cv2 is installed, otherwise reproduced locally byte-for-byte).

An optional ``caption`` (or ``text``) entry in the ``.npz`` is displayed as the
reference; ``--text "..."`` takes precedence.  The reference is shown only
when it is actually present -- nothing is fabricated.

Usage
-----
::

    python scripts/infer_landmarks.py \\
        --checkpoint runs/smoke/model.pt --npz poses.npz

    python scripts/infer_landmarks.py \\
        --checkpoint runs/smoke/model.pt --npz poses.npz \\
        --text "the reference caption"

Exit codes: ``0`` success, ``1`` user/input error (missing checkpoint or npz,
bad shape, config mismatch).  The model's output is a *greedy decode* of the
loaded checkpoint: with the smoke checkpoint the text is meaningless because
that checkpoint was trained on random synthetic poses.

Note on normalization: the checkpoint contract carries no training mean/std, so
no feature normalization is applied.  The smoke checkpoint was trained on raw
(synthetic) poses, so raw input is correct for it.  A checkpoint trained on real
data would need the training statistics; this CLI intentionally does not guess.

The module is import-safe: ``main(argv)`` returns an int and nothing runs at
import time except the repo-root ``sys.path`` adjustment.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The real model class (never forked) and its token ids.
from train_seq2seq_diagnostic import (  # noqa: E402
    EOS,
    PAD,
    SOS,
    UNK,
    ASLSeq2Seq,
)

POSE_DIM = 1629  # 543 landmarks * 3 streams (pos, vel, acc)
PRE_MOTION_DIM = 543  # 142 landmarks * 3 coords, zero-padded
DEFAULT_CHECKPOINT = "runs/smoke/model.pt"
DEFAULT_MAX_LEN = 32
# The model's learned positional embedding holds 512 slots; greedy_decode
# feeds it at most max_len + 1 tokens (SOS + generated), so cap below that.
MAX_DECODE_LEN = 510

_CONFIG_KEYS = (
    "hidden_dim",
    "enc_layers",
    "dec_layers",
    "num_heads",
    "input_features",
    "max_tgt",
)


class InferenceError(RuntimeError):
    """User-facing input/config error; ``main`` turns it into exit code 1."""


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------


def _add_motion_features(seq: np.ndarray) -> np.ndarray:
    """Expand ``[T, 543]`` landmarks to ``[T, 1629]`` with vel + acceleration.

    Mirrors ``add_motion_features`` in ``scrape_youtube_asl.py``.  That module
    imports cv2 at module import time and cv2 is an optional extra, so try the
    real helper lazily and fall back to this equivalent copy when it is not
    importable (keeping the inference path dependency-free).
    """
    try:
        from scrape_youtube_asl import add_motion_features as _impl
    except ImportError:
        _impl = None
    if _impl is not None:
        return _impl(seq)

    # Fallback: same three lines as scrape_youtube_asl.add_motion_features.
    vel = np.zeros_like(seq)
    vel[1:] = seq[1:] - seq[:-1]
    acc = np.zeros_like(seq)
    acc[1:] = vel[1:] - vel[:-1]
    return np.concatenate([seq, vel, acc], axis=1)


def load_poses(npz_path: str | Path) -> tuple[np.ndarray, str | None]:
    """Load ``[T, 1629]`` poses (and an optional reference caption) from an npz.

    Accepts ``[T, 1629]`` directly or ``[T, 543]`` pre-motion landmarks.
    Returns ``(poses, caption_or_None)``.
    """
    path = Path(npz_path)
    if not path.is_file():
        raise InferenceError(
            f"pose file not found: {path}\n"
            "  Produce one with: python scrape_youtube_asl.py --ids ... "
            "--output-dir ..."
        )

    try:
        with np.load(path, allow_pickle=False) as npz:
            files = list(npz.files)
            if "poses" not in files:
                raise InferenceError(
                    f"{path} has no 'poses' array (found: {files or 'nothing'})\n"
                    "  Expected an .npz from scrape_youtube_asl.py (key: poses)."
                )
            poses = np.asarray(npz["poses"], dtype=np.float32)
            caption = None
            for key in ("caption", "text"):
                if key in files:
                    value = npz[key]
                    if value.shape == ():
                        caption = str(value.item())
                    break
    except OSError as exc:
        raise InferenceError(f"could not read {path}: {exc}") from exc

    if poses.ndim != 2:
        raise InferenceError(f"poses in {path} must be 2-D [T, features], got shape {poses.shape}")
    n_frames, n_features = poses.shape
    if n_frames < 1:
        raise InferenceError(f"poses in {path} has no frames (shape {poses.shape})")

    if n_features == PRE_MOTION_DIM:
        poses = _add_motion_features(poses)
    elif n_features != POSE_DIM:
        raise InferenceError(
            f"poses in {path} have {n_features} features per frame; expected "
            f"{POSE_DIM} (pos+vel+acc) or {PRE_MOTION_DIM} (pre-motion landmarks)\n"
            "  Check that the .npz is a pose trajectory from "
            "scrape_youtube_asl.py."
        )

    if caption is not None:
        caption = caption.strip() or None
    return poses, caption


def load_checkpoint(checkpoint_path: str | Path) -> dict:
    """Load and validate a checkpoint produced under the documented contract."""
    path = Path(checkpoint_path)
    if not path.is_file():
        raise InferenceError(
            f"checkpoint not found: {path}\n"
            "  Train a CPU smoke checkpoint first:\n"
            "    python scripts/smoke_train.py --out runs/smoke"
        )
    try:
        checkpoint = torch.load(path, map_location="cpu")
    except Exception as exc:  # corrupt pickle, torch version, wrong file type...
        raise InferenceError(f"could not load checkpoint {path}: {exc}") from exc

    if not isinstance(checkpoint, dict):
        raise InferenceError(f"{path} is not a checkpoint dict (got {type(checkpoint).__name__})")
    missing = [key for key in ("model_state", "vocab", "config") if key not in checkpoint]
    if missing:
        raise InferenceError(
            f"checkpoint {path} is missing keys: {', '.join(missing)}\n"
            "  Expected {model_state, vocab, config}; see scripts/smoke_train.py."
        )
    if not isinstance(checkpoint["model_state"], dict) or not checkpoint["model_state"]:
        raise InferenceError(f"checkpoint {path} has an empty/invalid 'model_state'")
    if not isinstance(checkpoint["vocab"], dict) or not checkpoint["vocab"]:
        raise InferenceError(f"checkpoint {path} has an empty/invalid 'vocab'")
    if not isinstance(checkpoint["config"], dict):
        raise InferenceError(f"checkpoint {path} has an invalid 'config'")
    return checkpoint


# ---------------------------------------------------------------------------
# Model rebuilding / decoding
# ---------------------------------------------------------------------------


def resolve_device(name: str) -> torch.device:
    try:
        device = torch.device(name)
    except (RuntimeError, ValueError) as exc:
        raise InferenceError(
            f"invalid --device {name!r}: {exc}\n  Use --device cpu (the default) "
            "or a device name torch understands."
        ) from exc
    if device.type == "cuda" and not torch.cuda.is_available():
        print("[infer] CUDA requested but unavailable; falling back to CPU", file=sys.stderr)
        return torch.device("cpu")
    return device


def _vocab_size(checkpoint: dict) -> int:
    """Authoritative vocab size: the checkpoint's embedding rows, if present."""
    state = checkpoint["model_state"]
    embed = state.get("tgt_embed.weight")
    if embed is not None:
        return int(embed.shape[0])
    return max(int(i) for i in checkpoint["vocab"].values()) + 1


def build_model(checkpoint: dict, device: torch.device) -> ASLSeq2Seq:
    """Rebuild the real ``ASLSeq2Seq`` from a checkpoint's saved config."""
    config = checkpoint["config"]
    missing = [key for key in _CONFIG_KEYS if key not in config]
    if missing:
        raise InferenceError(
            "checkpoint config is missing keys: "
            f"{', '.join(missing)}; expected {', '.join(_CONFIG_KEYS)}"
        )

    input_dim = int(config["input_features"])
    if input_dim != POSE_DIM:
        raise InferenceError(
            f"checkpoint expects {input_dim} input features but this CLI decodes "
            f"{POSE_DIM}-dim pose trajectories (pos+vel+acc)\n"
            "  Load a checkpoint trained on the 1629-dim representation."
        )

    vocab_size = _vocab_size(checkpoint)
    vocab_ids = [int(i) for i in checkpoint["vocab"].values()]
    if min(vocab_ids) < 0 or max(vocab_ids) >= vocab_size:
        raise InferenceError(
            f"checkpoint vocab ids exceed the model embedding size "
            f"({min(vocab_ids)}..{max(vocab_ids)} vs vocab_size={vocab_size})"
        )

    # TemporalPoseEncoder registers a fixed-size sinusoidal positional-encoding
    # buffer whose rows depend on the training script's `max_len`.  Rebuild it
    # from the checkpoint (older checkpoints used 8192, smoke_train uses 512),
    # so a strict load works regardless of the training script's choice.
    pe = checkpoint["model_state"].get("encoder.pe")
    encoder_max_len = int(pe.shape[1]) if pe is not None and pe.ndim == 3 else 8192

    try:
        model = ASLSeq2Seq(
            vocab_size=vocab_size,
            input_dim=input_dim,
            hidden_dim=int(config["hidden_dim"]),
            enc_layers=int(config["enc_layers"]),
            dec_layers=int(config["dec_layers"]),
            num_heads=int(config["num_heads"]),
            dropout=0.0,  # inference only
            max_len=encoder_max_len,
        )
    except Exception as exc:  # invalid architecture in the checkpoint config
        raise InferenceError(
            f"could not rebuild ASLSeq2Seq from checkpoint config {config}: {exc}"
        ) from exc

    try:
        model.load_state_dict(checkpoint["model_state"], strict=True)
    except RuntimeError as exc:
        raise InferenceError(
            "checkpoint 'model_state' does not match the architecture rebuilt "
            f"from its config {config}\n  {exc}"
        ) from exc

    model.to(device)
    model.eval()
    return model


def decode_poses(
    model: ASLSeq2Seq,
    poses: np.ndarray,
    vocab: dict[str, int],
    device: torch.device | str = "cpu",
    max_len: int = DEFAULT_MAX_LEN,
) -> str:
    """Greedily decode one sentence from a ``[T, 1629]`` pose trajectory."""
    if not 1 <= max_len <= MAX_DECODE_LEN:
        raise InferenceError(f"--max-len must be between 1 and {MAX_DECODE_LEN} (got {max_len})")
    poses = np.asarray(poses, dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] != POSE_DIM:
        raise InferenceError(f"decode_poses expects [T, {POSE_DIM}] poses, got shape {poses.shape}")
    device = torch.device(device)
    src = torch.from_numpy(np.ascontiguousarray(poses)).unsqueeze(0).to(device)

    with torch.no_grad():
        token_ids = model.greedy_decode(src, src_key_padding_mask=None, max_len=max_len)

    inverse_vocab = {int(idx): word for word, idx in vocab.items()}
    words = []
    for token_id in token_ids[0].tolist():
        if token_id == EOS:
            break
        if token_id in (PAD, SOS, UNK):
            continue
        word = inverse_vocab.get(int(token_id))
        if word:
            words.append(word)
    return " ".join(words)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Greedily decode English text from a MediaPipe pose .npz "
        "with the ASL landmark->text seq2seq model.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help=f"Checkpoint .pt (default: {DEFAULT_CHECKPOINT})",
    )
    parser.add_argument(
        "--npz", required=True, help=".npz with a 'poses' array [T,1629] or [T,543]"
    )
    parser.add_argument(
        "--text", default=None, help="Reference caption to print alongside the prediction"
    )
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--max-len",
        type=int,
        default=DEFAULT_MAX_LEN,
        help=f"Max decoded tokens (default: {DEFAULT_MAX_LEN})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        device = resolve_device(args.device)
        checkpoint = load_checkpoint(args.checkpoint)
        poses, npz_caption = load_poses(args.npz)
        model = build_model(checkpoint, device)
        prediction = decode_poses(
            model,
            poses,
            checkpoint["vocab"],
            device=device,
            max_len=args.max_len,
        )
    except InferenceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130

    reference = args.text or npz_caption
    print(f"prediction: {prediction or '(empty)'}")
    if reference:
        print(f"reference: {reference}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
