"""Shared pytest fixtures for the ASL_PROJECT test suite.

All fixtures are CPU-only, dependency-free (no dataset, no checkpoint, no
network) and synthetic.  Heavy artifacts are built lazily so a slow/missing
optional dependency skips (never fails) the tests that need it.

Layout facts encoded by the synthetic data:
  * a pose frame is 142 landmarks * 3 coords = 426 floats, zero-padded to 543;
  * motion features stack position + velocity + acceleration -> 1629 dims.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# Flat-layout repo: make root modules importable before pyproject tooling lands.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- real data layout constants --------------------------------------------
POSE_DIM = 1629            # 543 * 3 (pos + vel + acc)
LANDMARK_DIM = 426         # 142 landmarks * 3 coordinates
PADDED_LANDMARK_DIM = 543  # 426 real floats + zero padding
SPECIAL_TOKENS = ("<pad>", "<sos>", "<eos>", "<unk>")
TOY_WORDS = ("hello", "world", "sign", "asl", "test", "good", "morning", "thanks")


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def build_toy_poses(n_frames: int = 24, seed: int = 0,
                    batch_size: int | None = None) -> torch.Tensor:
    """Build realistic ``[T, 1629]`` (or ``[B, T, 1629]``) pose tensor(s).

    The first 426 columns are real "landmark" values, columns 426:543 are the
    zero padding, and the rest are finite-difference velocity/acceleration.
    """
    def _one(sample_seed: int) -> np.ndarray:
        rng = np.random.RandomState(sample_seed)
        pos = np.zeros((n_frames, PADDED_LANDMARK_DIM), dtype=np.float32)
        pos[:, :LANDMARK_DIM] = rng.randn(n_frames, LANDMARK_DIM).astype(np.float32)
        vel = np.zeros_like(pos)
        vel[1:] = pos[1:] - pos[:-1]
        acc = np.zeros_like(pos)
        acc[1:] = vel[1:] - vel[:-1]
        return np.concatenate([pos, vel, acc], axis=1)

    if batch_size is None:
        return torch.from_numpy(_one(seed))
    return torch.from_numpy(np.stack([_one(seed + i) for i in range(batch_size)]))


def _write_segment_h5(path: Path, vocab: dict[str, int], rng: np.random.RandomState,
                      words: tuple[str, ...], sizes: dict[str, int],
                      max_frames: int) -> dict:
    """Write an H5 matching build_segment_h5.py's schema exactly."""
    h5py = pytest.importorskip("h5py")
    texts_by_split: dict[str, list[str]] = {}
    all_train_clips: list[np.ndarray] = []

    with h5py.File(path, "w") as f:
        for split, n in sizes.items():
            group = f.create_group(split)
            clips = group.create_group("clips")
            texts = group.create_group("texts")
            split_texts: list[str] = []
            for i in range(n):
                n_frames = int(rng.randint(16, max_frames + 1))
                clip = (rng.randn(n_frames, POSE_DIM) * 0.5).astype(np.float32)
                clips.create_dataset(str(i), data=clip)
                if split == "train":
                    all_train_clips.append(clip)
                n_words = int(rng.randint(2, 5))
                text = " ".join(str(w) for w in rng.choice(words, size=n_words))
                split_texts.append(text)
                texts.create_dataset(str(i), data=text.encode("utf-8"))
            group.create_dataset("n", data=n)
            texts_by_split[split] = split_texts

        stacked = np.concatenate(all_train_clips, axis=0)
        mean = stacked.mean(axis=0).astype(np.float32)
        std = (stacked.std(axis=0) + 1e-6).astype(np.float32)
        f.create_dataset("stats/mean", data=mean)
        f.create_dataset("stats/std", data=std)

    return {
        "path": path,
        "vocab": vocab,
        "words": words,
        "sizes": sizes,
        "texts": texts_by_split,
        "mean": mean,
        "std": std,
        "max_frames": max_frames,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def toy_poses():
    """Factory fixture: ``toy_poses(n_frames=24, seed=0, batch_size=None)``."""
    return build_toy_poses


@pytest.fixture
def toy_segment_h5(tmp_path):
    """Factory fixture writing a synthetic segment H5 (train/val + stats).

    ``toy_segment_h5(n_train=5, n_val=3, seed=0, words=TOY_WORDS, max_frames=40)``
    returns ``{path, vocab, words, sizes, texts, mean, std, max_frames}``.
    """

    def _make(n_train: int = 5, n_val: int = 3, seed: int = 0,
              words: tuple[str, ...] = TOY_WORDS, max_frames: int = 40) -> dict:
        rng = np.random.RandomState(seed)
        vocab = {token: i for i, token in enumerate(SPECIAL_TOKENS)}
        for word in words:
            vocab[word] = len(vocab)
        path = tmp_path / f"segments_{seed}.h5"
        return _write_segment_h5(path, vocab, rng, words,
                                 {"train": n_train, "val": n_val}, max_frames)

    return _make


@pytest.fixture
def toy_poses_h5(tmp_path):
    """Factory fixture writing a synthetic YouTube-ASL poses H5.

    Matches build_youtube_asl_h5.py: ``/poses/{vid}``, ``/captions/{vid}``,
    ``/segments/{vid}`` and a variable-length-bytes ``/index``.
    Returns ``{path, video_ids, lengths}``.
    """

    def _make(video_lengths: tuple[int, ...] = (24, 40, 16), seed: int = 0) -> dict:
        h5py = pytest.importorskip("h5py")
        rng = np.random.RandomState(seed)
        path = tmp_path / f"poses_{seed}.h5"
        video_ids = [f"vid_{i:02d}" for i in range(len(video_lengths))]

        with h5py.File(path, "w") as f:
            poses = f.create_group("poses")
            captions = f.create_group("captions")
            segments = f.create_group("segments")
            for video_id, n_frames in zip(video_ids, video_lengths):
                poses.create_dataset(
                    video_id,
                    data=(rng.randn(n_frames, POSE_DIM) * 0.5).astype(np.float32),
                )
                captions.create_dataset(
                    video_id, data=f"toy caption for {video_id}".encode())
                segments.create_dataset(
                    video_id,
                    data=json.dumps([{"start": 0.0, "end": 1.0, "text": "toy"}]).encode(),
                )
            dtype = h5py.special_dtype(vlen=bytes)
            index = f.create_dataset("index", shape=(len(video_ids),), dtype=dtype)
            for i, video_id in enumerate(video_ids):
                index[i] = video_id.encode("utf-8")

        return {"path": path, "video_ids": video_ids,
                "lengths": tuple(int(n) for n in video_lengths)}

    return _make


@pytest.fixture
def tiny_seq2seq():
    """Real ``ASLSeq2Seq`` with tiny dims (vocab ~20) for fast CPU tests."""
    mod = pytest.importorskip("train_seq2seq_diagnostic")
    model = mod.ASLSeq2Seq(
        vocab_size=20,
        input_dim=POSE_DIM,
        hidden_dim=32,
        enc_layers=1,
        dec_layers=1,
        num_heads=4,
        dropout=0.0,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def smoke_run(tmp_path_factory) -> SimpleNamespace:
    """Run scripts/smoke_train.py once (subprocess, tiny settings) for reuse."""
    out_dir = tmp_path_factory.mktemp("smoke_train") / "out"
    script = REPO_ROOT / "scripts" / "smoke_train.py"
    assert script.is_file(), f"missing smoke-train script: {script}"
    env = dict(os.environ)
    env.setdefault("PYTHONHASHSEED", "0")
    proc = subprocess.run(
        [sys.executable, str(script), "--steps", "4", "--out", str(out_dir)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=120,
        env=env,
    )
    return SimpleNamespace(proc=proc, out_dir=out_dir,
                           metrics_path=out_dir / "metrics.json",
                           checkpoint_path=out_dir / "model.pt")
