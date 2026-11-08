"""Tests for ``scripts/infer_landmarks.py`` against the smoke-train checkpoint.

If the inference CLI has not been implemented yet, the whole module is skipped
gracefully (the contract refers to it as a separate deliverable).  When it is
present, these tests load the real checkpoint written by
``scripts/smoke_train.py`` and decode a synthetic ``.npz`` pose trajectory --
no dataset, no network, no GPU.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
INFER_SCRIPT = REPO_ROOT / "scripts" / "infer_landmarks.py"

if not INFER_SCRIPT.is_file():  # pragma: no cover - only before the CLI lands
    pytest.skip(
        "scripts/infer_landmarks.py is not implemented yet; "
        "skipping checkpoint/inference tests",
        allow_module_level=True,
    )

POSE_DIM = 1629
PRE_MOTION_DIM = 543


@pytest.fixture(scope="module")
def infer():
    """Import scripts/infer_landmarks.py by path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("infer_landmarks_under_test", INFER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smoke_checkpoint(smoke_run):
    """The checkpoint produced by the real smoke-train subprocess."""
    assert smoke_run.checkpoint_path.is_file(), "smoke-train model.pt missing"
    return torch.load(smoke_run.checkpoint_path, map_location="cpu")


@pytest.fixture(scope="module")
def npz_poses(tmp_path_factory):
    """Synthetic [T, 1629] pose trajectory .npz with a caption."""
    out = tmp_path_factory.mktemp("infer_npz")
    rng = np.random.RandomState(0)
    poses = rng.randn(24, POSE_DIM).astype(np.float32)
    path = out / "poses.npz"
    np.savez(path, poses=poses, caption="a toy caption")
    return path, poses


# ---------------------------------------------------------------------------
# Checkpoint loading / model rebuild
# ---------------------------------------------------------------------------

class TestCheckpoint:
    def test_load_checkpoint_accepts_smoke_artifact(self, infer, smoke_run):
        ckpt = infer.load_checkpoint(smoke_run.checkpoint_path)

        assert set(ckpt) >= {"model_state", "vocab", "config"}
        assert ckpt["source"] == "smoke_train"

    def test_build_model_matches_checkpoint_config(self, infer, smoke_run):
        ckpt = infer.load_checkpoint(smoke_run.checkpoint_path)
        model = infer.build_model(ckpt, torch.device("cpu"))

        assert isinstance(model, infer.ASLSeq2Seq)
        assert model.vocab_size == len(ckpt["vocab"])
        assert model.hidden_dim == ckpt["config"]["hidden_dim"]
        assert not model.training, "inference model must be in eval mode"
        assert sum(p.numel() for p in model.parameters()) > 0

    def test_missing_checkpoint_is_an_error(self, infer, tmp_path):
        with pytest.raises(infer.InferenceError):
            infer.load_checkpoint(tmp_path / "does_not_exist.pt")


# ---------------------------------------------------------------------------
# Pose loading
# ---------------------------------------------------------------------------

class TestLoadPoses:
    def test_loads_1629_dim_npz_with_caption(self, infer, npz_poses):
        path, poses = npz_poses
        loaded, caption = infer.load_poses(path)

        assert loaded.shape == (24, POSE_DIM)
        assert np.allclose(loaded, poses, atol=1e-6)
        assert caption == "a toy caption"

    def test_expands_543_dim_pre_motion_landmarks(self, infer, tmp_path):
        rng = np.random.RandomState(1)
        landmarks = rng.randn(12, PRE_MOTION_DIM).astype(np.float32)
        path = tmp_path / "pre_motion.npz"
        np.savez(path, poses=landmarks)

        loaded, caption = infer.load_poses(path)

        assert loaded.shape == (12, POSE_DIM)
        assert np.allclose(loaded[:, :PRE_MOTION_DIM], landmarks, atol=1e-6)
        assert np.isfinite(loaded).all()
        assert caption is None

    def test_rejects_wrong_feature_count(self, infer, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(path, poses=np.zeros((8, 100), dtype=np.float32))
        with pytest.raises(infer.InferenceError, match="features per frame"):
            infer.load_poses(path)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class TestDecode:
    def test_decode_poses_returns_clean_text(self, infer, smoke_checkpoint, npz_poses):
        path, _poses = npz_poses
        model = infer.build_model(smoke_checkpoint, torch.device("cpu"))
        poses, _caption = infer.load_poses(path)

        text = infer.decode_poses(
            model, poses, smoke_checkpoint["vocab"], device="cpu", max_len=6,
        )

        assert isinstance(text, str)
        tokens = text.split()
        assert len(tokens) <= 6, f"decoded more than --max-len tokens: {tokens}"
        for special in ("<pad>", "<sos>", "<eos>", "<unk>"):
            assert special not in tokens
        # Every emitted token must come from the checkpoint vocabulary.
        assert set(tokens).issubset(set(smoke_checkpoint["vocab"]))

    def test_decode_accepts_543_dim_npz_end_to_end(self, infer, smoke_checkpoint, tmp_path):
        rng = np.random.RandomState(2)
        path = tmp_path / "pre_motion.npz"
        np.savez(path, poses=rng.randn(18, PRE_MOTION_DIM).astype(np.float32))
        model = infer.build_model(smoke_checkpoint, torch.device("cpu"))
        poses, _caption = infer.load_poses(path)

        text = infer.decode_poses(model, poses, smoke_checkpoint["vocab"], max_len=4)

        assert isinstance(text, str)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

class TestCli:
    def test_cli_subprocess_on_synthetic_npz(self, smoke_run, npz_poses):
        npz_path, _poses = npz_poses
        proc = subprocess.run(
            [sys.executable, str(INFER_SCRIPT),
             "--checkpoint", str(smoke_run.checkpoint_path),
             "--npz", str(npz_path),
             "--max-len", "5"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            timeout=120,
        )

        assert proc.returncode == 0, (
            f"infer CLI failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
        assert "prediction:" in proc.stdout
        assert "reference: a toy caption" in proc.stdout

    def test_cli_missing_checkpoint_returns_1(self, infer, npz_poses, tmp_path):
        npz_path, _poses = npz_poses
        rc = infer.main([
            "--checkpoint", str(tmp_path / "missing.pt"),
            "--npz", str(npz_path),
        ])
        assert rc == 1, "user/input errors must exit with code 1"
