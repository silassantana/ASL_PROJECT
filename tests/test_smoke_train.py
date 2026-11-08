"""Contract tests for ``scripts/smoke_train.py`` (CPU, subprocess, tiny settings).

The session-scoped ``smoke_run`` fixture in ``conftest.py`` executes::

    python scripts/smoke_train.py --steps 4 --out <tmpdir>

and these tests assert the CLI exit code, the exact ``metrics.json`` key set,
the exact ``model.pt`` schema, that the checkpoint rebuilds the real model, and
that training is deterministic for a fixed seed.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "smoke_train.py"

METRIC_KEYS = {
    "seed", "steps", "batch_size", "device", "wall_s",
    "loss_start", "loss_end", "toy_token_accuracy",
}
CONFIG_KEYS = {
    "hidden_dim", "enc_layers", "dec_layers",
    "num_heads", "input_features", "max_tgt",
}
CHECKPOINT_KEYS = {"model_state", "vocab", "config", "source"}
POSE_DIM = 1629


def load_smoke_module():
    """Import scripts/smoke_train.py as a module (for in-process determinism)."""
    spec = importlib.util.spec_from_file_location("smoke_train_under_test", SMOKE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# CLI + artifact contract
# ---------------------------------------------------------------------------

class TestSmokeRun:
    def test_script_exists(self):
        assert SMOKE_SCRIPT.is_file(), f"missing {SMOKE_SCRIPT}"

    def test_subprocess_exits_zero(self, smoke_run):
        proc = smoke_run.proc
        assert proc.returncode == 0, (
            f"smoke_train.py failed (exit {proc.returncode})\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )

    def test_prints_progress_table(self, smoke_run):
        stdout = smoke_run.proc.stdout
        assert "loss" in stdout.lower()
        assert "loss_start" in stdout  # final summary line

    def test_metrics_json_contract(self, smoke_run):
        assert smoke_run.metrics_path.is_file(), "metrics.json was not written"
        metrics = json.loads(smoke_run.metrics_path.read_text(encoding="utf-8"))

        assert set(metrics) == METRIC_KEYS, (
            f"metrics.json keys must be exactly {sorted(METRIC_KEYS)}, "
            f"got {sorted(metrics)}"
        )
        assert metrics["seed"] == 42
        assert metrics["steps"] == 4
        assert metrics["batch_size"] == 8
        assert metrics["device"] == "cpu"
        assert isinstance(metrics["wall_s"], (int, float)) and metrics["wall_s"] >= 0
        assert isinstance(metrics["loss_start"], float)
        assert isinstance(metrics["loss_end"], float)
        assert isinstance(metrics["toy_token_accuracy"], float)

        # The synthetic task is learnable, so the loss must actually go down.
        assert metrics["loss_end"] < metrics["loss_start"], (
            f"no learning: loss_start={metrics['loss_start']}, "
            f"loss_end={metrics['loss_end']}"
        )
        assert 0.0 <= metrics["toy_token_accuracy"] <= 1.0

    def test_checkpoint_contract(self, smoke_run):
        assert smoke_run.checkpoint_path.is_file(), "model.pt was not written"
        # torch.load default (weights_only) must accept it.
        ckpt = torch.load(smoke_run.checkpoint_path, map_location="cpu")

        assert set(ckpt) == CHECKPOINT_KEYS
        assert ckpt["source"] == "smoke_train"
        assert set(ckpt["config"]) == CONFIG_KEYS
        assert ckpt["config"]["input_features"] == POSE_DIM
        for key in ("hidden_dim", "enc_layers", "dec_layers", "num_heads", "max_tgt"):
            assert isinstance(ckpt["config"][key], int), f"config[{key!r}] must be int"

        assert isinstance(ckpt["vocab"], dict) and ckpt["vocab"]
        assert all(isinstance(word, str) for word in ckpt["vocab"])
        assert all(isinstance(index, int) for index in ckpt["vocab"].values())
        assert len(ckpt["vocab"]) == 64  # default --vocab-size
        assert sorted(ckpt["vocab"].values()) == list(range(len(ckpt["vocab"])))

        assert isinstance(ckpt["model_state"], dict) and ckpt["model_state"]

    def test_checkpoint_rebuilds_real_model_and_decodes(self, smoke_run):
        """Mirror what scripts/infer_landmarks.py does with the checkpoint."""
        from train_seq2seq_diagnostic import ASLSeq2Seq

        ckpt = torch.load(smoke_run.checkpoint_path, map_location="cpu")
        config = ckpt["config"]
        model = ASLSeq2Seq(
            vocab_size=len(ckpt["vocab"]),
            input_dim=config["input_features"],
            hidden_dim=config["hidden_dim"],
            enc_layers=config["enc_layers"],
            dec_layers=config["dec_layers"],
            num_heads=config["num_heads"],
            dropout=0.0,
            max_len=512,  # matches scripts/infer_landmarks.py
        )
        model.load_state_dict(ckpt["model_state"], strict=True)  # validates schema
        model.eval()

        src = torch.randn(1, 24, POSE_DIM)
        with torch.no_grad():
            tokens = model.greedy_decode(src, max_len=8)
        assert tokens.shape[0] == 1
        assert 1 <= tokens.shape[1] <= 8
        assert tokens.dtype == torch.long


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    @pytest.mark.parametrize("seed", [3, 1234])
    def test_same_seed_same_metrics(self, tmp_path, seed):
        smoke = load_smoke_module()
        out_a = tmp_path / f"a_{seed}"
        out_b = tmp_path / f"b_{seed}"

        assert smoke.main(["--steps", "3", "--seed", str(seed), "--out", str(out_a)]) == 0
        assert smoke.main(["--steps", "3", "--seed", str(seed), "--out", str(out_b)]) == 0

        metrics_a = json.loads((out_a / "metrics.json").read_text())
        metrics_b = json.loads((out_b / "metrics.json").read_text())

        for key in ("seed", "steps", "batch_size", "device", "loss_start",
                    "loss_end", "toy_token_accuracy"):
            assert metrics_a[key] == metrics_b[key], f"non-deterministic {key!r}"

        state_a = torch.load(out_a / "model.pt", map_location="cpu")["model_state"]
        state_b = torch.load(out_b / "model.pt", map_location="cpu")["model_state"]
        assert state_a.keys() == state_b.keys()
        for key in state_a:
            assert torch.equal(state_a[key], state_b[key]), f"weights differ at {key!r}"

    def test_different_seed_changes_run(self, tmp_path):
        smoke = load_smoke_module()
        out_a = tmp_path / "seed_a"
        out_b = tmp_path / "seed_b"
        assert smoke.main(["--steps", "2", "--seed", "1", "--out", str(out_a)]) == 0
        assert smoke.main(["--steps", "2", "--seed", "2", "--out", str(out_b)]) == 0

        metrics_a = json.loads((out_a / "metrics.json").read_text())
        metrics_b = json.loads((out_b / "metrics.json").read_text())
        assert (metrics_a["loss_start"] != metrics_b["loss_start"]
                or metrics_a["toy_token_accuracy"] != metrics_b["toy_token_accuracy"])


# ---------------------------------------------------------------------------
# CLI argument validation
# ---------------------------------------------------------------------------

class TestCli:
    def test_custom_out_dir_and_seed(self, tmp_path):
        smoke = load_smoke_module()
        out = tmp_path / "custom"
        rc = smoke.main(["--steps", "2", "--batch-size", "4", "--seed", "7",
                         "--out", str(out), "--hidden-dim", "32",
                         "--vocab-size", "32"])
        assert rc == 0
        metrics = json.loads((out / "metrics.json").read_text())
        assert metrics["seed"] == 7
        assert metrics["steps"] == 2
        assert metrics["batch_size"] == 4

        ckpt = torch.load(out / "model.pt", map_location="cpu")
        assert ckpt["config"]["hidden_dim"] == 32
        assert len(ckpt["vocab"]) == 32

    def test_invalid_steps_raises(self, tmp_path):
        smoke = load_smoke_module()
        with pytest.raises(SystemExit):
            smoke.main(["--steps", "0", "--out", str(tmp_path / "bad")])

    def test_help_lists_documented_flags(self, capsys):
        smoke = load_smoke_module()
        with pytest.raises(SystemExit) as exc_info:
            smoke.parse_args(["--help"])
        assert exc_info.value.code == 0
        help_text = capsys.readouterr().out
        for flag in ("--steps", "--batch-size", "--out", "--seed", "--device",
                     "--hidden-dim", "--vocab-size"):
            assert flag in help_text, f"missing CLI flag {flag}"
        assert "40" in help_text  # documented --steps default

    def test_cpu_friendly_defaults(self):
        """The documented contract defaults must all be CPU-friendly."""
        smoke = load_smoke_module()
        args = smoke.parse_args([])
        assert args.steps == 40
        assert args.batch_size == 8
        assert args.out == "runs/smoke"
        assert args.seed == 42
        assert args.device == "cpu"
        assert args.hidden_dim == 64
        assert args.vocab_size == 64
