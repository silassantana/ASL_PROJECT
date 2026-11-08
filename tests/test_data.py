"""Dataset / collation / masking tests against synthetic H5s with the real schemas.

These tests are skipped (not failed) when h5py is unavailable, because both
dataset modules import h5py at module import time.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

h5py = pytest.importorskip("h5py")

from train_seq2seq_diagnostic import (  # noqa: E402
    PAD,
    UNK,
    SegmentDataset,
    build_vocab,
    collate_fn,
    encode,
    tokenize,
)
from train_temporal_encoder import (  # noqa: E402
    BucketBatchSampler,
    PoseH5Dataset,
    collate_poses,
    make_span_mask,
)

POSE_DIM = 1629


def close(dataset) -> None:
    """Close a lazily-opened H5 handle if the dataset opened one."""
    handle = getattr(dataset, "_h5", None)
    if handle is not None:
        handle.close()
        dataset._h5 = None


# ---------------------------------------------------------------------------
# SegmentDataset (seq2seq diagnostic, /train/clips + /train/texts + /stats)
# ---------------------------------------------------------------------------

class TestSegmentDataset:
    def test_len_item_shapes_and_text(self, toy_segment_h5):
        data = toy_segment_h5(n_train=5, n_val=3, seed=1)
        ds = SegmentDataset(str(data["path"]), "train", data["vocab"])

        try:
            assert len(ds) == 5
            clip, tokens, text = ds[2]

            assert isinstance(clip, torch.Tensor)
            assert clip.dtype == torch.float32
            assert clip.ndim == 2 and clip.shape[1] == POSE_DIM
            assert 16 <= clip.shape[0] <= data["max_frames"]

            assert text == data["texts"]["train"][2]
            assert tokens.dtype == torch.long
            assert tokens.tolist() == encode(text, data["vocab"])[:64]
        finally:
            close(ds)

    def test_max_frames_and_max_tokens_truncation(self, toy_segment_h5):
        data = toy_segment_h5(n_train=4, n_val=2, seed=2)
        ds = SegmentDataset(
            str(data["path"]), "train", data["vocab"],
            max_src_frames=10, max_tgt_tokens=2,
        )
        try:
            clip, tokens, _text = ds[0]
            with h5py.File(data["path"], "r") as f:
                full_frames = f["train/clips/0"].shape[0]
            assert clip.shape[0] == min(10, full_frames)
            assert len(tokens) <= 2
        finally:
            close(ds)

    def test_mean_std_normalization_clips(self, toy_segment_h5):
        data = toy_segment_h5(n_train=4, n_val=2, seed=3)
        ds = SegmentDataset(
            str(data["path"]), "train", data["vocab"],
            max_src_frames=10, mean=data["mean"], std=data["std"],
        )
        try:
            clip, _tokens, _text = ds[1]
            with h5py.File(data["path"], "r") as f:
                raw = f["train/clips/1"][:10].astype(np.float32)
            expected = np.clip((raw - data["mean"]) / data["std"], -10.0, 10.0)
            assert np.allclose(clip.numpy(), expected, atol=1e-5)
            assert np.isfinite(clip.numpy()).all()
        finally:
            close(ds)

    def test_loader_batches_are_well_formed(self, toy_segment_h5):
        data = toy_segment_h5(n_train=5, n_val=2, seed=4)
        ds = SegmentDataset(str(data["path"]), "train", data["vocab"])
        loader = DataLoader(ds, batch_size=3, collate_fn=collate_fn, num_workers=0)
        try:
            batches = list(loader)
        finally:
            close(ds)

        assert len(batches) == 2  # 5 samples / batch 3
        seen_texts: list[str] = []
        for src, tgt, src_lens, tgt_lens, src_pad, texts in batches:
            batch_size = len(texts)
            assert src.shape == (batch_size, int(src_lens.max().item()), POSE_DIM)
            assert src.dtype == torch.float32
            assert src_pad.shape == src.shape[:2]
            assert src_pad.dtype == torch.bool
            assert tgt.shape == (batch_size, int(tgt_lens.max().item()))
            assert tgt.dtype == torch.long
            assert torch.isfinite(src).all()

            for i in range(batch_size):
                src_len = int(src_lens[i].item())
                assert not src_pad[i, :src_len].any(), "real frames must not be masked"
                assert src_pad[i, src_len:].all(), "padded frames must be masked"
                assert (src[i, src_len:] == 0).all(), "padded frames must be zero"
                tgt_len = int(tgt_lens[i].item())
                assert (tgt[i, tgt_len:] == PAD).all()
            seen_texts.extend(texts)

        assert sorted(seen_texts) == sorted(data["texts"]["train"])

    def test_batch_feeds_tiny_seq2seq(self, toy_segment_h5, tiny_seq2seq):
        data = toy_segment_h5(n_train=4, n_val=2, seed=5)
        ds = SegmentDataset(str(data["path"]), "train", data["vocab"])
        try:
            loader = DataLoader(ds, batch_size=2, collate_fn=collate_fn)
            src, tgt, _src_lens, _tgt_lens, src_pad, _texts = next(iter(loader))
        finally:
            close(ds)

        with torch.no_grad():
            logits = tiny_seq2seq(src, tgt[:, :-1], src_key_padding_mask=src_pad)
        assert logits.shape == (src.size(0), tgt.size(1) - 1, tiny_seq2seq.vocab_size)


# ---------------------------------------------------------------------------
# Vocabulary helpers
# ---------------------------------------------------------------------------

class TestVocabularyHelpers:
    def test_tokenize_normalises_punctuation(self):
        assert tokenize("Hello, WORLD! 2x") == ["hello", "world", "2x"]
        assert tokenize("") == []

    def test_build_vocab_specials_and_frequency_order(self):
        vocab = build_vocab(["hello world", "hello sign"], max_vocab=100)
        assert list(vocab)[:4] == ["<pad>", "<sos>", "<eos>", "<unk>"]
        assert vocab["hello"] == 4  # most frequent word -> first free id
        assert set(["world", "sign"]).issubset(vocab)

    def test_build_vocab_respects_max_vocab(self):
        texts = [" ".join(f"w{i}" for i in range(50))]
        vocab = build_vocab(texts, max_vocab=10)
        assert len(vocab) == 10

    def test_encode_unknown_maps_to_unk(self):
        vocab = build_vocab(["hello world"], max_vocab=100)
        assert encode("hello zzz", vocab) == [vocab["hello"], UNK]


# ---------------------------------------------------------------------------
# PoseH5Dataset (/poses/{vid} + /captions + /index)
# ---------------------------------------------------------------------------

class TestPoseH5Dataset:
    def test_indexing_and_lengths(self, toy_poses_h5):
        data = toy_poses_h5(video_lengths=(24, 40, 16), seed=6)
        ds = PoseH5Dataset(str(data["path"]), data["video_ids"], max_frames=100)

        try:
            assert len(ds) == 3
            assert ds.video_ids == data["video_ids"]

            poses, video_id = ds[1]
            assert video_id == data["video_ids"][1]
            assert poses.dtype == torch.float32
            assert poses.shape == (40, POSE_DIM)
        finally:
            close(ds)

    def test_h5_schema_matches_builder(self, toy_poses_h5):
        data = toy_poses_h5(video_lengths=(20, 30), seed=7)
        with h5py.File(data["path"], "r") as f:
            stored_ids = [v.decode("utf-8") for v in f["index"][:]]
            assert stored_ids == data["video_ids"]
            for video_id, n_frames in zip(data["video_ids"], data["lengths"]):
                assert f["poses"][video_id].shape == (n_frames, POSE_DIM)
                caption = f["captions"][video_id][()].decode("utf-8")
                assert video_id in caption

    def test_max_frames_truncation(self, toy_poses_h5):
        data = toy_poses_h5(video_lengths=(24, 40), seed=8)
        ds = PoseH5Dataset(str(data["path"]), data["video_ids"], max_frames=10)
        try:
            poses, _video_id = ds[0]
            assert poses.shape == (10, POSE_DIM)
        finally:
            close(ds)

    def test_mean_std_normalization_clips(self, toy_poses_h5):
        data = toy_poses_h5(video_lengths=(24,), seed=9)
        mean = np.full(POSE_DIM, 0.5, dtype=np.float32)
        std = np.full(POSE_DIM, 2.0, dtype=np.float32)
        ds = PoseH5Dataset(str(data["path"]), data["video_ids"],
                           max_frames=100, mean=mean, std=std)
        try:
            poses, _video_id = ds[0]
            with h5py.File(data["path"], "r") as f:
                raw = f["poses"][data["video_ids"][0]][:].astype(np.float32)
            expected = np.clip((raw - mean) / std, -10.0, 10.0)
            assert np.allclose(poses.numpy(), expected, atol=1e-5)
        finally:
            close(ds)

    def test_loader_with_collate_poses_is_well_formed(self, toy_poses_h5):
        data = toy_poses_h5(video_lengths=(24, 40, 16), seed=10)
        ds = PoseH5Dataset(str(data["path"]), data["video_ids"], max_frames=100)
        loader = DataLoader(ds, batch_size=2, collate_fn=collate_poses)
        try:
            batches = list(loader)
        finally:
            close(ds)

        assert len(batches) == 2  # 3 samples / batch 2
        seen_videos: list[str] = []
        for poses, lengths, pad_mask, video_ids in batches:
            batch_size = poses.size(0)
            assert poses.shape == (batch_size, int(lengths.max().item()), POSE_DIM)
            assert pad_mask.shape == poses.shape[:2]
            assert pad_mask.dtype == torch.bool
            for i in range(batch_size):
                seq_len = int(lengths[i].item())
                assert not pad_mask[i, :seq_len].any()
                assert pad_mask[i, seq_len:].all()
                assert (poses[i, seq_len:] == 0).all()
            seen_videos.extend(video_ids)
        assert sorted(seen_videos) == sorted(data["video_ids"])


# ---------------------------------------------------------------------------
# Collation / masking / bucketing
# ---------------------------------------------------------------------------

class TestCollatePoses:
    def test_pads_to_batch_max_and_tracks_lengths(self):
        seqs = [torch.randn(20, POSE_DIM), torch.randn(35, POSE_DIM), torch.randn(17, POSE_DIM)]
        padded, lengths, pad_mask, vids = collate_poses(
            [(seq, f"v{i}") for i, seq in enumerate(seqs)]
        )

        assert padded.shape == (3, 35, POSE_DIM)
        assert lengths.tolist() == [20, 35, 17]
        assert vids == ["v0", "v1", "v2"]
        assert pad_mask.shape == (3, 35)
        assert pad_mask.dtype == torch.bool

        assert not pad_mask[0, :20].any()
        assert pad_mask[0, 20:].all()
        assert torch.equal(padded[0, :20], seqs[0])
        assert (padded[0, 20:] == 0).all()
        assert torch.equal(padded[1], seqs[1])
        assert torch.equal(padded[2, :17], seqs[2])


class TestMakeSpanMask:
    def test_only_real_frames_masked(self):
        np.random.seed(0)
        torch.manual_seed(0)
        lengths = torch.tensor([20, 16, 30])

        mask = make_span_mask(lengths, mask_ratio=0.2, span_len_min=3, span_len_max=6)

        assert mask.shape == (3, 30)
        assert mask.dtype == torch.bool
        assert mask.any()
        for i, seq_len in enumerate(lengths.tolist()):
            assert not mask[i, seq_len:].any(), "padding must never be masked"
            ratio = mask[i, :seq_len].float().mean().item()
            assert 0.05 <= ratio <= 0.6, f"mask ratio {ratio:.2f} out of bounds"

    def test_ratio_scales_with_length(self):
        np.random.seed(1)
        lengths = torch.tensor([100])
        low = make_span_mask(lengths, mask_ratio=0.1).sum().item()
        np.random.seed(1)
        high = make_span_mask(lengths, mask_ratio=0.4).sum().item()
        assert low < high


class TestBucketBatchSampler:
    def test_covers_every_index_once(self):
        lengths = [10, 12, 11, 40, 42, 41, 5, 6, 7, 30]
        sampler = BucketBatchSampler(lengths, batch_size=3, shuffle=True)
        batches = list(iter(sampler))

        assert len(batches) == math.ceil(len(lengths) / 3)
        assert all(1 <= len(batch) <= 3 for batch in batches)
        seen = sorted(index for batch in batches for index in batch)
        assert seen == list(range(len(lengths))), "sampler must not drop/duplicate samples"
