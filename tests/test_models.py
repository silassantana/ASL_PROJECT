"""Shape/forward/backward tests for sign_transformer.py (CPU, tiny models)."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from sign_transformer import (
    ChannelAttention,
    CrossModalFusion,
    PositionalEncoding,
    SignLanguageTransformer,
    prepare_targets,
)

NUM_CLASSES = 20
BATCH = 3


def make_tiny_transformer(**overrides) -> SignLanguageTransformer:
    """Small-but-real model: same code paths, cheap on CPU."""
    kwargs = dict(
        num_classes=NUM_CLASSES,
        input_features=64,
        hidden_dim=32,
        num_encoder_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout=0.0,
        max_glosses=8,
    )
    kwargs.update(overrides)
    return SignLanguageTransformer(**kwargs)


# ---------------------------------------------------------------------------
# PositionalEncoding
# ---------------------------------------------------------------------------

class TestPositionalEncoding:
    def test_shape_and_sinusoid_values(self):
        pe = PositionalEncoding(d_model=16, max_len=32)
        x = torch.zeros(2, 10, 16)
        out = pe(x)

        assert out.shape == x.shape
        assert torch.allclose(out, pe.pe[:10].unsqueeze(0).expand(2, -1, -1), atol=1e-6)
        # Classic sinusoid: position 0 -> sin=0 on even dims, cos=1 on odd dims.
        assert torch.allclose(pe.pe[0, 0::2], torch.zeros(8), atol=1e-6)
        assert torch.allclose(pe.pe[0, 1::2], torch.ones(8), atol=1e-6)
        # Positions add rather than overwrite the input.
        out_x = pe(torch.ones(1, 4, 16))
        assert torch.allclose(out_x, 1.0 + pe.pe[:4].unsqueeze(0), atol=1e-6)

    def test_expands_beyond_max_len(self):
        pe = PositionalEncoding(d_model=16, max_len=8)
        assert pe.pe.shape[0] == 8

        out = pe(torch.zeros(2, 20, 16))

        assert out.shape == (2, 20, 16)
        assert pe.pe.shape[0] >= 20, "buffer must grow to cover seq_len > max_len"
        # The expanded rows must equal a freshly computed encoding.
        assert torch.allclose(pe.pe[:20], pe._compute_pe(20, 16), atol=1e-6)
        assert torch.isfinite(out).all()

    def test_gradient_flows_through(self):
        pe = PositionalEncoding(d_model=16, max_len=8)
        x = torch.randn(1, 12, 16, requires_grad=True)
        pe(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


# ---------------------------------------------------------------------------
# CTC / decoder targets
# ---------------------------------------------------------------------------

class TestPrepareTargets:
    def test_sos_eos_shift_and_padding(self):
        labels = torch.tensor([[5, 6, 7, 8], [1, 2, 0, 0]])
        lengths = torch.tensor([4, 2])

        dec_in, dec_tgt, tgt_lens = prepare_targets(labels, lengths)

        # max_len + 1 columns; SOS column first.
        assert dec_in.shape == (2, 5)
        assert dec_tgt.shape == (2, 5)
        assert dec_in.dtype == torch.long and dec_tgt.dtype == torch.long
        assert dec_in[:, 0].tolist() == [1, 1]

        # Raw glosses are shifted by +3 (PAD=0, SOS=1, EOS=2 reserved).
        assert dec_in[0, 1:].tolist() == [8, 9, 10, 11]
        assert dec_in[1, 1:3].tolist() == [4, 5]
        assert dec_in[1, 3:].tolist() == [0, 0]

        assert dec_tgt[0].tolist() == [8, 9, 10, 11, 2]
        assert dec_tgt[1].tolist() == [4, 5, 2, 0, 0]
        assert tgt_lens.tolist() == [5, 3]

    def test_matches_teacher_forcing_shift(self):
        labels = torch.randint(0, NUM_CLASSES, (4, 6))
        lengths = torch.tensor([6, 4, 1, 3])
        dec_in, dec_tgt, tgt_lens = prepare_targets(labels, lengths)

        assert dec_in.size(1) == dec_tgt.size(1) == 7
        for i, length in enumerate(lengths.tolist()):
            assert dec_in[i, 0] == 1
            assert dec_in[i, 1:length + 1].tolist() == (labels[i, :length] + 3).tolist()
            assert dec_tgt[i, :length].tolist() == (labels[i, :length] + 3).tolist()
            assert dec_tgt[i, length] == 2
            assert (dec_tgt[i, length + 1:] == 0).all()
        assert tgt_lens.tolist() == [7, 5, 2, 4]


# ---------------------------------------------------------------------------
# SignLanguageTransformer forward / CTC head / backward
# ---------------------------------------------------------------------------

class TestSignLanguageTransformer:
    def test_forward_shape(self):
        model = make_tiny_transformer()
        feats = torch.randn(BATCH, 16, 64)
        labels = torch.randint(0, NUM_CLASSES, (BATCH, 5))
        lengths = torch.tensor([5, 4, 3])
        dec_in, dec_tgt, _ = prepare_targets(labels, lengths)

        logits = model(feats, dec_in)

        assert logits.shape == (BATCH, 6, NUM_CLASSES + 3)
        assert torch.isfinite(logits).all()

    def test_forward_with_padding_mask(self):
        model = make_tiny_transformer()
        feats = torch.randn(BATCH, 16, 64)
        src_pad = torch.zeros(BATCH, 16, dtype=torch.bool)
        src_pad[:, 12:] = True
        dec_in = torch.randint(0, NUM_CLASSES + 3, (BATCH, 4))

        logits = model(feats, dec_in, src_key_padding_mask=src_pad)

        assert logits.shape == (BATCH, 4, NUM_CLASSES + 3)
        assert torch.isfinite(logits).all()

    def test_cross_entropy_loss_backward(self):
        model = make_tiny_transformer()
        model.train()
        feats = torch.randn(BATCH, 12, 64)
        labels = torch.randint(0, NUM_CLASSES, (BATCH, 4))
        dec_in, dec_tgt, _ = prepare_targets(labels, torch.tensor([4, 3, 2]))

        logits = model(feats, dec_in)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                               dec_tgt.reshape(-1), ignore_index=0)
        loss.backward()

        assert torch.isfinite(loss)
        assert loss.item() > 0
        grads = [p.grad for p in model.parameters() if p.requires_grad]
        assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
                   for g in grads), "no gradients reached the model parameters"

    def test_ctc_log_probs_consistency(self):
        model = make_tiny_transformer()
        feats = torch.randn(BATCH, 14, 64)

        memory = model.encode(feats)
        log_probs = model.ctc_logits(memory)

        # CTCLoss wants [T, B, C] time-major with C = num_classes + 1 (blank=0).
        assert log_probs.shape == (14, BATCH, NUM_CLASSES + 1)
        assert torch.allclose(
            torch.logsumexp(log_probs, dim=-1), torch.zeros(14, BATCH), atol=1e-5
        )
        expected = torch.log_softmax(model.ctc_head(memory), dim=-1).transpose(0, 1)
        assert torch.allclose(log_probs, expected, atol=1e-6)

        targets = torch.randint(1, NUM_CLASSES + 1, (BATCH, 4))
        ctc_loss = nn.CTCLoss(blank=0, zero_infinity=True)(
            log_probs, targets,
            torch.full((BATCH,), 14), torch.full((BATCH,), 4),
        )
        assert torch.isfinite(ctc_loss)
        ctc_loss.backward()
        assert model.ctc_head.weight.grad is not None
        assert torch.isfinite(model.ctc_head.weight.grad).all()

    def test_generate_runs_and_respects_max_glosses(self):
        model = make_tiny_transformer(max_glosses=6)
        feats = torch.randn(2, 10, 64)
        with torch.no_grad():
            preds = model(feats)  # gloss_targets=None -> autoregressive generate
        assert preds.shape[0] == 2
        assert 1 <= preds.shape[1] <= 6
        assert preds.dtype == torch.long

    def test_ctc_head_uses_separate_index_space(self):
        model = make_tiny_transformer()
        assert model.ctc_head.out_features == NUM_CLASSES + 1  # blank + glosses
        assert model.gloss_embedding.num_embeddings == NUM_CLASSES + 3  # +PAD/SOS/EOS
        assert model.output_proj[-1].out_features == NUM_CLASSES + 3


# ---------------------------------------------------------------------------
# ChannelAttention / CrossModalFusion
# ---------------------------------------------------------------------------

class TestChannelAttention:
    def test_shape_and_gate_bounds(self):
        attn = ChannelAttention(channels=64, reduction=8)
        x = torch.randn(2, 9, 64)

        out = attn(x)

        assert out.shape == x.shape
        assert torch.isfinite(out).all()
        # The gate is a sigmoid in (0, 1), so the output cannot exceed |x|.
        assert (out.abs() <= x.abs() + 1e-6).all()

    def test_changes_with_input(self):
        attn = ChannelAttention(channels=32)
        x = torch.randn(1, 5, 32)
        assert not torch.allclose(attn(x), attn(x * 0.5))


class TestCrossModalFusion:
    # NOTE: the real model feeds both streams from the same [B, T, f] tensor, so
    # clip_x and keypoint_x always share T.  CrossModalFusion.forward concatenates
    # them for the fusion gate, which is only defined for matching T (the repo's
    # own usage); these tests mirror that usage.
    def test_shape(self):
        fusion = CrossModalFusion(hidden_dim=32, num_heads=4, dropout=0.0)
        clip_x = torch.randn(2, 8, 32)
        keypoint_x = torch.randn(2, 8, 32)

        fused = fusion(clip_x, keypoint_x)

        assert fused.shape == clip_x.shape
        assert torch.isfinite(fused).all()

    def test_shape_with_key_padding_mask(self):
        fusion = CrossModalFusion(hidden_dim=32, num_heads=4, dropout=0.0)
        clip_x = torch.randn(2, 8, 32)
        keypoint_x = torch.randn(2, 8, 32)
        mask = torch.zeros(2, 8, dtype=torch.bool)
        mask[0, 6:] = True  # last keypoint frames are padding

        fused = fusion(clip_x, keypoint_x, key_padding_mask=mask)

        assert fused.shape == (2, 8, 32)
        assert torch.isfinite(fused).all()

    def test_multimodal_transformer_branch(self):
        model = SignLanguageTransformer(
            num_classes=10,
            input_features=48,       # keypoint_dim + clip_dim
            hidden_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dropout=0.0,
            use_multimodal_fusion=True,
            keypoint_dim=32,
            clip_dim=16,
        )
        assert model.use_multimodal_fusion is True
        assert model.channel_attention is None
        assert model.multimodal_fusion is not None

        feats = torch.randn(2, 9, 48)
        dec_in, _, _ = prepare_targets(
            torch.randint(0, 10, (2, 4)), torch.tensor([4, 3])
        )
        logits = model(feats, dec_in)

        assert logits.shape == (2, 5, 13)
        assert torch.isfinite(logits).all()

    def test_channel_attention_in_multimodal_branch(self):
        model = SignLanguageTransformer(
            num_classes=10,
            input_features=48,
            hidden_dim=32,
            num_encoder_layers=1,
            num_decoder_layers=1,
            num_heads=4,
            dropout=0.0,
            use_channel_attention=True,
            use_multimodal_fusion=True,
            keypoint_dim=32,
            clip_dim=16,
        )
        assert model.keypoint_channel_attention is not None
        assert model.clip_channel_attention is not None
        logits = model(torch.randn(2, 7, 48), torch.randint(0, 13, (2, 3)))
        assert logits.shape == (2, 3, 13)

    def test_short_input_raises_for_multimodal(self):
        model = SignLanguageTransformer(
            num_classes=5, input_features=48, hidden_dim=32,
            num_encoder_layers=1, num_decoder_layers=1, num_heads=4,
            dropout=0.0, use_multimodal_fusion=True, keypoint_dim=32, clip_dim=16,
        )
        with pytest.raises(ValueError, match="input features"):
            model.encode(torch.randn(1, 4, 47))
