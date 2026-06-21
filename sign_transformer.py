#!/usr/bin/env python3
"""
Encoder-Decoder Transformer for Continuous Sign Recognition.

Based on:
- "Sign Language Transformers" (CVPR 2023)
- "STMC: Spatial-Temporal Multi-Cue Encoder for Continuous SLR" (CVPR 2024)

This REPLACES CTC with an autoregressive decoder that predicts glosses one-by-one.
Much more stable than CTC for continuous sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding with dynamic expansion."""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = self._compute_pe(max_len, d_model)
        self.register_buffer('pe', pe)
    
    def _compute_pe(self, length, d_model):
        """Compute positional encoding for given length."""
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe
    
    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        
        # Expand positional encoding if needed
        if seq_len > self.pe.size(0):
            new_max_len = max(seq_len, self.pe.size(0) * 2)
            new_pe = self._compute_pe(new_max_len, self.d_model).to(self.pe.device)
            self.register_buffer('pe', new_pe)
        
        return x + self.pe[:seq_len, :].unsqueeze(0)


class ChannelAttention(nn.Module):
    """Lightweight channel attention over feature dimensions."""
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x):
        # x: [batch, seq_len, channels]
        pooled = x.mean(dim=1)
        weights = torch.sigmoid(self.fc2(F.relu(self.fc1(pooled))))
        return x * weights.unsqueeze(1)


class CrossModalFusion(nn.Module):
    """Fuse CLIP and keypoint streams with cross-attention."""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm = nn.LayerNorm(hidden_dim)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.fusion_norm = nn.LayerNorm(hidden_dim)

    def forward(self, clip_x, keypoint_x, key_padding_mask=None):
        """Return CLIP features enriched with keypoint context."""
        cross_out, _ = self.cross_attn(
            query=clip_x,
            key=keypoint_x,
            value=keypoint_x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        clip_enriched = self.cross_norm(clip_x + cross_out)
        gate = self.fusion_gate(torch.cat([clip_enriched, keypoint_x], dim=-1))
        fused = gate * clip_enriched + (1.0 - gate) * keypoint_x
        return self.fusion_norm(fused)


class SignLanguageTransformer(nn.Module):
    """
    Encoder-Decoder Transformer for continuous sign language recognition.
    
    Architecture:
    1. Video features → Encoder (temporal modeling)
    2. Encoder output → Decoder (gloss sequence generation)
    3. Autoregressive: predicts glosses one at a time
    """
    
    def __init__(self, num_classes, input_features=512, hidden_dim=256,
                 num_encoder_layers=4, num_decoder_layers=4, num_heads=8,
                 dropout=0.1, max_glosses=100, use_channel_attention=False,
                 attention_reduction=8, use_multimodal_fusion=None,
                 keypoint_dim=1629, clip_dim=512):
        super().__init__()
        
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.max_glosses = max_glosses
        self.use_channel_attention = use_channel_attention
        self.keypoint_dim = int(keypoint_dim)
        self.clip_dim = int(clip_dim)
        self.use_multimodal_fusion = (
            bool(use_multimodal_fusion)
            if use_multimodal_fusion is not None
            else input_features == (self.keypoint_dim + self.clip_dim)
        )
        
        # Special tokens
        self.pad_idx = 0
        self.sos_idx = 1  # Start of sequence
        self.eos_idx = 2  # End of sequence
        # Actual glosses start at index 3
        
        # ===== ENCODER =====
        # Optional channel attention for single-stream features or per modality.
        self.channel_attention = None
        self.keypoint_channel_attention = None
        self.clip_channel_attention = None
        if self.use_channel_attention:
            if self.use_multimodal_fusion:
                self.keypoint_channel_attention = ChannelAttention(
                    self.keypoint_dim,
                    reduction=attention_reduction,
                )
                self.clip_channel_attention = ChannelAttention(
                    self.clip_dim,
                    reduction=attention_reduction,
                )
            else:
                self.channel_attention = ChannelAttention(
                    input_features,
                    reduction=attention_reduction
                )

        if self.use_multimodal_fusion:
            self.keypoint_proj = nn.Sequential(
                nn.Linear(self.keypoint_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.clip_proj = nn.Sequential(
                nn.Linear(self.clip_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.multimodal_fusion = CrossModalFusion(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            self.fusion_input_norm = nn.LayerNorm(hidden_dim)
        else:
            # Projects video features to hidden dimension
            self.input_proj = nn.Sequential(
                nn.Linear(input_features, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim)
            )
        
        self.pos_encoder = PositionalEncoding(hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LN for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        
        # ===== DECODER =====
        # Embeds gloss indices
        self.gloss_embedding = nn.Embedding(num_classes + 3, hidden_dim)  # +3 for PAD, SOS, EOS
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        
        # ===== OUTPUT =====
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes + 3)
        )

        # ===== CTC HEAD (auxiliary, operates directly on encoder memory) =====
        # Separate index space from the decoder vocab: CTC index 0 = blank,
        # gloss g (0-indexed) -> CTC index g+1. Size = num_classes + 1.
        self.ctc_head = nn.Linear(hidden_dim, num_classes + 1)

        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize parameters with Xavier."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def encode(self, video_features, src_key_padding_mask=None):
        """
        Encode video features.
        
        Args:
            video_features: [batch, time, features]
            src_key_padding_mask: [batch, time] True where padded
        
        Returns:
            memory: [batch, time, hidden_dim]
        """
        if self.use_multimodal_fusion:
            if video_features.size(-1) < (self.keypoint_dim + self.clip_dim):
                raise ValueError(
                    f"Expected at least {self.keypoint_dim + self.clip_dim} input features "
                    f"for multimodal fusion, got {video_features.size(-1)}"
                )
            keypoint_features = video_features[..., :self.keypoint_dim]
            clip_features = video_features[..., self.keypoint_dim:self.keypoint_dim + self.clip_dim]

            if self.keypoint_channel_attention is not None:
                keypoint_features = self.keypoint_channel_attention(keypoint_features)
            if self.clip_channel_attention is not None:
                clip_features = self.clip_channel_attention(clip_features)

            keypoint_x = self.pos_encoder(self.keypoint_proj(keypoint_features))
            clip_x = self.pos_encoder(self.clip_proj(clip_features))
            x = self.multimodal_fusion(
                clip_x=clip_x,
                keypoint_x=keypoint_x,
                key_padding_mask=src_key_padding_mask,
            )
            x = self.fusion_input_norm(x)
        else:
            if self.channel_attention is not None:
                video_features = self.channel_attention(video_features)
            x = self.input_proj(video_features)
            x = self.pos_encoder(x)

        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return memory
    
    def decode_step(self, memory, tgt_input, tgt_mask=None, memory_key_padding_mask=None):
        """
        Single decoder step.
        
        Args:
            memory: Encoder output [batch, src_len, hidden_dim]
            tgt_input: Target sequence [batch, tgt_len]
            tgt_mask: Causal mask [tgt_len, tgt_len]
            memory_key_padding_mask: [batch, src_len] True where padded
        
        Returns:
            logits: [batch, tgt_len, num_classes + 3]
        """
        # Embed target
        tgt = self.gloss_embedding(tgt_input)
        tgt = self.pos_encoder(tgt)
        
        # Decode
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask,
                              memory_key_padding_mask=memory_key_padding_mask)
        
        # Project to vocabulary
        logits = self.output_proj(output)
        
        return logits
    
    def ctc_logits(self, memory):
        """
        CTC logits from encoder memory.

        Args:
            memory: [batch, src_len, hidden_dim]

        Returns:
            log_probs: [src_len, batch, num_classes + 1] (time-major, as required
                       by nn.CTCLoss), already log_softmax'd.
        """
        logits = self.ctc_head(memory)  # [batch, src_len, num_classes + 1]
        log_probs = torch.log_softmax(logits, dim=-1)
        return log_probs.transpose(0, 1)  # [src_len, batch, num_classes + 1]

    def forward(self, video_features, gloss_targets=None, src_key_padding_mask=None, return_memory=False):
        """
        Forward pass.
        
        Args:
            video_features: [batch, time, features]
            gloss_targets: [batch, max_glosses] - includes SOS at start, EOS at end
            src_key_padding_mask: [batch, time] True where padded
            return_memory: if True, also return the encoder memory (for auxiliary
                            losses such as CTC). Only affects the teacher-forcing
                            (training) branch.
        
        Returns:
            logits: [batch, max_glosses, num_classes + 3]
            memory (optional): [batch, time, hidden_dim], only if return_memory=True
                                and gloss_targets is not None
        """
        batch_size = video_features.size(0)
        device = video_features.device
        
        # Encode video
        memory = self.encode(video_features, src_key_padding_mask=src_key_padding_mask)
        
        if gloss_targets is not None:
            # Training mode: teacher forcing
            tgt_len = gloss_targets.size(1)
            
            # Create causal mask
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
            
            # Decode
            logits = self.decode_step(memory, gloss_targets, tgt_mask,
                                      memory_key_padding_mask=src_key_padding_mask)
            
            if return_memory:
                return logits, memory
            return logits
        else:
            # Inference mode: autoregressive generation
            return self.generate(memory, memory_key_padding_mask=src_key_padding_mask)
    
    def generate(self, memory, max_length=None, memory_key_padding_mask=None,
                 repetition_penalty=1.5, no_repeat_ngram_size=3):
        """
        Autoregressive generation (inference).
        
        Args:
            memory: Encoder output [batch, src_len, hidden_dim]
            max_length: Maximum glosses to generate
            memory_key_padding_mask: [batch, src_len] True where padded
            repetition_penalty: Penalty factor for already-generated tokens (>1 = less repetition)
            no_repeat_ngram_size: Block any n-gram from repeating (0 to disable)
        
        Returns:
            predictions: [batch, generated_length]
        """
        if max_length is None:
            max_length = self.max_glosses
        
        batch_size = memory.size(0)
        device = memory.device
        
        # Start with SOS token
        generated = torch.full((batch_size, 1), self.sos_idx, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length):
            tgt_len = generated.size(1)
            tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(device)
            
            # Predict next token
            logits = self.decode_step(memory, generated, tgt_mask,
                                      memory_key_padding_mask=memory_key_padding_mask)
            next_logits = logits[:, -1, :]  # [batch, vocab]

            # Repetition penalty: reduce logits for tokens already generated
            if repetition_penalty != 1.0:
                for b in range(batch_size):
                    if finished[b]:
                        continue
                    prev_tokens = generated[b].unique()
                    for tok in prev_tokens:
                        if next_logits[b, tok] > 0:
                            next_logits[b, tok] /= repetition_penalty
                        else:
                            next_logits[b, tok] *= repetition_penalty

            # N-gram blocking: prevent any n-gram from appearing twice
            if no_repeat_ngram_size > 0 and generated.size(1) >= no_repeat_ngram_size:
                for b in range(batch_size):
                    if finished[b]:
                        continue
                    gen_list = generated[b].tolist()
                    ngram_prefix = tuple(gen_list[-(no_repeat_ngram_size - 1):])
                    blocked = set()
                    for i in range(len(gen_list) - no_repeat_ngram_size + 1):
                        if tuple(gen_list[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                            blocked.add(gen_list[i + no_repeat_ngram_size - 1])
                    for tok in blocked:
                        next_logits[b, tok] = float('-inf')

            next_token = next_logits.argmax(dim=-1, keepdim=True)  # [batch, 1]
            
            # Force finished sequences to emit PAD
            next_token[finished] = self.pad_idx
            
            # Append to sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Track which sequences just emitted EOS
            finished = finished | (next_token.squeeze(-1) == self.eos_idx)
            if finished.all():
                break
        
        # Remove SOS token
        return generated[:, 1:]
    
    @staticmethod
    def generate_square_subsequent_mask(sz):
        """Generate causal mask for decoder."""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask


def prepare_targets(labels, label_lengths, sos_idx=1, eos_idx=2, pad_idx=0):
    """
    Prepare targets for encoder-decoder training.
    
    Args:
        labels: [batch, max_label_len] - raw gloss labels
        label_lengths: [batch] - actual lengths
        sos_idx: Start of sequence token
        eos_idx: End of sequence token
        pad_idx: Padding token
    
    Returns:
        decoder_input: [batch, max_len + 1] - targets with SOS prepended
        decoder_target: [batch, max_len + 1] - targets with EOS appended
        target_lengths: [batch] - lengths including SOS/EOS
    """
    batch_size = labels.size(0)
    device = labels.device
    max_len = label_lengths.max().item()
    
    # Adjust indices: original labels use 0-indexed glosses
    # We need to shift them by +3 to make room for PAD(0), SOS(1), EOS(2)
    adjusted_labels = labels.clone()
    for i in range(batch_size):
        L = label_lengths[i].item()
        adjusted_labels[i, :L] = labels[i, :L] + 3  # Shift by 3
    
    # Create decoder input (SOS + glosses)
    decoder_input = torch.full((batch_size, max_len + 1), pad_idx, dtype=torch.long, device=device)
    decoder_input[:, 0] = sos_idx
    for i in range(batch_size):
        L = label_lengths[i].item()
        decoder_input[i, 1:L+1] = adjusted_labels[i, :L]
    
    # Create decoder target (glosses + EOS)
    decoder_target = torch.full((batch_size, max_len + 1), pad_idx, dtype=torch.long, device=device)
    for i in range(batch_size):
        L = label_lengths[i].item()
        decoder_target[i, :L] = adjusted_labels[i, :L]
        decoder_target[i, L] = eos_idx
    
    # Target lengths include EOS
    target_lengths = label_lengths + 1
    
    return decoder_input, decoder_target, target_lengths


if __name__ == '__main__':
    # Test the model
    batch_size = 4
    seq_len = 16
    input_features = 512
    num_classes = 100
    max_glosses = 20
    
    model = SignLanguageTransformer(
        num_classes=num_classes,
        input_features=input_features,
        hidden_dim=256,
        num_encoder_layers=4,
        num_decoder_layers=4
    )
    
    # Test input
    video_features = torch.randn(batch_size, seq_len, input_features)
    
    # Test labels (raw 0-indexed glosses)
    labels = torch.randint(0, num_classes, (batch_size, max_glosses))
    label_lengths = torch.randint(5, max_glosses, (batch_size,))
    
    # Prepare targets
    decoder_input, decoder_target, target_lengths = prepare_targets(labels, label_lengths)
    
    # Forward pass
    logits = model(video_features, decoder_input)
    
    print("✓ Model test passed!")
    print(f"Video features: {video_features.shape}")
    print(f"Decoder input: {decoder_input.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Expected: [batch={batch_size}, max_glosses={max_glosses+1}, vocab={num_classes+3}]")
    
    # Test inference
    predictions = model.generate(model.encode(video_features), max_length=10)
    print(f"Inference predictions: {predictions.shape}")
