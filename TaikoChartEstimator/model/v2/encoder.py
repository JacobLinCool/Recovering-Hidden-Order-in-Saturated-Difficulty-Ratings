"""Transformer event-window encoder used by the paper's sequence model."""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape [batch, seq_len, d_model]
        """
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class ContinuousFeatureEncoder(nn.Module):
    """
    Encodes continuous features to d_model dimension.
    Uses learned linear projections with optional normalization.

    V2 Features (5 total):
    - beat_position (within measure, 0-1)
    - duration (for long notes)
    - bpm (normalized)
    - scroll (normalized)
    - local_density (notes/sec, normalized)
    """

    def __init__(
        self,
        n_continuous: int = 5,  # beat_pos, duration, bpm, scroll, local_density
        d_model: int = 256,
        use_layernorm: bool = True,
    ):
        super().__init__()
        self.projection = nn.Linear(n_continuous, d_model)
        self.layernorm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Continuous features [batch, seq_len, n_continuous]
        """
        return self.layernorm(self.projection(x))


class InstanceEncoder(nn.Module):
    """
    Encodes a sequence of event tokens to a fixed-size vector.

    Input: Token sequence [batch, seq_len, 6]
        - Column 0: note_type (discrete, 0-9)
        - Column 1: beat_position (continuous, 0-1)
        - Column 2: duration (continuous, normalized)
        - Column 3: bpm (continuous, normalized)
        - Column 4: scroll (continuous, normalized)
        - Column 5: local_density (continuous, normalized)

    Output: Instance embedding [batch, d_model]
    """

    def __init__(
        self,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_feedforward: int = 512,
        dropout: float = 0.1,
        n_note_types: int = 10,  # 9 types + padding
        max_seq_len: int = 128,
    ):
        """
        Initialize instance encoder.

        Args:
            d_model: Model dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            d_feedforward: Feedforward dimension
            dropout: Dropout rate
            n_note_types: Number of note type categories
            max_seq_len: Maximum sequence length
        """
        super().__init__()

        self.d_model = d_model

        # Discrete feature embedding (note type)
        self.type_embedding = nn.Embedding(n_note_types, d_model, padding_idx=9)

        # Continuous feature encoder (5 features in v2)
        self.continuous_encoder = ContinuousFeatureEncoder(
            n_continuous=5,  # beat_pos, duration, bpm, scroll, local_density
            d_model=d_model,
        )

        # Feature fusion
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.fusion_norm = nn.LayerNorm(d_model)

        # Positional encoding for at most one event window.
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
        )

        # Output projection
        self.output_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode token sequence to vector.

        Args:
            tokens: Token tensor [batch, seq_len, 6]
            mask: Attention mask [batch, seq_len], 1 for valid, 0 for padding

        Returns:
            Instance embedding [batch, d_model]
        """
        if mask.shape != tokens.shape[:2] or not torch.all(mask.sum(dim=1) > 0):
            raise ValueError("each event window needs a nonempty aligned mask")

        # Split discrete and continuous features
        note_types = tokens[:, :, 0].long()  # [batch, seq_len]
        continuous_feats = tokens[:, :, 1:]  # [batch, seq_len, 5]

        # Embed discrete features
        type_emb = self.type_embedding(note_types)  # [batch, seq_len, d_model]

        # Encode continuous features
        cont_emb = self.continuous_encoder(
            continuous_feats
        )  # [batch, seq_len, d_model]

        # Fuse embeddings
        fused = self.fusion(torch.cat([type_emb, cont_emb], dim=-1))
        fused = self.fusion_norm(fused)  # [batch, seq_len, d_model]

        # Add positional encoding
        fused = self.pos_encoder(fused)

        # True means padding is ignored by self-attention.
        attn_mask = mask == 0

        # Apply transformer
        encoded = self.transformer(fused, src_key_padding_mask=attn_mask)

        # The frozen study uses the mean over valid events.
        mask_expanded = mask.unsqueeze(-1)
        output = (encoded * mask_expanded).sum(dim=1) / mask_expanded.sum(
            dim=1
        )
        return self.output_norm(output)
