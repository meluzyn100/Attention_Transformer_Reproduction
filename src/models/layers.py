import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward


class EncoderLayer(nn.Module):
    """Single encoder layer with Post-LN (Add & Norm after each sub-layer).

    Implements: self-attention -> add & norm -> feed-forward -> add & norm
    (matches Vaswani et al. 2017 ordering)
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        # Self-attention sub-layer
        attn_out = self.self_attn(x, x, x, mask=src_mask)
        # Add & Norm
        x = self.norm1(x + self.dropout(attn_out))

        # Feed-forward sub-layer
        ff_out = self.feed_forward(x)
        # Add & Norm
        x = self.norm2(x + self.dropout(ff_out))

        return x


class DecoderLayer(nn.Module):
    """Single decoder layer with masked self-attention, cross-attention, and FFN.

    Order: masked self-attn -> add & norm -> cross-attn -> add & norm -> FFN -> add & norm
    Uses Post-LN (LayerNorm after the residual addition).
    """

    def __init__(self, d_model: int, h: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.masked_self_attn = MultiHeadAttention(d_model, h, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Masked self-attention
        self_attn_out = self.masked_self_attn(x, x, x, mask=tgt_mask)
        # Add & Norm
        x = self.norm1(x + self.dropout(self_attn_out))

        # Cross-attention over encoder outputs
        cross_attn_out = self.cross_attn(x, enc_output, enc_output, mask=src_mask)
        # Add & Norm
        x = self.norm2(x + self.dropout(cross_attn_out))

        # Position-wise feed-forward
        ff_out = self.feed_forward(x)
        # Add & Norm
        x = self.norm3(x + self.dropout(ff_out))

        return x
