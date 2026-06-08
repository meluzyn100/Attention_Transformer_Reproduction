import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    return_attention: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    # scores shape: (batch, h, seq_q, seq_k)
    # Attention(Q, K, V) = softmax((Q . K^T) / sqrt(d_k)). V
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    # Masking is applied before softmax to ensure masked positions have zero attention weights
    if mask is not None:
        mask = mask.to(dtype=torch.bool)
        # scores = scores.masked_fill(~mask, -float('inf'))
        scores = scores.masked_fill(~mask, -torch.finfo(scores.dtype).max)

        # Softmax will convert -inf to 0, effectively zeroing out masked positions
    attention_weights = F.softmax(scores, dim=-1)
    attention_weights = torch.nan_to_num(attention_weights, nan=0.0, posinf=0.0, neginf=0.0)

    # Apply dropout to attention weights during training
    attention_weights = F.dropout(attention_weights, p=dropout_p, training=training)
    output = torch.matmul(attention_weights, v)

    if return_attention:
        return output, attention_weights

    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % h != 0:
            raise ValueError("d_model must be divisible by h")

        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h
        self.dropout = dropout

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _prepare_mask(
        self,
        mask: torch.Tensor,
        batch_size: int,
        seq_q: int,
        seq_k: int,
    ) -> torch.Tensor:
        if mask.dim() == 2:
            # (batch, seq_k) -> (batch, 1, seq_q, seq_k)
            mask = mask.unsqueeze(1).unsqueeze(1).expand(batch_size, 1, seq_q, seq_k)
        elif mask.dim() == 3:
            # (batch, seq_q, seq_k) -> (batch, 1, seq_q, seq_k)
            mask = mask.unsqueeze(1)
        elif mask.dim() == 4:
            # (batch, 1|h, seq_q|1, seq_k)
            if mask.size(0) != batch_size or mask.size(-1) != seq_k:
                raise ValueError("mask has incompatible shape")
            if mask.size(-2) == 1 and seq_q != 1:
                mask = mask.expand(batch_size, mask.size(1), seq_q, seq_k)
        else:
            raise ValueError("mask must have 2, 3, or 4 dimensions")

        if mask.dim() != 4:
            raise ValueError("mask must be 4D after preparation")
        if (
            mask.size(0) != batch_size
            or mask.size(-2) != seq_q
            or mask.size(-1) != seq_k
        ):
            raise ValueError("mask has incompatible shape")
        if mask.size(1) not in (1, self.h):
            raise ValueError("mask has incompatible shape")

        return mask

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, seq_q, seq_k = q.size(0), q.size(1), k.size(1)

        q_proj = self.W_q(q).view(batch_size, seq_q, self.h, self.d_k).transpose(1, 2)
        k_proj = self.W_k(k).view(batch_size, seq_k, self.h, self.d_k).transpose(1, 2)
        v_proj = self.W_v(v).view(batch_size, seq_k, self.h, self.d_k).transpose(1, 2)

        attn_mask = None
        if mask is not None:
            attn_mask = self._prepare_mask(mask, batch_size, seq_q, seq_k)

        attended = scaled_dot_product_attention(
            q_proj,
            k_proj,
            v_proj,
            mask=attn_mask,
            dropout_p=self.dropout,
            training=self.training,
        )

        attended = (
            attended.transpose(1, 2).contiguous().view(batch_size, seq_q, self.d_model)
        )

        return self.W_o(attended)
