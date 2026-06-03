import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        # Compute the positional encodings using sine and cosine functions of different frequencies
        # The formula is:
        #   PE(pos, 2i) = sin(pos / (10000^(2i/d_model))),
        #   PE(pos, 2i+1) = cos(pos / (10000^(2i/d_model))),
        # a^b = exp(b * log(a)) is used to compute the denominator efficiently
        d_div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * d_div_term)
        pe[:, 1::2] = torch.cos(position * d_div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)
