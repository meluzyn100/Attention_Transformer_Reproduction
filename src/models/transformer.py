import torch
import torch.nn as nn

from .embeddings import TokenEmbedding
from .layers import DecoderLayer, EncoderLayer
from .positional_encoding import SinusoidalPositionalEncoding


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embedding(src)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, src_mask=src_mask)

        return x


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        tgt: torch.Tensor,
        enc_output: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedding(tgt)
        x = self.positional_encoding(x)

        for layer in self.layers:
            x = layer(x, enc_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask)

        return x


class Transformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_layers: int = 6,
        h: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 5000,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            h=h,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            h=h,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
        )
        self.generator = nn.Linear(d_model, vocab_size)

        self.apply(initialize_weights)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor | None = None,
        tgt_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        enc_output = self.encoder(src, src_mask=src_mask)
        dec_output = self.decoder(
            tgt,
            enc_output=enc_output,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        return self.generator(dec_output)
