import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .embeddings import TokenEmbedding
from .layers import DecoderLayer, EncoderLayer
from .positional_encoding import (
    IdentityEncoding,
    LearnedPositionalEncoding,
    SinusoidalPositionalEncoding,
)


def _build_pe(
    kind: str,
    d_model: int,
    dropout: float,
    max_len: int,
) -> torch.nn.Module:
    if kind == "sinusoidal":
        return SinusoidalPositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )
    if kind == "learned":
        return LearnedPositionalEncoding(
            d_model=d_model, dropout=dropout, max_len=max_len
        )
    if kind == "none":
        return IdentityEncoding(dropout=dropout)
    raise ValueError(
        f"positional_encoding must be 'sinusoidal', 'learned', or 'none', got {kind!r}"
    )


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
        positional_encoding: str = "sinusoidal",
        activation_checkpointing: bool = False,
        activation_checkpointing_reentrant: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = _build_pe(
            positional_encoding, d_model, dropout, max_len
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.activation_checkpointing = activation_checkpointing
        self.activation_checkpointing_reentrant = activation_checkpointing_reentrant

    def forward(
        self, src: torch.Tensor, src_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.embedding(src)
        x = self.positional_encoding(x)

        for layer in self.layers:
            if self.activation_checkpointing and self.training:
                x = checkpoint(
                    lambda h, m, layer=layer: layer(h, src_mask=m),
                    x,
                    src_mask,
                    use_reentrant=self.activation_checkpointing_reentrant,
                )
            else:
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
        positional_encoding: str = "sinusoidal",
        activation_checkpointing: bool = False,
        activation_checkpointing_reentrant: bool = False,
    ) -> None:
        super().__init__()
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = _build_pe(
            positional_encoding, d_model, dropout, max_len
        )
        self.layers = nn.ModuleList(
            [
                DecoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.activation_checkpointing = activation_checkpointing
        self.activation_checkpointing_reentrant = activation_checkpointing_reentrant

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
            if self.activation_checkpointing and self.training:
                x = checkpoint(
                    lambda h, enc, src_m, tgt_m, layer=layer: layer(
                        h,
                        enc_output=enc,
                        src_mask=src_m,
                        tgt_mask=tgt_m,
                    ),
                    x,
                    enc_output,
                    src_mask,
                    tgt_mask,
                    use_reentrant=self.activation_checkpointing_reentrant,
                )
            else:
                x = layer(
                    x, enc_output=enc_output, src_mask=src_mask, tgt_mask=tgt_mask
                )

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
        positional_encoding: str = "sinusoidal",
        activation_checkpointing: bool = False,
        activation_checkpointing_reentrant: bool = False,
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
            positional_encoding=positional_encoding,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_reentrant=activation_checkpointing_reentrant,
        )
        self.decoder = Decoder(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=num_layers,
            h=h,
            d_ff=d_ff,
            dropout=dropout,
            max_len=max_len,
            positional_encoding=positional_encoding,
            activation_checkpointing=activation_checkpointing,
            activation_checkpointing_reentrant=activation_checkpointing_reentrant,
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
