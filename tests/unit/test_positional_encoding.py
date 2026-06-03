import torch

from src.models.positional_encoding import SinusoidalPositionalEncoding


def test_positional_encoding_differs_between_positions():
    layer = SinusoidalPositionalEncoding(d_model=6, dropout=0.0, max_len=10)
    x = torch.zeros(1, 4, 6)

    out = layer(x)

    assert not torch.allclose(out[:, 0, :], out[:, 1, :])


def test_positional_encoding_is_identical_across_batch_for_same_positions():
    layer = SinusoidalPositionalEncoding(d_model=6, dropout=0.0, max_len=10)
    x = torch.zeros(2, 4, 6)

    out = layer(x)

    assert torch.allclose(out[0], out[1])
