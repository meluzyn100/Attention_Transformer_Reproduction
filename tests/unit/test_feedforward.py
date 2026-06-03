import torch

from src.models.feedforward import PositionWiseFeedForward


def test_position_wise_feed_forward_output_shape():
    batch, seq, d_model, d_ff = 2, 4, 8, 16
    x = torch.randn(batch, seq, d_model)
    layer = PositionWiseFeedForward(d_model=d_model, d_ff=d_ff, dropout=0.0)

    out = layer(x)

    assert out.shape == (batch, seq, d_model)


def test_position_wise_feed_forward_is_nonlinear():
    layer = PositionWiseFeedForward(d_model=2, d_ff=2, dropout=0.0)
    with torch.no_grad():
        layer.linear1.weight.copy_(torch.eye(2))
        layer.linear1.bias.zero_()
        layer.linear2.weight.copy_(torch.eye(2))
        layer.linear2.bias.zero_()

    x = torch.tensor([[[1.0, -1.0]]])
    y = torch.tensor([[[-1.0, 1.0]]])

    assert not torch.allclose(layer(x + y), layer(x) + layer(y))
