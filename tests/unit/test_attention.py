import torch

from src.models.attention import MultiHeadAttention, scaled_dot_product_attention


def test_multi_head_attention_output_shape():
    batch, seq, d_model, h = 2, 6, 16, 4
    x = torch.randn(batch, seq, d_model)
    layer = MultiHeadAttention(d_model=d_model, h=h, dropout=0.0)

    out = layer(x, x, x)

    assert out.shape == (batch, seq, d_model)


def test_scaled_attention_mask_zeros_out_masked_weights():
    q = torch.tensor([[[1.0, 0.0]]])
    k = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [999.0, 999.0]]])
    mask = torch.tensor([[[True, True, False]]])

    _, weights = scaled_dot_product_attention(
        q,
        k,
        v,
        mask=mask,
        dropout_p=0.0,
        training=False,
        return_attention=True,
    )

    assert torch.allclose(weights[..., 2], torch.zeros_like(weights[..., 2]), atol=1e-7)


def test_multi_head_attention_backward_pass():
    batch, seq, d_model, h = 2, 5, 16, 4
    x = torch.randn(batch, seq, d_model, requires_grad=True)
    layer = MultiHeadAttention(d_model=d_model, h=h, dropout=0.1)

    out = layer(x, x, x)
    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()