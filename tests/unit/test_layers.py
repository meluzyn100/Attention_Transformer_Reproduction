import torch

from src.models.layers import DecoderLayer, EncoderLayer


def test_encoder_layer_output_shape_and_backward():
    batch, seq, d_model, h, d_ff = 2, 7, 16, 4, 32
    x = torch.randn(batch, seq, d_model, requires_grad=True)
    layer = EncoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=0.0)

    out = layer(x, src_mask=None)

    assert out.shape == (batch, seq, d_model)

    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_decoder_layer_output_shape_and_masks():
    batch, src_seq, tgt_seq, d_model, h, d_ff = 2, 5, 6, 16, 4, 32
    # decoder input (target embeddings)
    x = torch.randn(batch, tgt_seq, d_model, requires_grad=True)
    # encoder output
    enc = torch.randn(batch, src_seq, d_model)

    layer = DecoderLayer(d_model=d_model, h=h, d_ff=d_ff, dropout=0.0)

    # causal (look-ahead) mask for target: allow attending only to previous and current positions
    causal = torch.tril(torch.ones(tgt_seq, tgt_seq)).bool().unsqueeze(0)
    causal = causal.expand(batch, tgt_seq, tgt_seq)

    # source padding mask (all True = no padding)
    src_mask = torch.ones(batch, src_seq).bool()

    out = layer(x, enc_output=enc, src_mask=src_mask, tgt_mask=causal)

    assert out.shape == (batch, tgt_seq, d_model)

    loss = out.mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
