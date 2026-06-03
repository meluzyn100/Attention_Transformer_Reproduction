import torch

from src.models.transformer import Transformer


def test_transformer_forward_shape_and_trainable_parameters():
    batch_size, src_len, tgt_len = 2, 10, 12
    vocab_size = 50

    model = Transformer(
        vocab_size=vocab_size,
        d_model=32,
        num_layers=6,
        h=4,
        d_ff=64,
        dropout=0.0,
    )

    src = torch.randint(0, vocab_size, (batch_size, src_len))
    tgt = torch.randint(0, vocab_size, (batch_size, tgt_len))
    src_mask = torch.ones(batch_size, src_len, dtype=torch.bool)
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, dtype=torch.bool)).unsqueeze(0)
    tgt_mask = tgt_mask.expand(batch_size, -1, -1)

    logits = model(src, tgt, src_mask=src_mask, tgt_mask=tgt_mask)

    assert logits.shape == (batch_size, tgt_len, vocab_size)
    assert all(parameter.requires_grad for parameter in model.parameters())