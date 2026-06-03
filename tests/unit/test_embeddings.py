import math

import torch

from src.models.embeddings import TokenEmbedding


def test_token_embedding_scales_outputs():
    layer = TokenEmbedding(num_embeddings=4, d_model=6)
    with torch.no_grad():
        layer.weight.copy_(torch.arange(24, dtype=torch.float32).view(4, 6))

    indices = torch.tensor([[0, 1], [2, 3]])
    raw = layer.weight[indices]
    out = layer(indices)

    assert torch.allclose(out, raw * math.sqrt(6))
