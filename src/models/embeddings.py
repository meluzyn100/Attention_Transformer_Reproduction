import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Embedding):
    def __init__(self, num_embeddings: int, d_model: int) -> None:
        super().__init__(num_embeddings=num_embeddings, embedding_dim=d_model)
        self.d_model = d_model

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Scale the embeddings by sqrt(d_model) to maintain consistent variance
        return super().forward(input) * math.sqrt(self.d_model)
