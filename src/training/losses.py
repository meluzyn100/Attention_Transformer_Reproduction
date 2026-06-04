from __future__ import annotations

import torch
from torch import nn


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Cross-entropy with fixed label smoothing built on KLDivLoss.

    Soft targets are built as:
    - smoothing / (vocab_size - 1) for all non-target classes
    - confidence (1 - smoothing) for the target class
    """

    def __init__(
        self,
        vocab_size: int,
        smoothing: float = 0.1,
        ignore_index: int | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 1:
            raise ValueError("vocab_size must be > 1")
        if not (0.0 <= smoothing < 1.0):
            raise ValueError("smoothing must be in [0, 1)")

        self.vocab_size = vocab_size
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.ignore_index = ignore_index
        self.kl_div = nn.KLDivLoss(reduction="batchmean")

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.dim() < 2:
            raise ValueError("logits must have shape (..., vocab_size)")
        if logits.size(-1) != self.vocab_size:
            raise ValueError(
                f"Expected logits last dim == {self.vocab_size}, got {logits.size(-1)}"
            )

        logits = logits.reshape(-1, logits.size(-1))
        target = target.reshape(-1)

        valid_mask = torch.ones_like(target, dtype=torch.bool)
        if self.ignore_index is not None:
            valid_mask = target != self.ignore_index

        if not valid_mask.any():
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        target = target[valid_mask]

        log_probs = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            soft_targets = torch.full_like(
                log_probs,
                fill_value=self.smoothing / (self.vocab_size - 1),
            )
            soft_targets.scatter_(1, target.unsqueeze(1), self.confidence)

        return self.kl_div(log_probs, soft_targets)
