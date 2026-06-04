"""Seq2Seq trainer for the Transformer translation model.

Extends the generic Trainer by handling:
- Teacher-forcing input shift: decoder receives tgt[:,:-1], loss targets tgt[:,1:]
- Mask creation (src padding mask + tgt causal mask)
- Correct model.forward(**kwargs) dispatch
"""
from __future__ import annotations

from typing import Any

import torch

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None

from src.data.dataset import create_src_mask, create_tgt_mask

from .trainer import Trainer


class TranslationTrainer(Trainer):
    """Trainer wired for (src_tokens, tgt_tokens) batches from TranslationDataset."""

    def _unpack_batch(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            src = batch.get("src") or batch.get("source")
            tgt = batch.get("tgt") or batch.get("target") or batch.get("labels")
            if src is None or tgt is None:
                raise KeyError("Batch dict must contain 'src'/'source' and 'tgt'/'target' keys")
            return src, tgt
        raise TypeError(
            f"Unsupported batch type {type(batch)}. "
            "Expected (src, tgt) tuple or dict with src/tgt keys."
        )

    def _seq2seq_step(
        self, batch: Any
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_tokens, tgt_tokens = self._unpack_batch(batch)

        # Teacher forcing: decoder sees tgt[:,:-1], loss targets tgt[:,1:]
        tgt_in = tgt_tokens[:, :-1]
        labels = tgt_tokens[:, 1:]

        src_mask = create_src_mask(src_tokens).to(self.device)
        tgt_mask = create_tgt_mask(tgt_in).to(self.device)

        logits = self.model(src_tokens, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
        # logits: (B, T-1, vocab_size) — reshape for loss
        return logits, labels

    def train_step(self, batch: Any) -> tuple[float, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = self._move_to_device(batch)
        logits, labels = self._seq2seq_step(batch)

        loss = self.criterion(logits, labels)
        loss.backward()

        grad_norm = self._compute_grad_norm()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.state.global_step += 1
        self._log_metrics(loss=float(loss.item()), grad_norm=grad_norm)
        return float(loss.item()), float(grad_norm)

    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = self._move_to_device(batch)
            logits, labels = self._seq2seq_step(batch)
            loss = self.criterion(logits, labels)
            running_loss += float(loss.item())
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        if self.use_mlflow and mlflow is not None:
            mlflow.log_metric("val/loss", avg_loss, step=self.state.global_step)
        return avg_loss
