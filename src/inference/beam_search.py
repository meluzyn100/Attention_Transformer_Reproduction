from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from src.data.dataset import create_src_mask, create_tgt_mask
from src.data.tokenizer import BOS_ID, EOS_ID, PAD_ID


@dataclass(slots=True)
class _Beam:
    tokens: list[int]
    log_prob: float
    finished: bool = False


class BeamSearch:
    def __init__(
        self,
        model: torch.nn.Module,
        beam_size: int = 4,
        alpha: float = 0.6,
        max_len: int = 256,
        bos_id: int = BOS_ID,
        eos_id: int = EOS_ID,
        pad_id: int = PAD_ID,
        device: str | torch.device | None = None,
    ) -> None:
        if beam_size <= 0:
            raise ValueError("beam_size must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if max_len < 2:
            raise ValueError("max_len must be at least 2")

        self.model = model
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_len = max_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id
        self.device = (
            torch.device(device)
            if device is not None
            else next(model.parameters()).device
        )

    def _length_penalty(self, length: int) -> float:
        return ((5.0 + length) ** self.alpha) / ((5.0 + 1.0) ** self.alpha)

    def _score(self, log_prob: float, length: int) -> float:
        return log_prob / self._length_penalty(length)

    def _finalize(self, tokens: Iterable[int]) -> list[int]:
        result = list(tokens)
        if result and result[0] == self.bos_id:
            result = result[1:]
        if self.eos_id in result:
            result = result[: result.index(self.eos_id)]
        return result

    @torch.no_grad()
    def search(
        self,
        src_tokens: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> list[int]:
        if src_tokens.dim() == 1:
            src_tokens = src_tokens.unsqueeze(0)
        if src_tokens.dim() != 2:
            raise ValueError(
                "src_tokens must have shape (batch, src_len) or (src_len,)"
            )
        if src_tokens.size(0) != 1:
            raise ValueError("BeamSearch.search expects a single source example")

        src_tokens = src_tokens.to(self.device)
        src_mask = (
            create_src_mask(src_tokens)
            if src_mask is None
            else src_mask.to(self.device)
        )

        self.model.eval()
        encoder = getattr(self.model, "encoder")
        decoder = getattr(self.model, "decoder")
        generator = getattr(self.model, "generator")

        enc_output = encoder(src_tokens, src_mask=src_mask)
        active = [_Beam(tokens=[self.bos_id], log_prob=0.0)]
        finished: list[_Beam] = []

        for _ in range(self.max_len - 1):
            unfinished = [beam for beam in active if not beam.finished]
            if not unfinished:
                break

            padded = torch.full(
                (len(unfinished), max(len(beam.tokens) for beam in unfinished)),
                self.pad_id,
                dtype=torch.long,
                device=self.device,
            )
            for row_index, beam in enumerate(unfinished):
                padded[row_index, : len(beam.tokens)] = torch.tensor(
                    beam.tokens, dtype=torch.long, device=self.device
                )

            repeated_enc = enc_output.repeat(len(unfinished), 1, 1)
            repeated_src_mask = src_mask.repeat(len(unfinished), 1, 1, 1)
            tgt_mask = create_tgt_mask(padded)
            decoder_output = decoder(
                padded,
                enc_output=repeated_enc,
                src_mask=repeated_src_mask,
                tgt_mask=tgt_mask,
            )
            logits = generator(decoder_output[:, -1, :])
            log_probs = torch.log_softmax(logits, dim=-1)

            candidates: list[_Beam] = []
            for row_index, beam in enumerate(unfinished):
                values, indices = torch.topk(log_probs[row_index], self.beam_size)
                for token_log_prob, token_id in zip(
                    values.tolist(), indices.tolist(), strict=True
                ):
                    next_tokens = [*beam.tokens, int(token_id)]
                    candidate = _Beam(
                        tokens=next_tokens,
                        log_prob=beam.log_prob + float(token_log_prob),
                        finished=int(token_id) == self.eos_id,
                    )
                    if candidate.finished:
                        finished.append(candidate)
                    else:
                        candidates.append(candidate)

            if not candidates:
                break

            candidates.sort(
                key=lambda beam: self._score(beam.log_prob, len(beam.tokens) - 1),
                reverse=True,
            )
            active = candidates[: self.beam_size]

        pool = finished or active
        best = max(
            pool, key=lambda beam: self._score(beam.log_prob, len(beam.tokens) - 1)
        )
        return self._finalize(best.tokens)
