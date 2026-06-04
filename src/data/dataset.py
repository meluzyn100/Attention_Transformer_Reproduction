from typing import Sequence

import torch
from torch.utils.data import Dataset

from .tokenizer import PAD_ID, SharedBPETokenizer


class TranslationDataset(Dataset[tuple[list[int], list[int]]]):
    def __init__(
        self,
        source_texts: Sequence[str],
        target_texts: Sequence[str],
        tokenizer: SharedBPETokenizer,
    ) -> None:
        if len(source_texts) != len(target_texts):
            raise ValueError("source_texts and target_texts must have the same length")

        self.source_texts = list(source_texts)
        self.target_texts = list(target_texts)
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.source_texts)

    def __getitem__(self, index: int) -> tuple[list[int], list[int]]:
        source_ids = self.tokenizer.encode(self.source_texts[index])
        target_ids = self.tokenizer.encode(self.target_texts[index])
        return source_ids, target_ids


def collate_fn(
    batch: Sequence[tuple[list[int], list[int]]], pad_id: int = PAD_ID
) -> tuple[torch.Tensor, torch.Tensor]:
    source_sequences, target_sequences = zip(*batch, strict=True)

    source_max_len = max(len(sequence) for sequence in source_sequences)
    target_max_len = max(len(sequence) for sequence in target_sequences)

    source_batch = torch.full((len(batch), source_max_len), pad_id, dtype=torch.long)
    target_batch = torch.full((len(batch), target_max_len), pad_id, dtype=torch.long)

    for row_index, sequence in enumerate(source_sequences):
        source_batch[row_index, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.long
        )
    for row_index, sequence in enumerate(target_sequences):
        target_batch[row_index, : len(sequence)] = torch.tensor(
            sequence, dtype=torch.long
        )

    return source_batch, target_batch


def create_src_mask(src_tokens: torch.Tensor) -> torch.Tensor:
    if src_tokens.dim() != 2:
        raise ValueError("src_tokens must have shape (batch, src_len)")

    return src_tokens.ne(PAD_ID).unsqueeze(1).unsqueeze(2)


def create_tgt_mask(tgt_tokens: torch.Tensor) -> torch.Tensor:
    if tgt_tokens.dim() != 2:
        raise ValueError("tgt_tokens must have shape (batch, tgt_len)")

    batch_size, tgt_len = tgt_tokens.shape
    valid_tokens = tgt_tokens.ne(PAD_ID)
    key_padding_mask = valid_tokens.unsqueeze(1).unsqueeze(2)
    query_padding_mask = valid_tokens.unsqueeze(1).unsqueeze(3)
    look_ahead_mask = (
        torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt_tokens.device, dtype=torch.bool)
        )
        .unsqueeze(0)
        .unsqueeze(0)
    )

    return (
        key_padding_mask
        & query_padding_mask
        & look_ahead_mask.expand(batch_size, 1, tgt_len, tgt_len)
    )
