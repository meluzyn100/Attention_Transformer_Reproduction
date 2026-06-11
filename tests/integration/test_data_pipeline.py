from functools import partial

import pytest
import torch
from torch.utils.data import DataLoader

from src.data.dataset import (
    TranslationDataset,
    collate_fn,
    create_src_mask,
    create_tgt_mask,
)
from src.data.preprocess import normalize_text
from src.data.tokenizer import BOS_ID, EOS_ID, PAD_ID, SharedBPETokenizer

pytest.importorskip("datasets")


def test_wmt14_data_pipeline_round_trip(tmp_path):
    from datasets import load_dataset

    split = load_dataset("wmt/wmt14", "de-en", split="train[:100]")

    source_texts = [normalize_text(example["translation"]["de"]) for example in split]
    target_texts = [normalize_text(example["translation"]["en"]) for example in split]

    source_path = tmp_path / "train.de.txt"
    target_path = tmp_path / "train.en.txt"
    source_path.write_text("\n".join(source_texts) + "\n", encoding="utf-8")
    target_path.write_text("\n".join(target_texts) + "\n", encoding="utf-8")

    tokenizer = SharedBPETokenizer(vocab_size=32_000)
    tokenizer.train([source_path, target_path])

    assert tokenizer.pad_id == PAD_ID
    assert tokenizer.bos_id == BOS_ID
    assert tokenizer.eos_id == EOS_ID

    dataset = TranslationDataset(source_texts, target_texts, tokenizer)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_id=tokenizer.pad_id),
    )

    source_batch, target_batch = next(iter(loader))

    assert source_batch.dtype == torch.long
    assert target_batch.dtype == torch.long
    assert source_batch.shape[0] == 8
    assert target_batch.shape[0] == 8
    assert source_batch.ndim == 2
    assert target_batch.ndim == 2

    source_mask = create_src_mask(source_batch)
    target_mask = create_tgt_mask(target_batch)

    assert source_mask.shape == (8, 1, 1, source_batch.shape[1])
    assert target_mask.shape == (8, 1, target_batch.shape[1], target_batch.shape[1])
    assert source_mask.dtype == torch.bool
    assert target_mask.dtype == torch.bool
    assert source_mask.any()
    assert target_mask.any()

    encoded_source = tokenizer.encode(source_texts[0])
    decoded_source = tokenizer.decode(encoded_source)
    assert decoded_source
    assert encoded_source[0] == BOS_ID
    assert encoded_source[-1] == EOS_ID
