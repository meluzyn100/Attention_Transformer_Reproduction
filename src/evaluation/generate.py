from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import hydra
import torch
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from src.data.dataset import create_src_mask
from src.data.tokenizer import SharedBPETokenizer
from src.evaluation.metrics import compute_bleu
from src.inference.beam_search import BeamSearch
from src.models.transformer import Transformer


@dataclass(slots=True)
class TranslationExample:
    source: str
    reference: str | None = None


class TranslationTextDataset(Dataset[TranslationExample]):
    def __init__(self, examples: list[TranslationExample]) -> None:
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TranslationExample:
        return self.examples[index]


def load_examples(path: str | Path) -> list[TranslationExample]:
    path = Path(path)
    if path.suffix == ".jsonl":
        examples: list[TranslationExample] = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)["translation"]
                examples.append(TranslationExample(source=record["de"], reference=record.get("en")))
        return examples

    source_path = path
    reference_path = path.with_suffix(".en.txt")
    sources = source_path.read_text(encoding="utf-8").splitlines()
    references = reference_path.read_text(encoding="utf-8").splitlines() if reference_path.exists() else [None] * len(sources)
    return [TranslationExample(source=src, reference=ref) for src, ref in zip(sources, references, strict=True)]


def resolve_path(path: str | Path, root: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else root / path


def collate_sources(
    batch: Sequence[TranslationExample],
    tokenizer: SharedBPETokenizer,
) -> tuple[torch.Tensor, list[str | None]]:
    encoded = [tokenizer.encode(example.source) for example in batch]
    max_len = max(len(sequence) for sequence in encoded)
    src_batch = torch.full((len(encoded), max_len), tokenizer.pad_id, dtype=torch.long)

    for row_index, sequence in enumerate(encoded):
        src_batch[row_index, : len(sequence)] = torch.tensor(sequence, dtype=torch.long)

    references = [example.reference for example in batch]
    return src_batch, references


def extract_model_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if all(isinstance(key, str) for key in checkpoint.keys()):
        return checkpoint
    raise ValueError("Unsupported checkpoint format")


def load_model(cfg: DictConfig, map_location: str = "cpu") -> Transformer:
    model = instantiate(cfg.model)
    checkpoint_path = resolve_path(cfg.evaluation.checkpoint, Path(get_original_cwd()))
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    model.load_state_dict(extract_model_state(checkpoint))
    model.eval()
    return model


def generate_translations(
    model: Transformer,
    tokenizer: SharedBPETokenizer,
    examples: Iterable[TranslationExample],
    batch_size: int = 8,
    beam_size: int = 4,
    alpha: float = 0.6,
    max_len: int = 256,
    device: str | torch.device = "cpu",
) -> tuple[list[str], list[str]]:
    dataset = TranslationTextDataset(list(examples))
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_sources(batch, tokenizer),
    )

    search = BeamSearch(
        model=model.to(device),
        beam_size=beam_size,
        alpha=alpha,
        max_len=max_len,
        device=device,
    )

    hypotheses: list[str] = []
    references: list[str] = []
    for src_batch, batch_references in tqdm(loader, desc="generate", unit="batch"):
        src_batch = src_batch.to(device)
        for row_index in range(src_batch.size(0)):
            src_tokens = src_batch[row_index : row_index + 1]
            src_mask = create_src_mask(src_tokens)
            token_ids = search.search(src_tokens, src_mask=src_mask)
            hypotheses.append(tokenizer.decode(token_ids))
        references.extend(reference for reference in batch_references if reference is not None)

    return hypotheses, references


@hydra.main(config_path="../../configs", config_name="evaluate_base", version_base=None)
def main(cfg: DictConfig) -> None:
    root = Path(get_original_cwd())

    tokenizer = SharedBPETokenizer.load(
        resolve_path(cfg.data.tokenizer_vocab, root),
        resolve_path(cfg.data.tokenizer_merges, root),
    )
    model = load_model(cfg, map_location=str(cfg.evaluation.device))
    examples = load_examples(resolve_path(cfg.data.test_jsonl, root))
    hypotheses, references = generate_translations(
        model=model,
        tokenizer=tokenizer,
        examples=examples,
        batch_size=int(cfg.evaluation.batch_size),
        beam_size=int(cfg.evaluation.beam_size),
        alpha=float(cfg.evaluation.alpha),
        max_len=int(cfg.evaluation.max_len),
        device=str(cfg.evaluation.device),
    )

    output_path = resolve_path(cfg.evaluation.output, root)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(hypotheses) + "\n", encoding="utf-8")

    if references and len(references) == len(hypotheses):
        print(f"BLEU: {compute_bleu(hypotheses, references):.2f}")
    print(f"Saved hypotheses to {output_path}")


if __name__ == "__main__":
    main()
