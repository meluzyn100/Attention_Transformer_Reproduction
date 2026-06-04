import json
from pathlib import Path
from typing import Iterable


def normalize_text(text: str, lowercase: bool = False) -> str:
    normalized = text.strip()
    if lowercase:
        normalized = normalized.lower()
    return " ".join(normalized.split())


def load_parallel_corpus(
    raw_dir: str | Path, split: str
) -> tuple[list[str], list[str]]:
    raw_path = Path(raw_dir) / f"{split}.jsonl"
    source_texts: list[str] = []
    target_texts: list[str] = []

    with raw_path.open("r", encoding="utf-8") as file_handle:
        for line in file_handle:
            if not line.strip():
                continue
            record = json.loads(line)
            translation = record["translation"]
            source_texts.append(translation["de"])
            target_texts.append(translation["en"])

    return source_texts, target_texts


def write_parallel_corpus(
    source_texts: Iterable[str],
    target_texts: Iterable[str],
    output_dir: str | Path,
    split: str,
    lowercase: bool = False,
) -> tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_path = output_path / f"{split}.de.txt"
    target_path = output_path / f"{split}.en.txt"

    with (
        source_path.open("w", encoding="utf-8") as source_file,
        target_path.open("w", encoding="utf-8") as target_file,
    ):
        for source_text, target_text in zip(source_texts, target_texts, strict=True):
            source_file.write(normalize_text(source_text, lowercase=lowercase) + "\n")
            target_file.write(normalize_text(target_text, lowercase=lowercase) + "\n")

    return source_path, target_path
