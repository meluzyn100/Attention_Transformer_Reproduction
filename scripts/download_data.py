#!/usr/bin/env python3
"""Download and write WMT14 de-en raw data using HuggingFace datasets.

This replaces the previous shell wrapper that embedded Python.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def save_split(split, out_dir: Path, split_name: str) -> None:
    jsonl_path = out_dir / f"{split_name}.jsonl"
    src_path = out_dir / f"{split_name}.de.txt"
    tgt_path = out_dir / f"{split_name}.en.txt"

    with (
        jsonl_path.open("w", encoding="utf-8") as jsonl_file,
        src_path.open("w", encoding="utf-8") as src_file,
        tgt_path.open("w", encoding="utf-8") as tgt_file,
    ):
        for example in split:
            translation = example["translation"]
            record = {"translation": {"de": translation["de"], "en": translation["en"]}}
            jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")
            src_file.write(translation["de"].replace("\n", " ").strip() + "\n")
            tgt_file.write(translation["en"].replace("\n", " ").strip() + "\n")


def download_wmt14(output_dir: Path, splits: Iterable[str]) -> None:
    from datasets import load_dataset

    dataset = load_dataset("wmt14", "de-en")

    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        if split not in dataset:
            raise ValueError(
                f"Split {split!r} not found in dataset; available: {list(dataset.keys())}"
            )
        save_split(dataset[split], output_dir, split)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download WMT14 de-en raw data")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/wmt14_de_en"),
        help="Directory to write raw files into",
    )
    p.add_argument(
        "--splits",
        nargs="*",
        default=["train", "validation", "test"],
        help="Dataset splits to download (default: train validation test)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    download_wmt14(args.output_dir, args.splits)
    print(f"Saved WMT14 de-en raw data to {args.output_dir}")


if __name__ == "__main__":
    main()
