#!/usr/bin/env python3
"""Train a shared BPE tokenizer on parallel corpus files.

Writes `bpe_vocab.json` and `bpe_merges.txt` to the output directory.
"""


import sys
import argparse
from pathlib import Path
from typing import Iterable, List

# Make `src` importable when running this script directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.tokenizer import SharedBPETokenizer  # noqa: E402


def gather_files(raw_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in (".de.txt", ".en.txt"):
        files.extend(sorted(raw_dir.glob(f"*{ext}")))
    return files


def sample_lines(file_path: Path, max_lines: int) -> Path:
    # Create a temporary sampled file in-memory on disk next to output dir
    sampled = file_path.with_suffix(file_path.suffix + ".sample")
    with (
        file_path.open("r", encoding="utf-8") as src,
        sampled.open("w", encoding="utf-8") as out,
    ):
        for i, line in enumerate(src):
            if i >= max_lines:
                break
            out.write(line)
    return sampled


def train_from_raw(
    raw_dir: Path,
    output_dir: Path,
    vocab_size: int = 32000,
    sample_size: int | None = None,
) -> None:
    files = gather_files(raw_dir)
    if not files:
        raise RuntimeError(f"No raw files found in {raw_dir}")

    if sample_size is not None and sample_size > 0:
        # sample sample_size lines from each language file
        sampled_files: List[Path] = []
        for f in files:
            sampled_files.append(sample_lines(f, sample_size))
        files_to_train: Iterable[Path] = sampled_files
    else:
        files_to_train = files

    tokenizer = SharedBPETokenizer(vocab_size=vocab_size)
    tokenizer.train([str(p) for p in files_to_train])

    out_vocab, out_merges = tokenizer.save(output_dir)
    print(f"Saved vocab to {out_vocab} and merges to {out_merges}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train shared BPE tokenizer from raw parallel files"
    )
    p.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory with raw .de.txt and .en.txt files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to write bpe vocab and merges into",
    )
    p.add_argument("--vocab-size", type=int, default=32000)
    p.add_argument(
        "--sample-size",
        type=int,
        default=10000,
        help="Number of lines to sample from each file (use 0 or omit to use full files)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    sample_size = (
        args.sample_size if args.sample_size and args.sample_size > 0 else None
    )
    train_from_raw(
        args.raw_dir,
        args.output_dir,
        vocab_size=args.vocab_size,
        sample_size=sample_size,
    )


if __name__ == "__main__":
    main()
