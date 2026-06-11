#!/usr/bin/env python3
"""Average the latest N checkpoints and write a single averaged checkpoint."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import torch

_STEP_RE = re.compile(r"step_(\d+)\.pt$")
_EPOCH_RE = re.compile(r"epoch_(\d+)\.pt$")


def _checkpoint_sort_key(path: Path) -> tuple[int, int]:
    name = path.name
    step_match = _STEP_RE.search(name)
    if step_match:
        return (2, int(step_match.group(1)))

    epoch_match = _EPOCH_RE.search(name)
    if epoch_match:
        return (1, int(epoch_match.group(1)))

    return (0, int(path.stat().st_mtime))


def _extract_model_state(checkpoint: dict) -> dict[str, torch.Tensor]:
    if "model" in checkpoint and isinstance(checkpoint["model"], dict):
        return checkpoint["model"]
    if all(isinstance(key, str) for key in checkpoint.keys()):
        return checkpoint
    raise ValueError(
        "Unsupported checkpoint format: expected 'model' state_dict or raw state_dict"
    )


def _latest_checkpoints(checkpoint_dir: Path, n_last: int) -> list[Path]:
    candidates = [p for p in checkpoint_dir.glob("*.pt") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")

    candidates.sort(key=_checkpoint_sort_key)
    return candidates[-n_last:]


def average_checkpoints(input_paths: list[Path]) -> dict[str, torch.Tensor]:
    averaged: dict[str, torch.Tensor] = {}
    base_keys: set[str] | None = None
    floating_dtypes: dict[str, torch.dtype] = {}

    for checkpoint_index, checkpoint_path in enumerate(input_paths):
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model_state = _extract_model_state(checkpoint)

        current_keys = set(model_state.keys())
        if base_keys is None:
            base_keys = current_keys
        elif current_keys != base_keys:
            missing = sorted(base_keys - current_keys)
            extra = sorted(current_keys - base_keys)
            raise ValueError(
                "Checkpoint keys mismatch: " f"missing={missing[:5]} extra={extra[:5]}"
            )

        for key, tensor in model_state.items():
            if not torch.is_tensor(tensor):
                continue

            if tensor.is_floating_point() or tensor.is_complex():
                floating_dtypes.setdefault(key, tensor.dtype)
                value = tensor.detach().to(dtype=torch.float64)
                if key not in averaged:
                    averaged[key] = value.clone()
                else:
                    averaged[key].add_(value)
            elif checkpoint_index == 0:
                averaged[key] = tensor.detach().clone()

    divisor = float(len(input_paths))
    for key, tensor in averaged.items():
        if tensor.is_floating_point() or tensor.is_complex():
            averaged[key] = (tensor / divisor).to(dtype=floating_dtypes[key])

    return averaged


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("checkpoints/base_en_de"),
        help="Directory containing checkpoints (*.pt)",
    )
    parser.add_argument(
        "--n-last",
        type=int,
        default=5,
        help="Number of latest checkpoints to average",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("checkpoints/base_en_de/averaged.pt"),
        help="Output path for averaged checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_last <= 0:
        raise ValueError("--n-last must be > 0")

    checkpoint_paths = _latest_checkpoints(args.checkpoint_dir, args.n_last)
    averaged_model = average_checkpoints(checkpoint_paths)

    reference_payload = torch.load(checkpoint_paths[-1], map_location="cpu")
    if isinstance(reference_payload, dict) and "model" in reference_payload:
        output_payload = dict(reference_payload)
        output_payload["model"] = averaged_model
        output_payload.setdefault("meta", {})
        output_payload["meta"]["averaged_from"] = [
            str(path) for path in checkpoint_paths
        ]
    else:
        output_payload = averaged_model

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_payload, args.output)

    print(f"Averaged {len(checkpoint_paths)} checkpoints")
    for path in checkpoint_paths:
        print(f" - {path}")
    print(f"Saved averaged checkpoint to: {args.output}")


if __name__ == "__main__":
    main()
