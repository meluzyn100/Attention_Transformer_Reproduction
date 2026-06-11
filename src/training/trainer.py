import math
import os
import subprocess
import random
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
import mlflow

from src.data.dataset import create_src_mask, create_tgt_mask

try:
    import pynvml
except Exception:  # pragma: no cover - optional dependency
    pynvml = None


@dataclass
class TrainerState:
    global_step: int = 0
    epoch: int = 0
    skipped_steps: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: str | torch.device = "cpu",
        grad_clip_norm: float | None = None,
        accum_steps: int = 1,
        use_amp: bool = False,
        amp_dtype: str = "auto",
        use_mlflow: bool = True,
        rank: int = 0,
        world_size: int = 1,
        gpu_monitoring: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.accum_steps = max(1, int(accum_steps))
        self._accum_counter = 0
        self.use_amp = (
            use_amp and self.device.type == "cuda" and torch.cuda.is_available()
        )
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        self.use_mlflow = use_mlflow
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.gpu_monitoring = gpu_monitoring and self.device.type == "cuda"
        self._nvml_handle = None
        self.state = TrainerState()
        self.model.to(self.device)

        if self.gpu_monitoring and pynvml is not None:
            try:
                pynvml.nvmlInit()
                index = torch.cuda.current_device() if torch.cuda.is_available() else 0
                self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            except Exception:  # pragma: no cover - best effort
                self._nvml_handle = None

    def _resolve_amp_dtype(self, amp_dtype: str) -> torch.dtype | None:
        if not self.use_amp:
            return None
        dtype = str(amp_dtype).lower().strip()
        bf16_supported = torch.cuda.is_bf16_supported()
        if dtype == "auto":
            return torch.bfloat16 if bf16_supported else torch.float16
        if dtype in {"bf16", "bfloat16"}:
            if not bf16_supported:
                warnings.warn(
                    "bf16 not supported, falling back to fp16", RuntimeWarning
                )
                return torch.float16
            return torch.bfloat16
        if dtype in {"fp16", "float16", "half"}:
            return torch.float16
        raise ValueError(f"amp_dtype must be auto, bf16, or fp16, got {dtype}")

    def _amp_context(self) -> torch.amp.autocast_mode.autocast:
        return torch.autocast(
            device_type=self.device.type,
            enabled=self.use_amp,
            dtype=self.amp_dtype if self.use_amp else None,
        )

    def _move_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._move_to_device(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            result = [self._move_to_device(v) for v in value]
            return type(value)(result) if isinstance(value, tuple) else result
        return value

    def _split_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            target_key = next(
                (
                    k
                    for k in ("labels", "label", "target", "targets", "y")
                    if k in batch
                ),
                None,
            )
            if target_key is None:
                raise KeyError("batch dict missing target key (labels, target, etc.)")
            return {k: v for k, v in batch.items() if k != target_key}, batch[
                target_key
            ]
        raise TypeError("batch must be tuple(input, target) or dict with target key")

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output
        if isinstance(model_output, dict):
            for key in ("logits", "output", "outputs"):
                if key in model_output and isinstance(model_output[key], torch.Tensor):
                    return model_output[key]
        if isinstance(model_output, (tuple, list)) and isinstance(
            model_output[0], torch.Tensor
        ):
            return model_output[0]
        raise TypeError(f"could not extract logits from {type(model_output)}")

    def _forward(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, dict):
            output = self.model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            output = self.model(*inputs)
        else:
            output = self.model(inputs)
        return self._extract_logits(output)

    def _logits_and_targets(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        model_inputs, targets = self._split_batch(batch)
        logits = self._forward(model_inputs)
        return logits, targets

    def _loss_for_batch(self, batch: Any) -> torch.Tensor:
        logits, targets = self._logits_and_targets(batch)
        return self.criterion(logits, targets)

    def _compute_grad_norm(self) -> float:
        total_norm_sq = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.detach().norm(2).square().item()

        return total_norm_sq**0.5

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _log_metrics(self, loss: float, grad_norm: float | None = None) -> None:
        if not self.use_mlflow or self.rank != 0:
            return
        metrics = {
            "train/loss": loss,
            "train/lr": self._current_lr(),
        }
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm
        mlflow.log_metrics(metrics, step=self.state.global_step)

    def _model_state_dict(self) -> dict[str, Any]:
        module = self.model.module if hasattr(self.model, "module") else self.model
        return module.state_dict()

    def _load_model_state_dict(self, state_dict: dict[str, Any]) -> None:
        module = self.model.module if hasattr(self.model, "module") else self.model
        module.load_state_dict(state_dict)

    def _collect_rng_state(self) -> dict[str, Any]:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": None,
            "cuda_all": None,
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state(self.device)
            state["cuda_all"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, state: dict[str, Any] | None) -> None:
        if not state:
            return
        python_state = state.get("python")
        if python_state is not None:
            try:
                random.setstate(python_state)
            except Exception:
                warnings.warn("failed to restore python RNG state; continuing")
        numpy_state = state.get("numpy")
        if numpy_state is not None:
            try:
                np.random.set_state(numpy_state)
            except Exception:
                warnings.warn("failed to restore numpy RNG state; continuing")
        torch_state = state.get("torch")
        if torch_state is not None:
            try:
                if not isinstance(torch_state, torch.Tensor):
                    torch_state = torch.tensor(torch_state, dtype=torch.uint8)
                elif torch_state.dtype != torch.uint8:
                    torch_state = torch_state.to(dtype=torch.uint8)
                torch.set_rng_state(torch_state.cpu())
            except Exception:
                warnings.warn("failed to restore torch RNG state; continuing")
        if self.device.type == "cuda" and torch.cuda.is_available():
            cuda_state = state.get("cuda")
            if cuda_state is not None:
                try:
                    if not isinstance(cuda_state, torch.Tensor):
                        cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
                    elif cuda_state.dtype != torch.uint8:
                        cuda_state = cuda_state.to(dtype=torch.uint8)
                    torch.cuda.set_rng_state(cuda_state.cpu(), self.device)
                except Exception:
                    warnings.warn("failed to restore CUDA RNG state; continuing")
            cuda_all = state.get("cuda_all")
            if cuda_all is not None:
                try:
                    converted = []
                    for entry in cuda_all:
                        if not isinstance(entry, torch.Tensor):
                            entry = torch.tensor(entry, dtype=torch.uint8)
                        elif entry.dtype != torch.uint8:
                            entry = entry.to(dtype=torch.uint8)
                        converted.append(entry.cpu())
                    torch.cuda.set_rng_state_all(converted)
                except Exception:
                    warnings.warn("failed to restore CUDA(all) RNG state; continuing")

    def get_performance_metrics(
        self,
        *,
        samples: int,
        tokens: int,
        step_time_s: float,
        data_time_s: float,
    ) -> dict[str, float]:
        metrics = {
            "perf/step_time_s": float(step_time_s),
            "perf/data_time_s": float(data_time_s),
            "perf/samples_per_s": float(samples / max(step_time_s, 1.0e-8)),
            "perf/tokens_per_s": float(tokens / max(step_time_s, 1.0e-8)),
        }
        if self.device.type == "cuda" and torch.cuda.is_available():
            metrics["gpu/memory_allocated_mb"] = float(
                torch.cuda.memory_allocated(self.device) / (1024**2)
            )
            metrics["gpu/memory_reserved_mb"] = float(
                torch.cuda.memory_reserved(self.device) / (1024**2)
            )
            metrics["gpu/max_memory_allocated_mb"] = float(
                torch.cuda.max_memory_allocated(self.device) / (1024**2)
            )
            util = None
            if self._nvml_handle is not None and pynvml is not None:
                try:
                    util = float(
                        pynvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu
                    )
                except Exception:  # pragma: no cover - best effort
                    util = None
            elif self.gpu_monitoring:
                try:
                    output = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    )
                    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
                    rows = [
                        line.strip() for line in output.splitlines() if line.strip()
                    ]
                    if rows:
                        util = float(rows[min(local_rank, len(rows) - 1)])
                except Exception:  # pragma: no cover - best effort
                    util = None
            if util is not None:
                metrics["gpu/utilization_pct"] = util
        return metrics

    def _optimizer_step(self, raw_loss_value: float) -> tuple[float, float, bool]:
        self._accum_counter += 1
        if self._accum_counter < self.accum_steps:
            return raw_loss_value, 0.0, False

        if self.use_grad_scaler:
            self.scaler.unscale_(self.optimizer)

        grad_norm = self._compute_grad_norm()
        if not math.isfinite(grad_norm):
            self.state.skipped_steps += 1
            self._accum_counter = 0
            self.optimizer.zero_grad(set_to_none=True)
            if self.use_grad_scaler:
                self.scaler.update()
            return raw_loss_value, math.nan, False

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        optimizer_stepped = True
        if self.use_grad_scaler:
            previous_scale = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            optimizer_stepped = self.scaler.get_scale() >= previous_scale
        else:
            self.optimizer.step()

        self._accum_counter = 0

        if self.scheduler is not None and optimizer_stepped:
            self.scheduler.step()
        if not optimizer_stepped:
            self.state.skipped_steps += 1
            return raw_loss_value, grad_norm, False

        self.state.global_step += 1
        self._log_metrics(loss=raw_loss_value, grad_norm=grad_norm)
        return raw_loss_value, grad_norm, True

    def train_step(self, batch: Any) -> tuple[float, float, bool]:
        self.model.train()
        if self._accum_counter == 0:
            self.optimizer.zero_grad(set_to_none=True)

        batch = self._move_to_device(batch)
        with self._amp_context():
            loss = self._loss_for_batch(batch)

        if not torch.isfinite(loss):
            self.state.skipped_steps += 1
            self._accum_counter = 0
            self.optimizer.zero_grad(set_to_none=True)
            return math.nan, math.nan, False

        raw_loss_value = loss.item()
        loss = loss / self.accum_steps
        if self.use_grad_scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        return self._optimizer_step(raw_loss_value)

    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            if batch is None:
                continue
            batch = self._move_to_device(batch)
            with self._amp_context():
                loss = self._loss_for_batch(batch)
            running_loss += loss.item()
            n_batches += 1

        avg_loss = running_loss / max(1, n_batches)
        if self.use_mlflow and self.rank == 0:
            mlflow.log_metric("val/loss", avg_loss, step=self.state.global_step)
        return avg_loss

    def save_checkpoint(self, path: str | Path) -> None:
        if self.rank != 0:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self._model_state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "scaler": self.scaler.state_dict() if self.use_grad_scaler else None,
            "state": {
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
                "skipped_steps": self.state.skipped_steps,
                "accum_counter": self._accum_counter,
            },
            "rng_state": self._collect_rng_state(),
        }
        torch.save(payload, path)

    def load_checkpoint(
        self, path: str | Path, map_location: str | None = None
    ) -> None:
        checkpoint = torch.load(
            path,
            map_location=map_location or str(self.device),
            weights_only=False,
        )
        self._load_model_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])
        if self.use_grad_scaler and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])

        state = checkpoint.get("state", {})
        self.state.global_step = int(state.get("global_step", 0))
        self.state.epoch = int(state.get("epoch", 0))
        self.state.skipped_steps = int(state.get("skipped_steps", 0))
        self._accum_counter = int(state.get("accum_counter", 0))
        self._restore_rng_state(checkpoint.get("rng_state"))


class TranslationTrainer(Trainer):
    """Seq2Seq trainer for Transformer translation with teacher forcing."""

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]
        if isinstance(batch, dict):
            src = batch.get("src") or batch.get("source")
            tgt = batch.get("tgt") or batch.get("target") or batch.get("labels")
            if src is None or tgt is None:
                raise KeyError(
                    "Batch dict must contain 'src'/'source' and 'tgt'/'target' keys"
                )
            return src, tgt
        raise TypeError(
            f"Unsupported batch type {type(batch)}. "
            "Expected (src, tgt) tuple or dict with src/tgt keys."
        )

    def _logits_and_targets(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        src_tokens, tgt_tokens = self._unpack_batch(batch)

        tgt_in = tgt_tokens[:, :-1]
        labels = tgt_tokens[:, 1:]

        src_mask = create_src_mask(src_tokens).to(self.device)
        tgt_mask = create_tgt_mask(tgt_in).to(self.device)

        logits = self.model(src_tokens, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
        return logits, labels
