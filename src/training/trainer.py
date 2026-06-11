
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
import mlflow

from src.data.dataset import create_src_mask, create_tgt_mask

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
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.accum_steps = max(1, int(accum_steps))
        self._accum_counter = 0
        self.use_amp = use_amp and self.device.type == "cuda" and torch.cuda.is_available()
        self.amp_dtype = self._resolve_amp_dtype(amp_dtype)
        self.use_grad_scaler = self.use_amp and self.amp_dtype == torch.float16
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_grad_scaler)
        self.use_mlflow = use_mlflow
        self.state = TrainerState()
        self.model.to(self.device)

    def _resolve_amp_dtype(self, amp_dtype: str) -> torch.dtype | None:
        if not self.use_amp:
            return None
        dtype = str(amp_dtype).lower().strip()
        bf16_supported = torch.cuda.is_bf16_supported()
        if dtype == "auto":
            return torch.bfloat16 if bf16_supported else torch.float16
        if dtype in {"bf16", "bfloat16"}:
            if not bf16_supported:
                warnings.warn("bf16 not supported, falling back to fp16", RuntimeWarning)
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
            target_key = next((k for k in ("labels", "label", "target", "targets", "y") if k in batch), None)
            if target_key is None:
                raise KeyError("batch dict missing target key (labels, target, etc.)")
            return {k: v for k, v in batch.items() if k != target_key}, batch[target_key]
        raise TypeError("batch must be tuple(input, target) or dict with target key")

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output
        if isinstance(model_output, dict):
            for key in ("logits", "output", "outputs"):
                if key in model_output and isinstance(model_output[key], torch.Tensor):
                    return model_output[key]
        if isinstance(model_output, (tuple, list)) and isinstance(model_output[0], torch.Tensor):
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

        return total_norm_sq ** 0.5

    def _current_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]

    def _log_metrics(self, loss: float, grad_norm: float | None = None) -> None:
        if not self.use_mlflow:
            return
        metrics = {
            "train/loss": loss,
            "train/lr": self._current_lr(),
        }
        if grad_norm is not None:
            metrics["train/grad_norm"] = grad_norm
        mlflow.log_metrics(metrics, step=self.state.global_step)

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
        if self.use_mlflow:
            mlflow.log_metric("val/loss", avg_loss, step=self.state.global_step)
        return avg_loss

    def save_checkpoint(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler is not None else None,
            "state": {
                "global_step": self.state.global_step,
                "epoch": self.state.epoch,
            },
        }
        torch.save(payload, path)

    def load_checkpoint(self, path: str | Path, map_location: str | None = None) -> None:
        checkpoint = torch.load(path, map_location=map_location or str(self.device))
        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        if self.scheduler is not None and checkpoint.get("scheduler") is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler"])

        state = checkpoint.get("state", {})
        self.state.global_step = int(state.get("global_step", 0))
        self.state.epoch = int(state.get("epoch", 0))


class TranslationTrainer(Trainer):
    """Seq2Seq trainer for Transformer translation with teacher forcing."""

    def _unpack_batch(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
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

    def _logits_and_targets(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        src_tokens, tgt_tokens = self._unpack_batch(batch)

        tgt_in = tgt_tokens[:, :-1]
        labels = tgt_tokens[:, 1:]

        src_mask = create_src_mask(src_tokens).to(self.device)
        tgt_mask = create_tgt_mask(tgt_in).to(self.device)

        logits = self.model(src_tokens, tgt_in, src_mask=src_mask, tgt_mask=tgt_mask)
        return logits, labels
