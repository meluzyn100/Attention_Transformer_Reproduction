
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


@dataclass
class TrainerState:
    global_step: int = 0
    epoch: int = 0


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        device: str | torch.device = "cpu",
        grad_clip_norm: float | None = None,
        use_mlflow: bool = True,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.grad_clip_norm = grad_clip_norm
        self.use_mlflow = use_mlflow and (mlflow is not None)
        self.state = TrainerState()

        self.model.to(self.device)

    def _move_to_device(self, value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {k: self._move_to_device(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            moved = [self._move_to_device(v) for v in value]
            return type(value)(moved) if isinstance(value, tuple) else moved
        return value

    def _split_batch(self, batch: Any) -> tuple[Any, torch.Tensor]:
        if isinstance(batch, dict):
            target_key = None
            for key in ("labels", "label", "target", "targets", "y"):
                if key in batch:
                    target_key = key
                    break
            if target_key is None:
                raise KeyError("Could not infer target key from batch dict")
            target = batch[target_key]
            model_inputs = {k: v for k, v in batch.items() if k != target_key}
            return model_inputs, target

        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            return batch[0], batch[1]

        raise TypeError(
            "Unsupported batch format. Use dict with labels/target or tuple(input, target)."
        )

    def _extract_logits(self, model_output: Any) -> torch.Tensor:
        if isinstance(model_output, torch.Tensor):
            return model_output
        if isinstance(model_output, dict):
            for key in ("logits", "output", "outputs"):
                if key in model_output and isinstance(model_output[key], torch.Tensor):
                    return model_output[key]
            raise KeyError("Could not find logits tensor in model output dict")
        if isinstance(model_output, (list, tuple)) and len(model_output) > 0:
            if isinstance(model_output[0], torch.Tensor):
                return model_output[0]
        raise TypeError("Unsupported model output format for logits extraction")

    def _forward(self, inputs: Any) -> torch.Tensor:
        if isinstance(inputs, dict):
            output = self.model(**inputs)
        elif isinstance(inputs, (list, tuple)):
            output = self.model(*inputs)
        else:
            output = self.model(inputs)
        return self._extract_logits(output)

    def _compute_grad_norm(self) -> float:
        norms = []
        for param in self.model.parameters():
            if param.grad is not None:
                norms.append(param.grad.detach().norm(2))
        if not norms:
            return 0.0
        total_norm = torch.norm(torch.stack(norms), 2)
        return float(total_norm.item())

    def _current_lr(self) -> float:
        return float(self.optimizer.param_groups[0]["lr"])

    def _log_metrics(self, loss: float, grad_norm: float | None = None) -> None:
        if not self.use_mlflow:
            return
        metrics = {
            "train/loss": float(loss),
            "train/lr": self._current_lr(),
        }
        if grad_norm is not None:
            metrics["train/grad_norm"] = float(grad_norm)
        mlflow.log_metrics(metrics, step=self.state.global_step)

    def train_step(self, batch: Any) -> tuple[float, float]:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        batch = self._move_to_device(batch)
        model_inputs, targets = self._split_batch(batch)
        logits = self._forward(model_inputs)

        loss = self.criterion(logits, targets)
        loss.backward()

        grad_norm = self._compute_grad_norm()
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self.state.global_step += 1
        self._log_metrics(loss=float(loss.item()), grad_norm=grad_norm)
        return float(loss.item()), float(grad_norm)

    @torch.no_grad()
    def validate(self, dataloader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        running_loss = 0.0
        n_batches = 0

        for batch in dataloader:
            batch = self._move_to_device(batch)
            model_inputs, targets = self._split_batch(batch)
            logits = self._forward(model_inputs)
            loss = self.criterion(logits, targets)
            running_loss += float(loss.item())
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
