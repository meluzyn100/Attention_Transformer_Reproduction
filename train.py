"""Training entrypoint for the Attention Is All You Need Transformer.

Usage:
    python train.py                          # full training
    python train.py smoke.enabled=true       # smoke test (overfit 1 batch)
    python train.py training.device=cpu      # force CPU
    python train.py data.max_train_samples=1000  # small subset run
"""
from __future__ import annotations

import json
import random
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.data import (
    SharedBPETokenizer,
    TranslationDataset,
    collate_fn,
)
from src.training import (
    LabelSmoothingCrossEntropyLoss,
    get_noam_scheduler,
)
from src.training.translation_trainer import TranslationTrainer

try:
    import mlflow
except Exception:  # pragma: no cover
    mlflow = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cfg(cfg: DictConfig, key: str, default: Any = None) -> Any:
    return OmegaConf.select(cfg, key, default=default)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_jsonl(path: str | Path, max_samples: int | None = None) -> tuple[list[str], list[str]]:
    src_texts, tgt_texts = [], []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_samples is not None and i >= max_samples:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            t = record["translation"]
            src_texts.append(t["de"])
            tgt_texts.append(t["en"])
    return src_texts, tgt_texts


def _build_dataloaders(cfg: DictConfig, tokenizer: SharedBPETokenizer) -> tuple[DataLoader, DataLoader | None]:
    batch_size = int(_cfg(cfg, "data.batch_size", 32))
    num_workers = int(_cfg(cfg, "data.num_workers", 0))
    max_train = _cfg(cfg, "data.max_train_samples", None)
    if max_train is not None:
        max_train = int(max_train)

    train_src, train_tgt = _load_jsonl(cfg.data.train_jsonl, max_samples=max_train)
    train_ds = TranslationDataset(train_src, train_tgt, tokenizer)

    _collate = partial(collate_fn, pad_id=tokenizer.pad_id)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=_collate,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    val_jsonl = _cfg(cfg, "data.val_jsonl", None)
    if val_jsonl is not None and Path(val_jsonl).exists():
        val_src, val_tgt = _load_jsonl(val_jsonl)
        val_ds = TranslationDataset(val_src, val_tgt, tokenizer)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
        )

    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Trainer assembly
# ---------------------------------------------------------------------------

def _build_trainer(
    cfg: DictConfig,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
) -> TranslationTrainer:
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    vocab_size = int(_cfg(cfg, "model.vocab_size", 32000))
    smoothing = float(_cfg(cfg, "training.label_smoothing", 0.1))
    ignore_index = _cfg(cfg, "training.ignore_index", None)
    criterion = LabelSmoothingCrossEntropyLoss(
        vocab_size=vocab_size,
        smoothing=smoothing,
        ignore_index=ignore_index,
    )

    scheduler = None
    if bool(_cfg(cfg, "scheduler.use_noam", True)):
        d_model = int(_cfg(cfg, "scheduler.d_model", _cfg(cfg, "model.d_model", 512)))
        warmup = int(_cfg(cfg, "scheduler.warmup_steps", 4000))
        scheduler = get_noam_scheduler(optimizer, d_model=d_model, warmup_steps=warmup)

    device = _cfg(cfg, "training.device", "cuda" if torch.cuda.is_available() else "cpu")
    grad_clip = _cfg(cfg, "training.grad_clip_norm", None)

    return TranslationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        grad_clip_norm=float(grad_clip) if grad_clip is not None else None,
        use_mlflow=bool(_cfg(cfg, "mlflow.enabled", False)),
    )


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _init_mlflow(cfg: DictConfig) -> bool:
    if not bool(_cfg(cfg, "mlflow.enabled", False)) or mlflow is None:
        return False
    uri = _cfg(cfg, "mlflow.tracking_uri", None)
    if uri:
        mlflow.set_tracking_uri(uri)
    exp = _cfg(cfg, "mlflow.experiment_name", None)
    if exp:
        mlflow.set_experiment(exp)
    mlflow.start_run(run_name=_cfg(cfg, "mlflow.run_name", None))
    mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
    return True


def _close_mlflow(active: bool) -> None:
    if active and mlflow is not None:
        mlflow.end_run()


# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------

def _run_smoke_test(
    trainer: TranslationTrainer,
    train_loader: DataLoader,
    cfg: DictConfig,
) -> None:
    smoke_steps = int(_cfg(cfg, "smoke.steps", 50))
    batch = next(iter(train_loader))

    print(f"[smoke] overfitting 1 batch for {smoke_steps} steps")
    for step in range(smoke_steps):
        loss, grad_norm = trainer.train_step(batch)
        if step == 0 or (step + 1) % 10 == 0:
            print(
                f"[smoke] step={step + 1:>3}/{smoke_steps}  "
                f"loss={loss:.6f}  grad_norm={grad_norm:.4f}  "
                f"lr={trainer._current_lr():.2e}"
            )


def _run_training_loop(
    trainer: TranslationTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
) -> None:
    epochs = int(_cfg(cfg, "training.epochs", 10))
    validate_every = int(_cfg(cfg, "training.validate_every", 1))
    save_every = int(_cfg(cfg, "training.save_every", 1))
    log_every = int(_cfg(cfg, "training.log_every", 100))
    ckpt_dir = Path(_cfg(cfg, "training.checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(trainer.state.epoch, epochs):
        trainer.state.epoch = epoch
        running_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            loss, grad_norm = trainer.train_step(batch)
            running_loss += loss
            n_batches += 1
            if n_batches % log_every == 0:
                print(
                    f"epoch={epoch + 1}  step={trainer.state.global_step}  "
                    f"loss={loss:.6f}  grad_norm={grad_norm:.4f}  "
                    f"lr={trainer._current_lr():.2e}"
                )

        avg_loss = running_loss / max(1, n_batches)
        print(f"epoch={epoch + 1}  train_loss={avg_loss:.6f}")
        if mlflow is not None and trainer.use_mlflow:
            mlflow.log_metric("train/epoch_loss", avg_loss, step=trainer.state.global_step)

        if val_loader is not None and (epoch + 1) % validate_every == 0:
            val_loss = trainer.validate(val_loader)
            print(f"epoch={epoch + 1}  val_loss={val_loss:.6f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
            trainer.save_checkpoint(ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    set_seed(int(_cfg(cfg, "training.seed", 42)))

    tokenizer = SharedBPETokenizer.load(
        cfg.data.tokenizer_vocab,
        cfg.data.tokenizer_merges,
    )

    train_loader, val_loader = _build_dataloaders(cfg, tokenizer)
    trainer = _build_trainer(cfg, train_loader, val_loader)

    smoke_enabled = bool(_cfg(cfg, "smoke.enabled", False))
    print(
        f"Model params: {sum(p.numel() for p in trainer.model.parameters()):,}  "
        f"device={trainer.device}  smoke={smoke_enabled}"
    )

    mlflow_active = _init_mlflow(cfg)
    try:
        if smoke_enabled:
            _run_smoke_test(trainer, train_loader, cfg)
            smoke_ckpt = Path(_cfg(cfg, "smoke.checkpoint_path", "checkpoints/smoke.pt"))
            trainer.save_checkpoint(smoke_ckpt)
            print(f"saved smoke checkpoint: {smoke_ckpt}")
        else:
            _run_training_loop(trainer, train_loader, val_loader, cfg)
    finally:
        _close_mlflow(mlflow_active)


if __name__ == "__main__":
    main()

