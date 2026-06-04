"""Training entrypoint for the Attention Is All You Need Transformer.

Usage:
    python train.py                          # full training
    python train.py smoke.enabled=true       # smoke test (overfit 1 batch)
    python train.py training.device=cpu      # force CPU
    python train.py data.max_train_samples=1000  # small subset run
"""

import json
import random
import gc
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow.tracking import MlflowClient

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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _flatten_dict(data: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in data.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_dict(value, prefix=full_key))
        elif isinstance(value, list):
            flattened[full_key] = ",".join(str(item) for item in value)
        elif value is None:
            flattened[full_key] = "null"
        else:
            flattened[full_key] = str(value)
    return flattened


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
    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.num_workers)
    max_train = cfg.data.max_train_samples
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
    val_jsonl = cfg.data.val_jsonl
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

    vocab_size = int(cfg.model.vocab_size)
    smoothing = float(cfg.training.label_smoothing)
    ignore_index = cfg.training.ignore_index
    criterion = LabelSmoothingCrossEntropyLoss(
        vocab_size=vocab_size,
        smoothing=smoothing,
        ignore_index=ignore_index,
    )

    scheduler = None
    if bool(cfg.scheduler.use_noam):
        d_model = int(cfg.scheduler.d_model)
        warmup = int(cfg.scheduler.warmup_steps)
        scheduler = get_noam_scheduler(optimizer, d_model=d_model, warmup_steps=warmup)

    device = cfg.training.device
    grad_clip = cfg.training.grad_clip_norm

    return TranslationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        grad_clip_norm=float(grad_clip) if grad_clip is not None else None,
        use_mlflow=bool(cfg.mlflow.enabled),
    )


# ---------------------------------------------------------------------------
# MLflow helpers
# ---------------------------------------------------------------------------

def _init_mlflow(cfg: DictConfig) -> bool:
    if not bool(cfg.mlflow.enabled):
        return False

    project_root = Path(get_original_cwd())
    mlflow_base_dir = project_root / str(cfg.mlflow.base_dir)
    mlflow_base_dir.mkdir(parents=True, exist_ok=True)

    uri = cfg.mlflow.tracking_uri
    if uri is None:
        uri = f"sqlite:///{(mlflow_base_dir / 'mlflow.db').as_posix()}"
    if uri:
        mlflow.set_tracking_uri(uri)

    artifact_root = mlflow_base_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    exp = cfg.mlflow.experiment_name
    if exp:
        client = MlflowClient()
        experiment = client.get_experiment_by_name(exp)
        if experiment is None:
            client.create_experiment(exp, artifact_location=artifact_root.as_uri())
        mlflow.set_experiment(exp)
    mlflow.start_run(run_name=cfg.mlflow.run_name)

    tags = cfg.mlflow.tags or {}
    if tags:
        mlflow.set_tags({str(key): str(value) for key, value in tags.items()})

    params = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_params(_flatten_dict(params))
    return True


def _close_mlflow(active: bool) -> None:
    if active:
        mlflow.end_run()


def _log_checkpoint_artifact(cfg: DictConfig, checkpoint_path: Path) -> None:
    if not bool(cfg.mlflow.enabled):
        return
    if not bool(cfg.mlflow.log_checkpoints):
        return
    if checkpoint_path.exists():
        mlflow.log_artifact(str(checkpoint_path), artifact_path="checkpoints")


def _release_runtime_memory(*objects: Any) -> None:
    # Drop strong references first so Python/CUDA allocators can reclaim memory.
    for obj in objects:
        del obj

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# Training modes
# ---------------------------------------------------------------------------

def _run_smoke_test(
    trainer: TranslationTrainer,
    train_loader: DataLoader,
    cfg: DictConfig,
) -> None:
    smoke_steps = int(cfg.smoke.steps)
    batch = next(iter(train_loader))

    print(f"[smoke] overfitting 1 batch for {smoke_steps} steps")
    progress = tqdm(range(smoke_steps), desc="smoke", unit="step")
    for step in progress:
        loss, grad_norm = trainer.train_step(batch)
        progress.set_postfix(
            loss=f"{loss:.4f}",
            grad_norm=f"{grad_norm:.3f}",
            lr=f"{trainer._current_lr():.2e}",
        )
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
    epochs = int(cfg.training.epochs)
    validate_every = int(cfg.training.validate_every)
    save_every = int(cfg.training.save_every)
    log_every = int(cfg.training.log_every)
    ckpt_dir = Path(cfg.training.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(trainer.state.epoch, epochs):
        trainer.state.epoch = epoch
        running_loss = 0.0
        n_batches = 0

        train_progress = tqdm(
            train_loader,
            desc=f"epoch {epoch + 1}/{epochs}",
            unit="batch",
            leave=False,
        )
        for batch in train_progress:
            loss, grad_norm = trainer.train_step(batch)
            running_loss += loss
            n_batches += 1
            train_progress.set_postfix(
                loss=f"{loss:.4f}",
                grad_norm=f"{grad_norm:.3f}",
                lr=f"{trainer._current_lr():.2e}",
            )
            if n_batches % log_every == 0:
                print(
                    f"epoch={epoch + 1}  step={trainer.state.global_step}  "
                    f"loss={loss:.6f}  grad_norm={grad_norm:.4f}  "
                    f"lr={trainer._current_lr():.2e}"
                )

        avg_loss = running_loss / max(1, n_batches)
        print(f"epoch={epoch + 1}  train_loss={avg_loss:.6f}")
        if trainer.use_mlflow:
            mlflow.log_metric("train/epoch_loss", avg_loss, step=trainer.state.global_step)

        if val_loader is not None and (epoch + 1) % validate_every == 0:
            val_loss = trainer.validate(val_loader)
            print(f"epoch={epoch + 1}  val_loss={val_loss:.6f}")

        if (epoch + 1) % save_every == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch + 1:03d}.pt"
            trainer.save_checkpoint(ckpt_path)
            _log_checkpoint_artifact(cfg, ckpt_path)
            print(f"saved checkpoint: {ckpt_path}")


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.training.seed))

    tokenizer = None
    train_loader = None
    val_loader = None
    trainer = None
    mlflow_active = False
    try:
        tokenizer = SharedBPETokenizer.load(
            cfg.data.tokenizer_vocab,
            cfg.data.tokenizer_merges,
        )

        train_loader, val_loader = _build_dataloaders(cfg, tokenizer)
        trainer = _build_trainer(cfg, train_loader, val_loader)

        smoke_enabled = bool(cfg.smoke.enabled)
        print(
            f"Model params: {sum(p.numel() for p in trainer.model.parameters()):,}  "
            f"device={trainer.device}  smoke={smoke_enabled}"
        )

        mlflow_active = _init_mlflow(cfg)
        if smoke_enabled:
            _run_smoke_test(trainer, train_loader, cfg)
            smoke_ckpt = Path(cfg.smoke.checkpoint_path)
            trainer.save_checkpoint(smoke_ckpt)
            _log_checkpoint_artifact(cfg, smoke_ckpt)
            print(f"saved smoke checkpoint: {smoke_ckpt}")
        else:
            _run_training_loop(trainer, train_loader, val_loader, cfg)
    except KeyboardInterrupt:
        print("Training interrupted (Ctrl+C). Releasing CUDA memory cache...")
        raise SystemExit(130)
    finally:
        _close_mlflow(mlflow_active)
        _release_runtime_memory(trainer, train_loader, val_loader, tokenizer)


if __name__ == "__main__":
    main()

