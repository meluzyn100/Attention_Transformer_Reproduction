"""Training entrypoint for the Attention Is All You Need Transformer.

Usage:
    python train.py                          # full training
    python train.py --config-name smoke      # smoke test
    python train.py training.device=cpu      # force CPU
"""

import gc
import json
import math
import os
import random
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

from src.data import SharedBPETokenizer, TranslationDataset, collate_fn
from src.training import LabelSmoothingCrossEntropyLoss, TranslationTrainer, get_noam_scheduler

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_config(cfg: dict[str, Any], prefix: str = "") -> dict[str, str]:
    """Recursively flatten nested config dict."""
    result = {}
    for key, val in cfg.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            result.update(flatten_config(val, full_key))
        elif isinstance(val, list):
            result[full_key] = ",".join(str(v) for v in val)
        elif val is None:
            result[full_key] = "null"
        else:
            result[full_key] = str(val)
    return result


def load_jsonl(path: str | Path, max_samples: int | None = None) -> tuple[list[str], list[str]]:
    """Load source-target pairs from JSONL file."""
    src, tgt = [], []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples and i >= max_samples:
                break
            if not line.strip():
                continue
            record = json.loads(line)["translation"]
            src.append(record["de"])
            tgt.append(record["en"])
    return src, tgt


def build_dataloaders(cfg: DictConfig, tokenizer: SharedBPETokenizer) -> tuple[DataLoader, DataLoader | None]:
    """Build train and optional validation dataloaders."""
    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.num_workers)
    pin_memory = cfg.data.pin_memory
    persistent_workers = cfg.data.persistent_workers and num_workers > 0
    multiprocessing_context = cfg.data.multiprocessing_context if num_workers > 0 else None
    prefetch_factor = int(cfg.data.prefetch_factor) if num_workers > 0 else None
    max_train_samples = cfg.data.max_train_samples and int(cfg.data.max_train_samples)

    train_src, train_tgt = load_jsonl(cfg.data.train_jsonl, max_train_samples)
    train_ds = TranslationDataset(train_src, train_tgt, tokenizer)
    collate_batch = partial(collate_fn, pad_id=tokenizer.pad_id, max_length=cfg.model.max_len)
    
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        multiprocessing_context=multiprocessing_context,
        prefetch_factor=prefetch_factor,
    )

    val_loader = None
    if cfg.data.val_jsonl and Path(cfg.data.val_jsonl).exists():
        val_src, val_tgt = load_jsonl(cfg.data.val_jsonl)
        val_ds = TranslationDataset(val_src, val_tgt, tokenizer)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_batch,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            multiprocessing_context=multiprocessing_context,
            prefetch_factor=prefetch_factor,
        )
    return train_loader, val_loader


def build_trainer(cfg: DictConfig) -> TranslationTrainer:
    """Build trainer with model, optimizer, and loss."""
    model = instantiate(cfg.model)
    optimizer = instantiate(cfg.optimizer, params=model.parameters())
    criterion = LabelSmoothingCrossEntropyLoss(
        vocab_size=int(cfg.model.vocab_size),
        smoothing=float(cfg.training.label_smoothing),
        ignore_index=cfg.training.ignore_index,
    )

    scheduler = None
    if cfg.scheduler.use_noam:
        scheduler = get_noam_scheduler(
            optimizer,
            d_model=int(cfg.scheduler.d_model),
            warmup_steps=int(cfg.scheduler.warmup_steps),
            lr_scale=float(cfg.scheduler.lr_scale),
        )

    return TranslationTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=cfg.training.device,
        grad_clip_norm=float(cfg.training.grad_clip_norm) if cfg.training.grad_clip_norm else None,
        accum_steps=int(cfg.training.get("accum_steps", 1)),
        use_amp=cfg.training.use_amp,
        amp_dtype=str(cfg.training.get("amp_dtype", "auto")),
        use_mlflow=cfg.mlflow.enabled,
    )


def init_mlflow(cfg: DictConfig) -> bool:
    """Initialize MLflow tracking."""
    if not cfg.mlflow.enabled:
        return False

    base_dir = Path(get_original_cwd()) / cfg.mlflow.base_dir
    base_dir.mkdir(parents=True, exist_ok=True)

    uri = cfg.mlflow.tracking_uri or f"sqlite:///{(base_dir / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(uri)

    artifact_root = base_dir / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    if cfg.mlflow.experiment_name:
        client = MlflowClient()
        exp = client.get_experiment_by_name(cfg.mlflow.experiment_name)
        if not exp:
            client.create_experiment(cfg.mlflow.experiment_name, artifact_location=artifact_root.as_uri())
        mlflow.set_experiment(cfg.mlflow.experiment_name)

    mlflow.start_run(run_name=cfg.mlflow.run_name)
    if cfg.mlflow.tags:
        mlflow.set_tags({str(k): str(v) for k, v in cfg.mlflow.tags.items()})

    params = OmegaConf.to_container(cfg, resolve=True)
    mlflow.log_params(flatten_config(params))
    return True


def close_mlflow(active: bool) -> None:
    if active:
        mlflow.end_run()


def log_checkpoint(cfg: DictConfig, path: Path) -> None:
    if cfg.mlflow.enabled and cfg.mlflow.log_checkpoints and path.exists():
        mlflow.log_artifact(str(path), artifact_path="checkpoints")


def cleanup_memory(*objects: Any) -> None:
    for _ in objects:
        del _
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_smoke(trainer: TranslationTrainer, train_loader: DataLoader, cfg: DictConfig) -> None:
    """Overfit on 1 batch for smoke test."""
    steps = int(cfg.smoke.steps)
    batch = next((b for b in train_loader if b), None)
    if not batch:
        raise RuntimeError("no batches found")

    print(f"[smoke] overfitting 1 batch for {steps} steps")
    pbar = tqdm(range(steps), desc="smoke", unit="step")
    for step in pbar:
        loss, grad_norm, _ = trainer.train_step(batch)
        pbar.set_postfix(loss=f"{loss:.4f}", grad_norm=f"{grad_norm:.3f}", lr=f"{trainer._current_lr():.2e}")
        if step == 0 or (step + 1) % 10 == 0:
            print(f"[smoke] step={step+1:>3}/{steps}  loss={loss:.6f}  grad_norm={grad_norm:.4f}  lr={trainer._current_lr():.2e}")


def run_training(trainer: TranslationTrainer, train_loader: DataLoader, val_loader: DataLoader | None, cfg: DictConfig) -> None:
    """Main training loop."""
    epochs = int(cfg.training.epochs)
    validate_every = int(cfg.training.validate_every)
    save_every = int(cfg.training.save_every)
    log_every = int(cfg.training.log_every)
    checkpoint_every_steps = int(cfg.training.get("checkpoint_every_steps", 0))
    max_steps = cfg.training.get("max_steps") and int(cfg.training.max_steps)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    last_saved_step = -1
    for epoch in range(trainer.state.epoch, epochs):
        if max_steps and trainer.state.global_step >= max_steps:
            break

        trainer.state.epoch = epoch
        running_loss = 0.0
        num_batches = 0
        skipped_non_finite = 0
        skipped_since_backoff = 0
        max_skipped = int(cfg.training.max_skipped_steps_per_epoch)
        backoff_trigger = int(cfg.training.skip_backoff_trigger)
        backoff_factor = float(cfg.training.skip_backoff_factor)
        min_lr = float(cfg.training.min_lr)

        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{epochs}", unit="batch", leave=False)
        for batch in pbar:
            if not batch:
                continue
            loss, grad_norm, stepped = trainer.train_step(batch)

            if not (math.isfinite(loss) and math.isfinite(grad_norm)):
                skipped_non_finite += 1
                skipped_since_backoff += 1
                pbar.set_postfix(skipped=skipped_non_finite, lr=f"{trainer._current_lr():.2e}")
                if skipped_non_finite <= 5 or skipped_non_finite % 20 == 0:
                    print(f"epoch={epoch+1} step={trainer.state.global_step} skipped (loss={loss}, grad_norm={grad_norm})")
                if skipped_non_finite >= max_skipped:
                    raise RuntimeError(f"too many skipped: {skipped_non_finite} >= {max_skipped}")

                if skipped_since_backoff >= backoff_trigger:
                    old_lr = trainer._current_lr()
                    new_lr = max(min_lr, old_lr * backoff_factor)
                    for param_group in trainer.optimizer.param_groups:
                        param_group["lr"] = max(min_lr, param_group["lr"] * backoff_factor)
                    if trainer.scheduler and hasattr(trainer.scheduler, "base_lrs"):
                        trainer.scheduler.base_lrs = [max(min_lr, base * backoff_factor) for base in trainer.scheduler.base_lrs]
                    print(f"epoch={epoch+1} step={trainer.state.global_step} LR backoff: {old_lr:.2e} -> {new_lr:.2e}")
                    if trainer.use_mlflow:
                        mlflow.log_metric("train/adaptive_lr_backoff", new_lr, step=trainer.state.global_step)
                    skipped_since_backoff = 0
                continue

            running_loss += loss
            num_batches += 1
            pbar.set_postfix(loss=f"{loss:.4f}", grad_norm=f"{grad_norm:.3f}", lr=f"{trainer._current_lr():.2e}", skipped=skipped_non_finite)

            if stepped and checkpoint_every_steps > 0 and trainer.state.global_step % checkpoint_every_steps == 0:
                checkpoint_path = checkpoint_dir / f"step_{trainer.state.global_step:06d}.pt"
                trainer.save_checkpoint(checkpoint_path)
                log_checkpoint(cfg, checkpoint_path)
                last_saved_step = trainer.state.global_step
                print(f"saved checkpoint: {checkpoint_path}")

            if num_batches % log_every == 0:
                print(f"epoch={epoch+1} step={trainer.state.global_step} loss={loss:.6f} grad_norm={grad_norm:.4f} lr={trainer._current_lr():.2e}")

            if max_steps and trainer.state.global_step >= max_steps:
                break

        avg_loss = running_loss / max(1, num_batches)
        print(f"epoch={epoch+1} train_loss={avg_loss:.6f} finite={num_batches} skipped={skipped_non_finite}")
        if trainer.use_mlflow:
            mlflow.log_metric("train/epoch_loss", avg_loss, step=trainer.state.global_step)
            mlflow.log_metric("train/skipped_non_finite", skipped_non_finite, step=trainer.state.global_step)

        if val_loader and (epoch + 1) % validate_every == 0:
            val_loss = trainer.validate(val_loader)
            print(f"epoch={epoch+1} val_loss={val_loss:.6f}")

        if (epoch + 1) % save_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch+1:03d}.pt"
            trainer.save_checkpoint(checkpoint_path)
            log_checkpoint(cfg, checkpoint_path)
            print(f"saved checkpoint: {checkpoint_path}")

        if max_steps and trainer.state.global_step >= max_steps:
            break

    if max_steps and trainer.state.global_step != last_saved_step:
        checkpoint_path = checkpoint_dir / f"step_{trainer.state.global_step:06d}.pt"
        trainer.save_checkpoint(checkpoint_path)
        log_checkpoint(cfg, checkpoint_path)
        print(f"saved final checkpoint: {checkpoint_path}")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    set_seed(int(cfg.training.seed))
    tokenizer = train_loader = val_loader = trainer = None
    mlflow_ok = False
    try:
        tokenizer = SharedBPETokenizer.load(cfg.data.tokenizer_vocab, cfg.data.tokenizer_merges)
        train_loader, val_loader = build_dataloaders(cfg, tokenizer)
        trainer = build_trainer(cfg)
        print(f"Model: {sum(p.numel() for p in trainer.model.parameters()):,} params  device={trainer.device}  smoke={cfg.smoke.enabled}")
        
        mlflow_ok = init_mlflow(cfg)
        if cfg.smoke.enabled:
            run_smoke(trainer, train_loader, cfg)
            path = Path(cfg.smoke.checkpoint_path)
            trainer.save_checkpoint(path)
            log_checkpoint(cfg, path)
            print(f"saved smoke checkpoint: {path}")
        else:
            run_training(trainer, train_loader, val_loader, cfg)
    except KeyboardInterrupt:
        print("Training interrupted (Ctrl+C).")
        raise SystemExit(130)
    finally:
        close_mlflow(mlflow_ok)
        cleanup_memory(trainer, train_loader, val_loader, tokenizer)


if __name__ == "__main__":
    main()
