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
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DistributedSampler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import hydra
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
import mlflow
from mlflow.tracking import MlflowClient

from src.data import SharedBPETokenizer, TranslationDataset, collate_fn
from src.distributed import (
    all_reduce_mean,
    barrier,
    cleanup_distributed,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed,
    is_distributed,
    is_rank0,
)
from src.training import (
    LabelSmoothingCrossEntropyLoss,
    TranslationTrainer,
    get_noam_scheduler,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def set_seed(seed: int, rank: int = 0) -> None:
    seed = int(seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(cfg: DictConfig) -> str:
    requested = str(cfg.training.device)
    if not is_distributed() or requested == "cpu":
        return requested
    local_rank = get_local_rank()
    if requested.startswith("cuda"):
        torch.cuda.set_device(local_rank)
        return f"cuda:{local_rank}"
    return requested


def find_latest_checkpoint(checkpoint_dir: str | Path) -> Path | None:
    path = Path(checkpoint_dir)
    if not path.exists():
        return None
    preferred = []
    for pattern in ("step_*.pt", "epoch_*.pt"):
        preferred.extend(path.glob(pattern))
    if preferred:
        preferred = sorted(preferred, key=lambda p: p.stat().st_mtime)
        return preferred[-1]

    checkpoints = sorted(path.glob("*.pt"), key=lambda p: p.stat().st_mtime)
    return checkpoints[-1] if checkpoints else None


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


def load_jsonl(
    path: str | Path, max_samples: int | None = None
) -> tuple[list[str], list[str]]:
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


def build_dataloaders(
    cfg: DictConfig, tokenizer: SharedBPETokenizer
) -> tuple[DataLoader, DataLoader | None, DistributedSampler | None]:
    """Build train and optional validation dataloaders."""
    batch_size = int(cfg.data.batch_size)
    num_workers = int(cfg.data.num_workers)
    pin_memory = cfg.data.pin_memory
    persistent_workers = cfg.data.persistent_workers and num_workers > 0
    multiprocessing_context = (
        cfg.data.multiprocessing_context if num_workers > 0 else None
    )
    prefetch_factor = int(cfg.data.prefetch_factor) if num_workers > 0 else None
    max_train_samples = cfg.data.max_train_samples and int(cfg.data.max_train_samples)
    train_src, train_tgt = load_jsonl(cfg.data.train_jsonl, max_train_samples)
    train_ds = TranslationDataset(train_src, train_tgt, tokenizer)
    train_sampler = None
    if is_distributed():
        train_sampler = DistributedSampler(
            train_ds,
            num_replicas=get_world_size(),
            rank=get_rank(),
            shuffle=True,
            drop_last=False,
        )
    collate_batch = partial(
        collate_fn, pad_id=tokenizer.pad_id, max_length=cfg.model.max_len
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
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
    return train_loader, val_loader, train_sampler


def build_trainer(cfg: DictConfig) -> TranslationTrainer:
    """Build trainer with model, optimizer, and loss."""
    model = instantiate(cfg.model)
    if is_distributed() and get_world_size() > 1:
        if str(cfg.training.device).startswith("cuda"):
            model = DDP(
                model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
                find_unused_parameters=False,
            )
        else:
            model = DDP(model, find_unused_parameters=False)
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
        grad_clip_norm=(
            float(cfg.training.grad_clip_norm) if cfg.training.grad_clip_norm else None
        ),
        accum_steps=int(cfg.training.get("accum_steps", 1)),
        use_amp=cfg.training.use_amp,
        amp_dtype=str(cfg.training.get("amp_dtype", "auto")),
        use_mlflow=cfg.mlflow.enabled and is_rank0(),
        rank=get_rank(),
        world_size=get_world_size(),
        gpu_monitoring=bool(cfg.training.get("gpu_monitoring", True)),
    )


def init_mlflow(cfg: DictConfig) -> bool:
    """Initialize MLflow tracking."""
    if not cfg.mlflow.enabled or not is_rank0():
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
            client.create_experiment(
                cfg.mlflow.experiment_name, artifact_location=artifact_root.as_uri()
            )
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
    if (
        is_rank0()
        and cfg.mlflow.enabled
        and cfg.mlflow.log_checkpoints
        and path.exists()
    ):
        mlflow.log_artifact(str(path), artifact_path="checkpoints")


def cleanup_memory(*objects: Any) -> None:
    for _ in objects:
        del _
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


def run_smoke(
    trainer: TranslationTrainer, train_loader: DataLoader, cfg: DictConfig
) -> None:
    """Overfit on 1 batch for smoke test."""
    steps = int(cfg.smoke.steps)
    batch = next((b for b in train_loader if b), None)
    if not batch:
        raise RuntimeError("no batches found")

    rank0 = is_rank0()
    if rank0:
        print(f"[smoke] overfitting 1 batch for {steps} steps")
    pbar = tqdm(range(steps), desc="smoke", unit="step", disable=not rank0)
    for step in pbar:
        loss, grad_norm, _ = trainer.train_step(batch)
        if rank0:
            pbar.set_postfix(
                loss=f"{loss:.4f}",
                grad_norm=f"{grad_norm:.3f}",
                lr=f"{trainer._current_lr():.2e}",
            )
        if rank0 and (step == 0 or (step + 1) % 10 == 0):
            print(
                f"[smoke] step={step+1:>3}/{steps}  loss={loss:.6f}  grad_norm={grad_norm:.4f}  lr={trainer._current_lr():.2e}"
            )


def run_training(
    trainer: TranslationTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    cfg: DictConfig,
    train_sampler: DistributedSampler | None = None,
) -> None:
    """Main training loop."""
    epochs = int(cfg.training.epochs)
    validate_every = int(cfg.training.validate_every)
    save_every = int(cfg.training.save_every)
    log_every = int(cfg.training.log_every)
    checkpoint_every_steps = int(cfg.training.get("checkpoint_every_steps", 0))
    max_steps = cfg.training.get("max_steps") and int(cfg.training.max_steps)
    checkpoint_dir = Path(cfg.training.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    rank0 = is_rank0()

    last_saved_step = -1
    for epoch in range(trainer.state.epoch, epochs):
        if max_steps and trainer.state.global_step >= max_steps:
            break

        trainer.state.epoch = epoch
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        running_loss = 0.0
        num_batches = 0
        skipped_non_finite = 0
        skipped_since_backoff = 0
        max_skipped = int(cfg.training.max_skipped_steps_per_epoch)
        backoff_trigger = int(cfg.training.skip_backoff_trigger)
        backoff_factor = float(cfg.training.skip_backoff_factor)
        min_lr = float(cfg.training.min_lr)

        pbar = tqdm(
            train_loader,
            desc=f"epoch {epoch+1}/{epochs}",
            unit="batch",
            leave=False,
            disable=not rank0,
        )
        batch_fetch_started = time.perf_counter()
        for batch in pbar:
            data_time_s = time.perf_counter() - batch_fetch_started
            if not batch:
                batch_fetch_started = time.perf_counter()
                continue
            step_started = time.perf_counter()
            loss, grad_norm, stepped = trainer.train_step(batch)
            step_time_s = time.perf_counter() - step_started

            if not (math.isfinite(loss) and math.isfinite(grad_norm)):
                skipped_non_finite += 1
                skipped_since_backoff += 1
                pbar.set_postfix(
                    skipped=skipped_non_finite, lr=f"{trainer._current_lr():.2e}"
                )
                if rank0 and (skipped_non_finite <= 5 or skipped_non_finite % 20 == 0):
                    print(
                        f"epoch={epoch+1} step={trainer.state.global_step} skipped (loss={loss}, grad_norm={grad_norm})"
                    )
                if skipped_non_finite >= max_skipped:
                    raise RuntimeError(
                        f"too many skipped: {skipped_non_finite} >= {max_skipped}"
                    )

                if skipped_since_backoff >= backoff_trigger:
                    old_lr = trainer._current_lr()
                    new_lr = max(min_lr, old_lr * backoff_factor)
                    for param_group in trainer.optimizer.param_groups:
                        param_group["lr"] = max(
                            min_lr, param_group["lr"] * backoff_factor
                        )
                    if trainer.scheduler and hasattr(trainer.scheduler, "base_lrs"):
                        trainer.scheduler.base_lrs = [
                            max(min_lr, base * backoff_factor)
                            for base in trainer.scheduler.base_lrs
                        ]
                    if rank0:
                        print(
                        f"epoch={epoch+1} step={trainer.state.global_step} LR backoff: {old_lr:.2e} -> {new_lr:.2e}"
                        )
                    if trainer.use_mlflow:
                        mlflow.log_metric(
                            "train/adaptive_lr_backoff",
                            new_lr,
                            step=trainer.state.global_step,
                        )
                    skipped_since_backoff = 0
                batch_fetch_started = time.perf_counter()
                continue

            running_loss += loss
            num_batches += 1
            samples = int(batch[0].shape[0]) if isinstance(batch, (tuple, list)) else 0
            tokens = (
                int(batch[0].numel() + batch[1].numel())
                if isinstance(batch, (tuple, list))
                else 0
            )
            perf_metrics = trainer.get_performance_metrics(
                samples=samples,
                tokens=tokens,
                step_time_s=step_time_s,
                data_time_s=data_time_s,
            )

            if rank0:
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    grad_norm=f"{grad_norm:.3f}",
                    lr=f"{trainer._current_lr():.2e}",
                    tok_s=f"{perf_metrics['perf/tokens_per_s']:.0f}",
                    skipped=skipped_non_finite,
                )

            if stepped and trainer.use_mlflow and rank0:
                mlflow.log_metrics(perf_metrics, step=trainer.state.global_step)

            if (
                stepped
                and checkpoint_every_steps > 0
                and trainer.state.global_step % checkpoint_every_steps == 0
            ):
                checkpoint_path = (
                    checkpoint_dir / f"step_{trainer.state.global_step:06d}.pt"
                )
                trainer.save_checkpoint(checkpoint_path)
                barrier()
                log_checkpoint(cfg, checkpoint_path)
                last_saved_step = trainer.state.global_step
                if rank0:
                    print(f"saved checkpoint: {checkpoint_path}")

            if rank0 and num_batches % log_every == 0:
                print(
                    f"epoch={epoch+1} step={trainer.state.global_step} loss={loss:.6f} grad_norm={grad_norm:.4f} lr={trainer._current_lr():.2e}"
                )

            if max_steps and trainer.state.global_step >= max_steps:
                break
            batch_fetch_started = time.perf_counter()

        avg_loss = running_loss / max(1, num_batches)
        avg_loss = all_reduce_mean(avg_loss, trainer.device)
        if rank0:
            print(
                f"epoch={epoch+1} train_loss={avg_loss:.6f} finite={num_batches} skipped={skipped_non_finite}"
            )
        if trainer.use_mlflow:
            mlflow.log_metric(
                "train/epoch_loss", avg_loss, step=trainer.state.global_step
            )
            mlflow.log_metric(
                "train/skipped_non_finite",
                skipped_non_finite,
                step=trainer.state.global_step,
            )

        if val_loader and (epoch + 1) % validate_every == 0:
            val_loss = trainer.validate(val_loader)
            val_loss = all_reduce_mean(val_loss, trainer.device)
            if rank0:
                print(f"epoch={epoch+1} val_loss={val_loss:.6f}")
            if trainer.use_mlflow:
                mlflow.log_metric("val/loss_reduced", val_loss, step=trainer.state.global_step)

        if (epoch + 1) % save_every == 0:
            checkpoint_path = checkpoint_dir / f"epoch_{epoch+1:03d}.pt"
            trainer.save_checkpoint(checkpoint_path)
            barrier()
            log_checkpoint(cfg, checkpoint_path)
            if rank0:
                print(f"saved checkpoint: {checkpoint_path}")

        if max_steps and trainer.state.global_step >= max_steps:
            break

    if max_steps and trainer.state.global_step != last_saved_step:
        checkpoint_path = checkpoint_dir / f"step_{trainer.state.global_step:06d}.pt"
        trainer.save_checkpoint(checkpoint_path)
        barrier()
        log_checkpoint(cfg, checkpoint_path)
        if rank0:
            print(f"saved final checkpoint: {checkpoint_path}")


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    ddp_active = False
    tokenizer = train_loader = val_loader = train_sampler = trainer = None
    mlflow_ok = False
    try:
        ddp_active = bool(cfg.distributed.enabled) and init_distributed(
            backend=str(cfg.distributed.backend),
            timeout_minutes=int(cfg.distributed.timeout_minutes),
        )
        if bool(cfg.distributed.enabled) and not ddp_active and is_rank0():
            print("distributed.enabled=true but no distributed process group found; running single-process")
        cfg.training.device = resolve_device(cfg)
        rank = get_rank()
        set_seed(int(cfg.training.seed), rank=rank)

        tokenizer = SharedBPETokenizer.load(
            cfg.data.tokenizer_vocab, cfg.data.tokenizer_merges
        )
        train_loader, val_loader, train_sampler = build_dataloaders(cfg, tokenizer)
        trainer = build_trainer(cfg)
        if is_rank0():
            print(
                f"Model: {sum(p.numel() for p in trainer.model.parameters()):,} params  device={trainer.device}  smoke={cfg.smoke.enabled}  world_size={get_world_size()}"
            )

        resume_path = cfg.training.get("resume_from")
        should_auto_resume = bool(cfg.training.get("auto_resume_latest", True)) and not bool(cfg.smoke.enabled)
        resume_strict = bool(cfg.training.get("resume_strict", False))
        if should_auto_resume and not resume_path:
            latest = find_latest_checkpoint(cfg.training.checkpoint_dir)
            resume_path = str(latest) if latest else None
        if resume_path:
            try:
                trainer.load_checkpoint(resume_path)
                if is_rank0():
                    print(f"resumed from checkpoint: {resume_path}")
                barrier()
            except RuntimeError as exc:
                if resume_strict:
                    raise
                if is_rank0():
                    short_error = str(exc).splitlines()[0]
                    print(
                        "resume checkpoint incompatible with current model/config; "
                        f"starting from scratch (path={resume_path}, reason={short_error})"
                    )

        mlflow_ok = init_mlflow(cfg)
        if cfg.smoke.enabled:
            run_smoke(trainer, train_loader, cfg)
            path = Path(cfg.smoke.checkpoint_path)
            trainer.save_checkpoint(path)
            barrier()
            log_checkpoint(cfg, path)
            if is_rank0():
                print(f"saved smoke checkpoint: {path}")
        else:
            run_training(trainer, train_loader, val_loader, cfg, train_sampler=train_sampler)
    except KeyboardInterrupt:
        if is_rank0():
            print("Training interrupted (Ctrl+C).")
        raise SystemExit(130)
    finally:
        close_mlflow(mlflow_ok)
        cleanup_memory(trainer, train_loader, val_loader, tokenizer)
        if ddp_active:
            cleanup_distributed()


if __name__ == "__main__":
    main()
