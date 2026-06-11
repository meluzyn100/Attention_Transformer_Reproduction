from __future__ import annotations

import datetime
import os

import torch
import torch.distributed as dist


def get_rank() -> int:
    return int(os.environ.get("RANK", "0"))


def get_world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_rank0() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl", timeout_minutes: int = 30) -> bool:
    world_size = get_world_size()
    has_torchrun_env = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if world_size <= 1 and not has_torchrun_env:
        return False

    if is_distributed():
        return True

    kwargs = {
        "backend": backend,
        "timeout": datetime.timedelta(minutes=int(timeout_minutes)),
    }
    if backend == "nccl" and torch.cuda.is_available():
        kwargs["device_id"] = torch.device("cuda", get_local_rank())

    dist.init_process_group(**kwargs)
    return True


def cleanup_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_mean(value: float, device: torch.device) -> float:
    if not is_distributed():
        return float(value)
    t = torch.tensor(float(value), device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    t /= get_world_size()
    return float(t.item())
