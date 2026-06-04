
import torch


def get_noam_scheduler(
    optimizer: torch.optim.Optimizer,
    d_model: int,
    warmup_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Create the original Transformer Noam LR schedule.

    Returned lambda is step-based (optimizer step index), not epoch-based.
    """

    if d_model <= 0:
        raise ValueError("d_model must be positive")
    if warmup_steps <= 0:
        raise ValueError("warmup_steps must be positive")

    factor = d_model ** -0.5

    def lr_lambda(step: int) -> float:
        step = max(1, int(step))
        return factor * min(step ** -0.5, step * (warmup_steps ** -1.5))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
