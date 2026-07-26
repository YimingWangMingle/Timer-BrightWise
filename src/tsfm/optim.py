from __future__ import annotations

import math
from collections.abc import Iterable

import torch

from tsfm.train_config import TrainingConfig


def lr_multiplier(step: int, config: TrainingConfig) -> float:
    if step < 0:
        raise ValueError("step must be non-negative")
    if step < config.warmup_steps:
        return (step + 1) / max(1, config.warmup_steps)
    if config.total_steps == config.warmup_steps:
        return config.minimum_lr_ratio
    progress = (step - config.warmup_steps) / max(
        1, config.total_steps - config.warmup_steps - 1
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    return config.minimum_lr_ratio + (1.0 - config.minimum_lr_ratio) * cosine


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter], config: TrainingConfig
) -> torch.optim.AdamW:
    values = list(parameters)
    kwargs = {
        "lr": config.peak_lr,
        "betas": (config.beta1, config.beta2),
        "weight_decay": config.weight_decay,
    }
    if any(parameter.is_cuda for parameter in values):
        try:
            return torch.optim.AdamW(values, **kwargs, fused=True)
        except (RuntimeError, TypeError):
            pass
    return torch.optim.AdamW(
        values,
        lr=config.peak_lr,
        betas=(config.beta1, config.beta2),
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: TrainingConfig
) -> torch.optim.lr_scheduler.LambdaLR:
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, config)
    )
