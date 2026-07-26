from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    total_steps: int
    warmup_steps: int
    peak_lr: float
    minimum_lr_ratio: float = 0.1
    micro_batch_size: int = 32
    gradient_accumulation_steps: int = 8
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    gradient_clip: float = 1.0
    validation_interval: int = 250
    checkpoint_interval: int = 250
    logging_interval: int = 10
    seed: int = 2026
    num_workers: int = 8
    prefetch_factor: int = 2
    pin_memory: bool = True
    context_patches: int = 30
    precision: str = "bf16"

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 <= self.warmup_steps <= self.total_steps:
            raise ValueError("warmup_steps must be in [0, total_steps]")
        if self.peak_lr <= 0:
            raise ValueError("peak_lr must be positive")
        if not 0 < self.minimum_lr_ratio <= 1:
            raise ValueError("minimum_lr_ratio must be in (0, 1]")
        if self.validation_interval <= 0:
            raise ValueError("validation_interval must be positive")
        if self.checkpoint_interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        if min(
            self.micro_batch_size,
            self.gradient_accumulation_steps,
            self.gradient_clip,
            self.num_workers,
            self.prefetch_factor,
            self.context_patches,
            self.logging_interval,
        ) <= 0:
            raise ValueError("batch, loader, context, and clipping values must be positive")
        if self.precision != "bf16":
            raise ValueError("production validation precision must be bf16")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
