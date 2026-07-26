from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from tsfm.train_config import TrainingConfig

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ProductionTemplate:
    coverage_cycles: int
    global_batch_size: int
    micro_batch_candidates: tuple[int, ...]
    peak_lr: float
    minimum_lr_ratio: float
    warmup_ratio: float
    weight_decay: float
    beta1: float
    beta2: float
    gradient_clip: float
    validation_interval: int
    checkpoint_interval: int
    logging_interval: int
    seed: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    context_patches: int
    precision: str

    def __post_init__(self) -> None:
        if self.coverage_cycles <= 0 or self.global_batch_size <= 0:
            raise ValueError("coverage and global batch must be positive")
        if not self.micro_batch_candidates or any(
            value <= 0 for value in self.micro_batch_candidates
        ):
            raise ValueError("micro-batch candidates must be positive")
        if len(set(self.micro_batch_candidates)) != len(self.micro_batch_candidates):
            raise ValueError("micro-batch candidates must be unique")
        if self.peak_lr <= 0 or not 0 < self.minimum_lr_ratio <= 1:
            raise ValueError("learning-rate settings are invalid")
        if not 0 < self.warmup_ratio <= 1:
            raise ValueError("warmup_ratio must be in (0, 1]")
        if min(
            self.validation_interval,
            self.checkpoint_interval,
            self.logging_interval,
            self.num_workers,
            self.prefetch_factor,
            self.context_patches,
        ) <= 0:
            raise ValueError("interval and loader settings must be positive")
        if self.precision != "bf16":
            raise ValueError("production precision must be bf16")

    @classmethod
    def from_json(cls, path: str | Path) -> "ProductionTemplate":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        values["micro_batch_candidates"] = tuple(values["micro_batch_candidates"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class ResolvedTrainingPlan:
    coverage_cycles: int
    window_count: int
    world_size: int
    global_batch_size: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    total_real_samples: int
    total_padded_samples: int
    total_steps: int
    warmup_steps: int
    peak_lr: float
    minimum_lr_ratio: float
    weight_decay: float
    beta1: float
    beta2: float
    gradient_clip: float
    validation_interval: int
    checkpoint_interval: int
    logging_interval: int
    seed: int
    num_workers: int
    prefetch_factor: int
    pin_memory: bool
    context_patches: int
    precision: str
    digests: dict[str, str]

    @classmethod
    def resolve(
        cls,
        template: ProductionTemplate,
        *,
        window_count: int,
        micro_batch_size: int,
        world_size: int,
        digests: dict[str, str],
    ) -> "ResolvedTrainingPlan":
        if window_count <= 0 or world_size <= 0:
            raise ValueError("window_count and world_size must be positive")
        if micro_batch_size not in template.micro_batch_candidates:
            raise ValueError("micro-batch is not an approved candidate")
        denominator = micro_batch_size * world_size
        if template.global_batch_size % denominator:
            raise ValueError("global batch is not divisible by micro-batch times world size")
        for name, digest in digests.items():
            if not name or _SHA256_PATTERN.fullmatch(digest) is None:
                raise ValueError(f"invalid digest binding: {name}")

        total_real = window_count * template.coverage_cycles
        total_steps = math.ceil(total_real / template.global_batch_size)
        total_positions = total_steps * template.global_batch_size
        return cls(
            coverage_cycles=template.coverage_cycles,
            window_count=window_count,
            world_size=world_size,
            global_batch_size=template.global_batch_size,
            micro_batch_size=micro_batch_size,
            gradient_accumulation_steps=template.global_batch_size // denominator,
            total_real_samples=total_real,
            total_padded_samples=total_positions - total_real,
            total_steps=total_steps,
            warmup_steps=max(1, math.ceil(total_steps * template.warmup_ratio)),
            peak_lr=template.peak_lr,
            minimum_lr_ratio=template.minimum_lr_ratio,
            weight_decay=template.weight_decay,
            beta1=template.beta1,
            beta2=template.beta2,
            gradient_clip=template.gradient_clip,
            validation_interval=template.validation_interval,
            checkpoint_interval=template.checkpoint_interval,
            logging_interval=template.logging_interval,
            seed=template.seed,
            num_workers=template.num_workers,
            prefetch_factor=template.prefetch_factor,
            pin_memory=template.pin_memory,
            context_patches=template.context_patches,
            precision=template.precision,
            digests=dict(sorted(digests.items())),
        )

    def training_config(self) -> TrainingConfig:
        return TrainingConfig(
            total_steps=self.total_steps,
            warmup_steps=self.warmup_steps,
            peak_lr=self.peak_lr,
            minimum_lr_ratio=self.minimum_lr_ratio,
            micro_batch_size=self.micro_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            weight_decay=self.weight_decay,
            beta1=self.beta1,
            beta2=self.beta2,
            gradient_clip=self.gradient_clip,
            validation_interval=self.validation_interval,
            checkpoint_interval=self.checkpoint_interval,
            logging_interval=self.logging_interval,
            seed=self.seed,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            context_patches=self.context_patches,
            precision=self.precision,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "ResolvedTrainingPlan":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise ValueError("resolved training plan must be a JSON object")
        plan = cls(**document)
        if plan.to_dict() != document:
            raise ValueError("resolved training plan is not canonical")
        return plan

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def write_atomic(self, destination: str | Path) -> str:
        destination = Path(destination)
        payload = (
            json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
        return hashlib.sha256(payload).hexdigest()
