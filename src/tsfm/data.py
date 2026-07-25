from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass(slots=True)
class NormalizedBatch:
    context: torch.Tensor
    target: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor


class SyntheticTimeSeriesDataset(Dataset[torch.Tensor]):
    def __init__(self, num_samples: int, total_length: int, seed: int) -> None:
        if num_samples <= 0:
            raise ValueError("num_samples must be positive")
        if total_length <= 1:
            raise ValueError("total_length must be greater than one")
        self.num_samples = num_samples
        self.total_length = total_length
        self.seed = seed

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> torch.Tensor:
        if not 0 <= index < self.num_samples:
            raise IndexError(index)
        generator = torch.Generator().manual_seed(self.seed + 1_000_003 * index)
        time = torch.linspace(0.0, 1.0, self.total_length, dtype=torch.float32)
        amplitude = 0.5 + 1.5 * torch.rand((), generator=generator)
        cycles = 0.5 + 4.5 * torch.rand((), generator=generator)
        phase = 2.0 * torch.pi * torch.rand((), generator=generator)
        seasonal = amplitude * torch.sin(2.0 * torch.pi * cycles * time + phase)
        slope = 2.0 * torch.rand((), generator=generator) - 1.0
        intercept = torch.rand((), generator=generator) - 0.5
        trend = intercept + slope * time
        coefficient = 0.2 + 0.65 * torch.rand((), generator=generator)
        innovations = 0.15 * torch.randn(self.total_length, generator=generator)
        autoregressive = torch.empty(self.total_length, dtype=torch.float32)
        autoregressive[0] = innovations[0]
        for position in range(1, self.total_length):
            autoregressive[position] = (
                coefficient * autoregressive[position - 1] + innovations[position]
            )
        return seasonal + trend + autoregressive


def normalize_context_target(
    context: torch.Tensor, target: torch.Tensor
) -> NormalizedBatch:
    if context.ndim != 2 or target.ndim != 2:
        raise ValueError("context and target must have shape [batch, raw_length]")
    if context.shape[0] != target.shape[0]:
        raise ValueError("context and target batch sizes must match")
    if context.shape[-1] < 2:
        raise ValueError("context must contain at least two values")
    mean = context.mean(dim=-1, keepdim=True)
    scale = torch.sqrt(
        context.var(dim=-1, unbiased=False, keepdim=True) + 1e-5
    )
    return NormalizedBatch(
        context=(context - mean) / scale,
        target=(target - mean) / scale,
        mean=mean,
        scale=scale,
    )
