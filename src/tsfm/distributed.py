from __future__ import annotations

import os
from datetime import timedelta
from dataclasses import dataclass

import torch
from torch.nn.parallel import DistributedDataParallel


@dataclass(frozen=True, slots=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    is_main_process: bool

    @property
    def enabled(self) -> bool:
        return self.world_size > 1


def distributed_context_from_environment(
    *, initialize: bool = True
) -> DistributedContext:
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    if world_size <= 0 or not 0 <= rank < world_size:
        raise ValueError("invalid RANK/WORLD_SIZE environment")
    if local_rank < 0:
        raise ValueError("LOCAL_RANK must be non-negative")
    if initialize and world_size > 1:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL distributed training requires CUDA")
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", timeout=timedelta(minutes=10)
        )
    return DistributedContext(rank, local_rank, world_size, rank == 0)


def wrap_model(
    model: torch.nn.Module, context: DistributedContext
) -> torch.nn.Module:
    if not context.enabled:
        return model
    return DistributedDataParallel(
        model,
        device_ids=[context.local_rank],
        gradient_as_bucket_view=True,
    )


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    current = model
    while hasattr(current, "module") and isinstance(current.module, torch.nn.Module):
        current = current.module
    return current


def gather_objects(
    value: object, context: DistributedContext
) -> list[object] | None:
    if not context.enabled:
        return [value]
    gathered = [None] * context.world_size if context.is_main_process else None
    torch.distributed.gather_object(value, gathered, dst=0)
    return gathered


def destroy_distributed(context: DistributedContext) -> None:
    if context.enabled and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


def reduce_mean(value: float, context: DistributedContext) -> float:
    if not context.enabled:
        return value
    tensor = torch.tensor(value, device=torch.device("cuda", context.local_rank))
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor / context.world_size)


def barrier(context: DistributedContext) -> None:
    if context.enabled:
        torch.distributed.barrier()
