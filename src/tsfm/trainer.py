from __future__ import annotations

import contextlib
import json
import os
import platform
import random
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tsfm.checkpoint import CheckpointManager, TrainingState, load_training_checkpoint, save_diagnostic
from tsfm.optim import build_optimizer, build_scheduler
from tsfm.s3.sampling import CounterSampler
from tsfm.train_config import TrainingConfig


@dataclass(slots=True)
class TrainingReport:
    losses: list[float]
    checkpoint_path: Path | None
    optimizer_steps: int
    micro_batches: int
    loss_dtype: str
    consumed_samples: int
    source_counts: dict[str, int]


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def environment_report(device: torch.device) -> dict[str, object]:
    report = {"python": platform.python_version(), "torch": str(torch.__version__), "device": str(device), "cuda_runtime": str(torch.version.cuda)}
    if device.type == "cuda": report["gpu"] = torch.cuda.get_device_name(device)
    return report


def _autocast(device: torch.device, precision: str):
    if device.type == "cuda" and precision == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _loader(dataset: Dataset, config: TrainingConfig, consumed: int, rank: int, world_size: int) -> DataLoader:
    return DataLoader(dataset, batch_size=config.micro_batch_size, sampler=CounterSampler(consumed, rank, world_size), num_workers=config.num_workers, pin_memory=config.pin_memory, prefetch_factor=config.prefetch_factor, persistent_workers=False)


def bounded_batch_probe(model: torch.nn.Module, batch: dict[str, object], device: torch.device, precision: str = "bf16") -> dict[str, float]:
    model = model.to(device).train(); context = batch["context"].to(device); target = batch["target"].to(device)
    if device.type == "cuda": torch.cuda.reset_peak_memory_stats(device)
    model.zero_grad(set_to_none=True)
    with _autocast(device, precision): loss = model(context, labels=target).loss
    if loss is None or not torch.isfinite(loss): raise FloatingPointError("bounded probe produced non-finite loss")
    loss.backward()
    peak = float(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0.0
    model.zero_grad(set_to_none=True)
    return {"loss": float(loss.detach()), "peak_allocated_bytes": peak}


def _distributed_mean(value: float, device: torch.device, world_size: int) -> float:
    if world_size == 1 or not torch.distributed.is_initialized(): return value
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    return float(tensor / world_size)


def run_training(*, model: torch.nn.Module, dataset: Dataset, training_config: TrainingConfig, output_dir: str | Path, manifest_checksum: str, config_snapshots: dict[str, object], device: torch.device, resume: str | Path | None = None, total_steps_override: int | None = None, rank: int = 0, world_size: int = 1, is_main_process: bool = True) -> TrainingReport:
    target_steps = total_steps_override or training_config.total_steps
    if not 0 < target_steps <= training_config.total_steps: raise ValueError("requested steps must be within configured total_steps")
    output = Path(output_dir)
    if is_main_process: output.mkdir(parents=True, exist_ok=True)
    model = model.to(device); optimizer = build_optimizer(model.parameters(), training_config); scheduler = build_scheduler(optimizer, training_config)
    state = TrainingState(0, 0, {"next_sample": 0})
    if resume is not None:
        state = load_training_checkpoint(resume, model=model, optimizer=optimizer, scheduler=scheduler, expected_config_snapshots=config_snapshots, expected_manifest_checksum=manifest_checksum)
    if state.global_step >= target_steps: raise ValueError("checkpoint is already at or beyond requested steps")
    iterator = iter(_loader(dataset, training_config, state.consumed_samples, rank, world_size))
    manager = CheckpointManager(output, keep_latest=2) if is_main_process else None
    losses: list[float] = []; source_counts: Counter[str] = Counter(); micro_batches = 0; last_path: Path | None = None; loss_dtype = ""; consumed = state.consumed_samples
    started = time.perf_counter(); model.train()
    for step in range(state.global_step + 1, target_steps + 1):
        optimizer.zero_grad(set_to_none=True); accumulated = 0.0; last_batch = None
        for _ in range(training_config.gradient_accumulation_steps):
            batch = next(iterator); last_batch = batch; context = batch["context"].to(device, non_blocking=True); target = batch["target"].to(device, non_blocking=True)
            with _autocast(device, training_config.precision): loss = model(context, labels=target).loss
            if loss is None or not torch.isfinite(loss):
                if is_main_process: save_diagnostic(output / "diagnostic.pt", step=step, loss=float("nan") if loss is None else float(loss.detach()), source_ids=list(batch["source_id"]), record_ids=list(batch["record_id"]), gradient_norm=None, config_paths={})
                raise FloatingPointError(f"non-finite loss at step {step}; record_id={batch['record_id'][0]}")
            loss_dtype = str(loss.dtype); (loss / training_config.gradient_accumulation_steps).backward(); accumulated += float(loss.detach()); micro_batches += 1
            consumed += context.shape[0] * world_size; source_counts.update(str(item) for item in batch["source_id"])
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.gradient_clip)
        if not torch.isfinite(grad_norm):
            if is_main_process and last_batch is not None: save_diagnostic(output / "diagnostic.pt", step=step, loss=accumulated, source_ids=list(last_batch["source_id"]), record_ids=list(last_batch["record_id"]), gradient_norm=float(grad_norm), config_paths={})
            raise FloatingPointError(f"non-finite gradients at step {step}")
        optimizer.step(); scheduler.step()
        average_loss = _distributed_mean(accumulated / training_config.gradient_accumulation_steps, device, world_size); losses.append(average_loss)
        current_state = TrainingState(step, consumed, {"next_sample": consumed})
        if is_main_process and manager is not None and (step % training_config.checkpoint_interval == 0 or step == target_steps):
            last_path = manager.save(step=step, validation_metric=average_loss, model=model, optimizer=optimizer, scheduler=scheduler, state=current_state, config_snapshots=config_snapshots, manifest_checksum=manifest_checksum, environment=environment_report(device))
        if world_size > 1 and torch.distributed.is_initialized(): torch.distributed.barrier()
    if is_main_process:
        assert last_path is not None
        report = {"elapsed_seconds": time.perf_counter()-started, "losses": losses, "consumed_samples": consumed, "source_counts": dict(source_counts), "environment": environment_report(device)}
        temporary = output / ".training-report.json.tmp"; temporary.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8"); os.replace(temporary, output / "training-report.json")
    return TrainingReport(losses, last_path, target_steps-state.global_step, micro_batches, loss_dtype, consumed, dict(source_counts))
