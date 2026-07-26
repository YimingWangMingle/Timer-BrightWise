from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset

from tsfm.trainer import TrainingReport, run_training
from tsfm.validation import evaluate_model


def _write_validation(path: Path, entry: dict[str, object]) -> None:
    existing = path.read_text(encoding="utf-8") if path.is_file() else ""
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(existing + json.dumps(entry, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def primary_validation_metric(views: dict[str, dict[str, float]]) -> float:
    required = {"val_heldout", "val_temporal"}
    if set(views) != required:
        raise ValueError("both validation views are required for primary metric")
    return sum(views[name]["normalized_mse"] for name in sorted(required)) / 2.0


def run_validated_training(
    *, model: torch.nn.Module, train_dataset: Dataset,
    validation_datasets: dict[str, Dataset], training_config,
    output_dir: str | Path, manifest_checksum: str,
    config_snapshots: dict[str, object], device: torch.device,
    resume: str | Path | None = None, validation_batches: int = 4,
    rank: int = 0, world_size: int = 1, is_main_process: bool = True,
) -> TrainingReport:
    if set(validation_datasets) != {"val_heldout", "val_temporal"}:
        raise ValueError("both val_heldout and val_temporal datasets are required")
    if training_config.validation_interval != training_config.checkpoint_interval:
        raise ValueError("validation_interval must equal checkpoint_interval")
    output = Path(output_dir)
    current_resume = Path(resume) if resume is not None else None
    current_step = 0
    if current_resume is not None:
        payload = torch.load(current_resume, map_location="cpu", weights_only=True)
        current_step = int(payload["training_state"]["global_step"])
    best_metric = float("inf")
    best_state = output / "validation-best.json"
    if best_state.is_file(): best_metric = float(json.loads(best_state.read_text())["metric"])
    combined_losses: list[float] = []; total_micro = 0; total_optimizer = 0; final_report = None
    while current_step < training_config.total_steps:
        endpoint = min(current_step + training_config.validation_interval, training_config.total_steps)
        report = run_training(model=model, dataset=train_dataset, training_config=training_config, output_dir=output, manifest_checksum=manifest_checksum, config_snapshots=config_snapshots, device=device, resume=current_resume, total_steps_override=endpoint, rank=rank, world_size=world_size, is_main_process=is_main_process)
        combined_losses.extend(report.losses); total_micro += report.micro_batches; total_optimizer += report.optimizer_steps
        current_step = endpoint; current_resume = output / f"step-{endpoint:06d}.pt"
        if world_size > 1 and torch.distributed.is_initialized(): torch.distributed.barrier()
        views = {name: evaluate_model(model, dataset, device=device, batch_size=training_config.micro_batch_size, batches=validation_batches, precision=training_config.precision, rank=rank, world_size=world_size) for name, dataset in validation_datasets.items()}
        metric = primary_validation_metric(views)
        if is_main_process:
            entry = {"step": endpoint, "validation_normalized_mse": metric, "views": views}
            output.mkdir(parents=True, exist_ok=True); _write_validation(output / "validation-report.jsonl", entry)
            if metric < best_metric:
                temporary = output / ".best.pt.tmp"; shutil.copyfile(current_resume, temporary); os.replace(temporary, output / "best.pt")
                state_tmp = output / ".validation-best.json.tmp"; state_tmp.write_text(json.dumps({"metric": metric, "step": endpoint})+"\n", encoding="utf-8"); os.replace(state_tmp, best_state); best_metric = metric
        if world_size > 1 and torch.distributed.is_initialized(): torch.distributed.barrier()
        final_report = report
    assert final_report is not None
    final_report.losses = combined_losses; final_report.micro_batches = total_micro; final_report.optimizer_steps = total_optimizer
    return final_report
