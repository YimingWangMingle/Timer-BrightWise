from __future__ import annotations

import json
import os
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


@dataclass(frozen=True, slots=True)
class TrainingState:
    global_step: int
    consumed_samples: int
    sampler_state: dict[str, int]


def _rng_state() -> dict[str, object]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "name": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _restore_rng(state: dict[str, object]) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["name"],
            numpy_state["keys"].numpy(),
            numpy_state["position"],
            numpy_state["has_gauss"],
            numpy_state["cached_gaussian"],
        )
    )
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available() and state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    state: TrainingState,
    config_snapshots: dict[str, object],
    manifest_checksum: str,
    environment: dict[str, object],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(
        {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "training_state": asdict(state),
            "rng": _rng_state(),
            "config_snapshots": config_snapshots,
            "manifest_checksum": manifest_checksum,
            "environment": {key: str(value) for key, value in environment.items()},
        },
        temporary,
    )
    os.replace(temporary, destination)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    expected_config_snapshots: dict[str, object],
    expected_manifest_checksum: str,
) -> TrainingState:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format_version")
    if payload["config_snapshots"] != expected_config_snapshots:
        raise ValueError("checkpoint configuration mismatch")
    if payload["manifest_checksum"] != expected_manifest_checksum:
        raise ValueError("checkpoint manifest checksum mismatch")
    model.load_state_dict(payload["model"])
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    _restore_rng(payload["rng"])
    return TrainingState(**payload["training_state"])


class CheckpointManager:
    def __init__(self, output_dir: str | Path, keep_latest: int = 2) -> None:
        if keep_latest <= 0:
            raise ValueError("keep_latest must be positive")
        self.output_dir = Path(output_dir)
        self.keep_latest = keep_latest
        self.metric_path = self.output_dir / "best-metric.json"
        self.best_metric = float("inf")
        if self.metric_path.is_file():
            self.best_metric = float(
                json.loads(self.metric_path.read_text(encoding="utf-8"))["metric"]
            )

    def save(self, step: int, validation_metric: float, **components) -> Path:
        step_path = self.output_dir / f"step-{step:06d}.pt"
        save_training_checkpoint(step_path, **components)
        if validation_metric < self.best_metric:
            best_path = self.output_dir / "best.pt"
            temporary = self.output_dir / ".best.pt.tmp"
            shutil.copyfile(step_path, temporary)
            os.replace(temporary, best_path)
            self.best_metric = validation_metric
            metric_tmp = self.output_dir / ".best-metric.json.tmp"
            metric_tmp.write_text(
                json.dumps({"metric": validation_metric}) + "\n",
                encoding="utf-8",
            )
            os.replace(metric_tmp, self.metric_path)
        step_paths = sorted(self.output_dir.glob("step-*.pt"))
        for obsolete in step_paths[: -self.keep_latest]:
            obsolete.unlink()
        return step_path


def save_diagnostic(
    path: str | Path,
    *,
    step: int,
    loss: float,
    source_ids: list[str],
    record_ids: list[str],
    gradient_norm: float | None,
    config_paths: dict[str, str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(
        {
            "step": step,
            "loss": loss,
            "source_ids": source_ids,
            "record_ids": record_ids,
            "gradient_norm": gradient_norm,
            "config_paths": config_paths,
        },
        temporary,
    )
    os.replace(temporary, destination)


def save_checkpoint(
    path: str | Path, model: nn.Module, optimizer: Optimizer, step: int
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    torch.save(
        {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
        temporary,
    )
    os.replace(temporary, destination)


def load_checkpoint(
    path: str | Path, model: nn.Module, optimizer: Optimizer
) -> int:
    checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["step"])
