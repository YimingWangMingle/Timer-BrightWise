from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from tsfm.config import TimerConfig
from tsfm.metrics import denormalized_error_metrics
from tsfm.model import TimerModel
from tsfm.train_config import TrainingConfig
from tsfm.trainer import run_training


class DeterministicWindowDataset(Dataset):
    def __len__(self) -> int:
        return 10_000

    def __getitem__(self, index: int):
        base = torch.arange(12, dtype=torch.float32) + index * 0.01
        context = base[:8]
        target = base[4:12]
        mean = context.mean().reshape(1)
        scale = torch.sqrt(context.var(unbiased=False) + 1e-5).reshape(1)
        return {
            "context": (context - mean) / scale,
            "target": (target - mean) / scale,
            "mean": mean,
            "scale": scale,
            "source_id": "synthetic",
            "record_id": f"synthetic/{index}",
            "sample_index": index,
            "window_start": 0,
        }


def model_config() -> TimerConfig:
    return TimerConfig(
        input_token_len=4,
        output_token_len=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
    )


def training_config() -> TrainingConfig:
    return TrainingConfig(
        total_steps=3,
        warmup_steps=1,
        peak_lr=1e-3,
        micro_batch_size=2,
        gradient_accumulation_steps=2,
        validation_interval=1,
        checkpoint_interval=1,
        num_workers=1,
        prefetch_factor=1,
        pin_memory=False,
        context_patches=2,
    )


def run(model, output_dir: Path, *, steps: int, resume: Path | None = None):
    config = training_config()
    return run_training(
        model=model,
        dataset=DeterministicWindowDataset(),
        training_config=config,
        output_dir=output_dir,
        manifest_checksum="d" * 64,
        config_snapshots={
            "model": {"hidden_size": 8},
            "data": {"fixture": True},
            "training": config.to_dict(),
        },
        device=torch.device("cpu"),
        total_steps_override=steps,
        resume=resume,
    )


def test_accumulation_steps_once_and_keeps_fp32_loss(tmp_path) -> None:
    torch.manual_seed(2026)
    report = run(TimerModel(model_config()), tmp_path, steps=2)

    assert report.optimizer_steps == 2
    assert report.micro_batches == 4
    assert report.loss_dtype == "torch.float32"
    assert report.checkpoint_path.is_file()


def test_interrupted_resume_reproduces_next_loss(tmp_path) -> None:
    torch.manual_seed(2026)
    interrupted = run(TimerModel(model_config()), tmp_path / "first", steps=2)
    resumed = run(
        TimerModel(model_config()),
        tmp_path / "resumed",
        steps=3,
        resume=interrupted.checkpoint_path,
    )
    torch.manual_seed(2026)
    continuous = run(TimerModel(model_config()), tmp_path / "continuous", steps=3)

    assert resumed.losses[-1] == pytest.approx(continuous.losses[-1], abs=1e-7)


def test_denormalized_metrics_are_fp32_and_exact() -> None:
    predictions = torch.tensor([[0.0, 1.0]], dtype=torch.bfloat16)
    targets = torch.tensor([[1.0, 1.0]], dtype=torch.bfloat16)
    mean = torch.tensor([[10.0]])
    scale = torch.tensor([[2.0]])

    metrics = denormalized_error_metrics(predictions, targets, mean, scale)

    assert metrics["mse"].dtype == torch.float32
    assert float(metrics["mse"]) == pytest.approx(2.0)
    assert float(metrics["mae"]) == pytest.approx(1.0)
