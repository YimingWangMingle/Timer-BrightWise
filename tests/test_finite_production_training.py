from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from tsfm.s3.finite_sampling import AffineCoverageSampler
from tsfm.train_config import TrainingConfig
from tsfm.trainer import run_training


class IndexedWindows(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int) -> dict[str, object]:
        values = torch.tensor([1.0, 2.0])
        return {
            "context": values,
            "target": values,
            "source_id": f"sample-{index}",
            "record_id": str(index),
        }


class ScalarModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, context, labels=None):
        loss = ((context * self.weight - labels) ** 2).mean()
        return type("Output", (), {"loss": loss})()


def _config() -> TrainingConfig:
    return TrainingConfig(
        total_steps=6,
        warmup_steps=1,
        peak_lr=1e-3,
        micro_batch_size=1,
        gradient_accumulation_steps=1,
        validation_interval=1,
        checkpoint_interval=1,
        num_workers=1,
        prefetch_factor=1,
        pin_memory=False,
        context_patches=1,
    )


def _plan() -> dict[str, object]:
    return {
        "coverage_cycles": 2,
        "window_count": 3,
        "world_size": 1,
        "global_batch_size": 1,
        "total_real_samples": 6,
        "total_padded_samples": 0,
        "total_steps": 6,
        "seed": 2026,
    }


def _run(output: Path, *, steps: int, resume: Path | None = None):
    config = _config()
    plan = _plan()
    return run_training(
        model=ScalarModel(),
        dataset=IndexedWindows(),
        training_config=config,
        output_dir=output,
        manifest_checksum="a" * 64,
        config_snapshots={"training": config.to_dict(), "resolved_plan": plan},
        device=torch.device("cpu"),
        resume=resume,
        total_steps_override=steps,
        resolved_plan=plan,
    )


def test_finite_training_resumes_at_global_sample_position(tmp_path: Path) -> None:
    expected = list(
        AffineCoverageSampler(
            window_count=3,
            cycles=2,
            seed=2026,
            global_batch_size=1,
        )
    )
    first = _run(tmp_path / "first", steps=2)
