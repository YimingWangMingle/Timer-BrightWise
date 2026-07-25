from pathlib import Path

import pytest
import torch

from tsfm.optim import build_optimizer, lr_multiplier
from tsfm.train_config import TrainingConfig

CONFIG_ROOT = Path(__file__).parents[1] / "configs" / "training"


def test_5090_configs_match_approved_runs() -> None:
    run26 = TrainingConfig.from_json(CONFIG_ROOT / "rtx5090_26m.json")
    run95 = TrainingConfig.from_json(CONFIG_ROOT / "rtx5090_95m.json")

    assert (run26.total_steps, run26.warmup_steps, run26.peak_lr) == (
        2_000,
        100,
        3e-4,
    )
    assert (run95.total_steps, run95.warmup_steps, run95.peak_lr) == (
        500,
        50,
        2e-4,
    )
    assert run26.micro_batch_size * run26.gradient_accumulation_steps == 256
    assert run26.num_workers == 8
    assert run26.prefetch_factor == 2
    assert run26.context_patches == 30


def test_optimizer_and_schedule_match_approved_values() -> None:
    parameter = torch.nn.Parameter(torch.ones(1))
    config = TrainingConfig(
        total_steps=100,
        warmup_steps=10,
        peak_lr=3e-4,
        validation_interval=10,
        checkpoint_interval=10,
    )
    optimizer = build_optimizer([parameter], config)

    assert optimizer.defaults["betas"] == (0.9, 0.95)
    assert optimizer.defaults["weight_decay"] == 0.1
    assert lr_multiplier(0, config) == pytest.approx(0.1)
    assert lr_multiplier(9, config) == pytest.approx(1.0)
    assert lr_multiplier(99, config) == pytest.approx(0.1)


def test_training_config_rejects_non_divisible_intervals() -> None:
    with pytest.raises(ValueError, match="validation_interval"):
        TrainingConfig(
            total_steps=501,
            warmup_steps=50,
            peak_lr=2e-4,
            validation_interval=250,
            checkpoint_interval=167,
        )
