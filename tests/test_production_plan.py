from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsfm.production_plan import ProductionTemplate, ResolvedTrainingPlan
from tsfm.train_config import TrainingConfig

ROOT = Path(__file__).parents[1]


def test_production_template_resolves_padding_steps_and_warmup() -> None:
    template = ProductionTemplate.from_json(
        ROOT / "configs" / "training" / "h100_307m_production.json"
    )
    plan = ResolvedTrainingPlan.resolve(
        template,
        window_count=10_001,
        micro_batch_size=256,
        world_size=4,
        digests={
            "model": "a" * 64,
            "source": "b" * 64,
            "manifest": "c" * 64,
            "packages": "d" * 64,
        },
    )

    assert plan.global_batch_size == 4096
    assert plan.gradient_accumulation_steps == 4
    assert plan.total_real_samples == 30_003
    assert plan.total_padded_samples == 2_765
    assert plan.total_steps == 8
    assert plan.warmup_steps == 1
    assert plan.minimum_lr_ratio == pytest.approx(0.1)
    config = plan.training_config()
    assert config.micro_batch_size == 256
    assert config.gradient_accumulation_steps == 4
    assert config.peak_lr == pytest.approx(5e-5)
    assert config.logging_interval == 10


def test_resolved_plan_writes_canonical_json_atomically(tmp_path: Path) -> None:
    template = ProductionTemplate.from_json(
        ROOT / "configs" / "training" / "h100_307m_production.json"
    )
    plan = ResolvedTrainingPlan.resolve(
        template,
        window_count=100,
        micro_batch_size=128,
        world_size=4,
        digests={"model": "a" * 64},
    )
    destination = tmp_path / "resolved.json"

    digest = plan.write_atomic(destination)

    assert len(digest) == 64
    assert json.loads(destination.read_text(encoding="utf-8")) == plan.to_dict()
    assert not list(tmp_path.glob("*.tmp"))


def test_resolution_rejects_unapproved_batch_and_invalid_digest() -> None:
    template = ProductionTemplate.from_json(
        ROOT / "configs" / "training" / "h100_307m_production.json"
    )
    with pytest.raises(ValueError, match="micro-batch"):
        ResolvedTrainingPlan.resolve(
            template,
            window_count=1,
            micro_batch_size=32,
            world_size=4,
            digests={},
        )
    with pytest.raises(ValueError, match="digest"):
        ResolvedTrainingPlan.resolve(
            template,
            window_count=1,
            micro_batch_size=64,
            world_size=4,
            digests={"model": "not-a-digest"},
        )


def test_training_intervals_need_not_divide_total_steps() -> None:
    config = TrainingConfig(
        total_steps=2001,
        warmup_steps=40,
        peak_lr=5e-5,
        validation_interval=2000,
        checkpoint_interval=2000,
    )
    assert config.total_steps == 2001
