import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from tsfm.checkpoint import (
    CheckpointManager,
    TrainingState,
    load_training_checkpoint,
    save_training_checkpoint,
)
from tsfm.config import TimerConfig
from tsfm.model import TimerModel
from tsfm.optim import build_optimizer, build_scheduler
from tsfm.train_config import TrainingConfig


def components():
    model_config = TimerConfig(
        input_token_len=4,
        output_token_len=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
    )
    training_config = TrainingConfig(
        total_steps=10,
        warmup_steps=2,
        peak_lr=1e-3,
        micro_batch_size=2,
        gradient_accumulation_steps=1,
        validation_interval=5,
        checkpoint_interval=5,
        num_workers=1,
    )
    model = TimerModel(model_config)
    optimizer = build_optimizer(model.parameters(), training_config)
    scheduler = build_scheduler(optimizer, training_config)
    snapshots = {
        "model": {"hidden_size": 8},
        "data": {"manifest": "fixture"},
        "training": training_config.to_dict(),
    }
    return model, optimizer, scheduler, snapshots


def test_full_checkpoint_restores_rng_scheduler_and_sampler_state(tmp_path) -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    model, optimizer, scheduler, snapshots = components()
    state = TrainingState(3, 6, {"next_sample": 6})
    path = tmp_path / "step-000003.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=state,
        config_snapshots=snapshots,
        manifest_checksum="a" * 64,
        environment={"torch": torch.__version__},
    )
    expected = (random.random(), float(np.random.rand()), torch.rand(1))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restored_model, restored_optimizer, restored_scheduler, _ = components()
    loaded = load_training_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_config_snapshots=snapshots,
        expected_manifest_checksum="a" * 64,
    )
    actual = (random.random(), float(np.random.rand()), torch.rand(1))

    assert loaded == state
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_checkpoint_refuses_configuration_mismatch(tmp_path) -> None:
    model, optimizer, scheduler, snapshots = components()
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=TrainingState(0, 0, {"next_sample": 0}),
        config_snapshots=snapshots,
        manifest_checksum="b" * 64,
        environment={},
    )
    changed = {**snapshots, "data": {"manifest": "different"}}

    with pytest.raises(ValueError, match="configuration mismatch"):
        load_training_checkpoint(
            path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_config_snapshots=changed,
            expected_manifest_checksum="b" * 64,
        )


def test_manager_keeps_best_and_two_latest(tmp_path) -> None:
    model, optimizer, scheduler, snapshots = components()
    manager = CheckpointManager(tmp_path, keep_latest=2)
    base_state = TrainingState(0, 0, {"next_sample": 0})
    for step, metric in [(1, 4.0), (2, 3.0), (3, 5.0), (4, 2.0)]:
        manager.save(
            step=step,
            validation_metric=metric,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            state=replace(
                base_state,
                global_step=step,
                consumed_samples=step * 2,
                sampler_state={"next_sample": step * 2},
            ),
            config_snapshots=snapshots,
            manifest_checksum="c" * 64,
            environment={},
        )

    assert sorted(path.name for path in tmp_path.glob("*.pt")) == [
        "best.pt",
        "step-000003.pt",
        "step-000004.pt",
    ]
