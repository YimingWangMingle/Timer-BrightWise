from __future__ import annotations

import contextlib

import torch

from tsfm.checkpoint import TrainingState, save_training_checkpoint
from tsfm.distributed import DistributedContext, unwrap_model
from tsfm.optim import build_optimizer, build_scheduler
from tsfm.train_config import TrainingConfig
from tsfm.trainer import run_training


class Wrapped(torch.nn.Module):
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self.module = module


class CountingNoSyncModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.no_sync_entries = 0

    @contextlib.contextmanager
    def no_sync(self):
        self.no_sync_entries += 1
        yield

    def forward(self, context, labels=None):
        prediction = context * self.weight
        return type("Output", (), {"loss": ((prediction - labels) ** 2).mean()})()


class Windows(torch.utils.data.Dataset):
    def __len__(self):
        return 100

    def __getitem__(self, index):
        values = torch.tensor([1.0, 2.0])
        return {
            "context": values,
            "target": values,
            "source_id": "synthetic",
            "record_id": str(index),
        }


def _config(accumulation: int = 1) -> TrainingConfig:
    return TrainingConfig(
        total_steps=1,
        warmup_steps=0,
        peak_lr=1e-3,
        micro_batch_size=1,
        gradient_accumulation_steps=accumulation,
        validation_interval=1,
        checkpoint_interval=1,
        num_workers=1,
        prefetch_factor=1,
        pin_memory=False,
        context_patches=1,
    )


def test_checkpoint_saves_unwrapped_keys_and_all_rank_rng(tmp_path) -> None:
    inner = torch.nn.Linear(2, 2)
    model = Wrapped(inner)
    optimizer = build_optimizer(model.parameters(), _config())
    scheduler = build_scheduler(optimizer, _config())
    path = tmp_path / "checkpoint.pt"
    states = {rank: {"torch_cpu": torch.get_rng_state()} for rank in range(4)}

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        state=TrainingState(0, 0, {"next_sample": 0}),
        config_snapshots={},
        manifest_checksum="a" * 64,
        environment={},
        rank_rng_states=states,
        resolved_plan={"world_size": 4},
    )
    payload = torch.load(path, weights_only=True)

    assert unwrap_model(model) is inner
    assert all(not key.startswith("module.") for key in payload["model"])
    assert set(payload["rank_rng_states"]) == {0, 1, 2, 3}
    assert payload["resolved_plan"] == {"world_size": 4}


def test_accumulation_uses_no_sync_except_final_microbatch(tmp_path) -> None:
    model = CountingNoSyncModel()

    run_training(
        model=model,
        dataset=Windows(),
        training_config=_config(accumulation=4),
        output_dir=tmp_path,
        manifest_checksum="a" * 64,
        config_snapshots={},
        device=torch.device("cpu"),
        world_size=2,
        is_main_process=False,
    )

    assert model.no_sync_entries == 3


def test_distributed_context_helpers_do_not_require_initialized_group() -> None:
    context = DistributedContext(0, 0, 1, True)
    assert unwrap_model(torch.nn.Linear(1, 1)).__class__ is torch.nn.Linear
