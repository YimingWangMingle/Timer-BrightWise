import pytest
import torch
from torch.utils.data import Dataset

from tsfm.validated_training import primary_validation_metric
from tsfm.validation import evaluate_model


class RecordingWindows(Dataset):
    def __init__(self, seen):
        self.seen = seen

    def __len__(self):
        return 100

    def __getitem__(self, index):
        self.seen.append(index)
        values = torch.ones(4)
        return {
            "context": values,
            "target": values,
            "mean": torch.zeros(1),
            "scale": torch.ones(1),
        }


class IdentityModel(torch.nn.Module):
    def forward(self, context):
        return type("Output", (), {"predictions": context})()


def test_validation_partitions_positions_by_rank() -> None:
    seen = []
    evaluate_model(
        IdentityModel(),
        RecordingWindows(seen),
        device=torch.device("cpu"),
        batch_size=2,
        batches=2,
        precision="fp32",
        rank=1,
        world_size=4,
    )
    assert seen == [1, 5, 9, 13]


def test_best_checkpoint_uses_normalized_not_raw_mse() -> None:
    views = {
        "val_heldout": {"normalized_mse": 0.4, "mse": 1_000_000.0, "mae": 1.0},
        "val_temporal": {"normalized_mse": 0.6, "mse": 1.0, "mae": 1.0},
    }
    assert primary_validation_metric(views) == pytest.approx(0.5)
