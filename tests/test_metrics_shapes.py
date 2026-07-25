import pytest
import torch

from tsfm.metrics import denormalized_error_metrics


def test_denormalized_metrics_broadcast_per_batch_for_patch_predictions() -> None:
    predictions = torch.zeros(2, 3, 4)
    targets = torch.ones(2, 3, 4)
    mean = torch.tensor([[10.0], [100.0]])
    scale = torch.tensor([[2.0], [3.0]])

    metrics = denormalized_error_metrics(predictions, targets, mean, scale)

    assert float(metrics["mse"]) == pytest.approx((4.0 + 9.0) / 2.0)
    assert float(metrics["mae"]) == pytest.approx((2.0 + 3.0) / 2.0)
