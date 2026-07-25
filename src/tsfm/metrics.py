from __future__ import annotations

import torch


def denormalized_error_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    if mean.shape[0] != predictions.shape[0] or scale.shape[0] != predictions.shape[0]:
        raise ValueError("normalization statistics must match the batch size")
    predictions_fp32 = predictions.float()
    targets_fp32 = targets.float()
    mean_fp32 = mean.float()
    scale_fp32 = scale.float()
    while mean_fp32.ndim < predictions_fp32.ndim:
        mean_fp32 = mean_fp32.unsqueeze(-1)
    while scale_fp32.ndim < predictions_fp32.ndim:
        scale_fp32 = scale_fp32.unsqueeze(-1)
    restored_predictions = predictions_fp32 * scale_fp32 + mean_fp32
    restored_targets = targets_fp32 * scale_fp32 + mean_fp32
    error = restored_predictions - restored_targets
    return {
        "mse": error.square().mean(dtype=torch.float32),
        "mae": error.abs().mean(dtype=torch.float32),
    }
