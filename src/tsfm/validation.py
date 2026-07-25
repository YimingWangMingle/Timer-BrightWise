from __future__ import annotations

import contextlib

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from tsfm.metrics import denormalized_error_metrics


def evaluate_model(
    model: torch.nn.Module,
    dataset: Dataset,
    *,
    device: torch.device,
    batch_size: int,
    batches: int,
    precision: str,
) -> dict[str, float]:
    if batch_size <= 0 or batches <= 0:
        raise ValueError("batch_size and batches must be positive")
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    was_training = model.training
    model.eval()
    totals = {"normalized_mse": 0.0, "mse": 0.0, "mae": 0.0}
    seen = 0
    autocast = (
        torch.autocast("cuda", dtype=torch.bfloat16)
        if device.type == "cuda" and precision == "bf16"
        else contextlib.nullcontext()
    )
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= batches:
                break
            context = batch["context"].to(device)
            target = batch["target"].to(device)
            with autocast:
                predictions = model(context).predictions
            target_patches = target.reshape_as(predictions)
            normalized = F.mse_loss(predictions.float(), target_patches.float())
            raw = denormalized_error_metrics(
                predictions,
                target_patches,
                batch["mean"].to(device),
                batch["scale"].to(device),
            )
            count = context.shape[0]
            totals["normalized_mse"] += float(normalized) * count
            totals["mse"] += float(raw["mse"]) * count
            totals["mae"] += float(raw["mae"]) * count
            seen += count
    model.train(was_training)
    if seen == 0:
        raise ValueError("validation dataset produced no batches")
    return {name: value / seen for name, value in totals.items()}
