from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from tsfm.checkpoint import save_checkpoint
from tsfm.config import TimerConfig
from tsfm.data import SyntheticTimeSeriesDataset, normalize_context_target
from tsfm.model import TimerModel


@dataclass(frozen=True, slots=True)
class SmokeTrainingConfig:
    steps: int = 100
    batch_size: int = 8
    learning_rate: float = 3e-3
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    num_samples: int = 64
    context_patches: int = 4
    seed: int = 2026
    log_every: int = 10


@dataclass(frozen=True, slots=True)
class SmokeResult:
    initial_loss: float
    final_loss: float
    checkpoint_path: Path


def run_smoke(
    model_config: TimerConfig,
    training_config: SmokeTrainingConfig,
    output_dir: str | Path,
) -> SmokeResult:
    if model_config.input_token_len != model_config.output_token_len:
        raise ValueError("smoke training requires equal input and output patch lengths")
    if training_config.context_patches > model_config.max_position_embeddings:
        raise ValueError("context_patches exceeds max_position_embeddings")

    torch.manual_seed(training_config.seed)
    patch_length = model_config.input_token_len
    dataset = SyntheticTimeSeriesDataset(
        num_samples=training_config.num_samples,
        total_length=(training_config.context_patches + 1) * patch_length,
        seed=training_config.seed,
    )
    model = TimerModel(model_config)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )

    def prepare_batch(start: int) -> tuple[torch.Tensor, torch.Tensor]:
        indices = [
            (start + offset) % len(dataset)
            for offset in range(training_config.batch_size)
        ]
        series = torch.stack([dataset[index] for index in indices])
        context = series[:, :-patch_length]
        target = series[:, patch_length:]
        normalized = normalize_context_target(context, target)
        return normalized.context, normalized.target

    evaluation_context, evaluation_target = prepare_batch(0)

    def evaluation_loss() -> float:
        model.eval()
        with torch.no_grad():
            loss = model(evaluation_context, labels=evaluation_target).loss
        assert loss is not None
        return float(loss)

    initial_loss = evaluation_loss()
    model.train()
    for step in range(1, training_config.steps + 1):
        context, target = prepare_batch((step - 1) * training_config.batch_size)
        optimizer.zero_grad(set_to_none=True)
        loss = model(context, labels=target).loss
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), training_config.grad_clip)
        optimizer.step()
        if training_config.log_every and step % training_config.log_every == 0:
            print(f"step={step} loss={loss.detach().item():.6f}")

    final_loss = evaluation_loss()
    checkpoint_path = Path(output_dir) / "final.pt"
    save_checkpoint(checkpoint_path, model, optimizer, training_config.steps)
    return SmokeResult(
        initial_loss=initial_loss,
        final_loss=final_loss,
        checkpoint_path=checkpoint_path,
    )
