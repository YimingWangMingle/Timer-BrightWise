from pathlib import Path

import pytest
import torch
from torch import nn

from tsfm.config import TimerConfig, estimate_parameter_count
from tsfm.model import TimerModel


def tiny_config() -> TimerConfig:
    return TimerConfig(
        input_token_len=16,
        output_token_len=16,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=32,
    )


def test_timer_initialization_matches_released_contract() -> None:
    torch.manual_seed(2026)
    model = TimerModel(tiny_config())
    linear_weights = [
        module.weight.detach().flatten()
        for module in model.modules()
        if isinstance(module, nn.Linear)
    ]
    all_weights = torch.cat(linear_weights)

    assert abs(float(all_weights.mean())) < 0.003
    assert 0.017 < float(all_weights.std(unbiased=False)) < 0.023
    for module in model.modules():
        if isinstance(module, nn.Linear) and module.bias is not None:
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))
        if isinstance(module, nn.LayerNorm):
            torch.testing.assert_close(module.weight, torch.ones_like(module.weight))
            torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))


def test_next_patch_loss_reduces_in_fp32() -> None:
    model = TimerModel(tiny_config())
    predictions = torch.randn(2, 4, 16, dtype=torch.bfloat16)
    labels = torch.randn(2, 64, dtype=torch.float32)

    loss = model._next_patch_loss(predictions, labels)

    assert loss.dtype == torch.float32


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("timer_26m.json", 26_349_568),
        ("timer_95m.json", 94_635_008),
        ("timer_300m.json", 307_146_240),
    ],
)
def test_model_ladder_has_exact_parameter_counts(filename: str, expected: int) -> None:
    path = Path(__file__).parents[1] / "configs" / "model" / filename

    assert estimate_parameter_count(TimerConfig.from_json(path)) == expected
