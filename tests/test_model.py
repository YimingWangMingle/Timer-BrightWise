import pytest
import torch

from tsfm.config import TimerConfig
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


def test_forward_computes_next_patch_loss_and_gradients() -> None:
    torch.manual_seed(7)
    model = TimerModel(tiny_config())
    input_values = torch.randn(2, 64)
    labels = torch.randn(2, 64)

    output = model(input_values, labels=labels)
    output.loss.backward()

    assert output.predictions.shape == (2, 4, 16)
    assert output.loss.ndim == 0
    assert torch.isfinite(output.loss)
    assert model.patch_embedding.weight.grad is not None


def test_causal_attention_blocks_future_patch_changes() -> None:
    torch.manual_seed(11)
    model = TimerModel(tiny_config()).eval()
    original = torch.randn(1, 64)
    changed = original.clone()
    changed[:, -16:] += 100.0

    with torch.no_grad():
        original_predictions = model(original).predictions
        changed_predictions = model(changed).predictions

    torch.testing.assert_close(
        original_predictions[:, :-1],
        changed_predictions[:, :-1],
        rtol=0.0,
        atol=1e-6,
    )
    assert not torch.allclose(
        original_predictions[:, -1], changed_predictions[:, -1]
    )


def test_generate_returns_exact_requested_length() -> None:
    torch.manual_seed(13)
    model = TimerModel(tiny_config()).eval()
    context = torch.randn(2, 48)

    with torch.no_grad():
        generated = model.generate(context, prediction_length=21)

    assert generated.shape == (2, 21)
    assert torch.isfinite(generated).all()


def test_forward_rejects_partial_input_patch() -> None:
    model = TimerModel(tiny_config())

    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 31))
