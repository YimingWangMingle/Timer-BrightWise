import torch

from tsfm.checkpoint import load_checkpoint, save_checkpoint
from tsfm.config import TimerConfig
from tsfm.model import TimerModel


def tiny_config() -> TimerConfig:
    return TimerConfig(
        input_token_len=8,
        output_token_len=8,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=16,
    )


def test_checkpoint_round_trip_restores_predictions_optimizer_and_step(tmp_path) -> None:
    torch.manual_seed(31)
    model = TimerModel(tiny_config())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    inputs = torch.randn(2, 24)
    labels = torch.randn(2, 24)

    loss = model(inputs, labels=labels).loss
    loss.backward()
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(inputs).predictions.clone()

    checkpoint_path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint_path, model, optimizer, step=17)

    torch.manual_seed(99)
    restored = TimerModel(tiny_config()).eval()
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    step = load_checkpoint(checkpoint_path, restored, restored_optimizer)

    with torch.no_grad():
        actual = restored(inputs).predictions
    assert step == 17
    assert restored_optimizer.state_dict()["state"]
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
