import torch
from torch.utils.data import Dataset

from tsfm.config import TimerConfig
from tsfm.model import TimerModel
from tsfm.validation import evaluate_model


class ValidationWindows(Dataset):
    def __len__(self):
        return 8

    def __getitem__(self, index):
        values = torch.arange(12, dtype=torch.float32) + index
        context = values[:8]
        target = values[4:]
        mean = context.mean().reshape(1)
        scale = torch.sqrt(context.var(unbiased=False) + 1e-5).reshape(1)
        return {
            "context": (context - mean) / scale,
            "target": (target - mean) / scale,
            "mean": mean,
            "scale": scale,
            "source_id": "validation",
            "record_id": f"validation/{index}",
        }


def test_evaluate_model_reports_finite_fp32_normalized_and_raw_metrics() -> None:
    model = TimerModel(
        TimerConfig(
            input_token_len=4,
            output_token_len=4,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=8,
        )
    )

    metrics = evaluate_model(
        model,
        ValidationWindows(),
        device=torch.device("cpu"),
        batch_size=2,
        batches=2,
        precision="bf16",
    )

    assert set(metrics) == {"normalized_mse", "mse", "mae"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert all(isinstance(value, float) for value in metrics.values())
