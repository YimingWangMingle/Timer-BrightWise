import warnings

from tsfm.config import TimerConfig
from tsfm.training import SmokeTrainingConfig, run_smoke


def test_smoke_logging_does_not_convert_a_gradient_tensor_directly(tmp_path) -> None:
    model_config = TimerConfig(
        input_token_len=4,
        output_token_len=4,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=4,
    )
    training_config = SmokeTrainingConfig(
        steps=1,
        batch_size=2,
        learning_rate=1e-2,
        weight_decay=0.0,
        grad_clip=1.0,
        num_samples=2,
        context_patches=2,
        seed=5,
        log_every=1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        run_smoke(model_config, training_config, tmp_path)
