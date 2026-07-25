from tsfm.config import TimerConfig
from tsfm.training import SmokeTrainingConfig, run_smoke


def test_smoke_training_reduces_fixed_corpus_loss_and_saves_checkpoint(tmp_path) -> None:
    model_config = TimerConfig(
        input_token_len=8,
        output_token_len=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=8,
    )
    training_config = SmokeTrainingConfig(
        steps=30,
        batch_size=8,
        learning_rate=1e-2,
        weight_decay=0.0,
        grad_clip=1.0,
        num_samples=8,
        context_patches=3,
        seed=47,
        log_every=0,
    )

    result = run_smoke(model_config, training_config, tmp_path)

    assert result.final_loss < result.initial_loss
    assert result.checkpoint_path.is_file()
