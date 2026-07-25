import json

import pytest

from tsfm.config import TimerConfig, estimate_parameter_count


def test_tiny_config_has_expected_parameter_count() -> None:
    config = TimerConfig(
        input_token_len=96,
        output_token_len=96,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_position_embeddings=128,
    )

    assert estimate_parameter_count(config) == 354_304


def test_300m_config_is_in_target_range() -> None:
    config = TimerConfig(
        input_token_len=96,
        output_token_len=96,
        hidden_size=1536,
        intermediate_size=3072,
        num_hidden_layers=13,
        num_attention_heads=12,
        max_position_embeddings=10_000,
    )

    count = estimate_parameter_count(config)
    assert 300_000_000 <= count <= 315_000_000


def test_config_rejects_incompatible_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TimerConfig(hidden_size=130, num_attention_heads=4)


def test_config_loads_from_json(tmp_path) -> None:
    path = tmp_path / "model.json"
    path.write_text(
        json.dumps(
            {
                "input_token_len": 96,
                "output_token_len": 96,
                "hidden_size": 128,
                "intermediate_size": 256,
                "num_hidden_layers": 2,
                "num_attention_heads": 4,
                "max_position_embeddings": 128,
                "attention_dropout": 0.0,
            }
        ),
        encoding="utf-8",
    )

    config = TimerConfig.from_json(path)

    assert config.hidden_size == 128
    assert config.head_dim == 32
