from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class TimerConfig:
    input_token_len: int = 96
    output_token_len: int = 96
    hidden_size: int = 128
    intermediate_size: int = 256
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    max_position_embeddings: int = 128
    attention_dropout: float = 0.0
    rope_theta: float = 10_000.0
    norm_eps: float = 1e-5

    def __post_init__(self) -> None:
        integer_fields = (
            "input_token_len",
            "output_token_len",
            "hidden_size",
            "intermediate_size",
            "num_hidden_layers",
            "num_attention_heads",
            "max_position_embeddings",
        )
        for name in integer_fields:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.head_dim % 2 != 0:
            raise ValueError("attention head dimension must be even for RoPE")
        if not 0.0 <= self.attention_dropout < 1.0:
            raise ValueError("attention_dropout must be in [0, 1)")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads

    @classmethod
    def from_json(cls, path: str | Path) -> "TimerConfig":
        values: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**values)


def estimate_parameter_count(config: TimerConfig) -> int:
    """Count parameters for the HF-compatible Timer architecture analytically."""
    hidden = config.hidden_size
    intermediate = config.intermediate_size
    patch_parameters = (
        config.input_token_len * hidden
        + hidden * config.output_token_len
    )
    per_layer = (
        4 * hidden * hidden
        + 3 * hidden * intermediate
        + 7 * hidden
    )
    final_norm = 2 * hidden
    return patch_parameters + config.num_hidden_layers * per_layer + final_norm
