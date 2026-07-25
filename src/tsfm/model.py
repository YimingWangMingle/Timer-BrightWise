from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from tsfm.config import TimerConfig


@dataclass(slots=True)
class TimerOutput:
    predictions: torch.Tensor
    loss: torch.Tensor | None = None


def _rotate_half(values: torch.Tensor) -> torch.Tensor:
    first, second = values.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, theta: float) -> None:
        super().__init__()
        frequencies = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("frequencies", frequencies, persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        positions = torch.arange(
            query.shape[-2], device=query.device, dtype=self.frequencies.dtype
        )
        angles = torch.outer(positions, self.frequencies)
        angles = torch.cat((angles, angles), dim=-1)
        cosine = angles.cos().to(dtype=query.dtype)[None, None, :, :]
        sine = angles.sin().to(dtype=query.dtype)[None, None, :, :]
        return (
            query * cosine + _rotate_half(query) * sine,
            key * cosine + _rotate_half(key) * sine,
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TimerConfig) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.hidden_size = config.hidden_size
        self.attention_dropout = config.attention_dropout
        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.rotary = RotaryEmbedding(config.head_dim, config.rope_theta)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = hidden_states.shape

        def split_heads(values: torch.Tensor) -> torch.Tensor:
            return values.view(
                batch_size, sequence_length, self.num_heads, self.head_dim
            ).transpose(1, 2)

        query = split_heads(self.q_proj(hidden_states))
        key = split_heads(self.k_proj(hidden_states))
        value = split_heads(self.v_proj(hidden_states))
        query, key = self.rotary(query, key)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, self.hidden_size
        )
        return self.o_proj(attended)


class GatedMLP(nn.Module):
    def __init__(self, config: TimerConfig) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            config.hidden_size, config.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            config.intermediate_size, config.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        )


class DecoderLayer(nn.Module):
    def __init__(self, config: TimerConfig) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(config)
        self.mlp = GatedMLP(config)
        self.attention_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.attention_norm(
            hidden_states + self.attention(hidden_states)
        )
        return self.mlp_norm(hidden_states + self.mlp(hidden_states))


class TimerModel(nn.Module):
    def __init__(self, config: TimerConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embedding = nn.Linear(
            config.input_token_len, config.hidden_size, bias=False
        )
        self.layers = nn.ModuleList(
            DecoderLayer(config) for _ in range(config.num_hidden_layers)
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=config.norm_eps)
        self.output_projection = nn.Linear(
            config.hidden_size, config.output_token_len, bias=False
        )
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def _validate_input(self, input_values: torch.Tensor) -> None:
        if input_values.ndim != 2:
            raise ValueError("input_values must have shape [batch, raw_length]")
        if input_values.shape[-1] % self.config.input_token_len != 0:
            raise ValueError("raw input length must be divisible by input_token_len")
        patch_count = input_values.shape[-1] // self.config.input_token_len
        if patch_count > self.config.max_position_embeddings:
            raise ValueError("input contains more patches than max_position_embeddings")

    def _next_patch_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        expected_length = predictions.shape[1] * predictions.shape[2]
        if labels.shape != (predictions.shape[0], expected_length):
            raise ValueError(
                "labels must have shape [batch, patch_count * output_token_len]"
            )
        return F.mse_loss(
            predictions.float(), labels.reshape_as(predictions).float()
        )

    def forward(
        self, input_values: torch.Tensor, labels: torch.Tensor | None = None
    ) -> TimerOutput:
        self._validate_input(input_values)
        patch_count = input_values.shape[-1] // self.config.input_token_len
        patches = input_values.reshape(
            input_values.shape[0], patch_count, self.config.input_token_len
        )
        hidden_states = self.patch_embedding(patches)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        predictions = self.output_projection(self.final_norm(hidden_states))
        loss = self._next_patch_loss(predictions, labels) if labels is not None else None
        return TimerOutput(predictions=predictions, loss=loss)

    def generate(
        self, input_values: torch.Tensor, prediction_length: int
    ) -> torch.Tensor:
        self._validate_input(input_values)
        if prediction_length <= 0:
            raise ValueError("prediction_length must be positive")
        if self.config.input_token_len != self.config.output_token_len:
            raise ValueError(
                "rolling generation requires equal input and output patch lengths"
            )

        generated: list[torch.Tensor] = []
        remaining = prediction_length
        context = input_values
        while remaining > 0:
            next_patch = self(context).predictions[:, -1, :]
            generated.append(next_patch)
            context = torch.cat((context, next_patch), dim=-1)
            remaining -= self.config.output_token_len
        return torch.cat(generated, dim=-1)[:, :prediction_length]
