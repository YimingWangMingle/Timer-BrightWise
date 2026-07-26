from __future__ import annotations

from types import SimpleNamespace

import tsfm.optim as optim
from tsfm.train_config import TrainingConfig


def _config() -> TrainingConfig:
    return TrainingConfig(total_steps=1, warmup_steps=0, peak_lr=1e-3)


def test_cuda_optimizer_requests_fused_adamw(monkeypatch) -> None:
    calls = []
    sentinel = object()

    def adamw(parameters, **kwargs):
        calls.append((parameters, kwargs))
        return sentinel

    monkeypatch.setattr(optim.torch.optim, "AdamW", adamw)
    parameter = SimpleNamespace(is_cuda=True)

    assert optim.build_optimizer([parameter], _config()) is sentinel
    assert calls[0][1]["fused"] is True


def test_fused_optimizer_retries_without_fused_when_unsupported(monkeypatch) -> None:
    calls = []
    sentinel = object()

    def adamw(parameters, **kwargs):
        calls.append(kwargs)
        if "fused" in kwargs:
            raise TypeError("fused unsupported")
        return sentinel

    monkeypatch.setattr(optim.torch.optim, "AdamW", adamw)

    assert optim.build_optimizer([SimpleNamespace(is_cuda=True)], _config()) is sentinel
    assert calls == [
        {
            "lr": 1e-3,
            "betas": (0.9, 0.95),
            "weight_decay": 0.1,
            "fused": True,
        },
        {"lr": 1e-3, "betas": (0.9, 0.95), "weight_decay": 0.1},
    ]
