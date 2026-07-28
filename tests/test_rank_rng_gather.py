import torch

from tsfm.trainer import _gather_rank_rng_states


def test_single_rank_rng_gather_returns_rank_key() -> None:
    states = _gather_rank_rng_states(rank=0, world_size=1)
    assert states is not None
    assert set(states) == {0}
    assert "torch_cpu" in states[0]


def test_distributed_rng_gather_transports_serialized_bytes(monkeypatch) -> None:
    known_state = {"torch_cpu": torch.arange(8, dtype=torch.uint8)}

    monkeypatch.setattr("tsfm.trainer.capture_rng_state", lambda: known_state)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def gather_object(value, gathered, dst):
        rank, payload = value
        assert rank == 0
        assert isinstance(payload, bytes)
        assert dst == 0
        for item_rank in range(4):
            gathered[item_rank] = (item_rank, payload)

    monkeypatch.setattr(torch.distributed, "gather_object", gather_object)

    states = _gather_rank_rng_states(rank=0, world_size=4)

    assert states is not None
    assert set(states) == {0, 1, 2, 3}
    for state in states.values():
        assert torch.equal(state["torch_cpu"], known_state["torch_cpu"])
