from tsfm.trainer import _gather_rank_rng_states


def test_single_rank_rng_gather_returns_rank_key() -> None:
    states = _gather_rank_rng_states(rank=0, world_size=1)
    assert states is not None
    assert set(states) == {0}
    assert "torch_cpu" in states[0]
