from tsfm.distributed import distributed_context_from_environment
from tsfm.s3.sampling import CounterSampler


def test_sixteen_ranks_partition_global_sample_indices() -> None:
    batches = [
        list(CounterSampler(0, rank=rank, world_size=16).take(32))
        for rank in range(16)
    ]
    flattened = [value for batch in batches for value in batch]

    assert len(flattened) == len(set(flattened)) == 512
    assert sorted(flattened) == list(range(512))


def test_torchrun_environment_marks_only_rank_zero_as_main(
    monkeypatch,
) -> None:
    monkeypatch.setenv("RANK", "7")
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("WORLD_SIZE", "16")

    context = distributed_context_from_environment(initialize=False)

    assert context.rank == 7
    assert context.local_rank == 3
    assert context.world_size == 16
    assert not context.is_main_process
