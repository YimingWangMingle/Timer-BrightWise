from __future__ import annotations

from collections import Counter
from pathlib import PurePosixPath

import pytest

from tsfm.s3.finite_sampling import AffineCoverageSampler, FiniteWindowIndex
from tsfm.s3.records import ManifestRecord, SplitRegion


def _record(name: str, length: int) -> ManifestRecord:
    return ManifestRecord(
        1,
        "utsd",
        "UTSD-12G",
        name,
        "0",
        PurePosixPath("shards/a.npy"),
        0,
        length,
        None,
        "all",
        "a" * 64,
    )


def test_finite_index_maps_every_window_across_regions() -> None:
    records = [_record("a", 8), _record("b", 10)]
    regions = [
        SplitRegion(0, "train", 1, 7),
        SplitRegion(1, "val_temporal", 0, 10),
        SplitRegion(1, "train", 2, 9),
    ]
    index = FiniteWindowIndex(records, regions, sample_length=4)

    assert index.window_count == 7
    assert [
        (index.sample(value).region_index, index.sample(value).window_start)
        for value in range(index.window_count)
    ] == [
        (0, 1),
        (0, 2),
        (0, 3),
        (2, 2),
        (2, 3),
        (2, 4),
        (2, 5),
    ]
    with pytest.raises(IndexError):
        index.sample(7)


def test_affine_sampler_covers_each_window_once_per_cycle() -> None:
    sampler = AffineCoverageSampler(
        window_count=11,
        cycles=3,
        seed=2026,
        global_batch_size=1,
    )
    values = list(sampler)

    assert sampler.total_real == 33
    assert sampler.total_padded == 0
    assert sampler.total_positions == 33
    for cycle in range(3):
        assert sorted(values[cycle * 11 : (cycle + 1) * 11]) == list(range(11))
    assert values == list(
        AffineCoverageSampler(11, 3, 2026, global_batch_size=1)
    )


def test_affine_sampler_padding_rank_partition_and_resume_are_deterministic() -> None:
    rank0 = AffineCoverageSampler(
        5, 2, 7, rank=0, world_size=2, global_batch_size=4
    )
    rank1 = AffineCoverageSampler(
        5, 2, 7, rank=1, world_size=2, global_batch_size=4
    )
    combined = [value for pair in zip(rank0, rank1) for value in pair]

    assert rank0.total_real == 10
    assert rank0.total_padded == 2
    assert rank0.total_positions == 12
    assert Counter(combined[:10]) == Counter({0: 2, 1: 2, 2: 2, 3: 2, 4: 2})
    resumed = AffineCoverageSampler(
        5,
        2,
        7,
        start_position=4,
        rank=0,
        world_size=1,
        global_batch_size=4,
    )
    full = AffineCoverageSampler(5, 2, 7, global_batch_size=4)
    assert list(resumed) == list(full)[4:]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"window_count": 0, "cycles": 1, "seed": 1},
        {"window_count": 1, "cycles": 0, "seed": 1},
        {
            "window_count": 1,
            "cycles": 1,
            "seed": 1,
            "world_size": 3,
            "global_batch_size": 4,
        },
    ],
)
def test_affine_sampler_rejects_invalid_plan(kwargs) -> None:
    with pytest.raises(ValueError):
        AffineCoverageSampler(**kwargs)
