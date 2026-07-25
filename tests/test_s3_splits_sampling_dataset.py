from collections import Counter
from pathlib import PurePosixPath

import numpy as np

from tsfm.s3.dataset import S3WindowDataset
from tsfm.s3.records import ManifestRecord, PreparedSegment, SplitRegion
from tsfm.s3.sampling import CounterSampler, HierarchicalIndex
from tsfm.s3.shards import pack_segments
from tsfm.s3.splits import build_split_regions


def record(
    source: str, series: str, length: int = 40_000, channel: str = "0"
) -> ManifestRecord:
    return ManifestRecord(
        1,
        source,
        f"{source}-dataset",
        series,
        channel,
        PurePosixPath("shards/a.npy"),
        0,
        length,
        None,
        "all",
        "a" * 64,
    )


def test_heldout_is_stable_and_keeps_all_channels_together() -> None:
    records = [record("utsd", f"s{index}") for index in range(10)]
    records.append(record("utsd", "s0", channel="1"))

    first = build_split_regions(records, sample_length=2_976, seed=2026)
    second = build_split_regions(records, sample_length=2_976, seed=2026)

    assert first == second
    heldout = {
        region.record_index
        for region in first
        if region.split == "val_heldout"
    }
    assert (0 in heldout) == (10 in heldout)
    assert heldout


def test_temporal_regions_share_no_point_and_short_series_is_train_only() -> None:
    regions = build_split_regions(
        [record("utsd", "long")],
        sample_length=2_976,
        heldout_fraction=0.0,
    )
    train = next(region for region in regions if region.split == "train")
    temporal = next(
        region for region in regions if region.split == "val_temporal"
    )
    assert train.stop <= temporal.start
    assert train.stop - train.start >= 2_976
    assert temporal.stop - temporal.start >= 2_976

    short = build_split_regions(
        [record("utsd", "short", length=5_000)],
        sample_length=2_976,
        heldout_fraction=0.0,
    )
    assert [(region.split, region.start, region.stop) for region in short] == [
        ("train", 0, 5_000)
    ]


def test_hierarchy_is_deterministic_and_balances_sources() -> None:
    records = [record("large", f"s{index}", 10_000) for index in range(20)]
    records.append(record("small", "only", 10_000))
    regions = [
        SplitRegion(index, "train", 0, item.length)
        for index, item in enumerate(records)
    ]
    hierarchy = HierarchicalIndex(
        records, regions, "train", sample_length=2_976, seed=2026
    )

    keys = [hierarchy.sample(index) for index in range(10_000)]

    assert keys == [hierarchy.sample(index) for index in range(10_000)]
    counts = Counter(
        records[regions[key.region_index].record_index].source_id
        for key in keys
    )
    assert 0.47 < counts["large"] / len(keys) < 0.53
    assert all(
        regions[key.region_index].start
        <= key.window_start
        <= regions[key.region_index].stop - 2_976
        for key in keys
    )


def test_counter_sampler_resume_and_rank_partition_do_not_overlap() -> None:
    rank0 = list(CounterSampler(100, rank=0, world_size=2).take(3))
    rank1 = list(CounterSampler(100, rank=1, world_size=2).take(3))

    assert rank0 == [100, 102, 104]
    assert rank1 == [101, 103, 105]
    assert not set(rank0) & set(rank1)


def test_window_has_shifted_target_and_context_only_statistics(tmp_path) -> None:
    values = np.arange(40, dtype=np.float32)
    segment = PreparedSegment(
        "utsd", "weather", "linear", "0", 0, values, "H"
    )
    records, _ = pack_segments([segment], tmp_path, max_shard_bytes=1_024)
    regions = [SplitRegion(0, "train", 0, 40)]
    hierarchy = HierarchicalIndex(
        records, regions, "train", sample_length=16, seed=7
    )
    dataset = S3WindowDataset(
        tmp_path,
        records,
        regions,
        hierarchy,
        patch_length=4,
        context_patches=3,
    )

    item = dataset[0]
    start = item["window_start"]
    context = values[start : start + 12]
    target = values[start + 4 : start + 16]
    mean = context.mean()
    scale = np.sqrt(context.var() + 1e-5)

    np.testing.assert_allclose(item["context"].numpy(), (context - mean) / scale)
    np.testing.assert_allclose(item["target"].numpy(), (target - mean) / scale)
    assert np.isclose(float(item["mean"]), mean)
    assert np.isclose(float(item["scale"]), scale)
    assert item["source_id"] == "utsd"
