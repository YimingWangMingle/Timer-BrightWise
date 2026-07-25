from dataclasses import replace
from pathlib import PurePosixPath

import numpy as np
import pytest

from tsfm.s3.manifest import load_manifest, write_manifest_atomic
from tsfm.s3.records import ManifestRecord, PreparedSegment
from tsfm.s3.shards import pack_segments, read_segment_mmap


def prepared(name: str, values: list[float]) -> PreparedSegment:
    return PreparedSegment(
        source_id="utsd",
        dataset_id="weather",
        series_id=name,
        channel_id="0",
        segment_index=0,
        values=np.asarray(values, dtype=np.float32),
        frequency="H",
    )


def manifest_record() -> ManifestRecord:
    return ManifestRecord(
        format_version=1,
        source_id="utsd",
        dataset_id="weather",
        series_id="station-1",
        channel_id="0",
        relative_shard_path=PurePosixPath("shards/shard-00000.npy"),
        offset=2,
        length=10,
        frequency="H",
        split_group="all",
        checksum="a" * 64,
    )


def test_manifest_round_trip_uses_portable_relative_paths(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    digest = write_manifest_atomic(path, [manifest_record()])

    assert load_manifest(path, expected_checksum=digest) == [manifest_record()]
    assert "\\" not in path.read_text(encoding="utf-8")
    assert len(digest) == 64


def test_manifest_rejects_version_checksum_and_parent_path(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest_atomic(path, [manifest_record()])
    with pytest.raises(ValueError, match="manifest checksum"):
        load_manifest(path, expected_checksum="0" * 64)

    write_manifest_atomic(path, [replace(manifest_record(), format_version=2)])
    with pytest.raises(ValueError, match="format_version"):
        load_manifest(path)

    unsafe = replace(
        manifest_record(), relative_shard_path=PurePosixPath("../escape.npy")
    )
    write_manifest_atomic(path, [unsafe])
    with pytest.raises(ValueError, match="relative_shard_path"):
        load_manifest(path)


def test_shards_roll_over_and_are_memory_mapped(tmp_path) -> None:
    records, digest = pack_segments(
        [prepared("a", [0, 1, 2, 3]), prepared("b", [4, 5, 6, 7])],
        tmp_path,
        max_shard_bytes=16,
    )

    assert [record.relative_shard_path.as_posix() for record in records] == [
        "shards/shard-00000.npy",
        "shards/shard-00001.npy",
    ]
    assert len(digest) == 64
    values = read_segment_mmap(tmp_path, records[1])
    assert isinstance(values, np.memmap)
    np.testing.assert_array_equal(
        values, np.array([4, 5, 6, 7], dtype=np.float32)
    )
    assert not list(tmp_path.rglob("*.tmp*"))


def test_pack_rejects_one_segment_larger_than_shard_limit(tmp_path) -> None:
    with pytest.raises(ValueError, match="segment exceeds"):
        pack_segments(
            [prepared("a", [0, 1, 2, 3, 4])],
            tmp_path,
            max_shard_bytes=16,
        )
