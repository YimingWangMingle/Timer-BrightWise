from __future__ import annotations

import hashlib
import os
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

import numpy as np

from tsfm.s3.manifest import write_manifest_atomic
from tsfm.s3.records import ManifestRecord, PreparedSegment


def _value_checksum(values: np.ndarray) -> str:
    little_endian = values.astype("<f4", copy=False)
    return hashlib.sha256(little_endian.tobytes()).hexdigest()


def pack_segments(
    segments: Iterable[PreparedSegment],
    processed_root: str | Path,
    max_shard_bytes: int = 2_000_000_000,
) -> tuple[list[ManifestRecord], str]:
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive")
    root = Path(processed_root)
    shard_dir = root / "shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    records: list[ManifestRecord] = []
    pending: list[PreparedSegment] = []
    pending_bytes = 0
    shard_index = 0

    def flush() -> None:
        nonlocal pending, pending_bytes, shard_index
        if not pending:
            return
        relative = PurePosixPath("shards") / f"shard-{shard_index:05d}.npy"
        destination = root.joinpath(*relative.parts)
        temporary = destination.with_name(f".{destination.name}.tmp.npy")
        total_values = sum(len(item.values) for item in pending)
        mapped = np.lib.format.open_memmap(
            temporary,
            mode="w+",
            dtype=np.float32,
            shape=(total_values,),
        )
        offset = 0
        for item in pending:
            values = np.asarray(item.values, dtype=np.float32)
            mapped[offset : offset + len(values)] = values
            records.append(
                ManifestRecord(
                    format_version=1,
                    source_id=item.source_id,
                    dataset_id=item.dataset_id,
                    series_id=item.series_id,
                    channel_id=item.channel_id,
                    relative_shard_path=relative,
                    offset=offset,
                    length=len(values),
                    frequency=item.frequency,
                    split_group="all",
                    checksum=_value_checksum(values),
                )
            )
            offset += len(values)
        mapped.flush()
        del mapped
        os.replace(temporary, destination)
        pending = []
        pending_bytes = 0
        shard_index += 1

    for item in segments:
        item_bytes = len(item.values) * np.dtype(np.float32).itemsize
        if item_bytes > max_shard_bytes:
            raise ValueError(
                f"{item.record_id}: segment exceeds max_shard_bytes"
            )
        if pending and pending_bytes + item_bytes > max_shard_bytes:
            flush()
        pending.append(item)
        pending_bytes += item_bytes
    flush()
    digest = write_manifest_atomic(root / "manifest.jsonl", records)
    return records, digest


def read_segment_mmap(
    processed_root: str | Path, record: ManifestRecord
) -> np.ndarray:
    path = Path(processed_root).joinpath(*record.relative_shard_path.parts)
    shard = np.load(path, mmap_mode="r", allow_pickle=False)
    stop = record.offset + record.length
    if record.offset < 0 or stop > len(shard):
        raise ValueError(f"{record.record_id}: segment crosses shard boundary")
    return shard[record.offset:stop]
