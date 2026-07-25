from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath

import numpy as np


@dataclass(frozen=True, slots=True)
class RawSeries:
    source_id: str
    dataset_id: str
    series_id: str
    values: np.ndarray
    frequency: str | None


@dataclass(frozen=True, slots=True)
class PreparedSegment:
    source_id: str
    dataset_id: str
    series_id: str
    channel_id: str
    segment_index: int
    values: np.ndarray
    frequency: str | None

    @property
    def record_id(self) -> str:
        return f"{self.source_id}/{self.dataset_id}/{self.series_id}/{self.channel_id}/{self.segment_index}"


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    format_version: int
    source_id: str
    dataset_id: str
    series_id: str
    channel_id: str
    relative_shard_path: PurePosixPath
    offset: int
    length: int
    frequency: str | None
    split_group: str
    checksum: str

    @property
    def record_id(self) -> str:
        return f"{self.source_id}/{self.dataset_id}/{self.series_id}/{self.channel_id}/{self.offset}"


@dataclass(frozen=True, slots=True)
class SplitRegion:
    record_index: int
    split: str
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class SampleKey:
    sample_index: int
    region_index: int
    window_start: int
