from __future__ import annotations

import bisect
import hashlib
import math
from collections.abc import Iterator

from torch.utils.data import Sampler

from tsfm.s3.records import ManifestRecord, SampleKey, SplitRegion


class FiniteWindowIndex:
    def __init__(
        self,
        records: list[ManifestRecord],
        regions: list[SplitRegion],
        sample_length: int,
        split: str = "train",
    ) -> None:
        if sample_length <= 0:
            raise ValueError("sample_length must be positive")
        self.records = records
        self.regions = regions
        self.sample_length = sample_length
        self.split = split
        self.region_indices: list[int] = []
        self.cumulative_windows: list[int] = []
        total = 0
        for region_index, region in enumerate(regions):
            if not 0 <= region.record_index < len(records):
                raise ValueError(f"region has invalid record index: {region.record_index}")
            available = region.stop - region.start - sample_length + 1
            if region.split != split or available <= 0:
                continue
            total += available
            self.region_indices.append(region_index)
            self.cumulative_windows.append(total)
        if total == 0:
            raise ValueError(f"split {split!r} has no finite windows")
        self.window_count = total

    def sample(self, canonical_index: int) -> SampleKey:
        if not 0 <= canonical_index < self.window_count:
            raise IndexError(canonical_index)
        prefix_index = bisect.bisect_right(
            self.cumulative_windows, canonical_index
        )
        previous = 0 if prefix_index == 0 else self.cumulative_windows[prefix_index - 1]
        region_index = self.region_indices[prefix_index]
        region = self.regions[region_index]
        return SampleKey(
            sample_index=canonical_index,
            region_index=region_index,
            window_start=region.start + canonical_index - previous,
        )


class AffineCoverageSampler(Sampler[int]):
    def __init__(
        self,
        window_count: int,
        cycles: int,
        seed: int,
        start_position: int = 0,
        rank: int = 0,
        world_size: int = 1,
        global_batch_size: int = 1,
    ) -> None:
        if window_count <= 0:
            raise ValueError("window_count must be positive")
        if cycles <= 0:
            raise ValueError("cycles must be positive")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid rank or world_size")
        if global_batch_size <= 0 or global_batch_size % world_size != 0:
            raise ValueError("global_batch_size must be divisible by world_size")
        self.window_count = window_count
        self.cycles = cycles
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.global_batch_size = global_batch_size
        self.total_real = window_count * cycles
        self.total_positions = (
            (self.total_real + global_batch_size - 1) // global_batch_size
        ) * global_batch_size
        self.total_padded = self.total_positions - self.total_real
        if not 0 <= start_position <= self.total_positions:
            raise ValueError("start_position is outside the finite plan")
        self.start_position = start_position

    def _parameters(self, cycle: int) -> tuple[int, int]:
        payload = f"{self.seed}:{cycle}".encode("ascii")
        digest = hashlib.sha256(payload).digest()
        multiplier = int.from_bytes(digest[:8], "big") % self.window_count
        if multiplier == 0:
            multiplier = 1
        while math.gcd(multiplier, self.window_count) != 1:
            multiplier = (multiplier + 1) % self.window_count
            if multiplier == 0:
                multiplier = 1
        increment = int.from_bytes(digest[8:16], "big") % self.window_count
        return multiplier, increment

    def canonical_index(self, position: int) -> int:
        if not 0 <= position < self.total_positions:
            raise IndexError(position)
        if position < self.total_real:
            cycle, offset = divmod(position, self.window_count)
        else:
            cycle = self.cycles - 1
            offset = position - self.total_real
        multiplier, increment = self._parameters(cycle)
        return (multiplier * offset + increment) % self.window_count

    def position_is_real(self, position: int) -> bool:
        if not 0 <= position < self.total_positions:
            raise IndexError(position)
        return position < self.total_real

    def __iter__(self) -> Iterator[int]:
        first = self.start_position + self.rank
        for position in range(first, self.total_positions, self.world_size):
            yield self.canonical_index(position)

    def __len__(self) -> int:
        first = self.start_position + self.rank
        if first >= self.total_positions:
            return 0
        return (self.total_positions - 1 - first) // self.world_size + 1
