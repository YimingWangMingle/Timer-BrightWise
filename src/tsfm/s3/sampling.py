from __future__ import annotations

import hashlib
import itertools
import random
from collections import defaultdict
from collections.abc import Iterator

from torch.utils.data import Sampler

from tsfm.s3.records import ManifestRecord, SampleKey, SplitRegion


def _seed_for(seed: int, sample_index: int) -> int:
    payload = f"{seed}:{sample_index}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


class HierarchicalIndex:
    def __init__(
        self,
        records: list[ManifestRecord],
        regions: list[SplitRegion],
        split: str,
        sample_length: int,
        seed: int,
    ) -> None:
        self.records = records
        self.regions = regions
        self.sample_length = sample_length
        self.seed = seed
        grouped: dict[str, list[int]] = defaultdict(list)
        for region_index, region in enumerate(regions):
            if (
                region.split == split
                and region.stop - region.start >= sample_length
            ):
                grouped[records[region.record_index].source_id].append(
                    region_index
                )
        if not grouped:
            raise ValueError(f"split {split!r} has no valid regions")
        self.sources = tuple(sorted(grouped))
        self.region_indices = {
            source: tuple(grouped[source]) for source in self.sources
        }

    def sample(self, sample_index: int) -> SampleKey:
        if sample_index < 0:
            raise IndexError(sample_index)
        rng = random.Random(_seed_for(self.seed, sample_index))
        source = self.sources[rng.randrange(len(self.sources))]
        choices = self.region_indices[source]
        region_index = choices[rng.randrange(len(choices))]
        region = self.regions[region_index]
        start = rng.randint(region.start, region.stop - self.sample_length)
        return SampleKey(sample_index, region_index, start)


class CounterSampler(Sampler[int]):
    def __init__(
        self, start_sample: int, rank: int = 0, world_size: int = 1
    ) -> None:
        if start_sample < 0:
            raise ValueError("start_sample must be non-negative")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError("invalid rank or world_size")
        self.start_sample = start_sample
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return itertools.count(
            self.start_sample + self.rank, self.world_size
        )

    def __len__(self) -> int:
        return 2**63 - 1

    def take(self, count: int) -> Iterator[int]:
        return itertools.islice(iter(self), count)
