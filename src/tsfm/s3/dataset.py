from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from tsfm.data import normalize_context_target
from tsfm.s3.records import ManifestRecord, SplitRegion
from tsfm.s3.sampling import HierarchicalIndex
from tsfm.s3.shards import read_segment_mmap


class S3WindowDataset(Dataset[dict[str, object]]):
    def __init__(
        self,
        processed_root: str | Path,
        records: list[ManifestRecord],
        regions: list[SplitRegion],
        hierarchy: HierarchicalIndex,
        patch_length: int = 96,
        context_patches: int = 30,
    ) -> None:
        self.processed_root = Path(processed_root)
        self.records = records
        self.regions = regions
        self.hierarchy = hierarchy
        self.patch_length = patch_length
        self.context_patches = context_patches
        expected = (context_patches + 1) * patch_length
        if hierarchy.sample_length != expected:
            raise ValueError(f"sample_length must be {expected}")

    def __len__(self) -> int:
        return 2**63 - 1

    def __getitem__(self, sample_index: int) -> dict[str, object]:
        key = self.hierarchy.sample(sample_index)
        region = self.regions[key.region_index]
        record = self.records[region.record_index]
        sample_length = (self.context_patches + 1) * self.patch_length
        stop = key.window_start + sample_length
        if key.window_start < region.start or stop > region.stop:
            raise ValueError(
                f"{record.record_id}: requested window crosses split boundary"
            )
        segment = read_segment_mmap(self.processed_root, record)
        window = torch.from_numpy(segment[key.window_start:stop].copy())
        context_length = self.context_patches * self.patch_length
        context = window[:context_length].unsqueeze(0)
        target = window[self.patch_length:].unsqueeze(0)
        normalized = normalize_context_target(context, target)
        return {
            "context": normalized.context.squeeze(0),
            "target": normalized.target.squeeze(0),
            "mean": normalized.mean.squeeze(0),
            "scale": normalized.scale.squeeze(0),
            "source_id": record.source_id,
            "record_id": record.record_id,
            "sample_index": sample_index,
            "window_start": key.window_start,
        }
