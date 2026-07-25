from __future__ import annotations

from pathlib import Path

from tsfm.s3.dataset import S3WindowDataset
from tsfm.s3.manifest import load_manifest, manifest_checksum
from tsfm.s3.sampling import HierarchicalIndex
from tsfm.s3.splits import build_split_regions


def build_s3_dataset(
    manifest_path: str | Path,
    *,
    split: str,
    seed: int,
    patch_length: int = 96,
    context_patches: int = 30,
) -> tuple[S3WindowDataset, str]:
    path = Path(manifest_path)
    records = load_manifest(path)
    digest = manifest_checksum(records)
    sample_length = (context_patches + 1) * patch_length
    regions = build_split_regions(records, sample_length=sample_length, seed=seed)
    hierarchy = HierarchicalIndex(
        records,
        regions,
        split,
        sample_length=sample_length,
        seed=seed,
    )
    return (
        S3WindowDataset(
            path.parent,
            records,
            regions,
            hierarchy,
            patch_length=patch_length,
            context_patches=context_patches,
        ),
        digest,
    )
