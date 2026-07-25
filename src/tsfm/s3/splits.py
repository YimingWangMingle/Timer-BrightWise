from __future__ import annotations

import hashlib
import math
from collections import defaultdict

from tsfm.s3.records import ManifestRecord, SplitRegion

SeriesIdentity = tuple[str, str, str]


def _identity(record: ManifestRecord) -> SeriesIdentity:
    return record.source_id, record.dataset_id, record.series_id


def _score(identity: SeriesIdentity, seed: int) -> int:
    payload = "\x1f".join((*identity, str(seed))).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def build_split_regions(
    records: list[ManifestRecord],
    sample_length: int = 2_976,
    heldout_fraction: float = 0.1,
    temporal_fraction: float = 0.1,
    seed: int = 2026,
) -> list[SplitRegion]:
    if sample_length <= 0:
        raise ValueError("sample_length must be positive")
    if not 0.0 <= heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be in [0, 1)")
    if not 0.0 < temporal_fraction < 1.0:
        raise ValueError("temporal_fraction must be in (0, 1)")

    identities_by_source: dict[str, set[SeriesIdentity]] = defaultdict(set)
    for item in records:
        identities_by_source[item.source_id].add(_identity(item))

    heldout: set[SeriesIdentity] = set()
    for identities in identities_by_source.values():
        ordered = sorted(identities, key=lambda item: (_score(item, seed), item))
        count = math.floor(len(ordered) * heldout_fraction)
        if heldout_fraction > 0.0 and len(ordered) >= 2:
            count = max(1, count)
        heldout.update(ordered[:count])

    regions: list[SplitRegion] = []
    for index, item in enumerate(records):
        if _identity(item) in heldout:
            if item.length >= sample_length:
                regions.append(SplitRegion(index, "val_heldout", 0, item.length))
            continue
        boundary = int(item.length * (1.0 - temporal_fraction))
        if boundary >= sample_length and item.length - boundary >= sample_length:
            regions.append(SplitRegion(index, "train", 0, boundary))
            regions.append(
                SplitRegion(index, "val_temporal", boundary, item.length)
            )
        elif item.length >= sample_length:
            regions.append(SplitRegion(index, "train", 0, item.length))
    return regions
