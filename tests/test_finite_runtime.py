from __future__ import annotations

import numpy as np

from tsfm.runtime import build_finite_s3_dataset
from tsfm.s3.records import PreparedSegment
from tsfm.s3.shards import pack_segments


def test_finite_runtime_dataset_has_exact_window_length(tmp_path) -> None:
    values = np.arange(40, dtype=np.float32)
    segment = PreparedSegment(
        "utsd", "UTSD-12G", "series", "0", 0, values, "h"
    )
    records, digest = pack_segments([segment], tmp_path, max_shard_bytes=1024)
    assert records

    dataset, loaded_digest, window_count = build_finite_s3_dataset(
        tmp_path / "manifest.jsonl",
        split="train",
        seed=2026,
        patch_length=4,
        context_patches=3,
    )

    assert loaded_digest == digest
    assert window_count == 25
    assert len(dataset) == 25
    assert dataset[0]["window_start"] == 0
    assert dataset[24]["window_start"] == 24
