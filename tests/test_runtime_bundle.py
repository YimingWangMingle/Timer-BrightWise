import numpy as np

from tsfm.runtime import build_s3_dataset
from tsfm.s3.records import PreparedSegment
from tsfm.s3.shards import pack_segments


def test_runtime_builds_training_dataset_from_manifest(tmp_path) -> None:
    values = np.arange(80, dtype=np.float32)
    records, digest = pack_segments(
        [PreparedSegment("utsd", "weather", "s1", "0", 0, values, "H")],
        tmp_path,
        max_shard_bytes=1_024,
    )
    assert records

    dataset, actual_digest = build_s3_dataset(
        tmp_path / "manifest.jsonl",
        split="train",
        seed=2026,
        patch_length=4,
        context_patches=3,
    )
    item = dataset[0]

    assert actual_digest == digest
    assert item["context"].shape == (12,)
    assert item["target"].shape == (12,)
