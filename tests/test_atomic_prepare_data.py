from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tsfm.conversion_state import validate_completed_conversion
from tsfm.s3.records import RawSeries

ROOT = Path(__file__).parents[1]


def _load_prepare_data_module():
    script = ROOT / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data_atomic", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_conversion_publishes_bound_utsd12g_directory(
    monkeypatch, tmp_path: Path
) -> None:
    prepare_data = _load_prepare_data_module()
    data_root = tmp_path / "tsfm-data"
    data_root.mkdir()
    source_manifest = "a" * 64
    inventory = {
        "format_version": 1,
        "projected_source_bytes": 5,
        "selected": [
            {
                "repository": "thuml/UTSD",
                "configuration": "UTSD-12G",
                "source_id": "utsd",
                "dataset_id": "UTSD-12G",
                "domain": "mixed-utsd",
                "split": "train",
                "revision": None,
                "file_format": "arrow",
                "data_files": [],
                "local_files": [str(tmp_path / "data-00000.arrow")],
                "estimated_source_bytes": 5,
                "source_manifest": source_manifest,
            }
        ],
    }
    (data_root / "inventory.json").write_text(
        json.dumps(inventory), encoding="utf-8"
    )
    policy = {
        "format_version": 1,
        "repositories": [
            {
                "source_id": "utsd",
                "repository": "thuml/UTSD",
                "mode": "local_arrow",
                "configurations": ["UTSD-12G"],
                "domains": {"UTSD-12G": "mixed-utsd"},
                "split": "train",
            }
        ],
        "selection": {
            "minimum_dataset_groups_by_source": {"utsd": 1},
            "minimum_domain_groups": 1,
            "maximum_source_cache_bytes": 1000,
            "maximum_processed_bytes": 1000,
            "minimum_segment_length": 4,
            "minimum_variance": 1e-8,
            "maximum_shard_bytes": 1000,
        },
    }
    config = tmp_path / "policy.json"
    config.write_text(json.dumps(policy, sort_keys=True), encoding="utf-8")
    args = SimpleNamespace(config=config)

    class FakeAdapter:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def iter_series(self):
            yield RawSeries(
                source_id="utsd",
                dataset_id="UTSD-12G",
                series_id="series",
                values=np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32),
                frequency="h",
            )

    monkeypatch.setattr(prepare_data, "_guard", lambda *args: SimpleNamespace(data_root=data_root))
    monkeypatch.setattr(prepare_data, "UTSDAdapter", FakeAdapter)

    prepare_data.convert(args, policy)

    final = data_root / "processed" / "utsd-12g"
    binding = validate_completed_conversion(
        final,
        source_manifest=source_manifest,
        policy_digest=hashlib.sha256(config.read_bytes()).hexdigest(),
    )
    assert binding.records == 1
    assert not list((data_root / "processed").glob(".utsd-12g.incomplete-*"))
