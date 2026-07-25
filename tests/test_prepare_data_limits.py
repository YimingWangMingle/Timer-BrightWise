import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]


def _load_prepare_data_module():
    script = ROOT / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data_limits", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy() -> dict:
    return {
        "format_version": 1,
        "repositories": [
            {
                "source_id": "source",
                "repository": "owner/repository",
                "configurations": ["group"],
                "domains": {"group": "domain"},
                "split": "train",
            }
        ],
        "selection": {
            "minimum_dataset_groups_by_source": {"source": 1},
            "minimum_domain_groups": 1,
            "maximum_source_cache_bytes": 10,
            "maximum_processed_bytes": 10,
        },
    }


@pytest.mark.parametrize(
    ("oversized_root", "setting"),
    [
        ("hf", "maximum_source_cache_bytes"),
        ("processed", "maximum_processed_bytes"),
    ],
)
def test_validation_report_enforces_independent_byte_limits(
    tmp_path, oversized_root, setting
) -> None:
    prepare_data = _load_prepare_data_module()
    hf_home = tmp_path / "hf"
    processed = tmp_path / "processed"
    hf_home.mkdir()
    processed.mkdir()
    (tmp_path / oversized_root / "payload.bin").write_bytes(b"x" * 11)
    inventory = prepare_data.build_inventory(
        _policy(),
        lambda *args, **kwargs: SimpleNamespace(
            info=SimpleNamespace(download_size=1, dataset_size=1)
        ),
        hf_home,
        "mirror",
        lambda _: None,
    )
    records = [SimpleNamespace(source_id="source", dataset_id="group")]

    with pytest.raises(OSError, match=setting):
        prepare_data.build_validation_report(
            _policy(), inventory, records, hf_home, processed
        )
