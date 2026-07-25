import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tsfm.s3.adapters import DatasetSpec, LOTSAAdapter


ROOT = Path(__file__).parents[1]
REVISION = "8191fd29eb5cf906ec55effca44d8059888b615d"
LOTSA_FILES = {
    "traffic_hourly": (
        "traffic_hourly/data-00000-of-00001.arrow",
        59_934_920,
    ),
    "beijing_air_quality": (
        "beijing_air_quality/data-00000-of-00001.arrow",
        18_516_128,
    ),
    "weather": ("weather/data-00000-of-00001.arrow", 171_846_176),
}


def _load_prepare_data_module():
    script = ROOT / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data_exact_arrow", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checked_policy() -> dict:
    return json.loads(
        (ROOT / "configs" / "data" / "server_validation.json").read_text(
            encoding="utf-8"
        )
    )


def test_policy_pins_exact_lotsa_arrow_files() -> None:
    policy = _checked_policy()
    lotsa = next(
        source for source in policy["repositories"] if source["source_id"] == "lotsa"
    )

    assert lotsa["revision"] == REVISION
    assert lotsa["file_format"] == "arrow"
    assert lotsa["data_files"] == {
        configuration: [{"path": path, "size": size}]
        for configuration, (path, size) in LOTSA_FILES.items()
    }


def test_inventory_probes_exact_lotsa_files_without_lotsa_builder(tmp_path) -> None:
    prepare_data = _load_prepare_data_module()
    builder_calls = []
    probe_calls = []

    def load_builder(repository, configuration, cache_dir):
        builder_calls.append((repository, configuration, cache_dir))
        if repository == "Salesforce/lotsa_data":
            raise AssertionError("LOTSA builder must not be called")
        return SimpleNamespace(
            info=SimpleNamespace(download_size=494_696_643, dataset_size=0)
        )

    def probe_file(url):
        probe_calls.append(url)
        for path, size in LOTSA_FILES.values():
            if url.endswith(path):
                return size
        raise AssertionError(f"unexpected URL: {url}")

    inventory = prepare_data.build_inventory(
        _checked_policy(),
        load_builder,
        tmp_path,
        "https://hf-mirror.com",
        lambda _: None,
        probe_file=probe_file,
    )

    assert builder_calls == [("thuml/UTSD", "UTSD-1G", tmp_path)]
    assert len(probe_calls) == 3
    assert all(REVISION in url for url in probe_calls)
    assert inventory["projected_source_bytes"] == 744_993_867
    lotsa_entries = [
        entry for entry in inventory["selected"] if entry["source_id"] == "lotsa"
    ]
    assert [entry["data_files"][0]["path"] for entry in lotsa_entries] == [
        LOTSA_FILES[configuration][0]
        for configuration in ("traffic_hourly", "beijing_air_quality", "weather")
    ]


def test_inventory_rejects_pinned_file_size_mismatch(tmp_path) -> None:
    prepare_data = _load_prepare_data_module()

    with pytest.raises(OSError, match="size mismatch.*traffic_hourly"):
        prepare_data.build_inventory(
            _checked_policy(),
            lambda *args, **kwargs: SimpleNamespace(
                info=SimpleNamespace(download_size=494_696_643, dataset_size=0)
            ),
            tmp_path,
            "https://hf-mirror.com",
            lambda _: None,
            probe_file=lambda _: 1,
        )


def test_lotsa_adapter_downloads_exact_files_then_uses_arrow_loader(tmp_path) -> None:
    download_calls = []
    loader_calls = []
    local_arrow = tmp_path / "cached.arrow"

    def downloader(**kwargs):
        download_calls.append(kwargs)
        return str(local_arrow)

    def loader(path, *, data_files, split, streaming, cache_dir):
        loader_calls.append((path, data_files, split, streaming, cache_dir))
        return [{"item_id": "series-1", "target": [1.0, 2.0], "freq": "H"}]

    remote_path = LOTSA_FILES["traffic_hourly"][0]
    spec = DatasetSpec(
        "Salesforce/lotsa_data",
        "traffic_hourly",
        "lotsa",
        "traffic_hourly",
        revision=REVISION,
        file_format="arrow",
        data_files=(remote_path,),
    )

    rows = list(
        LOTSAAdapter(
            spec,
            tmp_path,
            loader=loader,
            downloader=downloader,
        ).iter_series()
    )

    assert rows[0].series_id == "series-1"
    assert download_calls == [
        {
            "repo_id": "Salesforce/lotsa_data",
            "filename": remote_path,
            "repo_type": "dataset",
            "revision": REVISION,
            "cache_dir": tmp_path,
        }
    ]
    assert loader_calls == [
        (
            "arrow",
            {"train": [str(local_arrow)]},
            "train",
            True,
            tmp_path,
        )
    ]


def test_runbook_documents_pinned_exact_arrow_access() -> None:
    runbook = (ROOT / "docs" / "server-validation-runbook.md").read_text(
        encoding="utf-8"
    )

    assert REVISION in runbook
    assert "traffic_hourly/data-00000-of-00001.arrow" in runbook
    assert "without scanning the LOTSA repository" in runbook
