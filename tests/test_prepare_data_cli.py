import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).parents[1]


def _load_prepare_data_module():
    script = ROOT / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _policy() -> dict:
    return {
        "format_version": 1,
        "repositories": [
            {
                "source_id": "utsd",
                "repository": "thuml/UTSD",
                "configurations": ["UTSD-1G"],
                "domains": {"UTSD-1G": "mixed-utsd"},
                "split": "train",
            },
            {
                "source_id": "lotsa",
                "repository": "Salesforce/lotsa_data",
                "configurations": [
                    "traffic_hourly",
                    "beijing_air_quality",
                    "weather",
                ],
                "domains": {
                    "traffic_hourly": "traffic",
                    "beijing_air_quality": "air-quality",
                    "weather": "weather",
                },
                "split": "train",
            },
        ],
        "selection": {
            "minimum_dataset_groups_by_source": {"utsd": 1, "lotsa": 3},
            "minimum_domain_groups": 4,
            "maximum_source_cache_bytes": 2_000_000_000,
            "maximum_processed_bytes": 2_000_000_000,
        },
    }


def _builder(size: int):
    return SimpleNamespace(
        info=SimpleNamespace(download_size=size, dataset_size=size + 1)
    )


def test_checked_in_policy_declares_exactly_four_explicit_groups() -> None:
    policy = json.loads(
        (ROOT / "configs" / "data" / "server_validation.json").read_text(
            encoding="utf-8"
        )
    )
    groups = [
        (repository["repository"], configuration)
        for repository in policy["repositories"]
        for configuration in repository["configurations"]
    ]

    assert groups == [
        ("thuml/UTSD", "UTSD-1G"),
        ("Salesforce/lotsa_data", "traffic_hourly"),
        ("Salesforce/lotsa_data", "beijing_air_quality"),
        ("Salesforce/lotsa_data", "weather"),
    ]
    assert all(
        repository["configurations"] != "discover"
        for repository in policy["repositories"]
    )


def test_build_inventory_preserves_declaration_order_and_domains(tmp_path) -> None:
    prepare_data = _load_prepare_data_module()
    calls = []
    progress = []

    def load_builder(repository, configuration, cache_dir):
        calls.append((repository, configuration, cache_dir))
        return _builder(100 + len(calls))

    inventory = prepare_data.build_inventory(
        _policy(),
        load_builder,
        tmp_path / "hf-cache",
        "https://hf-mirror.com",
        progress.append,
    )

    assert [(item[0], item[1]) for item in calls] == [
        ("thuml/UTSD", "UTSD-1G"),
        ("Salesforce/lotsa_data", "traffic_hourly"),
        ("Salesforce/lotsa_data", "beijing_air_quality"),
        ("Salesforce/lotsa_data", "weather"),
    ]
    assert [item["domain"] for item in inventory["selected"]] == [
        "mixed-utsd",
        "traffic",
        "air-quality",
        "weather",
    ]
    assert inventory["entries"] == inventory["selected"]
    assert inventory["projected_source_bytes"] == 410
    assert all(str(tmp_path / "hf-cache") == str(item[2]) for item in calls)
    assert progress[0].endswith("utsd/thuml/UTSD/UTSD-1G")


def test_build_inventory_rejects_duplicate_repository_configuration() -> None:
    prepare_data = _load_prepare_data_module()
    policy = _policy()
    policy["repositories"][1]["configurations"].append("traffic_hourly")

    with pytest.raises(ValueError, match="duplicate.*traffic_hourly"):
        prepare_data.build_inventory(
            policy, lambda *args, **kwargs: _builder(1), None, "mirror", lambda _: None
        )


def test_build_inventory_error_names_repository_configuration_and_endpoint() -> None:
    prepare_data = _load_prepare_data_module()

    def failing_loader(repository, configuration, cache_dir):
        raise OSError("network unreachable")

    with pytest.raises(RuntimeError) as error:
        prepare_data.build_inventory(
            _policy(),
            failing_loader,
            "/cache",
            "https://hf-mirror.com",
            lambda _: None,
        )

    message = str(error.value)
    assert "thuml/UTSD" in message
    assert "UTSD-1G" in message
    assert "https://hf-mirror.com" in message


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda policy: policy["selection"][
                "minimum_dataset_groups_by_source"
            ].__setitem__("lotsa", 4),
            "lotsa",
        ),
        (
            lambda policy: policy["repositories"][1]["domains"].__setitem__(
                "weather", "traffic"
            ),
            "domain",
        ),
        (
            lambda policy: policy["selection"].__setitem__(
                "maximum_source_cache_bytes", 3
            ),
            "projected source",
        ),
    ],
)
def test_build_inventory_enforces_source_domain_and_projected_byte_quotas(
    mutate, message
) -> None:
    prepare_data = _load_prepare_data_module()
    policy = _policy()
    mutate(policy)

    with pytest.raises((ValueError, OSError), match=message):
        prepare_data.build_inventory(
            policy, lambda *args, **kwargs: _builder(1), None, "mirror", lambda _: None
        )


def test_validation_report_accounts_for_projected_cache_and_processed_bytes(
    tmp_path,
) -> None:
    prepare_data = _load_prepare_data_module()
    hf_home = tmp_path / "hf-home"
    processed = tmp_path / "processed"
    (hf_home / "datasets").mkdir(parents=True)
    (processed / "shards").mkdir(parents=True)
    (hf_home / "datasets" / "metadata.bin").write_bytes(b"a" * 11)
    (processed / "shards" / "shard.npy").write_bytes(b"b" * 17)
    inventory = prepare_data.build_inventory(
        _policy(), lambda *args, **kwargs: _builder(5), hf_home, "mirror", lambda _: None
    )
    records = [
        SimpleNamespace(source_id=item["source_id"], dataset_id=item["dataset_id"])
        for item in inventory["selected"]
    ]

    report = prepare_data.build_validation_report(
        _policy(), inventory, records, hf_home, processed
    )

    assert report["projected_source_bytes"] == 20
    assert report["hf_cache_bytes"] == 11
    assert report["processed_bytes"] == 17
    assert report["dataset_groups"] == {
        "lotsa": ["beijing_air_quality", "traffic_hourly", "weather"],
        "utsd": ["UTSD-1G"],
    }


def test_validation_report_rejects_selected_group_without_manifest_records(
    tmp_path,
) -> None:
    prepare_data = _load_prepare_data_module()
    inventory = prepare_data.build_inventory(
        _policy(), lambda *args, **kwargs: _builder(1), tmp_path, "mirror", lambda _: None
    )
    records = [
        SimpleNamespace(source_id=item["source_id"], dataset_id=item["dataset_id"])
        for item in inventory["selected"]
        if item["dataset_id"] != "weather"
    ]

    with pytest.raises(ValueError, match="lotsa/weather"):
        prepare_data.build_validation_report(
            _policy(), inventory, records, tmp_path / "hf", tmp_path / "processed"
        )


def test_server_runbook_documents_bounded_mirror_workflow() -> None:
    runbook = (ROOT / "docs" / "server-validation-runbook.md").read_text(
        encoding="utf-8"
    )

    assert "HF_ENDPOINT" in runbook
    assert "UTSD-1G" in runbook
    assert "traffic_hourly" in runbook
    assert "beijing_air_quality" in runbook
    assert "weather" in runbook
    assert "2,000,000,000" in runbook
    assert "inventory.json" in runbook
    assert runbook.index("inventory.json") < runbook.index("prepare_data.py convert")


def test_prepare_data_defaults_to_non_mutating_dry_run() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "prepare_data.py"),
            "--config",
            str(ROOT / "configs" / "data" / "server_validation.json"),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert "DRY RUN" in result.stdout
    assert "thuml/UTSD" in result.stdout
    assert "Salesforce/lotsa_data" in result.stdout
    assert not (ROOT / "data").exists()
