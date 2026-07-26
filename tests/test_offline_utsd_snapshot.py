from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from tsfm.artifacts import write_artifact_manifest
from tsfm.s3.adapters import DatasetSpec, UTSDAdapter

ROOT = Path(__file__).parents[1]
UTSD_REVISION = "7326ff5f4578da73d843fd675d760c6c6054017f"


def _load_prepare_data_module():
    script = ROOT / "scripts" / "prepare_data.py"
    spec = importlib.util.spec_from_file_location("prepare_data_offline_utsd", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _snapshot(source_root: Path) -> tuple[Path, tuple[Path, ...]]:
    snapshot = source_root / "raw" / "utsd" / "UTSD-12G"
    snapshot.mkdir(parents=True)
    arrow_files = (
        snapshot / "data-00000-of-00002.arrow",
        snapshot / "data-00001-of-00002.arrow",
    )
    arrow_files[0].write_bytes(b"arrow-a")
    arrow_files[1].write_bytes(b"arrow-b")
    (snapshot / "dataset_info.json").write_bytes(b"{}\n")
    (snapshot / "state.json").write_bytes(b"{}\n")
    write_artifact_manifest(snapshot, snapshot.parent / "UTSD-12G.sha256.json")
    return snapshot, arrow_files


def _fixture_policy(snapshot: Path) -> dict:
    policy = json.loads(
        (ROOT / "configs" / "data" / "utsd12g_production.json").read_text(
            encoding="utf-8"
        )
    )
    source = policy["repositories"][0]
    source["expected_files"] = 4
    source["expected_bytes"] = sum(
        path.stat().st_size for path in snapshot.iterdir() if path.is_file()
    )
    return policy


def test_local_utsd_adapter_uses_exact_arrow_files_without_hub(
    tmp_path: Path,
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    files = (tmp_path / "a.arrow", tmp_path / "b.arrow")
    for item in files:
        item.write_bytes(b"arrow")

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"item_id": "s", "freq": "h", "target": [1.0, 2.0, 3.0]}]

    spec = DatasetSpec(
        "thuml/UTSD",
        "UTSD-12G",
        "utsd",
        "UTSD-12G",
        file_format="arrow",
        local_files=tuple(str(item) for item in files),
    )
    rows = list(UTSDAdapter(spec, tmp_path / "cache", loader=loader).iter_series())

    assert rows[0].series_id == "s"
    assert calls == [
        (
            ("arrow",),
            {
                "data_files": {"train": [str(item.resolve()) for item in files]},
                "split": "train",
                "streaming": True,
                "cache_dir": tmp_path / "cache",
            },
        )
    ]


def test_production_policy_is_offline_utsd12g_only() -> None:
    policy = json.loads(
        (ROOT / "configs" / "data" / "utsd12g_production.json").read_text(
            encoding="utf-8"
        )
    )

    assert [
        (item["source_id"], item["configurations"])
        for item in policy["repositories"]
    ] == [("utsd", ["UTSD-12G"])]
    source = policy["repositories"][0]
    assert source["mode"] == "local_arrow"
    assert source["local_path"] == "raw/utsd/UTSD-12G"
    assert source["snapshot_revision"] == UTSD_REVISION
    assert source["expected_files"] == 82
    assert source["expected_bytes"] == 3_892_126_910
    assert "lotsa" not in json.dumps(policy).lower()


def test_local_inventory_verifies_snapshot_without_network_calls(
    tmp_path: Path,
) -> None:
    prepare_data = _load_prepare_data_module()
    source_root = tmp_path / "tsfm-data"
    snapshot, arrow_files = _snapshot(source_root)
    policy = _fixture_policy(snapshot)
    progress: list[str] = []

    def forbidden(*args, **kwargs):
        raise AssertionError("offline inventory must not access a Hub builder or URL")

    inventory = prepare_data.build_inventory(
        policy,
        forbidden,
        None,
        "unused-offline-endpoint",
        progress.append,
        probe_file=forbidden,
        source_root=source_root,
    )

    selected = inventory["selected"]
    assert len(selected) == 1
    assert selected[0]["local_files"] == [str(path.resolve()) for path in arrow_files]
    assert inventory["projected_source_bytes"] == policy["repositories"][0][
        "expected_bytes"
    ]
    assert progress == ["verifying utsd/thuml/UTSD/UTSD-12G"]


def test_local_inventory_requires_snapshot_manifest(tmp_path: Path) -> None:
    prepare_data = _load_prepare_data_module()
    source_root = tmp_path / "tsfm-data"
    snapshot, _ = _snapshot(source_root)
    (snapshot.parent / "UTSD-12G.sha256.json").unlink()
    policy = _fixture_policy(snapshot)

    with pytest.raises(FileNotFoundError):
        prepare_data.build_inventory(
            policy,
            None,
            None,
            "unused",
            lambda _: None,
            source_root=source_root,
        )
