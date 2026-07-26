from __future__ import annotations

import json
from pathlib import Path

import pytest

from tsfm.artifacts import verify_artifact_manifest, write_artifact_manifest


def _write_fixture(root: Path) -> None:
    root.mkdir()
    (root / "a.arrow").write_bytes(b"arrow-a")
    (root / "dataset_info.json").write_bytes(b"{}\n")


def test_manifest_round_trip_is_canonical(tmp_path: Path) -> None:
    root = tmp_path / "UTSD-12G"
    _write_fixture(root)
    manifest = tmp_path / "UTSD-12G.sha256.json"

    digest = write_artifact_manifest(root, manifest)

    assert verify_artifact_manifest(
        root, manifest, expected_files=2, expected_bytes=10
    ) == digest
    document = json.loads(manifest.read_text(encoding="utf-8"))
    assert document["format_version"] == 1
    assert [item["path"] for item in document["files"]] == [
        "a.arrow",
        "dataset_info.json",
    ]


def test_manifest_rejects_modified_missing_and_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "UTSD-12G"
    _write_fixture(root)
    manifest = tmp_path / "UTSD-12G.sha256.json"
    write_artifact_manifest(root, manifest)

    (root / "a.arrow").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch: a.arrow"):
        verify_artifact_manifest(root, manifest)

    (root / "a.arrow").unlink()
    with pytest.raises(ValueError, match="missing file: a.arrow"):
        verify_artifact_manifest(root, manifest)

    (root / "a.arrow").write_bytes(b"arrow-a")
    (root / "extra.arrow").write_bytes(b"x")
    with pytest.raises(ValueError, match="unlisted file: extra.arrow"):
        verify_artifact_manifest(root, manifest)


def test_manifest_rejects_wrong_expected_totals(tmp_path: Path) -> None:
    root = tmp_path / "UTSD-12G"
    _write_fixture(root)
    manifest = tmp_path / "UTSD-12G.sha256.json"
    write_artifact_manifest(root, manifest)

    with pytest.raises(ValueError, match="expected 82 files, found 2"):
        verify_artifact_manifest(root, manifest, expected_files=82)
    with pytest.raises(ValueError, match="expected 3892126910 bytes, found 10"):
        verify_artifact_manifest(root, manifest, expected_bytes=3_892_126_910)


def test_snapshot_builder_pins_revision_and_only_utsd12g(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import scripts.download_utsd12g_snapshot as command

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(command, "snapshot_download", lambda **kwargs: calls.append(kwargs))

    command.download_snapshot(tmp_path / "raw")

    assert calls == [
        {
            "repo_id": "thuml/UTSD",
            "repo_type": "dataset",
            "revision": "7326ff5f4578da73d843fd675d760c6c6054017f",
            "allow_patterns": ["UTSD-12G/*"],
            "local_dir": tmp_path / "raw",
        }
    ]
