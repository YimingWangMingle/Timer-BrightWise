from types import SimpleNamespace

import pytest

from tsfm.safety import audit_source_tree, validate_server_mutation


def test_missing_execute_flag_fails_before_creating_data_root(tmp_path) -> None:
    persistent = tmp_path / "persistent"
    repository = persistent / "project"
    repository.mkdir(parents=True)
    data_root = persistent / "data"

    with pytest.raises(PermissionError, match="--execute-server"):
        validate_server_mutation(
            False, repository, persistent, data_root, projected_bytes=0
        )

    assert not data_root.exists()


def test_data_root_inside_repository_is_rejected(tmp_path) -> None:
    repository = tmp_path / "project"
    repository.mkdir()

    with pytest.raises(ValueError, match="outside the repository"):
        validate_server_mutation(
            True,
            repository,
            tmp_path,
            repository / "data",
            projected_bytes=0,
            minimum_free_gib=0,
        )


def test_project_and_data_must_be_below_persistent_root(tmp_path) -> None:
    persistent = tmp_path / "persistent"
    persistent.mkdir()

    with pytest.raises(ValueError, match="persistent root"):
        validate_server_mutation(
            True,
            tmp_path / "elsewhere",
            persistent,
            persistent / "data",
            projected_bytes=0,
            minimum_free_gib=0,
        )


def test_projected_disk_use_preserves_reserve(tmp_path, monkeypatch) -> None:
    persistent = tmp_path / "persistent"
    repository = persistent / "project"
    repository.mkdir(parents=True)
    monkeypatch.setattr(
        "tsfm.safety.shutil.disk_usage",
        lambda _: SimpleNamespace(free=20 * 1024**3),
    )

    with pytest.raises(OSError, match="free-space reserve"):
        validate_server_mutation(
            True,
            repository,
            persistent,
            persistent / "data",
            projected_bytes=1,
        )


def test_source_audit_finds_generated_files_and_runtime_reference_imports(
    tmp_path,
) -> None:
    (tmp_path / "src" / "pkg").mkdir(parents=True)
    (tmp_path / "src" / "pkg" / "bad.py").write_text(
        "from third_party.timer import Model\n", encoding="utf-8"
    )
    (tmp_path / "weights.pt").write_bytes(b"not-a-real-weight")
    (tmp_path / "data").mkdir()

    findings = audit_source_tree(tmp_path)

    assert any("weights.pt" in finding for finding in findings)
    assert any("data" in finding for finding in findings)
    assert any("third_party" in finding for finding in findings)
