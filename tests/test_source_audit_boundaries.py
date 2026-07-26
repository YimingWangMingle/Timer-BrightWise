from tsfm.safety import audit_source_tree


UPLOAD_FILTER = """.venv/
outputs/
checkpoints/
data/
.pytest_cache/
__pycache__/
*.pt
*.pth
*.safetensors
"""


def test_audit_allows_config_data_and_ignored_caches(tmp_path) -> None:
    (tmp_path / ".uploadignore").write_text(UPLOAD_FILTER, encoding="utf-8")
    (tmp_path / "configs" / "data").mkdir(parents=True)
    (tmp_path / "configs" / "data" / "policy.json").write_text(
        "{}\n", encoding="utf-8"
    )
    (tmp_path / ".pytest_cache").mkdir()
    (tmp_path / "src" / "pkg" / "__pycache__").mkdir(parents=True)

    assert audit_source_tree(tmp_path) == []


def test_audit_requires_complete_upload_filter(tmp_path) -> None:
    (tmp_path / ".uploadignore").write_text("*.pt\n", encoding="utf-8")

    findings = audit_source_tree(tmp_path)
    assert any("upload filter" in finding for finding in findings)


def test_repository_upload_filter_excludes_offline_material() -> None:
    root = __import__("pathlib").Path(__file__).parents[1]
    configured = set(
        (root / ".uploadignore").read_text(encoding="utf-8").splitlines()
    )
    assert {
        "runtime-bundles/",
        "wheelhouse/",
        "*.sha256.json",
        "*.tar.zst",
    } <= configured
