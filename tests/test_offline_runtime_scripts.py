from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_runtime_requirements_pin_core_versions() -> None:
    text = (ROOT / "requirements/h100-py311-cu126.txt").read_text(
        encoding="utf-8"
    )
    for requirement in (
        "torch==2.7.1",
        "numpy==2.2.6",
        "datasets==3.6.0",
        "pyarrow==20.0.0",
        "psutil==7.0.0",
        "tqdm==4.67.1",
        "pytest==8.4.1",
        "setuptools==80.9.0",
        "wheel==0.45.1",
    ):
        assert requirement in text


def test_installer_is_offline_and_uses_root_work() -> None:
    text = (ROOT / "scripts/install_h100_offline.sh").read_text(
        encoding="utf-8"
    )
    assert "--no-index" in text
    assert "--find-links" in text
    assert "verify_artifact_manifest" in text
    assert "/root/work" in text
    assert "runtime-install-report.json" in text
    assert "package_manifest_digest" in text
    assert "spec_from_file_location" in text
    assert "curl " not in text and "wget " not in text


def test_builder_uses_linux_runtime_and_cuda126_torch_index() -> None:
    text = (ROOT / "scripts/build_h100_offline_bundle.sh").read_text(
        encoding="utf-8"
    )
    assert "Linux" in text and "x86_64" in text
    assert "https://download.pytorch.org/whl/cu126" in text
    assert "write_artifact_manifest" in text
    assert "h100-py311-cu126.txt" in text
    assert "RUNTIME_SUFFIX" in text
    assert "cpython-3.11${RUNTIME_SUFFIX}" in text
    assert "-type f -path '*/bin/python3.11'" not in text
    assert "python3.11" in text
    assert "spec_from_file_location" in text
