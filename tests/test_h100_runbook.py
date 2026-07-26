from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_h100_runbook_contains_one_launch_and_all_persistent_paths() -> None:
    text = (ROOT / "docs/h100-307m-runbook.md").read_text(encoding="utf-8")
    assert text.count("nohup bash scripts/launch_h100_307m.sh") == 1
    for value in (
        "/root/work/venvs/tsfm-h100",
        "/root/work/tsfm-data/raw/utsd/UTSD-12G",
        "/root/work/checkpoints/timer-307m-production",
        "preflight-report.json",
        "resolved-training-config.json",
    ):
        assert value in text
    assert "Windows" in text and "must not contain datasets" in text


def test_h100_runbook_documents_exact_hardware_and_resume_gate() -> None:
    text = (ROOT / "docs/h100-307m-runbook.md").read_text(encoding="utf-8")
    for value in (
        "4 x NVIDIA H100 80GB",
        "20",
        "two-step resume",
        "307,146,240",
        "3,892,126,910",
    ):
        assert value in text


def test_project_docs_route_production_to_h100_runbook() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    validation = (ROOT / "docs/server-validation-runbook.md").read_text(
        encoding="utf-8"
    )
    assert "docs/h100-307m-runbook.md" in readme
    assert "docs/h100-307m-runbook.md" in validation
