from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tsfm.pipeline import (
    PipelineBinding,
    PhaseResult,
    PipelineState,
    choose_production_resume,
    run_pipeline,
)


def _binding(value: str) -> PipelineBinding:
    digest = value * 64
    return PipelineBinding(digest, digest, digest, digest, digest, digest)


def test_pipeline_reuses_pass_only_for_identical_binding(tmp_path: Path) -> None:
    state = PipelineState(tmp_path / "pipeline-state.json")
    binding = _binding("a")
    state.record_phase("pass", binding, {"status": "PASS"})
    assert state.can_reuse_preflight(binding)
    assert not state.can_reuse_preflight(_binding("b"))


def test_nonempty_production_without_valid_checkpoint_fails(tmp_path: Path) -> None:
    output = tmp_path / "production"
    output.mkdir()
    (output / "partial.pt").write_bytes(b"broken")
    with pytest.raises(
        RuntimeError, match="nonempty production directory has no valid latest checkpoint"
    ):
        choose_production_resume(output)


def test_command_order_stops_before_production_on_failed_resume_gate(
    tmp_path: Path,
) -> None:
    phases = []

    class Runner:
        def run_phase(self, phase, options):
            phases.append(phase)
            return 1 if phase == "resume2" else 0

    options = SimpleNamespace(
        state_path=tmp_path / "state.json",
        binding=_binding("a"),
    )
    assert run_pipeline(options, runner=Runner()) == 1
    assert phases == [
        "runtime",
        "source",
        "conversion",
        "hardware",
        "nccl",
        "batch",
        "preflight20",
        "resume2",
    ]


def test_phase_record_contains_canonical_execution_evidence(tmp_path: Path) -> None:
    class Runner:
        def run_phase(self, phase, options):
            return PhaseResult(
                exit_code=0,
                started_at="2026-07-26T00:00:00Z",
                ended_at="2026-07-26T00:00:01Z",
                argv=(("python", phase),),
                report_path=f"/root/work/reports/{phase}.json",
            )

    options = SimpleNamespace(
        state_path=tmp_path / "state.json",
        binding=_binding("a"),
    )
    assert run_pipeline(options, runner=Runner()) == 0
    state = json.loads(options.state_path.read_text(encoding="utf-8"))
    runtime = state["phases"]["runtime"]["report"]
    assert runtime == {
        "argv": [["python", "runtime"]],
        "ended_at": "2026-07-26T00:00:01Z",
        "exit_code": 0,
        "report_path": "/root/work/reports/runtime.json",
        "started_at": "2026-07-26T00:00:00Z",
        "status": "PASS",
    }


def test_batch_phase_can_bind_the_newly_resolved_plan(tmp_path: Path) -> None:
    old = _binding("a")
    new = _binding("b")

    class Runner:
        def run_phase(self, phase, options):
            if phase == "batch":
                options.binding = new
            return 0

    options = SimpleNamespace(state_path=tmp_path / "state.json", binding=old)
    assert run_pipeline(options, runner=Runner()) == 0
    state = PipelineState.load(options.state_path)
    assert state["phases"]["pass"]["binding"] == {
        "packages": new.packages,
        "source": new.source,
        "processed": new.processed,
        "model": new.model,
        "plan": new.plan,
        "hardware": new.hardware,
    }


def test_h100_launchers_are_offline_and_use_four_fresh_ranks() -> None:
    root = Path(__file__).parents[1]
    command = (root / "scripts" / "h100_pipeline.py").read_text(encoding="utf-8")
    launcher = (root / "scripts" / "launch_h100_307m.sh").read_text(encoding="utf-8")
    assert "--nproc_per_node=4" in command
    assert "subprocess.run" in command
    assert "shell=True" not in command
    assert "HF_ENDPOINT" not in launcher
    assert "TSFM_PERSISTENT_ROOT=/root/work" in launcher
    assert "exec" in launcher
