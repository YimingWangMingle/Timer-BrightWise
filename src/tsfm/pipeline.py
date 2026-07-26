from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import torch

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
GATE_PHASES = (
    "runtime",
    "source",
    "conversion",
    "hardware",
    "nccl",
    "batch",
    "preflight20",
    "resume2",
)


@dataclass(frozen=True, slots=True)
class PipelineBinding:
    packages: str
    source: str
    processed: str
    model: str
    plan: str
    hardware: str

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"invalid pipeline digest: {name}")

@dataclass(frozen=True, slots=True)
class PhaseResult:
    exit_code: int
    started_at: str
    ended_at: str
    argv: tuple[tuple[str, ...], ...] = ()
    report_path: str | None = None



class PipelineState:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.document = self.load(self.path)

    @staticmethod
    def load(path: str | Path) -> dict:
        candidate = Path(path)
        if not candidate.is_file():
            return {"format_version": 1, "phases": {}}
        document = json.loads(candidate.read_text(encoding="utf-8"))
        if document.get("format_version") != 1 or not isinstance(
            document.get("phases"), dict
        ):
            raise ValueError("invalid pipeline state")
        return document

    def record_phase(
        self, name: str, binding: PipelineBinding, report: dict[str, object]
    ) -> None:
        self.document["phases"][name] = {
            "binding": asdict(binding),
            "report": report,
        }
        payload = json.dumps(self.document, sort_keys=True, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.tmp")
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, self.path)

    def can_reuse_preflight(self, binding: PipelineBinding) -> bool:
        phase = self.document["phases"].get("pass")
        return bool(
            phase
            and phase.get("binding") == asdict(binding)
            and phase.get("report", {}).get("status") == "PASS"
        )


def choose_production_resume(output_dir: str | Path) -> Path | None:
    output = Path(output_dir)
    if not output.exists() or not any(output.iterdir()):
        return None
    for candidate in sorted(output.glob("step-*.pt"), reverse=True):
        try:
            payload = torch.load(candidate, map_location="cpu", weights_only=True)
            if (
                payload.get("format_version") == 1
                and int(payload["training_state"]["global_step"]) >= 0
            ):
                return candidate
        except (OSError, RuntimeError, KeyError, TypeError, ValueError):
            continue
    raise RuntimeError(
        "nonempty production directory has no valid latest checkpoint"
    )


class PhaseRunner(Protocol):
    def run_phase(self, phase: str, options: object) -> int | PhaseResult: ...


def _phase_report(
    result: int | PhaseResult, *, success_status: str = "PASS"
) -> tuple[int, dict[str, object]]:
    if isinstance(result, int):
        return result, {
            "status": success_status if result == 0 else "FAIL",
            "exit_code": result,
        }
    report: dict[str, object] = {
        "status": success_status if result.exit_code == 0 else "FAIL",
        "exit_code": result.exit_code,
        "started_at": result.started_at,
        "ended_at": result.ended_at,
        "argv": [list(command) for command in result.argv],
        "report_path": result.report_path,
    }
    return result.exit_code, report


def run_pipeline(options: object, *, runner: PhaseRunner) -> int:
    state = PipelineState(options.state_path)
    if not state.can_reuse_preflight(options.binding):
        for phase in GATE_PHASES:
            exit_code, report = _phase_report(runner.run_phase(phase, options))
            state.record_phase(phase, options.binding, report)
            if exit_code != 0:
                return exit_code
        state.record_phase("pass", options.binding, {"status": "PASS"})
    exit_code, report = _phase_report(
        runner.run_phase("production", options), success_status="STARTED"
    )
    state.record_phase(
        "production",
        options.binding,
        report,
    )
    return exit_code
