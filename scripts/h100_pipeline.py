from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tsfm.config import TimerConfig
from tsfm.model import TimerModel
from tsfm.pipeline import (
    PhaseResult,
    PipelineBinding,
    choose_production_resume,
    run_pipeline,
)
from tsfm.preflight import collect_environment_report
from tsfm.production_plan import ProductionTemplate, ResolvedTrainingPlan
from tsfm.runtime import build_finite_s3_dataset
from tsfm.train_config import TrainingConfig

EXPECTED_PARAMETERS = 307_146_240
EXPECTED_WORLD_SIZE = 4
MINIMUM_H100_MEMORY_BYTES = 79 * 1024**3
ZERO_DIGEST = "0" * 64


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_digest(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError(f"expected JSON object: {path}")
    return document


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _hardware_digest(report: dict[str, object]) -> str:
    facts = {
        "cuda_runtime": report.get("cuda_runtime"),
        "driver": report.get("driver"),
        "gpus": report.get("gpus", []),
    }
    return _json_digest(facts)


def _runtime_package_digest(report_path: Path) -> str:
    if not report_path.is_file():
        return ZERO_DIGEST
    report = _read_json(report_path)
    digest = report.get("package_manifest_digest")
    return digest if isinstance(digest, str) and len(digest) == 64 else ZERO_DIGEST


@dataclass(slots=True)
class PipelineOptions:
    project_root: Path
    persistent_root: Path
    python: Path
    torchrun: Path
    state_path: Path
    reports: Path
    data_root: Path
    source_root: Path
    source_manifest: Path
    processed_root: Path
    processed_manifest: Path
    runtime_report: Path
    model_config: Path
    data_config: Path
    production_template: Path
    preflight_template: Path
    resolved_plan: Path
    preflight_config: Path
    preflight_output: Path
    production_output: Path
    binding: PipelineBinding

    @classmethod
    def create(cls, project_root: Path, persistent_root: Path) -> "PipelineOptions":
        project_root = project_root.resolve()
        persistent_root = persistent_root.resolve()
        data_root = persistent_root / "tsfm-data"
        reports = persistent_root / "reports" / "timer-307m"
        source_root = data_root / "raw" / "utsd" / "UTSD-12G"
        source_manifest = source_root.parent / "UTSD-12G.sha256.json"
        processed_root = data_root / "processed" / "utsd-12g"
        processed_manifest = processed_root / "manifest.jsonl"
        runtime_report = persistent_root / "runtime-bundles" / "runtime-install-report.json"
        model_config = project_root / "configs" / "model" / "timer_300m.json"
        resolved_plan = reports / "resolved-training-config.json"
        hardware = collect_environment_report(persistent_root)
        binding = PipelineBinding(
            packages=_runtime_package_digest(runtime_report),
            source=_sha256_file(source_manifest) if source_manifest.is_file() else ZERO_DIGEST,
            processed=_sha256_file(processed_manifest) if processed_manifest.is_file() else ZERO_DIGEST,
            model=_sha256_file(model_config),
            plan=_sha256_file(resolved_plan) if resolved_plan.is_file() else ZERO_DIGEST,
            hardware=_hardware_digest(hardware),
        )
        venv_bin = persistent_root / "venvs" / "tsfm-h100" / "bin"
        return cls(
            project_root=project_root,
            persistent_root=persistent_root,
            python=venv_bin / "python",
            torchrun=venv_bin / "torchrun",
            state_path=reports / "pipeline-state.json",
            reports=reports,
            data_root=data_root,
            source_root=source_root,
            source_manifest=source_manifest,
            processed_root=processed_root,
            processed_manifest=processed_manifest,
            runtime_report=runtime_report,
            model_config=model_config,
            data_config=project_root / "configs" / "data" / "utsd12g_production.json",
            production_template=project_root / "configs" / "training" / "h100_307m_production.json",
            preflight_template=project_root / "configs" / "training" / "h100_307m_preflight.json",
            resolved_plan=resolved_plan,
            preflight_config=reports / "preflight-training-config.json",
            preflight_output=persistent_root / "checkpoints" / "timer-307m-preflight",
            production_output=persistent_root / "checkpoints" / "timer-307m-production",
            binding=binding,
        )


class H100PhaseRunner:
    def __init__(self) -> None:
        self.selected_micro_batch: int | None = None

    @staticmethod
    def _execute(options: PipelineOptions, commands: list[list[str]]) -> int:
        for command in commands:
            completed = subprocess.run(command, cwd=options.project_root, check=False)
            if completed.returncode != 0:
                return completed.returncode
        return 0

    @staticmethod
    def _torchrun(options: PipelineOptions, script: str, *arguments: str) -> list[str]:
        return [
            str(options.torchrun),
            "--standalone",
            "--nproc_per_node=4",
            str(options.project_root / "scripts" / script),
            *arguments,
        ]

    def run_phase(self, phase: str, options: PipelineOptions) -> PhaseResult:
        started = _utc_now()
        commands: list[list[str]] = []
        report_path: Path | None = None
        try:
            handler = getattr(self, f"_{phase}")
            commands, exit_code, report_path = handler(options)
        except Exception as error:
            exit_code = 1
            report_path = options.reports / f"{phase}-error.json"
            _write_json_atomic(
                report_path,
                {"phase": phase, "status": "FAIL", "error": str(error)},
            )
            print(f"{phase} failed: {error}", file=sys.stderr, flush=True)
        return PhaseResult(
            exit_code=exit_code,
            started_at=started,
            ended_at=_utc_now(),
            argv=tuple(tuple(item) for item in commands),
            report_path=str(report_path) if report_path is not None else None,
        )

    def _runtime(self, options: PipelineOptions):
        report = _read_json(options.runtime_report)
        if not str(report.get("python", "")).startswith("3.11."):
            raise RuntimeError("offline runtime is not CPython 3.11")
        if options.binding.packages == ZERO_DIGEST:
            raise RuntimeError("runtime report has no package manifest digest")
        if not options.python.is_file() or not options.torchrun.is_file():
            raise RuntimeError("offline venv executables are missing")
        return [], 0, options.runtime_report

    def _source(self, options: PipelineOptions):
        command = [
            str(options.python),
            str(options.project_root / "scripts" / "prepare_data.py"),
            "discover",
            f"--config={options.data_config}",
            f"--persistent-root={options.persistent_root}",
            f"--data-root={options.data_root}",
            f"--source-root={options.data_root}",
            "--execute-server",
        ]
        exit_code = self._execute(options, [command])
        if exit_code == 0:
            options.binding = replace(
                options.binding, source=_sha256_file(options.source_manifest)
            )
        return [command], exit_code, options.data_root / "inventory.json"

    def _conversion(self, options: PipelineOptions):
        base = [
            f"--config={options.data_config}",
            f"--persistent-root={options.persistent_root}",
            f"--data-root={options.data_root}",
            f"--source-root={options.data_root}",
            "--execute-server",
        ]
        commands: list[list[str]] = []
        if not (options.processed_root / "conversion-complete.json").is_file():
            commands.append(
                [str(options.python), str(options.project_root / "scripts" / "prepare_data.py"), "convert", *base]
            )
        commands.append(
            [str(options.python), str(options.project_root / "scripts" / "prepare_data.py"), "validate", *base]
        )
        exit_code = self._execute(options, commands)
        if exit_code == 0:
            options.binding = replace(
                options.binding, processed=_sha256_file(options.processed_manifest)
            )
        return commands, exit_code, options.data_root / "conversion-report.json"

    def _hardware(self, options: PipelineOptions):
        command = [
            str(options.python),
            str(options.project_root / "scripts" / "server_preflight.py"),
            f"--persistent-root={options.persistent_root}",
            f"--report-dir={options.reports}",
            "--expected-gpu=H100",
            "--expected-gpu-count=4",
            f"--minimum-gpu-memory-bytes={MINIMUM_H100_MEMORY_BYTES}",
        ]
        exit_code = self._execute(options, [command])
        report_path = options.reports / "server-preflight.json"
        if exit_code == 0:
            options.binding = replace(
                options.binding, hardware=_hardware_digest(_read_json(report_path))
            )
        return [command], exit_code, report_path

    def _nccl(self, options: PipelineOptions):
        report_path = options.reports / "nccl-probe.json"
        command = self._torchrun(
            options,
            "nccl_probe.py",
            "--expected-world-size=4",
            f"--report-dir={options.reports}",
        )
        return [command], self._execute(options, [command]), report_path

    def _batch(self, options: PipelineOptions):
        config = TimerConfig.from_json(options.model_config)
        parameter_count = sum(item.numel() for item in TimerModel(config).parameters())
        if parameter_count != EXPECTED_PARAMETERS:
            raise RuntimeError(
                f"expected {EXPECTED_PARAMETERS} parameters, found {parameter_count}"
            )
        template = ProductionTemplate.from_json(options.production_template)
        commands: list[list[str]] = []
        for candidate in template.micro_batch_candidates:
            command = self._torchrun(
                options,
                "batch_probe.py",
                f"--model-config={options.model_config}",
                f"--manifest={options.processed_manifest}",
                f"--micro-batch-size={candidate}",
                f"--report-dir={options.reports}",
                "--expected-world-size=4",
                f"--seed={template.seed}",
            )
            commands.append(command)
            if self._execute(options, [command]) == 0:
                report = _read_json(options.reports / f"batch-probe-{candidate}.json")
                if report.get("status") == "PASS":
                    self.selected_micro_batch = candidate
                    break
        if self.selected_micro_batch is None:
            return commands, 1, options.reports
        _, manifest_digest, window_count = build_finite_s3_dataset(
            options.processed_manifest,
            split="train",
            seed=template.seed,
            patch_length=config.input_token_len,
            context_patches=template.context_patches,
        )
        plan = ResolvedTrainingPlan.resolve(
            template,
            window_count=window_count,
            micro_batch_size=self.selected_micro_batch,
            world_size=EXPECTED_WORLD_SIZE,
            digests={
                "model": options.binding.model,
                "source": options.binding.source,
                "manifest": manifest_digest,
                "packages": options.binding.packages,
            },
        )
        plan_digest = plan.write_atomic(options.resolved_plan)
        options.binding = replace(options.binding, plan=plan_digest)
        self._write_preflight_config(options, plan)
        return commands, 0, options.resolved_plan

    @staticmethod
    def _write_preflight_config(
        options: PipelineOptions, plan: ResolvedTrainingPlan
    ) -> None:
        values = _read_json(options.preflight_template)
        resume_steps = int(values.pop("resume_steps"))
        values["total_steps"] = int(values["total_steps"]) + resume_steps
        values["micro_batch_size"] = plan.micro_batch_size
        values["gradient_accumulation_steps"] = plan.gradient_accumulation_steps
        config = TrainingConfig(**values)
        _write_json_atomic(options.preflight_config, config.to_dict())

    def _preflight20(self, options: PipelineOptions):
        command = self._torchrun(
            options,
            "train.py",
            "run",
            f"--model-config={options.model_config}",
            f"--training-config={options.preflight_config}",
            f"--manifest={options.processed_manifest}",
            f"--output-dir={options.preflight_output}",
            "--steps=20",
            "--device=cuda",
        )
        return [command], self._execute(options, [command]), options.preflight_output / "step-000020.pt"

    def _resume2(self, options: PipelineOptions):
        checkpoint = options.preflight_output / "step-000020.pt"
        command = self._torchrun(
            options,
            "train.py",
            "resume-check",
            f"--model-config={options.model_config}",
            f"--training-config={options.preflight_config}",
            f"--manifest={options.processed_manifest}",
            f"--output-dir={options.preflight_output}",
            f"--resume={checkpoint}",
            "--steps=2",
            "--device=cuda",
        )
        exit_code = self._execute(options, [command])
        report_path = options.reports / "preflight-report.json"
        if exit_code == 0:
            _write_json_atomic(
                report_path,
                {
                    "status": "PASS",
                    "binding": asdict(options.binding),
                    "micro_batch_size": self.selected_micro_batch,
                    "checkpoint": str(options.preflight_output / "step-000022.pt"),
                },
            )
        return [command], exit_code, report_path

    def _production(self, options: PipelineOptions):
        if not options.resolved_plan.is_file():
            raise RuntimeError("resolved training plan is missing")
        resume = choose_production_resume(options.production_output)
        command = self._torchrun(
            options,
            "train.py",
            "run",
            f"--model-config={options.model_config}",
            f"--resolved-plan={options.resolved_plan}",
            f"--manifest={options.processed_manifest}",
            f"--output-dir={options.production_output}",
            "--device=cuda",
            *([f"--resume={resume}"] if resume is not None else []),
        )
        return [command], self._execute(options, [command]), options.production_output / "training-report.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gate and launch Timer 307M on four H100 GPUs")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--persistent-root", type=Path, default=Path("/root/work"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    options = PipelineOptions.create(args.project_root, args.persistent_root)
    options.reports.mkdir(parents=True, exist_ok=True)
    return run_pipeline(options, runner=H100PhaseRunner())


if __name__ == "__main__":
    raise SystemExit(main())
