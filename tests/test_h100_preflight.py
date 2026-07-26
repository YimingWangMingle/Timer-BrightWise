from __future__ import annotations

import scripts.nccl_probe as nccl_probe
from tsfm.preflight import validate_server_report


def _report(count: int) -> dict:
    return {
        "gpus": [
            {
                "index": index,
                "name": "NVIDIA H100 80GB HBM3",
                "total_memory_bytes": 80 * 1024**3,
                "bf16_supported": True,
            }
            for index in range(count)
        ],
        "cgroup_memory_bytes": 128 * 1024**3,
        "disk": {"free_bytes": 3 * 1024**4},
    }


def test_h100_gate_requires_four_matching_80gb_devices() -> None:
    errors = validate_server_report(
        _report(3),
        expected_gpu="H100",
        expected_gpu_count=4,
        minimum_gpu_memory_bytes=80 * 1024**3,
    )
    assert "expected 4 GPUs, found 3" in errors


def test_h100_gate_rejects_name_and_memory_mismatch() -> None:
    report = _report(4)
    report["gpus"][2]["name"] = "RTX 5090"
    report["gpus"][3]["total_memory_bytes"] = 79 * 1024**3
    errors = validate_server_report(
        report,
        expected_gpu="H100",
        expected_gpu_count=4,
        minimum_gpu_memory_bytes=80 * 1024**3,
    )
    assert any("GPU 2" in error and "H100" in error for error in errors)
    assert any("GPU 3" in error and "memory" in error for error in errors)


def test_nccl_probe_rejects_wrong_world_size(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert nccl_probe.main(
        ["--expected-world-size", "4", "--report-dir", str(tmp_path)]
    ) == 2
