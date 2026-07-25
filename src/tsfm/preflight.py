from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
from pathlib import Path

import torch
import torch.nn.functional as F


def read_cgroup_memory_limit(path: str | Path) -> int | None:
    candidate = Path(path)
    if not candidate.is_file():
        return None
    value = candidate.read_text(encoding="ascii").strip()
    return None if value == "max" else int(value)


def _driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.splitlines()[0].strip()
    except (FileNotFoundError, subprocess.SubprocessError, IndexError):
        return None


def collect_environment_report(persistent_root: str | Path) -> dict[str, object]:
    root = Path(persistent_root).resolve()
    usage = shutil.disk_usage(root)
    cuda = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda_runtime": torch.version.cuda,
        "driver": _driver_version(),
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "bf16_supported": torch.cuda.is_bf16_supported() if cuda else False,
        "cgroup_memory_bytes": read_cgroup_memory_limit(
            "/sys/fs/cgroup/memory.max"
        ),
        "disk": {
            "path": str(root),
            "total_bytes": usage.total,
            "used_bytes": usage.used,
            "free_bytes": usage.free,
        },
    }


def sdpa_bf16_probe() -> dict[str, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BF16 SDPA probe")
    device = torch.device("cuda", 0)
    query = torch.randn(2, 8, 30, 64, device=device, dtype=torch.bfloat16, requires_grad=True)
    key = torch.randn_like(query, requires_grad=True)
    value = torch.randn_like(query, requires_grad=True)
    output = F.scaled_dot_product_attention(query, key, value, is_causal=True)
    loss = output.float().square().mean()
    loss.backward()
    if not all(tensor.grad is not None and torch.isfinite(tensor.grad).all() for tensor in (query, key, value)):
        raise FloatingPointError("BF16 SDPA produced non-finite gradients")
    return {
        "loss": float(loss.detach()),
        "peak_allocated_bytes": float(torch.cuda.max_memory_allocated(device)),
    }


def run_server_preflight(
    persistent_root: str | Path,
    report_dir: str | Path,
    expected_gpu: str,
) -> Path:
    report = collect_environment_report(persistent_root)
    errors = []
    if report["gpu"] is None or expected_gpu.lower() not in str(report["gpu"]).lower():
        errors.append(f"expected GPU containing {expected_gpu!r}, got {report['gpu']!r}")
    if not report["bf16_supported"]:
        errors.append("BF16 is not supported")
    memory = report["cgroup_memory_bytes"]
    if memory is not None and memory < 80 * 1024**3:
        errors.append("cgroup memory limit is below 80 GiB")
    if report["disk"]["free_bytes"] < 20 * 1024**3:
        errors.append("persistent disk has less than 20 GiB free")
    if not errors:
        report["sdpa_bf16"] = sdpa_bf16_probe()
    report["errors"] = errors
    destination = Path(report_dir) / "server-preflight.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
    if errors:
        raise RuntimeError("; ".join(errors))
    return destination
