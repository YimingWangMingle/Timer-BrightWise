from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tsfm.distributed import destroy_distributed, distributed_context_from_environment


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe four-rank NCCL collectives")
    parser.add_argument("--expected-world-size", type=int, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    actual_world_size = int(os.getenv("WORLD_SIZE", "1"))
    if actual_world_size != args.expected_world_size:
        print(
            f"expected WORLD_SIZE={args.expected_world_size}, got {actual_world_size}",
            file=sys.stderr,
        )
        return 2
    context = distributed_context_from_environment(initialize=True)
    try:
        device = torch.device("cuda", context.local_rank)
        total = torch.tensor(float(context.rank), device=device)
        torch.distributed.all_reduce(total)
        gathered: list[list[int] | None] = [None] * context.world_size
        sample_ids = list(range(context.rank, context.rank + 16 * context.world_size, context.world_size))
        torch.distributed.all_gather_object(gathered, sample_ids)
        flat = [item for values in gathered for item in values]
        if float(total) != sum(range(context.world_size)):
            raise RuntimeError("NCCL all_reduce returned an unexpected sum")
        if len(flat) != len(set(flat)):
            raise RuntimeError("rank sample partitions overlap")
        if context.is_main_process:
            report = {
                "world_size": context.world_size,
                "all_reduce_sum": float(total),
                "sample_ids": gathered,
                "status": "PASS",
            }
            destination = args.report_dir / "nccl-probe.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, destination)
        torch.distributed.barrier()
        return 0
    finally:
        destroy_distributed(context)
