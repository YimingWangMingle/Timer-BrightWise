from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tsfm.config import TimerConfig
from tsfm.distributed import destroy_distributed, distributed_context_from_environment, wrap_model
from tsfm.model import TimerModel
from tsfm.runtime import build_finite_s3_dataset
from tsfm.s3.sampling import CounterSampler


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe a real four-rank H100 batch")
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-world-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(os.getenv("WORLD_SIZE", "1")) != args.expected_world_size:
        return 2
    context = distributed_context_from_environment(initialize=True)
    try:
        device = torch.device("cuda", context.local_rank)
        config = TimerConfig.from_json(args.model_config)
        dataset, manifest_digest, _ = build_finite_s3_dataset(
            args.manifest,
            split="train",
            seed=args.seed,
            patch_length=config.input_token_len,
            context_patches=30,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.micro_batch_size,
            sampler=CounterSampler(0, context.rank, context.world_size),
            num_workers=0,
            pin_memory=True,
        )
        model = wrap_model(TimerModel(config).to(device), context).train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, fused=True)
        batch = next(iter(loader))
        torch.cuda.reset_peak_memory_stats(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(
                batch["context"].to(device, non_blocking=True),
                labels=batch["target"].to(device, non_blocking=True),
            ).loss
        if loss is None or not torch.isfinite(loss):
            raise FloatingPointError("batch probe produced non-finite loss")
        loss.backward()
        optimizer.step()
        peak = torch.tensor(
            float(torch.cuda.max_memory_reserved(device)), device=device
        )
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        total = torch.cuda.get_device_properties(device).total_memory
        passed = float(peak) < 0.8 * total
        if context.is_main_process:
            report = {
                "micro_batch_size": args.micro_batch_size,
                "loss": float(loss.detach()),
                "max_memory_reserved_bytes": int(float(peak)),
                "memory_limit_bytes": int(0.8 * total),
                "manifest_checksum": manifest_digest,
                "status": "PASS" if passed else "FAIL",
            }
            destination = args.report_dir / f"batch-probe-{args.micro_batch_size}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(f".{destination.name}.tmp")
            temporary.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, destination)
        torch.distributed.barrier()
        return 0 if passed else 1
    finally:
        destroy_distributed(context)


if __name__ == "__main__":
    raise SystemExit(main())
