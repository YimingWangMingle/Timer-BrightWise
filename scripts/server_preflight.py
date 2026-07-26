from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tsfm.preflight import run_server_preflight


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate CUDA/BF16 and persistent resources")
    parser.add_argument("--persistent-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--expected-gpu", default="RTX 5090")
    parser.add_argument("--expected-gpu-count", type=int, default=1)
    parser.add_argument("--minimum-gpu-memory-bytes", type=int, default=0)
    args = parser.parse_args()
    path = run_server_preflight(
        args.persistent_root, args.report_dir, args.expected_gpu,
        expected_gpu_count=args.expected_gpu_count,
        minimum_gpu_memory_bytes=args.minimum_gpu_memory_bytes,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
