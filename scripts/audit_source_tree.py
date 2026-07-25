from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tsfm.safety import audit_source_tree


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a source tree before upload")
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    findings = audit_source_tree(args.root)
    if findings:
        for finding in findings:
            print(f"ERROR: {finding}")
        return 1
    print("source tree audit passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
