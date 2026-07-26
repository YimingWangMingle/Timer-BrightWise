from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from huggingface_hub import snapshot_download
except ImportError as import_error:

    def snapshot_download(**_: Any) -> str:
        raise RuntimeError(
            "huggingface_hub is required on the networked Linux download server"
        ) from import_error


from tsfm.artifacts import verify_artifact_manifest, write_artifact_manifest

UTSD_REPOSITORY = "thuml/UTSD"
UTSD_REVISION = "7326ff5f4578da73d843fd675d760c6c6054017f"
UTSD_ALLOW_PATTERNS = ["UTSD-12G/*"]
UTSD_EXPECTED_FILES = 82
UTSD_EXPECTED_BYTES = 3_892_126_910


def download_snapshot(destination: Path, *, endpoint: str | None = None) -> Path:
    destination = destination.resolve()
    arguments: dict[str, object] = {
        "repo_id": UTSD_REPOSITORY,
        "repo_type": "dataset",
        "revision": UTSD_REVISION,
        "allow_patterns": UTSD_ALLOW_PATTERNS,
        "local_dir": destination,
    }
    if endpoint is not None:
        arguments["endpoint"] = endpoint
    snapshot_download(**arguments)
    return destination / "UTSD-12G"


def build_snapshot(
    destination: Path,
    manifest: Path,
    *,
    endpoint: str | None = None,
) -> str:
    snapshot = download_snapshot(destination, endpoint=endpoint)
    digest = write_artifact_manifest(snapshot, manifest)
    verified = verify_artifact_manifest(
        snapshot,
        manifest,
        expected_files=UTSD_EXPECTED_FILES,
        expected_bytes=UTSD_EXPECTED_BYTES,
    )
    if verified != digest:
        raise RuntimeError("artifact manifest digest changed during verification")
    return digest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download the pinned UTSD-12G snapshot on a networked Linux server"
    )
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--endpoint")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if not sys.platform.startswith("linux"):
        raise RuntimeError("UTSD-12G downloads are allowed only on a Linux server")
    args = parse_args(argv)
    destination = args.destination.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest is not None
        else destination / "UTSD-12G.sha256.json"
    )
    if "HF_HOME" not in os.environ:
        print("warning: HF_HOME is not set; using the Hugging Face default cache")
    digest = build_snapshot(destination, manifest, endpoint=args.endpoint)
    print(f"UTSD-12G snapshot verified: {digest}")
    print(f"snapshot: {destination / 'UTSD-12G'}")
    print(f"manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
