from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_MARKER_NAME = "conversion-complete.json"


@dataclass(frozen=True, slots=True)
class ConversionBinding:
    source_manifest: str
    processed_manifest: str
    policy_digest: str
    records: int
    processed_bytes: int

    def __post_init__(self) -> None:
        for name in ("source_manifest", "processed_manifest", "policy_digest"):
            value = getattr(self, name)
            if _SHA256_PATTERN.fullmatch(value) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
        if self.records < 0:
            raise ValueError("records must be non-negative")
        if self.processed_bytes < 0:
            raise ValueError("processed_bytes must be non-negative")


def _canonical_payload(binding: ConversionBinding) -> bytes:
    document = {"format_version": 1, **asdict(binding)}
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _processed_bytes(root: Path) -> int:
    return sum(
        path.stat().st_size
        for path in root.rglob("*")
        if path.is_file() and path.name != _MARKER_NAME
    )


def begin_conversion(final_root: Path) -> Path:
    final_root = final_root.resolve()
    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = final_root.with_name(
        f".{final_root.name}.incomplete-{uuid.uuid4().hex}"
    )
    staging.mkdir()
    return staging


def publish_conversion(
    staging: Path, final_root: Path, binding: ConversionBinding
) -> Path:
    staging = staging.resolve()
    final_root = final_root.resolve()
    if not staging.is_dir():
        raise ValueError(f"conversion staging directory is missing: {staging}")
    if staging.parent != final_root.parent or staging == final_root:
        raise ValueError("conversion staging and destination must be sibling directories")
    if final_root.exists() and (
        not final_root.is_dir() or any(final_root.iterdir())
    ):
        raise FileExistsError(f"nonempty production destination: {final_root}")

    manifest = staging / "manifest.jsonl"
    if not manifest.is_file():
        raise ValueError("processed manifest is missing from conversion staging")
    if _file_digest(manifest) != binding.processed_manifest:
        raise ValueError("processed manifest binding mismatch")
    if _processed_bytes(staging) != binding.processed_bytes:
        raise ValueError("processed byte binding mismatch")

    marker = staging / _MARKER_NAME
    temporary = marker.with_name(f".{marker.name}.tmp")
    temporary.write_bytes(_canonical_payload(binding))
    os.replace(temporary, marker)
    if final_root.exists():
        final_root.rmdir()
    os.replace(staging, final_root)
    return final_root


def _read_binding(marker: Path) -> ConversionBinding:
    try:
        document: Any = json.loads(marker.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("conversion completion marker is invalid JSON") from error
    if not isinstance(document, dict) or document.get("format_version") != 1:
        raise ValueError("unsupported conversion completion marker")
    expected_keys = {
        "format_version",
        "source_manifest",
        "processed_manifest",
        "policy_digest",
        "records",
        "processed_bytes",
    }
    if set(document) != expected_keys:
        raise ValueError("conversion completion marker fields are invalid")
    binding = ConversionBinding(
        source_manifest=document["source_manifest"],
        processed_manifest=document["processed_manifest"],
        policy_digest=document["policy_digest"],
        records=document["records"],
        processed_bytes=document["processed_bytes"],
    )
    if marker.read_bytes() != _canonical_payload(binding):
        raise ValueError("conversion completion marker is not canonical")
    return binding


def validate_completed_conversion(
    final_root: Path,
    *,
    source_manifest: str,
    policy_digest: str,
) -> ConversionBinding:
    final_root = final_root.resolve()
    marker = final_root / _MARKER_NAME
    if not marker.is_file():
        raise ValueError("conversion completion marker is missing")
    binding = _read_binding(marker)
    if binding.source_manifest != source_manifest:
        raise ValueError("source manifest binding mismatch")
    if binding.policy_digest != policy_digest:
        raise ValueError("policy digest binding mismatch")

    manifest = final_root / "manifest.jsonl"
    if not manifest.is_file() or _file_digest(manifest) != binding.processed_manifest:
        raise ValueError("processed manifest binding mismatch")
    if _processed_bytes(final_root) != binding.processed_bytes:
        raise ValueError("processed byte binding mismatch")
    record_count = sum(1 for line in manifest.open("rb") if line.strip())
    if record_count != binding.records:
        raise ValueError("processed record count binding mismatch")
    return binding
