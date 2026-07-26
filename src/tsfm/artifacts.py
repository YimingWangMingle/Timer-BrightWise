from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    path: str
    size: int
    sha256: str


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _artifact_entries(root: Path) -> list[ArtifactEntry]:
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")

    entries: list[ArtifactEntry] = []
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError(f"artifact symlink is forbidden: {relative}")
        if path.is_file():
            entries.append(
                ArtifactEntry(
                    path=relative,
                    size=path.stat().st_size,
                    sha256=_file_digest(path),
                )
            )
    return entries


def _canonical_payload(entries: list[ArtifactEntry]) -> bytes:
    document = {
        "format_version": 1,
        "files": [asdict(entry) for entry in entries],
    }
    return (
        json.dumps(document, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _parse_manifest(payload: bytes) -> list[ArtifactEntry]:
    try:
        document: Any = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("artifact manifest is not valid UTF-8 JSON") from error
    if not isinstance(document, dict) or document.get("format_version") != 1:
        raise ValueError("unsupported artifact manifest format")
    raw_files = document.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("artifact manifest files must be a list")

    entries: list[ArtifactEntry] = []
    seen: set[str] = set()
    for raw_entry in raw_files:
        if not isinstance(raw_entry, dict) or set(raw_entry) != {
            "path",
            "size",
            "sha256",
        }:
            raise ValueError("artifact manifest contains an invalid file entry")
        path = raw_entry["path"]
        size = raw_entry["size"]
        sha256 = raw_entry["sha256"]
        if not isinstance(path, str):
            raise ValueError("artifact path must be a string")
        pure_path = PurePosixPath(path)
        if (
            not path
            or pure_path.is_absolute()
            or "\\" in path
            or any(part in {"", ".", ".."} for part in pure_path.parts)
        ):
            raise ValueError(f"unsafe artifact path: {path}")
        if path in seen:
            raise ValueError(f"duplicate artifact path: {path}")
        if not isinstance(size, int) or isinstance(size, bool) or size < 0:
            raise ValueError(f"invalid artifact size: {path}")
        if not isinstance(sha256, str) or _SHA256_PATTERN.fullmatch(sha256) is None:
            raise ValueError(f"invalid artifact checksum: {path}")
        seen.add(path)
        entries.append(ArtifactEntry(path=path, size=size, sha256=sha256))

    entries.sort(key=lambda entry: entry.path)
    if payload != _canonical_payload(entries):
        raise ValueError("artifact manifest is not canonical")
    return entries


def write_artifact_manifest(root: Path, destination: Path) -> str:
    root = root.resolve()
    destination = destination.resolve()
    if destination == root or root in destination.parents:
        raise ValueError("artifact manifest must be outside the artifact root")

    payload = _canonical_payload(_artifact_entries(root))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return hashlib.sha256(payload).hexdigest()


def verify_artifact_manifest(
    root: Path,
    manifest: Path,
    *,
    expected_files: int | None = None,
    expected_bytes: int | None = None,
) -> str:
    if expected_files is not None and expected_files < 0:
        raise ValueError("expected_files must be non-negative")
    if expected_bytes is not None and expected_bytes < 0:
        raise ValueError("expected_bytes must be non-negative")

    payload = manifest.read_bytes()
    declared = _parse_manifest(payload)
    actual = _artifact_entries(root.resolve())
    actual_by_path = {entry.path: entry for entry in actual}
    declared_paths = {entry.path for entry in declared}

    for entry in declared:
        found = actual_by_path.get(entry.path)
        if found is None:
            raise ValueError(f"missing file: {entry.path}")
        if found.size != entry.size:
            raise ValueError(f"size mismatch: {entry.path}")
        if found.sha256 != entry.sha256:
            raise ValueError(f"checksum mismatch: {entry.path}")

    extras = sorted(set(actual_by_path) - declared_paths)
    if extras:
        raise ValueError(f"unlisted file: {extras[0]}")
    if expected_files is not None and len(actual) != expected_files:
        raise ValueError(f"expected {expected_files} files, found {len(actual)}")
    total_bytes = sum(entry.size for entry in actual)
    if expected_bytes is not None and total_bytes != expected_bytes:
        raise ValueError(f"expected {expected_bytes} bytes, found {total_bytes}")
    return hashlib.sha256(payload).hexdigest()
