from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path, PurePosixPath

from tsfm.s3.records import ManifestRecord

FORMAT_VERSION = 1


def _canonical_line(record: ManifestRecord) -> str:
    values = asdict(record)
    values["relative_shard_path"] = record.relative_shard_path.as_posix()
    return (
        json.dumps(
            values,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        + "\n"
    )


def manifest_checksum(records: list[ManifestRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(_canonical_line(record).encode("utf-8"))
    return digest.hexdigest()


def write_manifest_atomic(
    path: str | Path, records: list[ManifestRecord]
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(_canonical_line(record))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return manifest_checksum(records)


def load_manifest(
    path: str | Path, expected_checksum: str | None = None
) -> list[ManifestRecord]:
    records: list[ManifestRecord] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            values = json.loads(line)
            if values.get("format_version") != FORMAT_VERSION:
                raise ValueError(
                    f"line {line_number}: unsupported format_version"
                )
            relative = PurePosixPath(values["relative_shard_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"line {line_number}: unsafe relative_shard_path"
                )
            values["relative_shard_path"] = relative
            record = ManifestRecord(**values)
            if record.offset < 0 or record.length <= 0:
                raise ValueError(f"line {line_number}: invalid offset or length")
            if len(record.checksum) != 64:
                raise ValueError(f"line {line_number}: invalid segment checksum")
            records.append(record)
    actual = manifest_checksum(records)
    if expected_checksum is not None and actual != expected_checksum:
        raise ValueError(
            f"manifest checksum mismatch: expected {expected_checksum}, got {actual}"
        )
    return records
