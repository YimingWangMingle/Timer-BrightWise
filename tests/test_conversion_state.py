from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tsfm.conversion_state import (
    ConversionBinding,
    begin_conversion,
    publish_conversion,
    validate_completed_conversion,
)


def _binding(manifest: Path) -> ConversionBinding:
    return ConversionBinding(
        source_manifest="a" * 64,
        processed_manifest=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        policy_digest="b" * 64,
        records=1,
        processed_bytes=manifest.stat().st_size,
    )


def test_conversion_is_accepted_only_after_atomic_completion(tmp_path: Path) -> None:
    final = tmp_path / "processed" / "utsd-12g"
    staging = begin_conversion(final)
    manifest = staging / "manifest.jsonl"
    manifest.write_bytes(b"record\n")

    with pytest.raises(ValueError, match="conversion completion marker is missing"):
        validate_completed_conversion(
            staging, source_manifest="a" * 64, policy_digest="b" * 64
        )

    binding = _binding(manifest)
    assert publish_conversion(staging, final, binding) == final
    assert (
        validate_completed_conversion(
            final, source_manifest="a" * 64, policy_digest="b" * 64
        )
        == binding
    )
    assert not staging.exists()


def test_publish_refuses_nonempty_destination(tmp_path: Path) -> None:
    final = tmp_path / "processed" / "utsd-12g"
    final.mkdir(parents=True)
    (final / "unknown").write_text("do not overwrite", encoding="utf-8")
    staging = begin_conversion(final)
    manifest = staging / "manifest.jsonl"
    manifest.write_bytes(b"")

    with pytest.raises(FileExistsError, match="nonempty production destination"):
        publish_conversion(staging, final, _binding(manifest))

    assert (final / "unknown").read_text(encoding="utf-8") == "do not overwrite"
    assert staging.exists()


def test_completed_conversion_rejects_binding_and_manifest_mismatch(
    tmp_path: Path,
) -> None:
    final = tmp_path / "processed" / "utsd-12g"
    staging = begin_conversion(final)
    manifest = staging / "manifest.jsonl"
    manifest.write_bytes(b"record\n")
    publish_conversion(staging, final, _binding(manifest))

    with pytest.raises(ValueError, match="source manifest binding mismatch"):
        validate_completed_conversion(
            final, source_manifest="c" * 64, policy_digest="b" * 64
        )
    with pytest.raises(ValueError, match="policy digest binding mismatch"):
        validate_completed_conversion(
            final, source_manifest="a" * 64, policy_digest="d" * 64
        )

    (final / "manifest.jsonl").write_bytes(b"tampered\n")
    with pytest.raises(ValueError, match="processed manifest binding mismatch"):
        validate_completed_conversion(
            final, source_manifest="a" * 64, policy_digest="b" * 64
        )
