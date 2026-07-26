from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any
from urllib.parse import quote
from urllib.request import Request, urlopen

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from tsfm.artifacts import verify_artifact_manifest
from tsfm.s3.adapters import DatasetSpec, LOTSAAdapter, UTSDAdapter
from tsfm.s3.manifest import load_manifest
from tsfm.s3.segments import finite_univariate_segments
from tsfm.s3.shards import pack_segments, read_segment_mmap
from tsfm.safety import validate_server_mutation

FileProbe = Callable[[str], int]


def _atomic_json(path: Path, values: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(values, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def _directory_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _load_policy(path: Path) -> dict:
    policy = json.loads(path.read_text(encoding="utf-8"))
    if policy.get("format_version") != 1:
        raise ValueError("unsupported data policy format_version")
    return policy


def _guard(args: argparse.Namespace, projected_bytes: int = 0):
    if args.persistent_root is None or args.data_root is None:
        raise ValueError("persistent root and data root are required for server actions")
    return validate_server_mutation(
        args.execute_server,
        PROJECT_ROOT,
        args.persistent_root,
        args.data_root,
        projected_bytes,
    )


def _declared_entries(policy: dict) -> list[dict]:
    entries = []
    seen = set()
    for source in policy["repositories"]:
        configurations = source["configurations"]
        if isinstance(configurations, str):
            raise ValueError(
                f"{source['repository']}: configurations must be an explicit list"
            )
        domains = source["domains"]
        configured_files = source.get("data_files")
        for configuration in configurations:
            key = (source["repository"], configuration)
            if key in seen:
                raise ValueError(
                    "duplicate repository/configuration group: "
                    f"{source['repository']}/{configuration}"
                )
            seen.add(key)
            if configuration not in domains:
                raise ValueError(
                    f"{source['repository']}/{configuration}: domain is not declared"
                )
            mode = source.get("mode", "hub")
            if mode not in {"hub", "local_arrow"}:
                raise ValueError(f"unsupported dataset mode: {mode}")
            data_files = []
            if configured_files is not None:
                data_files = configured_files.get(configuration, [])
                if not data_files:
                    raise ValueError(
                        f"{source['repository']}/{configuration}: data_files are not declared"
                    )
                if not source.get("revision") or not source.get("file_format"):
                    raise ValueError(
                        f"{source['repository']}/{configuration}: exact files require "
                        "revision and file_format"
                    )
            if mode == "local_arrow":
                local_path = source.get("local_path")
                relative = Path(local_path) if isinstance(local_path, str) else None
                if (
                    relative is None
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or source.get("file_format") != "arrow"
                ):
                    raise ValueError(
                        f"{source['repository']}/{configuration}: invalid local Arrow policy"
                    )
                if data_files:
                    raise ValueError("local Arrow policy cannot declare remote data_files")
                if not source.get("snapshot_revision"):
                    raise ValueError("local Arrow policy requires snapshot_revision")
                if int(source.get("expected_files", 0)) <= 0:
                    raise ValueError("local Arrow policy requires expected_files")
                if int(source.get("expected_bytes", 0)) <= 0:
                    raise ValueError("local Arrow policy requires expected_bytes")
            entries.append(
                {
                    "source_id": source["source_id"],
                    "repository": source["repository"],
                    "configuration": configuration,
                    "dataset_id": configuration,
                    "domain": domains[configuration],
                    "split": source["split"],
                    "revision": source.get("revision"),
                    "file_format": source.get("file_format"),
                    "data_files": data_files,
                    "mode": mode,
                    "local_path": source.get("local_path"),
                    "snapshot_revision": source.get("snapshot_revision"),
                    "expected_files": source.get("expected_files"),
                    "expected_bytes": source.get("expected_bytes"),
                }
            )
    return entries


def _validate_inventory_quotas(
    policy: dict, entries: list[dict], projected_source_bytes: int
) -> None:
    selection = policy["selection"]
    source_counts = Counter(entry["source_id"] for entry in entries)
    for source_id, minimum in selection[
        "minimum_dataset_groups_by_source"
    ].items():
        actual = source_counts[source_id]
        if actual < minimum:
            raise ValueError(
                f"source {source_id} has {actual} dataset groups; minimum is {minimum}"
            )
    domains = {entry["domain"] for entry in entries}
    minimum_domains = selection["minimum_domain_groups"]
    if len(domains) < minimum_domains:
        raise ValueError(
            f"domain quota has {len(domains)} groups; minimum is {minimum_domains}"
        )
    maximum_source_bytes = selection["maximum_source_cache_bytes"]
    if projected_source_bytes > maximum_source_bytes:
        raise OSError(
            f"projected source data is {projected_source_bytes} bytes; "
            f"limit is {maximum_source_bytes}"
        )


def _resolve_url(endpoint: str, repository: str, revision: str, path: str) -> str:
    return (
        f"{endpoint.rstrip('/')}/datasets/{quote(repository, safe='/')}/resolve/"
        f"{quote(revision, safe='')}/{quote(path, safe='/')}"
    )


def _probe_file_size(url: str) -> int:
    timeout = int(os.environ.get("HF_HUB_ETAG_TIMEOUT", "30"))
    request = Request(url, method="HEAD")
    with urlopen(request, timeout=timeout) as response:
        value = response.headers.get("X-Linked-Size") or response.headers.get(
            "Content-Length"
        )
    if value is None:
        raise OSError("HEAD response did not include a content length")
    return int(value)


def build_inventory(
    policy: dict,
    load_builder: Callable[..., Any],
    cache_dir: str | Path | None,
    endpoint: str,
    progress: Callable[[str], None],
    probe_file: FileProbe | None = None,
    source_root: str | Path | None = None,
) -> dict:
    declared = _declared_entries(policy)
    entries = []
    probe = probe_file or _probe_file_size
    for entry in declared:
        repository = entry["repository"]
        configuration = entry["configuration"]
        local_files: list[str] = []
        source_manifest = None
        if entry["mode"] == "local_arrow":
            if source_root is None:
                raise ValueError("source_root is required for local Arrow inventory")
            resolved_source_root = Path(source_root).resolve()
            snapshot = (resolved_source_root / entry["local_path"]).resolve()
            if resolved_source_root != snapshot and resolved_source_root not in snapshot.parents:
                raise ValueError(f"local snapshot escapes source_root: {snapshot}")
            progress(f"verifying {entry['source_id']}/{repository}/{configuration}")
            manifest = snapshot.parent / f"{snapshot.name}.sha256.json"
            source_manifest = verify_artifact_manifest(
                snapshot,
                manifest,
                expected_files=int(entry["expected_files"]),
                expected_bytes=int(entry["expected_bytes"]),
            )
            arrow_paths = sorted(snapshot.glob("data-*.arrow"))
            if not arrow_paths:
                raise ValueError(f"no Arrow shards found in local snapshot: {snapshot}")
            local_files = [str(path.resolve()) for path in arrow_paths]
            estimated = _directory_bytes(snapshot)
        elif entry["data_files"]:
            progress(f"probing {entry['source_id']}/{repository}/{configuration}")
            estimated = 0
            for file_spec in entry["data_files"]:
                url = _resolve_url(
                    endpoint,
                    repository,
                    entry["revision"],
                    file_spec["path"],
                )
                try:
                    actual_size = int(probe(url))
                except Exception as error:
                    raise RuntimeError(
                        "dataset file probe failed: "
                        f"repository={repository}, configuration={configuration}, "
                        f"revision={entry['revision']}, file={file_spec['path']}, "
                        f"HF_ENDPOINT={endpoint}: {error}"
                    ) from error
                expected_size = int(file_spec["size"])
                if actual_size != expected_size:
                    raise OSError(
                        f"size mismatch for {repository}/{configuration}/"
                        f"{file_spec['path']}: expected {expected_size}, got {actual_size}"
                    )
                estimated += actual_size
        else:
            progress(f"discovering {entry['source_id']}/{repository}/{configuration}")
            if load_builder is None:
                raise ValueError("load_builder is required for Hub inventory")
            try:
                builder = load_builder(repository, configuration, cache_dir=cache_dir)
            except Exception as error:
                raise RuntimeError(
                    "dataset metadata request failed: "
                    f"repository={repository}, configuration={configuration}, "
                    f"HF_ENDPOINT={endpoint}: {error}"
                ) from error
            estimated = int(
                builder.info.download_size or builder.info.dataset_size or 0
            )
            if estimated < 0:
                raise ValueError(
                    f"{repository}/{configuration}: metadata byte estimate is negative"
                )
        entries.append(
            {
                **entry,
                "estimated_source_bytes": estimated,
                "local_files": local_files,
                "source_manifest": source_manifest,
            }
        )

    projected = sum(entry["estimated_source_bytes"] for entry in entries)
    _validate_inventory_quotas(policy, entries, projected)
    return {
        "format_version": 1,
        "entries": entries,
        "selected": list(entries),
        "projected_source_bytes": projected,
    }


def build_validation_report(
    policy: dict,
    inventory: dict,
    records: Iterable[Any],
    hf_home: str | Path,
    processed_root: str | Path,
) -> dict:
    records = list(records)
    selected = inventory["selected"]
    projected = int(inventory["projected_source_bytes"])
    _validate_inventory_quotas(policy, selected, projected)

    selected_groups = {
        (entry["source_id"], entry["dataset_id"]) for entry in selected
    }
    manifest_groups = {(record.source_id, record.dataset_id) for record in records}
    missing = sorted(selected_groups - manifest_groups)
    if missing:
        names = ", ".join(f"{source}/{dataset}" for source, dataset in missing)
        raise ValueError(
            f"selected dataset groups produced no manifest records: {names}"
        )

    selection = policy["selection"]
    hf_cache_bytes = _directory_bytes(Path(hf_home))
    processed_bytes = _directory_bytes(Path(processed_root))
    maximum_source_bytes = selection["maximum_source_cache_bytes"]
    if hf_cache_bytes > maximum_source_bytes:
        raise OSError(
            "maximum_source_cache_bytes exceeded: Hugging Face cache is "
            f"{hf_cache_bytes} bytes; limit is {maximum_source_bytes}"
        )
    maximum_processed_bytes = selection["maximum_processed_bytes"]
    if processed_bytes > maximum_processed_bytes:
        raise OSError(
            "maximum_processed_bytes exceeded: processed data is "
            f"{processed_bytes} bytes; limit is {maximum_processed_bytes}"
        )

    dataset_groups = {
        source_id: sorted(
            dataset_id
            for source, dataset_id in manifest_groups
            if source == source_id
        )
        for source_id in sorted({source for source, _ in manifest_groups})
    }
    return {
        "records": len(records),
        "dataset_groups": dataset_groups,
        "domains": sorted({entry["domain"] for entry in selected}),
        "projected_source_bytes": projected,
        "hf_cache_bytes": hf_cache_bytes,
        "processed_bytes": processed_bytes,
    }


def discover(args: argparse.Namespace, policy: dict) -> None:
    paths = _guard(args)
    local_only = all(
        source.get("mode", "hub") == "local_arrow"
        for source in policy["repositories"]
    )
    if local_only:
        load_builder = None
        hf_home = None
        endpoint = ""
        if args.source_root is None:
            raise ValueError("source root is required for local Arrow discovery")
    else:
        from datasets import load_dataset_builder

        hf_home = os.environ.get("HF_HOME")
        if not hf_home:
            raise ValueError("HF_HOME is required for server data discovery")
        endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
    inventory = build_inventory(
        policy,
        load_builder,
        hf_home,
        endpoint,
        lambda message: print(message, flush=True),
        source_root=args.source_root,
    )
    inventory["policy_path"] = str(args.config.resolve())
    paths.data_root.mkdir(parents=True, exist_ok=True)
    _atomic_json(paths.data_root / "inventory.json", inventory)
    print(f"wrote inventory with {len(inventory['selected'])} selected dataset groups")


def convert(args: argparse.Namespace, policy: dict) -> None:
    paths = _guard(args, policy["selection"]["maximum_processed_bytes"])
    inventory_path = paths.data_root / "inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    selected = inventory["selected"]
    _validate_inventory_quotas(
        policy, selected, int(inventory["projected_source_bytes"])
    )

    def segments():
        for entry in selected:
            spec = DatasetSpec(
                entry["repository"],
                entry["configuration"],
                entry["source_id"],
                entry["dataset_id"],
                entry["split"],
                revision=entry.get("revision"),
                file_format=entry.get("file_format"),
                data_files=tuple(
                    file_spec["path"] for file_spec in entry.get("data_files", [])
                ),
                local_files=tuple(entry.get("local_files", [])),
            )
            adapter_type = UTSDAdapter if entry["source_id"] == "utsd" else LOTSAAdapter
            if spec.local_files:
                cache_dir = paths.data_root / ".datasets-cache"
            else:
                hf_home = os.environ.get("HF_HOME")
                if not hf_home:
                    raise ValueError("HF_HOME is required for Hub conversion")
                cache_dir = Path(hf_home)
            adapter = adapter_type(spec, cache_dir)
            for series in adapter.iter_series():
                yield from finite_univariate_segments(
                    series,
                    policy["selection"]["minimum_segment_length"],
                    policy["selection"]["minimum_variance"],
                )

    processed = paths.data_root / "processed"
    records, digest = pack_segments(
        segments(), processed, policy["selection"]["maximum_shard_bytes"]
    )
    (processed / "manifest.sha256").write_text(digest + "\n", encoding="ascii")
    print(f"converted {len(records)} finite segments; manifest={digest}")


def validate(args: argparse.Namespace, policy: dict) -> None:
    paths = _guard(args)
    processed = paths.data_root / "processed"
    expected = (processed / "manifest.sha256").read_text(encoding="ascii").strip()
    records = load_manifest(processed / "manifest.jsonl", expected)
    for record in records:
        values = read_segment_mmap(processed, record)
        actual = hashlib.sha256(
            values.astype("<f4", copy=False).tobytes()
        ).hexdigest()
        if actual != record.checksum:
            raise ValueError(f"{record.record_id}: segment checksum mismatch")

    local_only = all(
        source.get("mode", "hub") == "local_arrow"
        for source in policy["repositories"]
    )
    source_cache = args.source_root if local_only else os.environ.get("HF_HOME")
    if not source_cache:
        requirement = "source root" if local_only else "HF_HOME"
        raise ValueError(f"{requirement} is required for server data validation")
    inventory = json.loads(
        (paths.data_root / "inventory.json").read_text(encoding="utf-8")
    )
    report = build_validation_report(
        policy, inventory, records, source_cache, processed
    )
    report["manifest_checksum"] = expected
    _atomic_json(paths.data_root / "conversion-report.json", report)
    print(json.dumps(report, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Guarded UTSD/LOTSA preparation")
    parser.add_argument("action", nargs="?", choices=("discover", "convert", "validate"))
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/data/server_validation.json",
    )
    parser.add_argument(
        "--persistent-root", type=Path, default=os.getenv("TSFM_PERSISTENT_ROOT")
    )
    parser.add_argument("--data-root", type=Path, default=os.getenv("TSFM_DATA_ROOT"))
    parser.add_argument(
        "--source-root",
        type=Path,
        default=os.getenv("TSFM_SOURCE_ROOT") or os.getenv("TSFM_DATA_ROOT"),
    )
    parser.add_argument("--execute-server", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = _load_policy(args.config)
    if args.action is None:
        repositories = ", ".join(item["repository"] for item in policy["repositories"])
        print(f"DRY RUN: repositories={repositories}")
        print(
            "No files or network were touched. Choose an action and pass "
            "--execute-server on the server."
        )
        return 0
    {"discover": discover, "convert": convert, "validate": validate}[
        args.action
    ](args, policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
