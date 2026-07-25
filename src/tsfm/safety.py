from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

REQUIRED_UPLOAD_FILTERS = {
    ".venv/", "outputs/", "checkpoints/", "data/", ".pytest_cache/",
    "__pycache__/", "*.pt", "*.pth", "*.safetensors",
}
MATERIAL_ROOT_DIRECTORIES = {"outputs", "checkpoints", "data"}
SKIPPED_SCAN_DIRECTORIES = {".venv", ".pytest_cache", "__pycache__", "third_party"}
EXCLUDED_SUFFIXES = {".pt", ".pth", ".safetensors"}
THIRD_PARTY_IMPORT = re.compile(r"^\s*(?:from|import)\s+third_party(?:\.|\s|$)", re.MULTILINE)


@dataclass(frozen=True, slots=True)
class SafePaths:
    repository_root: Path
    persistent_root: Path
    data_root: Path


def _below(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent); return True
    except ValueError:
        return False


def validate_server_mutation(execute_server: bool, repository_root: str | Path, persistent_root: str | Path, data_root: str | Path, projected_bytes: int, minimum_free_gib: int = 20) -> SafePaths:
    if not execute_server: raise PermissionError("data mutation requires --execute-server")
    if projected_bytes < 0 or minimum_free_gib < 0: raise ValueError("disk requirements must be non-negative")
    repository = Path(repository_root).resolve(); persistent = Path(persistent_root).resolve(); data = Path(data_root).resolve()
    if not _below(repository, persistent) or not _below(data, persistent):
        raise ValueError("project and data roots must be below the explicit persistent root")
    if _below(data, repository): raise ValueError("data root must remain outside the repository")
    required = projected_bytes + minimum_free_gib * 1024**3
    if shutil.disk_usage(persistent).free < required: raise OSError("operation would violate the persistent free-space reserve")
    return SafePaths(repository, persistent, data)


def audit_source_tree(root: str | Path) -> list[str]:
    base = Path(root).resolve(); findings: set[str] = set()
    filter_path = base / ".uploadignore"
    configured = {line.strip() for line in filter_path.read_text(encoding="utf-8").splitlines() if line.strip()} if filter_path.is_file() else set()
    missing = sorted(REQUIRED_UPLOAD_FILTERS - configured)
    if missing: findings.add(f"upload filter is missing: {', '.join(missing)}")
    for name in MATERIAL_ROOT_DIRECTORIES:
        if (base / name).exists(): findings.add(f"material root directory present: {name}")
    for path in base.rglob("*"):
        if not path.is_file(): continue
        relative = path.relative_to(base)
        if relative.parts and relative.parts[0] in SKIPPED_SCAN_DIRECTORIES: continue
        if any(part in {".pytest_cache", "__pycache__"} for part in relative.parts): continue
        if path.suffix.lower() in EXCLUDED_SUFFIXES: findings.add(f"excluded weight file present: {relative.as_posix()}")
        if path.suffix == ".py" and relative.parts and relative.parts[0] in {"src", "scripts"}:
            if THIRD_PARTY_IMPORT.search(path.read_text(encoding="utf-8")):
                findings.add(f"runtime third_party import: {relative.as_posix()}")
    return sorted(findings)
