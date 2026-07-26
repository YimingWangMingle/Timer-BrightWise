#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 BUNDLE_DIR VENV_DIR PROJECT_DIR" >&2
  exit 2
fi

BUNDLE_DIR="$(readlink -f "$1")"
VENV_ROOT="$(readlink -m "$2")"
PROJECT_ROOT="$(readlink -f "$3")"
PERSISTENT_ROOT=/root/work
RUNTIME_ROOT="$PERSISTENT_ROOT/runtime/cpython-3.11"
RUNTIME_REPORT="$PERSISTENT_ROOT/runtime-bundles/runtime-install-report.json"
BUNDLE_MANIFEST="${BUNDLE_DIR}.sha256.json"

case "$VENV_ROOT" in
  "$PERSISTENT_ROOT"/*) ;;
  *) echo "venv must be below /root/work" >&2; exit 2 ;;
esac
case "$PROJECT_ROOT" in
  "$PERSISTENT_ROOT"/*) ;;
  *) echo "project must be below /root/work" >&2; exit 2 ;;
esac
if [[ ! -f "$BUNDLE_MANIFEST" ]]; then
  echo "bundle manifest is missing: $BUNDLE_MANIFEST" >&2
  exit 2
fi
if [[ -e "$RUNTIME_ROOT" || -e "$VENV_ROOT" ]]; then
  echo "runtime or venv destination already exists; refusing to overwrite" >&2
  exit 2
fi

python3 -c \
  'import importlib.util, sys; from pathlib import Path; spec = importlib.util.spec_from_file_location("tsfm_artifacts", sys.argv[3]); module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); print(module.verify_artifact_manifest(Path(sys.argv[1]), Path(sys.argv[2])))' \
  "$BUNDLE_DIR" "$BUNDLE_MANIFEST" "$PROJECT_ROOT/src/tsfm/artifacts.py"

RUNTIME_ARCHIVE="$(find "$BUNDLE_DIR" -maxdepth 1 -type f -name 'cpython-3.11*' -print -quit)"
if [[ -z "$RUNTIME_ARCHIVE" ]]; then
  echo "CPython 3.11 runtime archive is missing from bundle" >&2
  exit 2
fi

mkdir -p "$PERSISTENT_ROOT/runtime" "$(dirname "$VENV_ROOT")" "$(dirname "$RUNTIME_REPORT")"
RUNTIME_TMP="$PERSISTENT_ROOT/runtime/.cpython-3.11.tmp-$$"
REPORT_TMP="${RUNTIME_REPORT}.tmp-$$"
trap 'rm -rf -- "$RUNTIME_TMP"; rm -f -- "$REPORT_TMP"' EXIT
mkdir "$RUNTIME_TMP"
case "$RUNTIME_ARCHIVE" in
  *.tar.zst) tar --zstd -xf "$RUNTIME_ARCHIVE" -C "$RUNTIME_TMP" ;;
  *.tar.gz|*.tgz) tar -xzf "$RUNTIME_ARCHIVE" -C "$RUNTIME_TMP" ;;
  *.tar.xz) tar -xJf "$RUNTIME_ARCHIVE" -C "$RUNTIME_TMP" ;;
  *.tar) tar -xf "$RUNTIME_ARCHIVE" -C "$RUNTIME_TMP" ;;
  *) echo "unsupported runtime archive" >&2; exit 2 ;;
esac
RUNTIME_PYTHON="$(find "$RUNTIME_TMP" -path '*/bin/python3.11' -print -quit)"
if [[ -z "$RUNTIME_PYTHON" ]]; then
  RUNTIME_PYTHON="$(find "$RUNTIME_TMP" -path '*/bin/python3' -print -quit)"
fi
if [[ -z "$RUNTIME_PYTHON" || "$($RUNTIME_PYTHON -c 'import platform; print(platform.python_version())')" != 3.11.* ]]; then
  echo "bundle runtime is not CPython 3.11" >&2
  exit 2
fi
mv "$RUNTIME_TMP" "$RUNTIME_ROOT"
RUNTIME_PYTHON="$(find "$RUNTIME_ROOT" -path '*/bin/python3.11' -print -quit)"
if [[ -z "$RUNTIME_PYTHON" ]]; then
  RUNTIME_PYTHON="$(find "$RUNTIME_ROOT" -path '*/bin/python3' -print -quit)"
fi

"$RUNTIME_PYTHON" -m venv --copies "$VENV_ROOT"
"$VENV_ROOT/bin/python" -m pip install \
  --no-index --find-links "$BUNDLE_DIR/wheelhouse" \
  -r "$BUNDLE_DIR/h100-py311-cu126.txt"
"$VENV_ROOT/bin/python" -m pip install \
  --no-index --no-build-isolation --no-deps -e "$PROJECT_ROOT"
"$VENV_ROOT/bin/python" -c 'import datasets, numpy, pyarrow, torch; assert torch.cuda.is_available()'

PACKAGE_MANIFEST_DIGEST="$(sha256sum "$BUNDLE_MANIFEST" | awk '{print $1}')"
export PACKAGE_MANIFEST_DIGEST PROJECT_ROOT REPORT_TMP
"$VENV_ROOT/bin/python" - <<'PY'
import importlib.metadata
import json
import os
import platform
from pathlib import Path

import torch

report = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda_runtime": torch.version.cuda,
    "package_manifest_digest": os.environ["PACKAGE_MANIFEST_DIGEST"],
    "project_version": importlib.metadata.version("tsfm"),
    "project_root": os.environ["PROJECT_ROOT"],
}
Path(os.environ["REPORT_TMP"]).write_text(
    json.dumps(report, sort_keys=True, separators=(",", ":")) + "\n",
    encoding="utf-8",
)
PY
mv "$REPORT_TMP" "$RUNTIME_REPORT"
trap - EXIT
echo "offline H100 runtime installed: $VENV_ROOT"
echo "runtime report: $RUNTIME_REPORT"
