#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 PYTHON_RUNTIME_TAR OUTPUT_DIR" >&2
  exit 2
fi
if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "x86_64" ]]; then
  echo "bundle construction requires Linux x86_64" >&2
  exit 2
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ARCHIVE="$(readlink -f "$1")"
OUTPUT_DIR="$(readlink -m "$2")"
OUTPUT_MANIFEST="${OUTPUT_DIR}.sha256.json"
REQUIREMENTS="$PROJECT_ROOT/requirements/h100-py311-cu126.txt"
TORCH_INDEX="https://download.pytorch.org/whl/cu126"
PYPI_INDEX="${PIP_INDEX_URL:-https://pypi.org/simple}"

if [[ ! -f "$RUNTIME_ARCHIVE" ]]; then
  echo "Python runtime archive is missing: $RUNTIME_ARCHIVE" >&2
  exit 2
fi
if [[ -e "$OUTPUT_MANIFEST" ]]; then
  echo "bundle manifest already exists: $OUTPUT_MANIFEST" >&2
  exit 2
fi
if [[ -d "$OUTPUT_DIR" && -n "$(find "$OUTPUT_DIR" -mindepth 1 -print -quit)" ]]; then
  echo "output directory must be empty: $OUTPUT_DIR" >&2
  exit 2
fi

BUILD_TMP="$(mktemp -d)"
trap 'rm -rf -- "$BUILD_TMP"' EXIT
mkdir -p "$BUILD_TMP/runtime" "$OUTPUT_DIR/wheelhouse"

case "$RUNTIME_ARCHIVE" in
  *.tar.zst) RUNTIME_SUFFIX=.tar.zst; tar --zstd -xf "$RUNTIME_ARCHIVE" -C "$BUILD_TMP/runtime" ;;
  *.tar.gz|*.tgz) RUNTIME_SUFFIX=.tar.gz; tar -xzf "$RUNTIME_ARCHIVE" -C "$BUILD_TMP/runtime" ;;
  *.tar.xz) RUNTIME_SUFFIX=.tar.xz; tar -xJf "$RUNTIME_ARCHIVE" -C "$BUILD_TMP/runtime" ;;
  *.tar) RUNTIME_SUFFIX=.tar; tar -xf "$RUNTIME_ARCHIVE" -C "$BUILD_TMP/runtime" ;;
  *) echo "unsupported Python runtime archive: $RUNTIME_ARCHIVE" >&2; exit 2 ;;
esac

RUNTIME_PYTHON="$(find "$BUILD_TMP/runtime" -path '*/bin/python3.11' -print -quit)"
if [[ -z "$RUNTIME_PYTHON" ]]; then
  RUNTIME_PYTHON="$(find "$BUILD_TMP/runtime" -path '*/bin/python3' -print -quit)"
fi
if [[ -z "$RUNTIME_PYTHON" || "$($RUNTIME_PYTHON -c 'import platform; print(platform.python_version())')" != 3.11.* ]]; then
  echo "supplied runtime is not CPython 3.11" >&2
  exit 2
fi

cp "$RUNTIME_ARCHIVE" "$OUTPUT_DIR/cpython-3.11${RUNTIME_SUFFIX}"
cp "$REQUIREMENTS" "$OUTPUT_DIR/h100-py311-cu126.txt"
grep -v '^torch==' "$REQUIREMENTS" > "$BUILD_TMP/non-torch-requirements.txt"
"$RUNTIME_PYTHON" -m pip download \
  --only-binary=:all: --dest "$OUTPUT_DIR/wheelhouse" \
  --index-url "$TORCH_INDEX" "torch==2.7.1"
"$RUNTIME_PYTHON" -m pip download \
  --only-binary=:all: --dest "$OUTPUT_DIR/wheelhouse" \
  --index-url "$PYPI_INDEX" --find-links "$OUTPUT_DIR/wheelhouse" \
  -r "$BUILD_TMP/non-torch-requirements.txt"

"$RUNTIME_PYTHON" -c \
  'import importlib.util, sys; from pathlib import Path; spec = importlib.util.spec_from_file_location("tsfm_artifacts", sys.argv[3]); module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module); print(module.write_artifact_manifest(Path(sys.argv[1]), Path(sys.argv[2])))' \
  "$OUTPUT_DIR" "$OUTPUT_MANIFEST" "$PROJECT_ROOT/src/tsfm/artifacts.py"
echo "offline H100 bundle: $OUTPUT_DIR"
echo "artifact manifest: $OUTPUT_MANIFEST"
