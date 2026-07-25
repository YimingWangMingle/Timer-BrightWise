#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: source scripts/bootstrap_autodl.sh PERSISTENT_ROOT" >&2
  return 2 2>/dev/null || exit 2
fi

export TSFM_PERSISTENT_ROOT="$(readlink -f "$1")"
export TSFM_DATA_ROOT="$TSFM_PERSISTENT_ROOT/tsfm-data"
export HF_HOME="$TSFM_PERSISTENT_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$TSFM_PERSISTENT_ROOT/cache/pip"
export TORCH_HOME="$TSFM_PERSISTENT_ROOT/cache/torch"
export TSFM_CHECKPOINT_ROOT="$TSFM_PERSISTENT_ROOT/checkpoints"

mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TSFM_CHECKPOINT_ROOT"

echo "TSFM_PERSISTENT_ROOT=$TSFM_PERSISTENT_ROOT"
echo "TSFM_DATA_ROOT=$TSFM_DATA_ROOT"
echo "TSFM_CHECKPOINT_ROOT=$TSFM_CHECKPOINT_ROOT"
