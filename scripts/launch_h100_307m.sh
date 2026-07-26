#!/usr/bin/env bash
set -euo pipefail

export TSFM_PERSISTENT_ROOT=/root/work
PROJECT_ROOT="${TSFM_PROJECT_ROOT:-/root/work/TimeSeriesFoundationModel}"
VENV_ROOT="$TSFM_PERSISTENT_ROOT/venvs/tsfm-h100"

cd "$PROJECT_ROOT"
source scripts/bootstrap_autodl.sh "$TSFM_PERSISTENT_ROOT"

if [[ ! -x "$VENV_ROOT/bin/python" || ! -x "$VENV_ROOT/bin/torchrun" ]]; then
  echo "offline H100 virtual environment is missing: $VENV_ROOT" >&2
  exit 2
fi

export PYTHONUNBUFFERED=1
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
mkdir -p "$TSFM_PERSISTENT_ROOT/logs"

exec "$VENV_ROOT/bin/python" scripts/h100_pipeline.py \
  --project-root "$PROJECT_ROOT" \
  --persistent-root "$TSFM_PERSISTENT_ROOT"
