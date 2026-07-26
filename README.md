# Time Series Foundation Model

This project implements a training pipeline for a Timer-style decoder-only
time-series foundation model that can be progressively scaled to 0.3B, 1B,
and 3B parameters. The current milestone includes the tested local pipeline
and an offline, resumable 307M production path for four H100 GPUs. It does not
include datasets or pretrained weights in Git and does not claim paper-quality
reproduction.

## Current Capabilities

- Continuous, non-overlapping patch tokens;
- RoPE and PyTorch SDPA causal self-attention;
- SiLU gated MLP and next-patch MSE;
- normalization based only on context statistics;
- deterministic synthetic time series;
- autoregressive forecasting with cropping to arbitrary lengths;
- atomic checkpoints for the model, optimizer, and training step;
- a single model class configured through JSON for tiny through 307M scales;
- S3 conversion for heterogeneous univariate series;
- deterministic finite coverage with exact resume positions; and
- BF16 DDP gates, immutable training plans, and portable checkpoints for the
  four-H100 production path.

The `third_party/` directory contains reference code for Timer, Timer-XL, and
OpenLTM only. None of it is imported at runtime.

## Local Validation

Run the following commands in PowerShell:

```powershell
cd D:\path\to\Timer-BrightWise
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\run_smoke.py --steps 100
```

The default model is defined in `configs/model/tiny.json`, and the training
settings come from `configs/training/smoke.json`. The checkpoint is written to
`outputs/smoke/final.pt`; this directory is ignored by Git.

`configs/model/timer_300m.json` resolves to exactly 307,146,240 parameters. Do
not instantiate this model locally.

The server workflow is documented in `docs/h100-307m-runbook.md`. Follow it
only after the complete local test suite and source-tree audit pass.

## Server-Only Boundaries

Local validation verifies model behavior, S3 conversion with synthetic
fixtures, finite sampling, planning, checkpointing, and orchestration without
downloading data. The following work must be completed on a server: UTSD-12G
snapshot construction and conversion, BF16, NCCL, four-H100 batch resolution,
the 20+2 resume gate, throughput measurement, and formal 307M training. Local
tests do not claim that the real H100 node has passed these gates.
