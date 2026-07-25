# Time Series Foundation Model

This project implements a training pipeline for a Timer-style decoder-only
time-series foundation model that can be progressively scaled to 0.3B, 1B,
and 3B parameters. The current milestone is a tiny end-to-end validation that
runs on CPU. It does not include real-world datasets, pretrained weights, or
production multi-GPU training.

## Current Capabilities

- Continuous, non-overlapping patch tokens;
- RoPE and PyTorch SDPA causal self-attention;
- SiLU gated MLP and next-patch MSE;
- normalization based only on context statistics;
- deterministic synthetic time series;
- autoregressive forecasting with cropping to arbitrary lengths;
- atomic checkpoints for the model, optimizer, and training step; and
- a single model class configured through JSON for both tiny and 0.3B scales.

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

`configs/model/timer_300m.json` defines approximately 307.1 million parameters
and is provided only for configuration parsing and scale validation. Do not
instantiate this model locally.

## Server-Only Boundaries

Local validation only verifies the forward pass, backward pass, causality,
normalization, generation, and checkpoint pipeline. The following work must be
completed on a server: UTSD and LOTSA download and preprocessing, BF16,
FlashAttention, NCCL/DDP or FSDP, H20 throughput testing, checkpoint-resume
stress testing, and formal 0.3B training.
