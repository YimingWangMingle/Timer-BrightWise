# Tiny Timer Smoke Test Design

## Goal

Build a CPU-friendly end-to-end smoke test for the future 0.3B, 1B, and 3B
time-series foundation models. The tiny model must use the same model, data,
loss, generation, and checkpoint code paths as the larger configurations.

## Scope

The first milestone includes:

- a Timer-style decoder-only Transformer;
- non-overlapping continuous patch tokens;
- RoPE causal self-attention through PyTorch SDPA;
- gated MLP blocks matching the public HF Timer 84M implementation;
- next-patch MSE training;
- deterministic synthetic time-series data;
- autoregressive rolling generation;
- checkpoint save and resume;
- CPU unit tests and a 100-step smoke command.

It deliberately excludes UTSD, LOTSA, DDP, BF16, FlashAttention extensions,
Timer-XL multivariate attention, and model weights.

## Architecture

`TimerModel` accepts normalized values shaped `[batch, raw_length]`. It splits
the input into non-overlapping patches of `input_token_len`, projects each
patch to the hidden dimension, applies causal decoder blocks, and projects
each hidden state to one predicted output patch.

Training examples contain `context_patches + 1` patches. The first
`context_patches` are model inputs and the final shifted sequence is the
ground-truth next-patch target. Every predicted position receives MSE
supervision.

The model API remains size-independent. Tiny, 84M, 0.3B, 1B, and 3B variants
are represented only by JSON configuration.

## Components

- `tsfm.config`: validated model configuration and analytical parameter count.
- `tsfm.model`: patch embedding, RoPE attention, gated MLP, decoder, loss, and generation.
- `tsfm.data`: deterministic synthetic sine/trend/AR sequences and context-only normalization.
- `tsfm.checkpoint`: atomic model/optimizer checkpoint persistence.
- `tsfm.training`: reusable training step and bounded CPU smoke loop.
- `scripts/run_smoke.py`: command-line entry point for the 100-step run.

## Correctness Requirements

- Input and output patch shapes must be explicit and validated.
- Changing a future patch must not change predictions at earlier positions.
- Loss must be finite and gradients must reach trainable parameters.
- Generation must return exactly the requested number of raw points.
- A restored checkpoint must reproduce predictions bit-for-bit on CPU.
- A fixed synthetic batch must achieve lower loss after the smoke run.
- The 0.3B configuration must estimate between 300M and 315M parameters
  without constructing the full model locally.

## Local Environment

Use a project-local `.venv` with Python 3.12, CPU PyTorch, and pytest. The
virtual environment, checkpoints, caches, and generated logs are never
committed.

## Server Boundary

The local milestone proves functional correctness only. BF16, NCCL, H20
throughput, distributed sampling, UTSD/LOTSA I/O, and multi-GPU checkpointing
remain server validation tasks.
