# Timer-Style S3 Server Validation Design

**Date:** 2026-07-24  
**Status:** Approved for implementation planning  
**Scope:** Local code development plus single-RTX-5090 server validation before 0.3B H20 training

## 1. Objective

Build a portable Timer-style pre-training system that validates the complete
path from heterogeneous time-series sources to stable BF16 next-patch training.
The system first validates a 26M model, then performs a short 95M scaling run,
and finally prepares the existing 307M configuration for H20 cluster training.

This phase proves engineering and mathematical correctness. It does not attempt
to train a useful foundation model to convergence.

## 2. Non-Negotiable Boundaries

### 2.1 Local machine

The local machine may contain only:

- source code, configuration, documentation, and tests;
- deterministic synthetic data generated during a test;
- temporary test fixtures created under the operating-system temporary folder;
- the already downloaded reference repositories under `third_party/`.

UTSD, LOTSA, other real datasets, converted shards, model weights, and formal
checkpoints must never be downloaded or stored locally. Tests must not access
the network or a Hugging Face cache.

### 2.2 Server

All mutable server artifacts live under the persistent data disk:

```text
/root/autodl-tmp/
|-- TimeSeriesFoundationModel/  # uploaded source tree
|-- tsfm-data/                  # raw, processed, and manifests
|-- cache/                      # pip, Hugging Face, and Torch caches
`-- checkpoints/                # training outputs
```

The 30GB container root filesystem is not used for project data, caches, or
checkpoints. The private data disk must retain at least 20GB free space.

### 2.3 Source synchronization

The complete source tree is uploaded, excluding generated and platform-specific
content:

```text
.venv/
outputs/
checkpoints/
data/
.pytest_cache/
__pycache__/
*.pt
*.pth
*.safetensors
```

The Windows virtual environment is never uploaded because it cannot run on
Ubuntu. Runtime paths must come from configuration or environment variables;
production code must contain no absolute Windows or AutoDL paths.

## 3. Target Environment

The validated single-GPU environment is:

| Component | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090, 32GB |
| Driver | 580.76.05 |
| PyTorch | 2.8.0+cu128 |
| CUDA runtime | 12.8 |
| Python | 3.12 |
| OS | Ubuntu 22.04 |
| BF16 | Supported |
| CPU allocation | 25 cores |
| Memory limit | 90GiB |
| Persistent data disk | 100GB XFS at `/root/autodl-tmp` |

The environment setup script configures `TSFM_DATA_ROOT`, `HF_HOME`,
`PIP_CACHE_DIR`, and `TORCH_HOME` beneath `/root/autodl-tmp`. Data download
commands require an explicit data root outside the repository and an explicit
execution flag; their default behavior is a dry run.

## 4. Model Architecture

All sizes use the same production `TimerModel` implementation:

- univariate input shaped `[batch, raw_length]`;
- non-overlapping continuous patches of 96 points;
- bias-free patch embedding;
- decoder-only causal self-attention;
- RoPE;
- biased Q/K/V and bias-free attention output projection;
- SiLU gated MLP;
- post-norm residual blocks;
- final LayerNorm;
- one bias-free 96-point output head;
- next-patch MSE.

Linear weights use `Normal(mean=0, std=0.02)` and linear biases use zero,
matching the released Timer 84M initialization. LayerNorm scale starts at one
and bias at zero.

### 4.1 Model ladder

| Name | Hidden | Intermediate | Layers | Heads | Parameters |
|---|---:|---:|---:|---:|---:|
| `timer_26m` | 512 | 1024 | 10 | 8 | 26,349,568 |
| `timer_95m` | 1024 | 2048 | 9 | 8 | 94,635,008 |
| `timer_300m` | 1536 | 3072 | 13 | 12 | 307,146,240 |

All three use patch length 96 and `max_position_embeddings=10000`. Validation
training uses 30 input patches, or 2,880 raw points. A training sample contains
31 patches so that the shifted target also has 30 patches.

The first release remains channel-independent and does not model cross-variable
relationships. Timer-XL-style multivariate attention is explicitly out of
scope.

## 5. S3 Data Architecture

The project implements a purpose-built S3 variant for heterogeneous, zero-shot
forecasting. It is not required to reproduce the ICML 2024 preprocessing
exactly.

```text
source adapter
  -> finite single-variable segments
  -> float32 memory-mapped shards
  -> JSONL manifest
  -> hierarchical sampler
  -> online context/target window
  -> context-only normalization
  -> next-patch batch
```

### 5.1 Source adapters

`UTSDAdapter` and `LOTSAAdapter` emit a common logical record. Multivariate
sources are split by channel. Irregular sources do not require time alignment;
v0 treats consecutive observations as consecutive positions and does not add a
timestamp embedding.

Non-finite observations split a series into finite contiguous segments. v0
does not interpolate gaps. A segment is retained only when it:

- contains at least 2,976 points;
- is finite after conversion to float32;
- has variance greater than `1e-8`.

### 5.2 Shards and manifest

Converted single-variable segments are concatenated into 1D float32 NPY shards
of at most 2GB each. Each source value occurs once in processed storage. The
manifest contains one JSON object per segment with:

```text
format_version
source_id
dataset_id
series_id
channel_id
relative_shard_path
offset
length
frequency
split_group
checksum
```

Paths are relative to `TSFM_DATA_ROOT`. Shards are opened with memory mapping;
the loader does not load a full corpus into RAM.

### 5.3 Splits

Two validation views are generated deterministically:

- `val_heldout`: 10% of complete series selected by a stable hash of source,
  dataset, and series identity;
- `val_temporal`: the final 10% of each sufficiently long non-heldout series.

Training windows are entirely before the temporal boundary. Temporal validation
windows are entirely after it, so no raw point is shared by training and
validation windows. A series too short to provide both full regions contributes
to training only. A source with at least two series contributes at least one
heldout series.

### 5.4 Hierarchical sampling

The sampler selects:

1. a source group with equal probability;
2. a record within that source with equal probability;
3. a valid window start uniformly within that record.

This prevents a large corpus from silently overwhelming smaller domains. The
sampler is deterministic from the global seed, process rank, worker id, and
sample counter. Every run records requested and achieved source proportions.

The server-validation corpus contains at least two dataset groups from UTSD and
two from LOTSA, spans at least four domain groups, uses no more than 20GB of raw
files, and uses no more than 20GB of processed shards.

### 5.5 Context-only normalization

For each sample, statistics are computed from the 2,880-point context only:

```text
mean = context.mean()
scale = sqrt(context.var(unbiased=False) + 1e-5)
normalized_context = (context - mean) / scale
normalized_target = (target - mean) / scale
```

The target never contributes to normalization statistics. Forecasts are
de-normalized with the saved context mean and scale for metrics and inference.
This removes the need for corpus statistics when forecasting a new series.

## 6. Training System

The single-GPU validation trainer uses:

- `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`;
- FP32 MSE reduction and FP32 reported metrics;
- AdamW with betas `(0.9, 0.95)`, weight decay `0.1`;
- gradient clipping at `1.0`;
- cosine learning-rate decay to 10% of the peak;
- deterministic seed `2026`;
- micro-batch size 32 and gradient accumulation 8;
- effective batch size 256 sequences;
- eight data-loader workers, pinned memory, and prefetch factor 2.

The 26M run uses peak learning rate `3e-4`, 100 warmup steps, and 2,000 total
steps. The 95M run uses peak learning rate `2e-4`, 50 warmup steps, and 500
total steps. Validation and checkpointing occur every 250 steps. The server
keeps the best checkpoint and the two most recent checkpoints.

Before either run, a bounded probe executes forward and backward on the exact
configured batch and context length. An out-of-memory result fails the probe
without silently changing the batch or accumulation settings.

### 6.1 Checkpoint contract

An atomic checkpoint contains:

- model, optimizer, and scheduler state;
- global step and consumed sample count;
- Python, NumPy, Torch CPU, and Torch CUDA RNG state;
- sampler state;
- model, data, and training configuration snapshots;
- source manifest checksum;
- software and hardware version report.

Resuming must continue from the next batch and reproduce the next loss within
the documented BF16 tolerance.

## 7. Error Handling and Safety

The system fails before mutation when:

- a real-data command runs without an explicit server execution flag;
- `TSFM_DATA_ROOT` is missing, resolves inside the repository, or is not on the
  persistent data disk;
- projected or current disk use would leave less than 20GB free;
- a manifest version or checksum does not match;
- a requested window crosses a segment or split boundary;
- loss or gradients become non-finite;
- a model configuration and checkpoint configuration differ.

Failures include the source id, record id, step, and configuration path needed
to reproduce the issue. A non-finite training failure writes a small diagnostic
state but does not overwrite the most recent valid checkpoint.

## 8. Validation Strategy

### 8.1 Local tests

Local tests use synthetic tensors and temporary NPY shards only. They cover:

- Timer initialization and analytical parameter counts;
- causal isolation, next-patch alignment, and generation;
- manifest round trip and cross-platform paths;
- multivariate-to-single-variable conversion;
- non-finite segmentation and short/constant filtering;
- temporal and heldout split isolation;
- hierarchical sampler determinism and proportions;
- proof that normalization uses context only;
- checkpoint and RNG round trip;
- refusal to download or use an unsafe data root.

### 8.2 Server checks

Server checks cover:

- RTX 5090 visibility and BF16 support;
- SDPA BF16 forward/backward;
- persistent-disk and memory-limit detection;
- data conversion throughput and manifest validation;
- a one-batch overfit check;
- 26M loss reduction over 2,000 steps;
- checkpoint interruption and resume;
- 95M stability over 500 steps;
- peak allocated GPU memory, sequences per second, patch tokens per second,
  data wait time, and achieved source distribution.

## 9. Acceptance Gates

The 26M stage passes only if:

- all local and server tests pass without warnings;
- no real dataset exists in the local project or upload package;
- BF16 loss and gradients remain finite for 2,000 steps;
- both validation views produce finite metrics and training loss is below its
  initial value;
- checkpoint resume passes;
- persistent disk retains at least 20GB free;
- no runtime module imports `third_party`.

The 95M stage passes only if the same production code runs for 500 steps with
no architecture-specific branch, no OOM, no non-finite value, and a valid
resumable checkpoint.

Only after both gates pass may the 307M configuration move to the H20 cluster.
The H20 phase begins with a 10-20-step distributed preflight before any formal
training allocation is consumed.

## 10. Delivery Phases

1. Harden the shared model and configuration contract locally.
2. Implement and test the S3 adapters, shard format, manifest, splits, sampler,
   and context normalization using synthetic fixtures.
3. Implement the BF16 trainer, scheduler, full checkpoint contract, metrics,
   disk guards, and server bootstrap scripts.
4. Upload the filtered source tree to the RTX 5090 server.
5. Download and convert the bounded UTSD/LOTSA validation subset on the server.
6. Execute and report the 26M validation run.
7. Execute and report the 95M scaling run.
8. Produce the 307M H20 launch package and distributed preflight checklist.

Formal 307M pre-training, Timer-XL multivariate attention, downstream
fine-tuning, anomaly detection, imputation, and full-corpus download are outside
this design.
