# H100 307M UTSD-12G Production Training Design

**Date:** 2026-07-26

**Status:** Approved design

## 1. Objective

Turn the existing Timer-style engineering path into a reproducible production
training path for the existing 307,146,240-parameter decoder-only model on one
node with four NVIDIA H100 80GB GPUs. The first production corpus is UTSD-12G
only. The H100 node must not require access to GitHub, Hugging Face, or PyPI.

The Windows development machine remains code-only. No dataset, converted S3
data, model weight, or production checkpoint may be downloaded to or created
on Windows.

## 2. Confirmed Server Boundary

The target is an Ubuntu 22.04.3 container on a host reporting Linux 4.18,
NVIDIA driver 575.57.08, and CUDA compatibility 12.9 through `nvidia-smi`.
The allocation is expected to expose four H100 80GB HBM3 GPUs. Persistent
storage is `/root/work`, with approximately 3.7 TB currently free. The
container reports approximately 1 TB of CPU memory.

The host kernel string does not need to match the Ubuntu user space. PyTorch
wheels carry their CUDA runtime, so the wheel CUDA version does not need to
equal the 12.9 compatibility value reported by `nvidia-smi`. The production
runtime is CPython 3.11 with PyTorch 2.7.1 CUDA 12.6. This combination supports
H100 compute capability, BF16, NCCL, and native scaled-dot-product attention
with the installed driver.

The H100 allocation must still prove that four GPUs are visible. The supplied
screenshot exposes only one GPU and therefore is hardware evidence, not proof
of the future four-GPU allocation.

## 3. Chosen Deployment Approach

Use an offline-first source bundle, dependency bundle, and data snapshot:

1. The code repository is uploaded without data, environments, caches, or
   checkpoints.
2. A networked Linux x86_64 server downloads the pinned UTSD-12G repository
   snapshot and builds the CPython 3.11/PyTorch wheelhouse.
3. That server creates SHA-256 manifests for both external bundles.
4. The bundles are transferred directly from the networked server to the H100
   environment. They do not pass through Windows.
5. The H100 node verifies both manifests, creates a fresh environment below
   `/root/work`, converts the local UTSD snapshot to S3, runs one gate, and
   starts production training.

Transferring a Hugging Face cache is explicitly rejected because cache links,
revisions, and machine-specific paths are fragile. A container image is not
required for this single-node job.

## 4. External Artifact Contract

### 4.1 UTSD-12G

The source revision is:

`7326ff5f4578da73d843fd675d760c6c6054017f`

Only `UTSD-12G/*` is selected. At that revision it contains 80 Arrow shards,
`dataset_info.json`, and `state.json`: 82 files totaling 3,892,126,910 bytes
(3.89 GB decimal, 3.62 GiB). The snapshot is stored at:

`/root/work/tsfm-data/raw/utsd/UTSD-12G`

The transfer package includes a sorted SHA-256 manifest with relative POSIX
paths, byte sizes, and hashes. Conversion is forbidden unless all 82 entries
match and no unlisted regular file appears under the snapshot root.

### 4.2 Offline Runtime

The runtime bundle targets Linux x86_64 and CPython 3.11. It contains a pinned
CPython 3.11 runtime, PyTorch 2.7.1 CUDA 12.6, all project server dependencies,
and their transitive wheels. Package versions and wheel hashes are recorded in
an immutable lock file. The bundle is installed without an index into:

`/root/work/venvs/tsfm-h100`

The wheelhouse is an external artifact and is never committed to Git.

## 5. Persistent Layout

```text
/root/work/TimeSeriesFoundationModel
/root/work/venvs/tsfm-h100
/root/work/runtime-bundles
/root/work/tsfm-data/raw/utsd/UTSD-12G
/root/work/tsfm-data/processed/utsd-12g
/root/work/checkpoints/timer-307m-preflight
/root/work/checkpoints/timer-307m-production
/root/work/logs
```

All virtual environments, package caches, source data, processed data,
checkpoints, and logs use `/root/work`. The overlay root filesystem is not used
for persistent artifacts.

## 6. Data Conversion and Split Contract

The UTSD adapter gains an explicit local Arrow snapshot mode. It reads the 80
local Arrow shards directly and performs no network lookup. Existing online
UTSD and exact-file LOTSA paths remain available, but the production policy
allows only source `utsd` and dataset group `UTSD-12G`.

S3 keeps the approved Timer-style contract:

- input and output patch length: 96 points;
- context: 30 patches, or 2,880 points;
- sample length: 31 patches, or 2,976 points;
- each prediction position learns the next 96-point patch;
- normalization is computed from the input context and applied to the target;
- finite values and minimum variance are enforced before storage.

The split seed is 2026. Ten percent of complete series identities are held out
for cross-series validation. For non-held-out series, the final ten percent of
time is temporal validation when both regions can form a complete sample. The
resulting report must disclose actual point and window counts for train,
held-out validation, and temporal validation instead of claiming an exact
percentage when short-series constraints alter it.

Conversion writes to an incomplete run directory. It atomically publishes the
manifest and a completion marker only after shard, manifest, split, checksum,
and quota validation succeeds. An incomplete directory is never accepted for
training and is not deleted automatically.

## 7. Finite Production Sampling

The current infinite with-replacement counter stream remains available for
small engineering probes but is not used for production.

Production defines every valid stride-one window in every training region as a
finite canonical index. A prefix-sum index maps a canonical number to its
region and raw start without materializing all windows. For each coverage
cycle, an affine permutation with a multiplier coprime to the window count and
an independently derived offset deterministically shuffles `[0, N)` in O(1)
memory. Ranks consume disjoint positions from that permutation.

One coverage cycle therefore visits every valid window once. Production runs
three cycles. Only the final global batch may be padded, with fewer than 4,096
repeated windows; the resolved plan records the exact padding count. The
checkpoint records cycle, permutation position, consumed real samples, padded
samples, and optimizer step so resume does not skip or replay completed work.

## 8. Model Contract

The model configuration remains:

| Field | Value |
| --- | ---: |
| Parameters | 307,146,240 |
| Hidden size | 1,536 |
| Intermediate size | 3,072 |
| Decoder layers | 13 |
| Attention heads | 12 |
| Head dimension | 128 |
| Input patch | 96 |
| Output patch | 96 |
| RoPE theta | 10,000 |
| Attention dropout | 0 |

There is one model implementation for tiny, 26M, 95M, and 307M variants. The
H100 work does not add a model-size branch. DDP replicates the 307M model on
each GPU; FSDP is unnecessary for four 80GB devices.

## 9. Resolved Production Training Plan

Stable choices are:

- world size: 4;
- precision: BF16 autocast with FP32 optimizer state;
- target global batch: 4,096 windows;
- optimizer: fused AdamW when supported, with a tested standard AdamW fallback;
- peak learning rate: 5e-5;
- warmup: first 2 percent of resolved optimizer steps;
- decay: cosine to 5e-6;
- betas: 0.9 and 0.95;
- weight decay: 0.1;
- gradient clipping: 1.0;
- coverage cycles: 3;
- seed: 2026.

The one-time gate tests per-GPU micro-batch candidates 512, 256, 128, and 64,
in that order. Each candidate runs in a fresh four-rank `torchrun` subprocess
so an out-of-memory failure cannot poison the CUDA or NCCL state used by the
next candidate. A candidate performs a real forward, backward, gradient clip,
and optimizer step. It is accepted only if all ranks stay below 80 percent of
physical GPU memory and report finite values. Gradient accumulation is then
`4096 / (micro_batch * 4)`. Every candidate divides the target exactly.

After conversion and batch resolution, rank zero writes
`resolved-training-config.json`. It includes window counts, coverage cycles,
padding, total optimizer steps, micro-batch, accumulation, global batch,
schedule steps, all fixed hyperparameters, model digest, source snapshot
digest, processed manifest digest, package lock digest, and world size. The
same immutable object is embedded in checkpoints.

## 10. One-Command Gate and Launch

The operator starts one top-level pipeline command under `nohup`. The pipeline
is phase-aware and performs:

1. verify the offline runtime and create or validate the fresh venv;
2. verify four H100 80GB GPUs, BF16, RAM, `/root/work` disk, and SDPA backward;
3. verify the UTSD-12G SHA-256 manifest;
4. convert the local snapshot or validate an already complete conversion;
5. run a four-rank NCCL collective test and capture GPU topology;
6. verify the exact 307,146,240 model parameter count;
7. resolve micro-batch and the immutable training plan;
8. run 20 real-data BF16 DDP optimizer steps;
9. save a preflight checkpoint and resume it for two more steps;
10. write an atomic `preflight-report.json` bound to all relevant digests;
11. start or resume production training only when every gate passed.

The preflight and production output directories are separate. A valid gate is
reused on pipeline restart if every bound digest and hardware fact still
matches. A server interruption during production resumes the production
checkpoint without rerunning the successful gate.

## 11. DDP and Checkpoint Requirements

DDP uses NCCL through `torchrun --standalone --nproc_per_node=4`. Gradient
accumulation uses `no_sync()` for all non-final micro-batches. Data workers are
persistent, host memory is pinned, and copies are non-blocking. Rank metrics,
source counts, sample counts, throughput, and peak memory are reduced into one
rank-zero report. Validation positions are partitioned across ranks and metric
numerators and denominators are all-reduced.

Checkpoint state is saved from the unwrapped model so exported weights do not
contain a `module.` prefix. A full checkpoint contains model, optimizer,
scheduler, per-rank RNG, sampler, resolved configuration, manifest digest,
package digest, world size, and environment report. Rank zero gathers the RNG
state from every rank, and each rank restores only its own state. Writes use a
temporary file and atomic replacement.

Production logs loss, learning rate, samples per second, patch predictions per
second, and per-rank peak memory every 10 steps. Validation and checkpointing
occur every 2,000 steps. The primary selection metric is the mean normalized
MSE across held-out-series and temporal views; raw-scale MSE and MAE remain
diagnostic metrics and do not choose the best checkpoint. Retention keeps the
latest three full checkpoints, the best validation checkpoint, and the final
checkpoint. Training completion also exports model-only weights and their
configuration.

## 12. Failure Policy

The pipeline stops before production for any data checksum mismatch, unlisted
source file, incomplete conversion, wrong GPU count or model, insufficient
memory or disk, failed BF16/SDPA/NCCL probe, non-finite preflight value,
checkpoint-resume mismatch, or digest mismatch.

Production saves a diagnostic and stops for non-finite loss or gradients. An
existing production directory with a valid latest checkpoint resumes. A
nonempty production directory without a valid checkpoint fails closed and is
never overwritten or deleted automatically.

## 13. Repository and Synchronization Boundary

Git contains source, JSON policies and configurations, lock metadata, scripts,
tests, and documentation. `.gitignore` and the source audit exclude data,
wheelhouses, runtime archives, virtual environments, caches, S3 shards, logs,
and checkpoints.

`D:\\学习\\TimeSeriesFoundationModel` and
`D:\\学习\\TimeSeriesFoundationModel-ServerUpload` are operationally required
to contain byte-identical runtime source after every implementation change.
The upload clone remains the Git repository used for publishing.

## 14. Verification and Acceptance

Local verification uses synthetic fixtures only and includes the complete unit
suite, local Arrow adapter tests, SHA manifest tests, finite sampler coverage
and resume tests, resolved-plan tests, DDP state tests, checkpoint portability
tests, one-command phase tests, and source-tree audit. No local test may fetch
or materialize UTSD, LOTSA, weights, or checkpoints.

The H100 acceptance gate requires:

- exactly four H100 80GB devices;
- successful BF16 SDPA backward on every rank;
- successful NCCL collectives across four ranks;
- exact source and processed-data digests;
- exact model parameter count;
- one resolved global batch of 4,096;
- 20 finite DDP steps followed by a two-step resume with exact optimizer-step,
  sampler-position, and digest continuity;
- disjoint sample positions across ranks;
- one valid rank-zero checkpoint and aggregate report;
- automatic transition into production only after the atomic PASS report.

This produces a complete engineering production run for the 307M model on
UTSD-12G. It does not claim Timer or Timer-XL quality parity. LOTSA expansion,
1B/3B scaling, cross-variable Timer-XL attention, downstream benchmarks, and
formal model release evaluation remain outside this design.
