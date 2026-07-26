# Four-H100 307M Production Runbook

This runbook starts the Timer-style 307M engineering production run on
4 x NVIDIA H100 80GB GPUs. It trains exactly 307,146,240 parameters from
scratch on the pinned UTSD-12G snapshot. It does not claim reproduction of
the Timer paper's quality or benchmark results.

## 1. Storage and Trust Boundary

Windows is code-only and must not contain datasets, model weights, formal
checkpoints, wheelhouses, or Python runtime archives. Build the data and
runtime artifacts on a networked Linux x86_64 machine, transfer them through
the approved channel, and place all mutable material under `/root/work` on
the H100 node.

Required H100 layout:

```text
/root/work/TimeSeriesFoundationModel
/root/work/runtime-bundles/h100-runtime-bundle
/root/work/runtime-bundles/h100-runtime-bundle.sha256.json
/root/work/tsfm-data/raw/utsd/UTSD-12G
/root/work/tsfm-data/raw/utsd/UTSD-12G.sha256.json
/root/work/venvs/tsfm-h100
/root/work/checkpoints/timer-307m-preflight
/root/work/checkpoints/timer-307m-production
```

Do not transfer Hugging Face caches. Transfer the repository snapshot itself,
including the 80 Arrow shards and two metadata files selected by the pinned
revision. Its strict artifact manifest covers 82 files and 3,892,126,910
bytes.

## 2. Networked Linux: Build UTSD-12G Snapshot

Use a temporary or persistent transfer root on a Linux x86_64 server with
Internet access. The following endpoint is optional; omit it when the normal
Hugging Face endpoint is reachable.

```bash
cd /srv/TimeSeriesFoundationModel
python3 -m venv /srv/venvs/tsfm-transfer
source /srv/venvs/tsfm-transfer/bin/activate
python -m pip install -e '.[server]'
mkdir -p /srv/tsfm-transfer/tsfm-data/raw/utsd
python scripts/download_utsd12g_snapshot.py \
  --destination /srv/tsfm-transfer/tsfm-data/raw/utsd \
  --manifest /srv/tsfm-transfer/tsfm-data/raw/utsd/UTSD-12G.sha256.json \
  --endpoint https://hf-mirror.com
```

The command must print `UTSD-12G snapshot verified`. It is pinned to revision
`7326ff5f4578da73d843fd675d760c6c6054017f` and refuses any file-count,
byte-count, checksum, missing-file, or extra-file mismatch.

## 3. Networked Linux: Build Offline Python Bundle

Obtain an approved Linux x86_64 CPython 3.11 standalone runtime archive. The
builder executes that runtime to verify its version, downloads the pinned
PyTorch 2.7.1 CUDA 12.6 wheel set plus all transitive dependencies, and writes
the SHA-256 manifest outside the bundle directory.

```bash
cd /srv/TimeSeriesFoundationModel
bash scripts/build_h100_offline_bundle.sh \
  /srv/input/cpython-3.11-linux-x86_64.tar.zst \
  /srv/tsfm-transfer/h100-runtime-bundle
```

Transfer these items without repacking or modifying their contents:

```text
/srv/tsfm-transfer/h100-runtime-bundle
/srv/tsfm-transfer/h100-runtime-bundle.sha256.json
/srv/tsfm-transfer/tsfm-data/raw/utsd/UTSD-12G
/srv/tsfm-transfer/tsfm-data/raw/utsd/UTSD-12G.sha256.json
```

## 4. H100 Node: Place and Verify Inputs

Upload the current source clone to `/root/work/TimeSeriesFoundationModel`.
Place the bundle and data exactly as shown in Section 1. Confirm that the
node exposes four GPUs before installation:

```bash
nvidia-smi -L
test "$(nvidia-smi -L | wc -l)" -eq 4
test -f /root/work/tsfm-data/raw/utsd/UTSD-12G.sha256.json
test -f /root/work/runtime-bundles/h100-runtime-bundle.sha256.json
```

The expected environment is Ubuntu 24.04 with an NVIDIA driver that supports
CUDA 12.6 user-space wheels. A newer compatible driver is acceptable. The
pipeline, rather than the OS image label, is the acceptance authority.

## 5. H100 Node: Offline Installation

Run the installer once on a fresh destination:

```bash
cd /root/work/TimeSeriesFoundationModel
bash scripts/install_h100_offline.sh \
  /root/work/runtime-bundles/h100-runtime-bundle \
  /root/work/venvs/tsfm-h100 \
  /root/work/TimeSeriesFoundationModel
```

The installer first verifies every bundle file, then installs with
`--no-index`. It does not contact PyPI, GitHub, or Hugging Face. It refuses to
overwrite an existing runtime or venv. Success creates
`/root/work/runtime-bundles/runtime-install-report.json`.

## 6. Start the One-Command Pipeline

Use this as the only production launch command:

```bash
cd /root/work/TimeSeriesFoundationModel
nohup bash scripts/launch_h100_307m.sh > /root/work/logs/h100-307m-pipeline.log 2>&1 &
```

Monitor without starting a second process:

```bash
jobs -l
tail -f /root/work/logs/h100-307m-pipeline.log
```

The command performs source verification, atomic conversion, exact hardware
checks, NCCL, batch selection, immutable plan resolution, a 20-step preflight,
a two-step resume, and then production. Each batch candidate runs in a fresh
four-rank process. The first candidate below 80 percent GPU memory is selected
from 512, 256, 128, and 64 samples per GPU.

## 7. Acceptance Evidence

Do not call the gate successful merely because the process is running. Inspect:

```bash
python -m json.tool /root/work/reports/timer-307m/pipeline-state.json
python -m json.tool /root/work/reports/timer-307m/preflight-report.json
python -m json.tool /root/work/reports/timer-307m/resolved-training-config.json
ls -lh /root/work/checkpoints/timer-307m-preflight
ls -lh /root/work/checkpoints/timer-307m-production
```

`preflight-report.json` must say `PASS`. The binding must match package,
UTSD source snapshot, processed manifest, model configuration, resolved plan,
and current hardware digests. The preflight directory must contain a valid
`step-000022.pt`, proving the 20 steps plus two-step resume. Production starts
only after that PASS record is written.

The resolved plan must report world size 4, global batch 4,096, BF16,
three coverage cycles, peak learning rate 5e-5, 2 percent warmup, and the
selected micro-batch. Checkpoints embed that same resolved plan.

## 8. Restart and Failure Policy

After a server interruption, rerun the single launch command from Section 6.
An identical PASS binding skips the completed gate. A valid latest production
`step-*.pt` resumes automatically from its saved optimizer, scheduler,
per-rank RNG, and finite sampler position.

Stop and investigate when any phase is `FAIL`, when a digest changes, when
fewer than four H100 GPUs are visible, when NCCL or BF16 SDPA fails, when all
batch candidates fail, or when loss/gradients become non-finite. A nonempty
production directory without a valid checkpoint fails closed.

Never delete, rename, or overwrite a failing source snapshot, processed data
directory, runtime bundle, gate directory, or production directory to force a
restart. Preserve the log and JSON reports, diagnose the mismatch, and use a
new explicitly named directory only after the cause is understood.
