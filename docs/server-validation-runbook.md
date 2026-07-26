# Server Validation Runbook

This file preserves the RTX 5090 validation workflow as engineering evidence.
Formal four-H100 307M production must use
`docs/h100-307m-runbook.md`. The H20 16-card section below is legacy
preflight-only and is not the current production recipe.

## 1. Local Gate

Do not download UTSD, LOTSA, model weights, or formal checkpoints locally.

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\audit_source_tree.py --root .
```

Upload the source tree using `.uploadignore`. Never upload `.venv`, caches,
`data`, `outputs`, `checkpoints`, or weight files.

## 2. RTX 5090 Bootstrap

The uploaded project must be `/root/autodl-tmp/TimeSeriesFoundationModel`.

```bash
cd /root/autodl-tmp/TimeSeriesFoundationModel
source /root/autodl-tmp/venvs/tsfm/bin/activate
source scripts/bootstrap_autodl.sh /root/autodl-tmp
python -m pip install -e '.[dev,server]'
python -m pytest -q
python scripts/server_preflight.py \
  --persistent-root "$TSFM_PERSISTENT_ROOT" \
  --report-dir "$TSFM_CHECKPOINT_ROOT/preflight" \
  --expected-gpu "RTX 5090"
```

The preflight must report CUDA 12.8, BF16 support, successful causal SDPA
forward/backward, at least 80 GiB cgroup memory, and at least 20 GiB free.

## 3. Bounded Server-Only Data Preparation

Clear inherited proxy variables and use the working Hugging Face mirror in
every new terminal:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_ETAG_TIMEOUT=30
export HF_HUB_DOWNLOAD_TIMEOUT=120
```

Review the non-mutating policy first:

```bash
python scripts/prepare_data.py --config configs/data/server_validation.json
```

The policy contains `UTSD-1G`, `traffic_hourly`, `beijing_air_quality`, and
`weather`. LOTSA is pinned to revision
`8191fd29eb5cf906ec55effca44d8059888b615d`. Discovery probes the declared
Arrow files without scanning the LOTSA repository:

```text
traffic_hourly/data-00000-of-00001.arrow
beijing_air_quality/data-00000-of-00001.arrow
weather/data-00000-of-00001.arrow
```

Run discovery with absolute paths so it is independent of shell activation:

```bash
mkdir -p /root/autodl-tmp/logs
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
nohup env \
  HF_HOME=/root/autodl-tmp/cache/huggingface \
  HF_ENDPOINT=https://hf-mirror.com \
  HF_HUB_ETAG_TIMEOUT=30 \
  HF_HUB_DOWNLOAD_TIMEOUT=120 \
  /root/autodl-tmp/venvs/tsfm/bin/python -u \
  /root/autodl-tmp/TimeSeriesFoundationModel/scripts/prepare_data.py discover \
  --config=/root/autodl-tmp/TimeSeriesFoundationModel/configs/data/server_validation.json \
  --persistent-root=/root/autodl-tmp \
  --data-root=/root/autodl-tmp/tsfm-data \
  --execute-server \
  > /root/autodl-tmp/logs/discover.log 2>&1 &
tail -f /root/autodl-tmp/logs/discover.log
```

Discovery is complete only when the log says it wrote four selected groups.
Before conversion, inspect `inventory.json`:

```bash
/root/autodl-tmp/venvs/tsfm/bin/python -m json.tool \
  /root/autodl-tmp/tsfm-data/inventory.json
du -sb /root/autodl-tmp/cache/huggingface
```

Do not run conversion unless the inventory contains one UTSD group, three
LOTSA groups, four distinct domains, the pinned LOTSA revision and exact file
paths, and `projected_source_bytes` no greater than 2,000,000,000. The actual
Hugging Face cache must also remain no greater than 2,000,000,000 bytes.

After that review gate passes, run conversion and validation:

```bash
python scripts/prepare_data.py convert \
  --config configs/data/server_validation.json \
  --persistent-root "$TSFM_PERSISTENT_ROOT" \
  --data-root "$TSFM_DATA_ROOT" \
  --execute-server
python scripts/prepare_data.py validate \
  --config configs/data/server_validation.json \
  --persistent-root "$TSFM_PERSISTENT_ROOT" \
  --data-root "$TSFM_DATA_ROOT" \
  --execute-server
```

Conversion uses `hf_hub_download` for only the pinned Arrow files and reads
their cached local paths with the Arrow loader. Do not train unless
`conversion-report.json` proves all four selected groups produced manifest
records, all checksums pass, `hf_cache_bytes` is no greater than
2,000,000,000, `processed_bytes` is no greater than 2,000,000,000, and at
least 20 GiB disk remains free.

## 4. 26M Validation

```bash
python scripts/train.py probe \
  --model-config configs/model/timer_26m.json \
  --training-config configs/training/rtx5090_26m.json \
  --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" \
  --output-dir "$TSFM_CHECKPOINT_ROOT/26m"
python scripts/train.py overfit-one-batch \
  --model-config configs/model/timer_26m.json \
  --training-config configs/training/rtx5090_26m.json \
  --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" \
  --output-dir "$TSFM_CHECKPOINT_ROOT/26m-overfit"
python scripts/train.py run \
  --model-config configs/model/timer_26m.json \
  --training-config configs/training/rtx5090_26m.json \
  --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" \
  --output-dir "$TSFM_CHECKPOINT_ROOT/26m"
```

Gate: 2,000 finite BF16 steps, final training loss below initial, valid
heldout/temporal metrics, best plus two latest checkpoints, and resume match.

## 5. 95M Validation

Use the same commands with `timer_95m.json`, `rtx5090_95m.json`, and output
`$TSFM_CHECKPOINT_ROOT/95m`. Gate: exact batch probe succeeds without changing
batch settings and 500 finite BF16 steps complete through the same model class.

## 6. 307M H20 Engineering Preflight

Only after both 5090 gates pass:

```bash
torchrun --standalone --nproc_per_node=16 scripts/train.py run \
  --model-config configs/model/timer_300m.json \
  --training-config configs/training/h20_300m_preflight.json \
  --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" \
  --output-dir "$TSFM_CHECKPOINT_ROOT/h20-300m-preflight"
```

Gate: 16 NCCL ranks, 20 finite BF16 steps, mutually exclusive sample indices,
one rank-zero checkpoint, and a valid aggregate report. This configuration is
an engineering preflight and is not a formal 307M pre-training recipe.
