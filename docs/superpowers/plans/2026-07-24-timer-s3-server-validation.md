# Timer-Style S3 Server Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a portable Timer-style path from heterogeneous UTSD/LOTSA series through S3 preprocessing to resumable BF16 validation on one RTX 5090, then package the unchanged 307M model class for a 16-card H20 preflight.

**Architecture:** Retain one channel-independent decoder-only `TimerModel` and scale it only through JSON. Add lazy source adapters, finite univariate segmentation, bounded atomic NPY shards, a versioned JSONL manifest, leak-free split regions, and counter-addressable hierarchical sampling. A separate production trainer consumes that data contract, computes loss and metrics in FP32 under BF16 autocast, and saves all state needed for deterministic continuation.

**Tech Stack:** Python 3.12, PyTorch 2.8/CUDA 12.8, NumPy, Hugging Face `datasets` as a server-only optional dependency, pytest, Bash, JSON/JSONL, NPY memory maps, NCCL/DDP.

## Global Constraints

- Never download or store real datasets, converted shards, model weights, or formal checkpoints on the local Windows machine.
- Local tests use only deterministic tensors and files below pytest `tmp_path`; they never access a network or Hugging Face cache.
- On the server, project, data, caches, and checkpoints resolve below the explicit `TSFM_PERSISTENT_ROOT`; on the validated AutoDL host it is `/root/autodl-tmp`.
- `TSFM_DATA_ROOT`, `HF_HOME`, `PIP_CACHE_DIR`, `TORCH_HOME`, and checkpoint roots are outside the repository and below the persistent root.
- Data mutation defaults to dry-run and requires `--execute-server`; guards run before any directory is created.
- Disk mutation must leave at least 20 GiB free. Validation raw data and processed shards are each capped at 20,000,000,000 bytes.
- Upload excludes `.venv/`, `outputs/`, `checkpoints/`, `data/`, `.pytest_cache/`, `__pycache__/`, `*.pt`, `*.pth`, and `*.safetensors`.
- Production Python contains no absolute Windows or AutoDL paths. `third_party/` remains reference-only and is never imported at runtime.
- Inputs are univariate `[batch, raw_length]`; patch length is 96; context is 30 patches/2,880 values; each sampled window is 31 patches/2,976 values.
- Architecture remains causal SDPA plus RoPE, biased Q/K/V, bias-free attention output, SiLU gated MLP, post-norm blocks, final LayerNorm, and bias-free next-patch head.
- Linear initialization is `Normal(0, 0.02)` with zero biases; LayerNorm starts with one weight and zero bias.
- Normalization is `sqrt(context.var(unbiased=False) + 1e-5)` and target values never influence statistics.
- Validation uses BF16 autocast, FP32 loss/metrics, AdamW `(0.9, 0.95)`, weight decay `0.1`, gradient clipping `1.0`, cosine decay to 10% peak LR, and seed `2026`.
- Preserve the existing CPU smoke command and all current tests.
- The project is not a Git repository. Do not initialize Git implicitly; run listed commits only after owner authorization.

---

## Planned File Structure

- `src/tsfm/model.py`, `config.py`: single model and scale configuration.
- `src/tsfm/data.py`: synthetic fixtures and shared normalization.
- `src/tsfm/s3/records.py`, `segments.py`, `adapters.py`: source-neutral S3 input contract.
- `src/tsfm/s3/manifest.py`, `shards.py`: portable atomic storage.
- `src/tsfm/s3/splits.py`, `sampling.py`, `dataset.py`: isolation and deterministic window loading.
- `src/tsfm/safety.py`: execution, path, free-space, and upload guards.
- `src/tsfm/train_config.py`, `optim.py`, `metrics.py`, `trainer.py`: production training.
- `src/tsfm/checkpoint.py`: full-state atomic checkpointing and retention.
- `scripts/prepare_data.py`, `server_preflight.py`, `train.py`, `audit_source_tree.py`, `bootstrap_autodl.sh`: operational entry points.
- `configs/data/server_validation.json`: bounded source-selection policy.
- `configs/model/timer_26m.json`, `timer_95m.json`, `timer_300m.json`: model ladder.
- `configs/training/rtx5090_26m.json`, `rtx5090_95m.json`, `h20_300m_preflight.json`: engineering runs.
- `docs/server-validation-runbook.md`: upload, validation gates, and H20 handoff.

### Task 1: Lock Initialization, FP32 Loss, and Model Sizes

**Files:** Modify `src/tsfm/model.py`, `tests/test_model.py`, `tests/test_config.py`; create `configs/model/timer_26m.json`, `configs/model/timer_95m.json`; verify `configs/model/timer_300m.json`.

**Interfaces:** Keep `TimerModel(TimerConfig)` and `TimerOutput`; add `TimerModel._next_patch_loss(predictions, labels)`; keep `TimerConfig.from_json` and `estimate_parameter_count`.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_model.py
def test_timer_initialization_and_fp32_loss() -> None:
    torch.manual_seed(2026)
    model = TimerModel(tiny_config())
    weights = torch.cat([m.weight.flatten() for m in model.modules() if isinstance(m, torch.nn.Linear)])
    assert abs(float(weights.mean())) < 0.003
    assert 0.017 < float(weights.std(unbiased=False)) < 0.023
    for module in model.modules():
        if isinstance(module, torch.nn.Linear) and module.bias is not None:
            assert torch.count_nonzero(module.bias) == 0
    loss = model._next_patch_loss(torch.randn(2, 4, 16).bfloat16(), torch.randn(2, 64))
    assert loss.dtype == torch.float32

# tests/test_config.py
@pytest.mark.parametrize("name,count", [
    ("timer_26m.json", 26_349_568), ("timer_95m.json", 94_635_008),
    ("timer_300m.json", 307_146_240),
])
def test_exact_model_ladder(name: str, count: int) -> None:
    path = Path(__file__).parents[1] / "configs/model" / name
    assert estimate_parameter_count(TimerConfig.from_json(path)) == count
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_model.py::test_timer_initialization_and_fp32_loss tests/test_config.py::test_exact_model_ladder -q`

Expected: FAIL because initialization/loss helper and two JSON files are absent.

- [ ] **Step 3: Implement minimally**

```python
# inside TimerModel.__init__, after modules are assigned
self.apply(self._initialize_weights)

@staticmethod
def _initialize_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

def _next_patch_loss(self, predictions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if labels.shape != (predictions.shape[0], predictions.shape[1] * predictions.shape[2]):
        raise ValueError("labels must have shape [batch, patch_count * output_token_len]")
    return F.mse_loss(predictions.float(), labels.reshape_as(predictions).float())
```

Use `_next_patch_loss` in `forward`. Write complete JSON objects with shared patch/max-position fields and `(hidden, intermediate, layers, heads)` equal to `(512,1024,10,8)`, `(1024,2048,9,8)`, and `(1536,3072,13,12)`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_model.py tests/test_config.py -q`

Expected: PASS without warnings, including causal isolation and generation.

- [ ] **Step 5: Commit after Git exists**

```bash
git add src/tsfm/model.py tests/test_model.py tests/test_config.py configs/model
git commit -m "feat: lock Timer model contract"
```

### Task 2: Define S3 Records, Adapters, Segmentation, and Normalization

**Files:** Create `src/tsfm/s3/{__init__,records,segments,adapters}.py`, `tests/s3/test_segments.py`, `tests/s3/test_adapters.py`; modify `src/tsfm/data.py`, `tests/test_data.py`, `pyproject.toml`.

**Interfaces:** `RawSeries(source_id,dataset_id,series_id,values,frequency)`, `PreparedSegment`, `DatasetSpec`, `UTSDAdapter.iter_series`, `LOTSAAdapter.iter_series`, `finite_univariate_segments`, and `NormalizedBatch.scale`.

- [ ] **Step 1: Write failing tests**

```python
def test_channels_and_nonfinite_gaps_become_independent_segments() -> None:
    values = np.column_stack((np.arange(8.), np.arange(8.) + 10)); values[3, 0] = np.nan
    raw = RawSeries("utsd", "weather", "s1", values, "H")
    got = list(finite_univariate_segments(raw, min_length=3, min_variance=1e-8))
    assert [(x.channel_id, x.values.tolist()) for x in got] == [
        ("0", [0.,1.,2.]), ("0", [4.,5.,6.,7.]), ("1", [10.,11.,12.,13.,14.,15.,16.,17.])]
    assert all(x.values.dtype == np.float32 for x in got)

def test_adapter_uses_injected_loader_without_datasets_import(tmp_path) -> None:
    loader = lambda *a, **k: [{"item_id":"x", "target":[1,2,3], "freq":"H"}]
    spec = DatasetSpec("thuml/UTSD", "UTSD-1G", "utsd", "weather")
    row = next(UTSDAdapter(spec, tmp_path, loader).iter_series())
    assert (row.series_id, row.values.tolist()) == ("x", [1,2,3])

def test_normalization_ignores_target_statistics() -> None:
    context = torch.tensor([[1.,3.,5.]])
    a = normalize_context_target(context, torch.tensor([[7.,9.]]))
    b = normalize_context_target(context, torch.tensor([[7000.,-9000.]]))
    torch.testing.assert_close(a.scale, torch.sqrt(context.var(-1, unbiased=False, keepdim=True) + 1e-5))
    torch.testing.assert_close(a.scale, b.scale)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/s3/test_segments.py tests/s3/test_adapters.py tests/test_data.py -q`

Expected: collection FAIL because S3 modules and `.scale` do not exist.

- [ ] **Step 3: Implement records and transformations**

```python
@dataclass(frozen=True, slots=True)
class RawSeries:
    source_id: str; dataset_id: str; series_id: str
    values: np.ndarray; frequency: str | None

@dataclass(frozen=True, slots=True)
class PreparedSegment:
    source_id: str; dataset_id: str; series_id: str; channel_id: str
    segment_index: int; values: np.ndarray; frequency: str | None

def finite_univariate_segments(series, min_length=2976, min_variance=1e-8):
    values = np.asarray(series.values)
    if values.ndim == 1: values = values[:, None]
    if values.ndim != 2: raise ValueError("values must be one- or two-dimensional")
    for channel in range(values.shape[1]):
        data = np.asarray(values[:, channel], dtype=np.float32)
        bounds = np.flatnonzero(np.diff(np.r_[False, np.isfinite(data), False])).reshape(-1, 2)
        kept = 0
        for start, stop in bounds:
            part = np.ascontiguousarray(data[start:stop])
            if len(part) >= min_length and float(np.var(part, dtype=np.float64)) > min_variance:
                yield PreparedSegment(series.source_id, series.dataset_id, series.series_id, str(channel), kept, part, series.frequency)
                kept += 1
```

Adapters accept an injected loader; otherwise they import `datasets.load_dataset` inside `iter_series`, request `streaming=True`, and map `target`, `item_id|id`, and `freq|frequency`. Add only `server = ["datasets>=3,<5", "psutil>=6,<8", "tqdm>=4.66,<5"]` as an optional extra. Replace normalization with:

```python
mean = context.mean(dim=-1, keepdim=True)
scale = torch.sqrt(context.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
return NormalizedBatch((context-mean)/scale, (target-mean)/scale, mean, scale)
```

- [ ] **Step 4: Verify GREEN without server dependencies**

Run: `python -m pytest tests/s3/test_segments.py tests/s3/test_adapters.py tests/test_data.py tests/test_training.py -q`

Expected: PASS with no network/cache access.

- [ ] **Step 5: Commit after Git exists**

```bash
git add src/tsfm/s3 src/tsfm/data.py pyproject.toml tests/s3 tests/test_data.py
git commit -m "feat: add S3 source contracts"
```

### Task 3: Add Atomic NPY Shards and Versioned JSONL Manifest

**Files:** Create `src/tsfm/s3/manifest.py`, `src/tsfm/s3/shards.py`, `tests/s3/test_manifest.py`, `tests/s3/test_shards.py`.

**Interfaces:** `ManifestRecord`, `write_manifest_atomic`, `load_manifest`, `manifest_checksum`, `pack_segments`, `read_segment_mmap`.

- [ ] **Step 1: Write failing storage tests**

```python
def test_shard_rollover_and_manifest_round_trip(tmp_path) -> None:
    segments = [prepared("a", [0,1,2,3]), prepared("b", [4,5,6,7])]
    records, digest = pack_segments(segments, tmp_path, max_shard_bytes=16)
    assert [r.relative_shard_path.as_posix() for r in records] == ["shards/shard-00000.npy", "shards/shard-00001.npy"]
    assert len(digest) == 64
    assert load_manifest(tmp_path / "manifest.jsonl", digest) == records
    values = read_segment_mmap(tmp_path, records[1])
    assert isinstance(values.base, np.memmap)
    np.testing.assert_array_equal(values, np.array([4,5,6,7], np.float32))

def test_manifest_rejects_version_checksum_and_parent_path(tmp_path) -> None:
    for mutation, message in [("version", "format_version"), ("checksum", "checksum"), ("path", "relative")]:
        with pytest.raises(ValueError, match=message): validate_mutated_manifest(tmp_path, mutation)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/s3/test_manifest.py tests/s3/test_shards.py -q`

Expected: collection FAIL because storage modules do not exist.

- [ ] **Step 3: Implement canonical storage**

```python
@dataclass(frozen=True, slots=True)
class ManifestRecord:
    format_version: int; source_id: str; dataset_id: str; series_id: str; channel_id: str
    relative_shard_path: PurePosixPath; offset: int; length: int
    frequency: str | None; split_group: str; checksum: str

def manifest_checksum(records):
    return hashlib.sha256("".join(canonical_json_line(x) for x in records).encode()).hexdigest()

def write_manifest_atomic(path, records):
    destination = Path(path); temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.writelines(canonical_json_line(x) for x in records); handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return manifest_checksum(records)
```

`canonical_json_line` uses sorted compact JSON and POSIX relative paths. `load_manifest` requires version 1, rejects absolute/`..` paths, checks expected digest, positive length, nonnegative offset, and 64-character segment checksum. `pack_segments` groups segments without exceeding 2,000,000,000 bytes, allocates each exact final shape through `np.lib.format.open_memmap` at `.tmp.npy`, copies values once, records offsets and SHA256 of little-endian float32 bytes, flushes, atomically renames, and finally writes `manifest.jsonl`. Reject a single oversized segment. `read_segment_mmap` uses `np.load(..., mmap_mode="r", allow_pickle=False)` and verifies bounds.

- [ ] **Step 4: Verify GREEN and atomic cleanup**

Run: `python -m pytest tests/s3/test_manifest.py tests/s3/test_shards.py -q`

Expected: PASS and `assert not list(tmp_path.rglob("*.tmp*"))` after successful writes.

- [ ] **Step 5: Commit after Git exists**

```bash
git add src/tsfm/s3/manifest.py src/tsfm/s3/shards.py tests/s3/test_manifest.py tests/s3/test_shards.py
git commit -m "feat: add portable S3 storage"
```

### Task 4: Implement Leak-Free Splits and Resumable Hierarchical Windows

**Files:** Create `src/tsfm/s3/splits.py`, `sampling.py`, `dataset.py`, `tests/s3/test_splits.py`, `test_sampling.py`, `test_dataset.py`.

**Interfaces:** `SplitRegion`, `SampleKey`, `build_split_regions`, `HierarchicalIndex.sample`, `CounterSampler`, `S3WindowDataset`.

- [ ] **Step 1: Write failing isolation and determinism tests**

```python
def test_temporal_regions_share_no_point_and_short_series_is_train_only() -> None:
    long_regions = build_split_regions([record("long", 40000)], heldout_fraction=0)
    train = next(x for x in long_regions if x.split == "train")
    val = next(x for x in long_regions if x.split == "val_temporal")
    assert train.stop <= val.start
    assert [x.split for x in build_split_regions([record("short", 5000)], heldout_fraction=0)] == ["train"]

def test_source_probability_is_equal_and_replayable() -> None:
    records = [record("large", str(i)) for i in range(20)] + [record("small", "x")]
    regions = [SplitRegion(i, "train", 0, 10000) for i in range(21)]
    index = HierarchicalIndex(records, regions, "train", 2976, 2026)
    keys = [index.sample(i) for i in range(10000)]
    assert keys == [index.sample(i) for i in range(10000)]
    share = sum(records[regions[k.region_index].record_index].source_id == "large" for k in keys) / len(keys)
    assert 0.47 < share < 0.53

def test_window_target_is_shifted_one_patch_and_uses_context_scale(tmp_path) -> None:
    dataset = synthetic_mmap_dataset(tmp_path, values=np.arange(40, dtype=np.float32), patch=4, context_patches=3)
    item = dataset[0]; start = item["window_start"]
    raw_context = np.arange(40, dtype=np.float32)[start:start+12]
    raw_target = np.arange(40, dtype=np.float32)[start+4:start+16]
    scale = np.sqrt(raw_context.var() + 1e-5)
    np.testing.assert_allclose(item["target"], (raw_target-raw_context.mean())/scale)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/s3/test_splits.py tests/s3/test_sampling.py tests/s3/test_dataset.py -q`

Expected: collection FAIL because split/sampling/dataset modules do not exist.

- [ ] **Step 3: Implement exact policies**

```python
def build_split_regions(records, sample_length=2976, heldout_fraction=.1, temporal_fraction=.1, seed=2026):
    identities = group_by_source((r.source_id, r.dataset_id, r.series_id) for r in records)
    heldout = set()
    for source_ids in identities.values():
        ordered = sorted(source_ids, key=lambda x: (stable_sha256_u64(x, seed), x))
        count = max(1, int(len(ordered)*heldout_fraction)) if heldout_fraction and len(ordered) >= 2 else int(len(ordered)*heldout_fraction)
        heldout.update(ordered[:count])
    result = []
    for index, record in enumerate(records):
        identity = (record.source_id, record.dataset_id, record.series_id)
        if identity in heldout:
            result.append(SplitRegion(index, "val_heldout", 0, record.length)); continue
        boundary = int(record.length*(1-temporal_fraction))
        if boundary >= sample_length and record.length-boundary >= sample_length:
            result.extend([SplitRegion(index,"train",0,boundary), SplitRegion(index,"val_temporal",boundary,record.length)])
        elif record.length >= sample_length:
            result.append(SplitRegion(index,"train",0,record.length))
    return result
```

`HierarchicalIndex` groups valid region indices by sorted source ID. For each global sample index it seeds a local `random.Random` from SHA256 of `seed:index`, chooses source uniformly, region uniformly inside source, and window start uniformly inside the region. `CounterSampler(start,rank,world_size)` yields `start+rank, start+rank+world_size, ...`. `S3WindowDataset[index]` mmap-slices 2,976 points, makes context `window[:2880]`, target `window[96:]`, normalizes both from context, and returns tensors plus mean, scale, source ID, record ID, sample index, and start. It raises before slicing across a region.

- [ ] **Step 4: Verify GREEN and resume addressing**

Run: `python -m pytest tests/s3/test_splits.py tests/s3/test_sampling.py tests/s3/test_dataset.py -q`

Expected: PASS; rank 0/1 counters from the same start have no overlap and a resumed start reproduces the expected next keys.

- [ ] **Step 5: Commit after Git exists**

```bash
git add src/tsfm/s3/splits.py src/tsfm/s3/sampling.py src/tsfm/s3/dataset.py tests/s3
git commit -m "feat: add isolated resumable S3 windows"
```

### Task 5: Guard Server Data Preparation and Upload Contents

**Files:** Create `src/tsfm/safety.py`, `scripts/prepare_data.py`, `scripts/audit_source_tree.py`, `configs/data/server_validation.json`, `tests/test_safety.py`, `tests/test_data_cli.py`, `tests/test_source_audit.py`; modify `.gitignore`.

**Interfaces:** `validate_server_mutation`, `audit_source_tree`, and `prepare_data.py {discover,convert,validate}`; no action defaults to policy-only dry run.

- [ ] **Step 1: Write failing refusal tests**

```python
def test_guard_fails_before_creating_data_root(tmp_path) -> None:
    persistent = tmp_path/"persistent"; repo = persistent/"project"; repo.mkdir(parents=True)
    data = persistent/"data"
    with pytest.raises(PermissionError, match="execute-server"):
        validate_server_mutation(False, repo, persistent, data, 0)
    assert not data.exists()

def test_guard_rejects_repo_data_and_insufficient_space(tmp_path, monkeypatch) -> None:
    repo = tmp_path/"project"; repo.mkdir()
    with pytest.raises(ValueError, match="outside"):
        validate_server_mutation(True, repo, tmp_path, repo/"data", 0)
    monkeypatch.setattr(shutil, "disk_usage", lambda p: SimpleNamespace(free=20*1024**3))
    with pytest.raises(OSError, match="free"):
        validate_server_mutation(True, repo, tmp_path, tmp_path/"data", 1)

def test_audit_rejects_weights_data_and_third_party_import(tmp_path) -> None:
    (tmp_path/"bad.pt").write_bytes(b"x")
    assert any("bad.pt" in x for x in audit_source_tree(tmp_path))
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_safety.py tests/test_data_cli.py tests/test_source_audit.py -q`

Expected: collection FAIL because guards and CLIs do not exist.

- [ ] **Step 3: Implement safety before mutation**

```python
def validate_server_mutation(execute_server, repository_root, persistent_root, data_root, projected_bytes, minimum_free_gib=20):
    if not execute_server: raise PermissionError("data mutation requires --execute-server")
    repo, persistent, data = map(lambda p: Path(p).resolve(), (repository_root,persistent_root,data_root))
    if not repo.is_relative_to(persistent) or not data.is_relative_to(persistent):
        raise ValueError("project and data must be below persistent root")
    if data.is_relative_to(repo): raise ValueError("data root must remain outside repository")
    required = projected_bytes + minimum_free_gib*1024**3
    if shutil.disk_usage(persistent).free < required: raise OSError("operation would violate free-space reserve")
    return SafePaths(repo,persistent,data)
```

The JSON policy names `thuml/UTSD` with `UTSD-1G`, `Salesforce/lotsa_data` with configuration discovery, seed 2026, two dataset groups per source, four domains total, raw/processed limits 20,000,000,000, segment length 2,976, variance `1e-8`, and shard limit 2,000,000,000. `discover` writes atomic inventory metadata; `convert` accepts only an inventory satisfying every quota and streams adapters into shards; `validate` verifies paths, bounds, checksums, quotas, and group/domain counts. Every mutating branch calls the guard before importing `datasets` or making directories. `audit_source_tree` checks all exclusions and rejects `third_party` imports below `src/` and `scripts/`.

- [ ] **Step 4: Verify GREEN locally with network disabled**

Run: `python -m pytest tests/test_safety.py tests/test_data_cli.py tests/test_source_audit.py -q`

Expected: PASS; `python scripts/prepare_data.py --config configs/data/server_validation.json` prints `DRY RUN` and creates nothing.

- [ ] **Step 5: Commit after Git exists**

```bash
git add src/tsfm/safety.py scripts/prepare_data.py scripts/audit_source_tree.py configs/data .gitignore tests
git commit -m "feat: enforce server-only data safety"
```

### Task 6: Add Production Training Configuration and Scheduler

**Files:** Create `src/tsfm/train_config.py`, `src/tsfm/optim.py`, `tests/test_train_config.py`, `tests/test_optim.py`, `configs/training/rtx5090_26m.json`, `rtx5090_95m.json`.

**Interfaces:** immutable `TrainingConfig.from_json`, `lr_multiplier`, `build_optimizer`, `build_scheduler`.

- [ ] **Step 1: Write failing exact-value tests**

```python
def test_approved_5090_configs() -> None:
    run26 = TrainingConfig.from_json(CONFIG/"rtx5090_26m.json")
    run95 = TrainingConfig.from_json(CONFIG/"rtx5090_95m.json")
    assert (run26.total_steps,run26.warmup_steps,run26.peak_lr) == (2000,100,3e-4)
    assert (run95.total_steps,run95.warmup_steps,run95.peak_lr) == (500,50,2e-4)
    assert run26.micro_batch_size*run26.gradient_accumulation_steps == 256

def test_optimizer_and_cosine_endpoints() -> None:
    config = TrainingConfig(total_steps=100,warmup_steps=10,validation_interval=10,checkpoint_interval=10)
    optimizer = build_optimizer([torch.nn.Parameter(torch.ones(1))],config)
    assert optimizer.defaults["betas"] == (0.9,0.95)
    assert lr_multiplier(0,config) == pytest.approx(.1)
    assert lr_multiplier(9,config) == pytest.approx(1.)
    assert lr_multiplier(99,config) == pytest.approx(.1)
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_train_config.py tests/test_optim.py -q`

Expected: collection FAIL because modules/configurations are absent.

- [ ] **Step 3: Implement validated configuration and math**

```python
@dataclass(frozen=True, slots=True)
class TrainingConfig:
    total_steps:int; warmup_steps:int; peak_lr:float
    minimum_lr_ratio:float=.1; micro_batch_size:int=32; gradient_accumulation_steps:int=8
    weight_decay:float=.1; beta1:float=.9; beta2:float=.95; gradient_clip:float=1.
    validation_interval:int=250; checkpoint_interval:int=250; seed:int=2026
    num_workers:int=8; prefetch_factor:int=2; pin_memory:bool=True; context_patches:int=30
    precision:str="bf16"

def lr_multiplier(step, c):
    if step < c.warmup_steps: return (step+1)/max(1,c.warmup_steps)
    progress = (step-c.warmup_steps)/max(1,c.total_steps-c.warmup_steps-1)
    return c.minimum_lr_ratio+(1-c.minimum_lr_ratio)*.5*(1+math.cos(math.pi*min(1.,progress)))

def build_optimizer(parameters,c):
    return torch.optim.AdamW(parameters,lr=c.peak_lr,betas=(c.beta1,c.beta2),weight_decay=c.weight_decay)
```

Validate positive values, BF16, warmup range, and intervals dividing total steps. Both JSONs explicitly set micro-batch 32, accumulation 8, workers 8, pinned memory true, prefetch 2, context 30, seed 2026; 26M uses `2000/100/0.0003`, 95M uses `500/50/0.0002`.

- [ ] **Step 4: Verify GREEN**

Run: `python -m pytest tests/test_train_config.py tests/test_optim.py -q`

Expected: PASS and the last scheduled LR is exactly 10% peak within float tolerance.

- [ ] **Step 5: Commit after Git exists**

```bash
git add src/tsfm/train_config.py src/tsfm/optim.py configs/training tests/test_train_config.py tests/test_optim.py
git commit -m "feat: add production training schedules"
```

### Task 7: Expand Checkpoints to Full Exact Continuation

**Files:** Modify `src/tsfm/checkpoint.py`, `tests/test_checkpoint.py`; create `tests/test_checkpoint_retention.py`.

**Interfaces:** `TrainingState(global_step,consumed_samples,sampler_state)`, `save_training_checkpoint`, `load_training_checkpoint`, `CheckpointManager`, `save_diagnostic`.

- [ ] **Step 1: Write failing state/RNG/compatibility tests**

```python
def test_checkpoint_restores_next_rng_and_sampler_state(tmp_path, components) -> None:
    random.seed(7); np.random.seed(7); torch.manual_seed(7)
    state = TrainingState(3,768,{"next_sample":768})
    save_training_checkpoint(tmp_path/"x.pt",state=state,manifest_checksum="a"*64,**components)
    expected = (random.random(),float(np.random.rand()),torch.rand(1))
    random.seed(99); np.random.seed(99); torch.manual_seed(99)
    loaded = load_training_checkpoint(tmp_path/"x.pt",expected_manifest_checksum="a"*64,**components.restore_args)
    actual = (random.random(),float(np.random.rand()),torch.rand(1))
    assert loaded == state and actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2],expected[2],rtol=0,atol=0)

def test_retention_keeps_best_and_two_latest(tmp_path, components) -> None:
    manager = CheckpointManager(tmp_path,keep_latest=2)
    for step,metric in [(250,4.),(500,3.),(750,5.),(1000,2.)]: manager.save(step,metric,**components)
    assert sorted(x.name for x in tmp_path.glob("*.pt")) == ["best.pt","step-000750.pt","step-001000.pt"]
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_checkpoint.py tests/test_checkpoint_retention.py -q`

Expected: FAIL because full state and manager do not exist.

- [ ] **Step 3: Implement atomic complete payloads**

```python
@dataclass(frozen=True,slots=True)
class TrainingState:
    global_step:int; consumed_samples:int; sampler_state:dict[str,int]

def save_training_checkpoint(path,*,model,optimizer,scheduler,state,config_snapshots,manifest_checksum,environment):
    payload={"format_version":1,"model":model.state_dict(),"optimizer":optimizer.state_dict(),
      "scheduler":scheduler.state_dict(),"training_state":asdict(state),"rng":capture_rng(),
      "config_snapshots":config_snapshots,"manifest_checksum":manifest_checksum,"environment":environment}
    destination=Path(path); temporary=destination.with_name(f".{destination.name}.tmp")
    torch.save(payload,temporary); os.replace(temporary,destination)

def load_training_checkpoint(path,*,model,optimizer,scheduler,expected_config_snapshots,expected_manifest_checksum):
    payload=torch.load(path,map_location="cpu",weights_only=True)
    if payload["config_snapshots"] != expected_config_snapshots: raise ValueError("checkpoint configuration mismatch")
    if payload["manifest_checksum"] != expected_manifest_checksum: raise ValueError("checkpoint manifest checksum mismatch")
    model.load_state_dict(payload["model"]); optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"]); restore_rng(payload["rng"])
    return TrainingState(**payload["training_state"])
```

`capture_rng/restore_rng` cover Python, NumPy, Torch CPU, and all CUDA generators using tensors/primitive containers compatible with `weights_only=True`. Manager writes new files before pruning. `save_diagnostic` stores step, loss, source/record IDs, gradient norm, and config paths but no model/optimizer tensors and never overwrites a valid checkpoint.

- [ ] **Step 4: Verify GREEN and existing smoke wrapper**

Run: `python -m pytest tests/test_checkpoint.py tests/test_checkpoint_retention.py tests/test_training.py -q`

Expected: PASS with exact next RNG values and config/manifest mismatch refusal.

- [ ] **Step 5: Commit after Git exists**

```bash
git add src/tsfm/checkpoint.py tests/test_checkpoint.py tests/test_checkpoint_retention.py
git commit -m "feat: add exact resumable checkpoints"
```

### Task 8: Implement BF16 Trainer, Metrics, Probe, and Resume

**Files:** Create `src/tsfm/metrics.py`, `src/tsfm/trainer.py`, `scripts/train.py`, `tests/test_metrics.py`, `tests/test_trainer.py`, `tests/test_resume.py`, `tests/test_train_cli.py`.

**Interfaces:** `evaluate_model`, `bounded_batch_probe`, `run_training`, `TrainingReport`; CLI accepts model/training/manifest/output/resume arguments and no implicit paths.

- [ ] **Step 1: Write failing trainer tests**

```python
def test_accumulation_steps_once_and_uses_fp32_loss(tmp_path, tiny_run) -> None:
    result = run_training(**tiny_run, total_steps_override=2, device=torch.device("cpu"))
    assert result.optimizer_steps == 2
    assert result.micro_batches == 2*tiny_run["training_config"].gradient_accumulation_steps
    assert result.loss_dtype == "torch.float32"

def test_interrupted_resume_reproduces_next_loss(tmp_path, tiny_run) -> None:
    first = run_training(**tiny_run,total_steps_override=2,output_dir=tmp_path/"first",device=torch.device("cpu"))
    resumed = run_training(**tiny_run,total_steps_override=3,resume=first.checkpoint_path,output_dir=tmp_path/"resume",device=torch.device("cpu"))
    continuous = run_training(**tiny_run,total_steps_override=3,output_dir=tmp_path/"continuous",device=torch.device("cpu"))
    assert resumed.losses[-1] == pytest.approx(continuous.losses[-1],abs=1e-7)

def test_nonfinite_loss_writes_diagnostic_not_checkpoint(tmp_path, tiny_run) -> None:
    tiny_run["model"].forward = nonfinite_forward
    with pytest.raises(FloatingPointError,match="record_id"):
        run_training(**tiny_run,output_dir=tmp_path,device=torch.device("cpu"))
    assert (tmp_path/"diagnostic.pt").is_file() and not list(tmp_path.glob("step-*.pt"))
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_metrics.py tests/test_trainer.py tests/test_resume.py tests/test_train_cli.py -q`

Expected: collection FAIL because production trainer/CLI do not exist.

- [ ] **Step 3: Implement the bounded production loop**

```python
for step in range(state.global_step+1, config.total_steps+1):
    optimizer.zero_grad(set_to_none=True)
    for _ in range(config.gradient_accumulation_steps):
        started=time.perf_counter(); batch=next(iterator); data_wait += time.perf_counter()-started
        context=batch["context"].to(device,non_blocking=True); target=batch["target"].to(device,non_blocking=True)
        with autocast_context(device,config.precision): loss=model(context,labels=target).loss
        if loss is None or not torch.isfinite(loss): fail_with_diagnostic(step,batch,loss)
        (loss/config.gradient_accumulation_steps).backward()
        consumed_samples += context.shape[0]*world_size
    grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config.gradient_clip)
    if not torch.isfinite(grad_norm): fail_with_diagnostic(step,batch,loss,grad_norm)
    optimizer.step(); scheduler.step()
```

The DataLoader uses `CounterSampler(consumed_samples,rank,world_size)`, eight workers, pinned memory, and prefetch two. CUDA uses `torch.autocast("cuda",dtype=torch.bfloat16)`; CPU tests use `nullcontext`. `bounded_batch_probe` runs the exact configured batch/context forward/backward, reports peak allocation, and re-raises OOM without changing batch settings. Validation computes normalized MSE and de-normalized FP32 MSE/MAE on fixed heldout and temporal sample indices. Report JSONL records loss, LR, gradient norm, peak memory, sequences/s, patch tokens/s, data wait, requested/achieved source proportions, hardware/software, and config paths. Validate/checkpoint every 250 steps and keep best plus two latest.

- [ ] **Step 4: Verify GREEN and all local tests**

Run: `python -m pytest -q`

Expected: every test PASS without warnings; no local real-data or formal-checkpoint path is created.

- [ ] **Step 5: Commit after Git exists**

```bash
git add src/tsfm/metrics.py src/tsfm/trainer.py scripts/train.py tests
git commit -m "feat: add BF16-ready resumable trainer"
```

### Task 9: Add Server Bootstrap and Hardware Preflight

**Files:** Create `scripts/bootstrap_autodl.sh`, `scripts/server_preflight.py`, `tests/test_server_preflight.py`; modify `README.md`.

**Interfaces:** sourced bootstrap takes one persistent-root argument; preflight writes JSON and exits nonzero unless CUDA, BF16, SDPA backward, cgroup memory, and persistent disk gates pass.

- [ ] **Step 1: Write failing parsers/tests**

```python
def test_cgroup_v2_limit_parser_handles_max_and_bytes(tmp_path) -> None:
    path=tmp_path/"memory.max"; path.write_text("96636764160\n")
    assert read_cgroup_memory_limit(path) == 90*1024**3
    path.write_text("max\n"); assert read_cgroup_memory_limit(path) is None

def test_environment_report_contains_required_keys(monkeypatch,tmp_path) -> None:
    report=collect_environment_report(tmp_path)
    assert {"python","torch","cuda_runtime","driver","gpu","bf16_supported","cgroup_memory_bytes","disk"} <= report.keys()
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_server_preflight.py -q`

Expected: collection FAIL because preflight functions do not exist.

- [ ] **Step 3: Implement explicit environment setup/checks**

```bash
# scripts/bootstrap_autodl.sh
set -euo pipefail
test "$#" -eq 1 || { echo "usage: source scripts/bootstrap_autodl.sh PERSISTENT_ROOT"; return 2; }
export TSFM_PERSISTENT_ROOT="$(readlink -f "$1")"
export TSFM_DATA_ROOT="$TSFM_PERSISTENT_ROOT/tsfm-data"
export HF_HOME="$TSFM_PERSISTENT_ROOT/cache/huggingface"
export PIP_CACHE_DIR="$TSFM_PERSISTENT_ROOT/cache/pip"
export TORCH_HOME="$TSFM_PERSISTENT_ROOT/cache/torch"
export TSFM_CHECKPOINT_ROOT="$TSFM_PERSISTENT_ROOT/checkpoints"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TORCH_HOME" "$TSFM_CHECKPOINT_ROOT"
```

Preflight obtains driver/GPU with PyTorch APIs, cgroup limit from `/sys/fs/cgroup/memory.max` when present, disk usage from the explicit root, and runs BF16 SDPA causal forward/backward. It asserts RTX 5090 visibility, BF16 support, at least 80 GiB cgroup memory, and at least 20 GiB remaining after projected data/output use. It writes atomically beneath the user-supplied report directory.

- [ ] **Step 4: Verify local parsers, then server hardware**

Local: `python -m pytest tests/test_server_preflight.py -q`

Server: `python scripts/server_preflight.py --persistent-root "$TSFM_PERSISTENT_ROOT" --report-dir "$TSFM_PERSISTENT_ROOT/checkpoints/preflight"`

Expected server report: RTX 5090, PyTorch 2.8.0+cu128, CUDA 12.8, BF16 true, about 90 GiB cgroup memory, SDPA backward PASS, and at least 20 GiB free.

- [ ] **Step 5: Commit after Git exists**

```bash
git add scripts/bootstrap_autodl.sh scripts/server_preflight.py tests/test_server_preflight.py README.md
git commit -m "feat: add AutoDL bootstrap and preflight"
```

### Task 10: Convert the Bounded Corpus and Validate 26M on RTX 5090

**Files:** Create `docs/server-validation-runbook.md`; server-generated files remain outside repository under `TSFM_DATA_ROOT` and `TSFM_CHECKPOINT_ROOT`.

**Interfaces:** consumes completed Tasks 1-9; produces locked inventory, validated manifest/report, 26M metrics, best/two-latest checkpoints, and resume evidence.

- [ ] **Step 1: Run the local pre-upload gate**

Run: `python -m pytest -q && python scripts/audit_source_tree.py --root .`

Expected: all tests PASS; audit reports zero excluded artifacts and zero runtime `third_party` imports. On PowerShell run the two commands separately.

- [ ] **Step 2: Upload filtered source and bootstrap server**

```bash
rsync -av --exclude-from=.uploadignore ./ root@SERVER:/root/autodl-tmp/TimeSeriesFoundationModel/
cd /root/autodl-tmp/TimeSeriesFoundationModel
source scripts/bootstrap_autodl.sh /root/autodl-tmp
python -m pip install -e '.[dev,server]'
python -m pytest -q
python scripts/server_preflight.py --persistent-root "$TSFM_PERSISTENT_ROOT" --report-dir "$TSFM_CHECKPOINT_ROOT/preflight"
```

Expected: upload has no excluded data/weights; server tests and preflight PASS.

- [ ] **Step 3: Discover, lock, convert, and validate only the bounded subset**

```bash
python scripts/prepare_data.py discover --config configs/data/server_validation.json --persistent-root "$TSFM_PERSISTENT_ROOT" --data-root "$TSFM_DATA_ROOT" --execute-server
python scripts/prepare_data.py convert --config configs/data/server_validation.json --persistent-root "$TSFM_PERSISTENT_ROOT" --data-root "$TSFM_DATA_ROOT" --execute-server
python scripts/prepare_data.py validate --config configs/data/server_validation.json --persistent-root "$TSFM_PERSISTENT_ROOT" --data-root "$TSFM_DATA_ROOT" --execute-server
```

Expected: inventory/validation report proves two UTSD groups, two LOTSA groups, four domains, raw at most 20 GB, processed at most 20 GB, and at least 20 GiB disk free. If discovery cannot satisfy these simultaneously, conversion exits before downloading the selection and records the rejected inventory.

- [ ] **Step 4: Probe, overfit one batch, train 26M, and resume once**

```bash
python scripts/train.py probe --model-config configs/model/timer_26m.json --training-config configs/training/rtx5090_26m.json --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" --output-dir "$TSFM_CHECKPOINT_ROOT/26m"
python scripts/train.py overfit-one-batch --model-config configs/model/timer_26m.json --training-config configs/training/rtx5090_26m.json --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" --output-dir "$TSFM_CHECKPOINT_ROOT/26m"
python scripts/train.py run --model-config configs/model/timer_26m.json --training-config configs/training/rtx5090_26m.json --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" --output-dir "$TSFM_CHECKPOINT_ROOT/26m"
python scripts/train.py resume-check --checkpoint "$TSFM_CHECKPOINT_ROOT/26m/step-001000.pt" --steps 2 --model-config configs/model/timer_26m.json --training-config configs/training/rtx5090_26m.json --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" --output-dir "$TSFM_CHECKPOINT_ROOT/26m-resume-check"
```

Expected: exact batch probe fits without automatic changes; one-batch loss falls; 2,000 BF16 steps have finite loss/gradients and final training loss below initial; both validation views are finite; resume next loss matches recorded uninterrupted loss within documented BF16 tolerance; best plus two latest checkpoints remain.

- [ ] **Step 5: Record the 26M gate, then commit the runbook after Git exists**

```bash
git add docs/server-validation-runbook.md
git commit -m "docs: add server validation runbook"
```

The runbook names every report/checkpoint path above, records SHA256 of source manifest and JSON configurations, and states that server outputs are evidence artifacts, never upload inputs.

### Task 11: Validate the 95M Scale Point Without a Model Branch

**Files:** Modify only `docs/server-validation-runbook.md` with measured evidence; use existing `src/tsfm/model.py` unchanged.

**Interfaces:** same `scripts/train.py` interface and manifest as 26M; JSON changes only.

- [ ] **Step 1: Prove no scale-specific production branch exists**

Run: `rg -n "26m|95m|300m|hidden_size ==|num_hidden_layers ==" src scripts`

Expected: no production conditional keyed to model name/size; only reports/config path text may match.

- [ ] **Step 2: Run exact 95M memory probe**

```bash
python scripts/train.py probe --model-config configs/model/timer_95m.json --training-config configs/training/rtx5090_95m.json --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" --output-dir "$TSFM_CHECKPOINT_ROOT/95m"
```

Expected: configured micro-batch 32/context 30 forward/backward succeeds in BF16 with no silent adjustment.

- [ ] **Step 3: Run 500 steps**

```bash
python scripts/train.py run --model-config configs/model/timer_95m.json --training-config configs/training/rtx5090_95m.json --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" --output-dir "$TSFM_CHECKPOINT_ROOT/95m"
```

Expected: 500 finite steps, checkpoints at 250/500, both validation views finite, no OOM, and a resumable best checkpoint.

- [ ] **Step 4: Re-run all acceptance audits**

Run: `python -m pytest -q`

Run: `python scripts/audit_source_tree.py --root .`

Expected: all tests PASS without warnings; disk still has at least 20 GiB free; runtime import audit remains clean.

- [ ] **Step 5: Record evidence and commit after Git exists**

```bash
git add docs/server-validation-runbook.md
git commit -m "docs: record 95M validation gate"
```

### Task 12: Package the 307M 16-Card H20 DDP Preflight

**Files:** Modify `src/tsfm/trainer.py`, `scripts/train.py`; create `configs/training/h20_300m_preflight.json`, `tests/test_distributed.py`; modify `docs/server-validation-runbook.md`.

**Interfaces:** environment-driven `rank/world_size/local_rank`, NCCL DDP, counter partitioning, rank-zero atomic checkpoints/reports; model code and manifest format stay unchanged.

- [ ] **Step 1: Write failing distributed partition tests**

```python
def test_sixteen_ranks_partition_global_sample_indices() -> None:
    batches = [list(CounterSampler(0,rank,16).take(32)) for rank in range(16)]
    flattened = [value for batch in batches for value in batch]
    assert len(flattened) == len(set(flattened)) == 512
    assert sorted(flattened) == list(range(512))

def test_only_rank_zero_writes_checkpoint(monkeypatch,tmp_path) -> None:
    monkeypatch.setenv("RANK","1"); monkeypatch.setenv("WORLD_SIZE","16"); monkeypatch.setenv("LOCAL_RANK","1")
    context = distributed_context_from_environment()
    assert not context.is_main_process
    assert checkpoint_writer_for(context,tmp_path) is None
```

- [ ] **Step 2: Verify RED**

Run: `python -m pytest tests/test_distributed.py -q`

Expected: FAIL because distributed context and rank-zero writer do not exist.

- [ ] **Step 3: Implement DDP coordination and preflight JSON**

```python
@dataclass(frozen=True,slots=True)
class DistributedContext:
    rank:int; local_rank:int; world_size:int; is_main_process:bool

def distributed_context_from_environment():
    rank=int(os.getenv("RANK","0")); local=int(os.getenv("LOCAL_RANK","0")); world=int(os.getenv("WORLD_SIZE","1"))
    if world > 1:
        torch.cuda.set_device(local); torch.distributed.init_process_group("nccl")
    return DistributedContext(rank,local,world,rank==0)
```

Wrap the same `TimerModel` in `DistributedDataParallel` only when world size exceeds one; use `CounterSampler(consumed_samples,rank,world_size)`, all-reduce scalar metrics, and perform checkpoints/reports only on rank zero after barriers. The preflight JSON sets total steps 20, warmup 2, peak LR `2e-4`, micro-batch 8, accumulation 1, context 30, seed 2026, BF16, validation/checkpoint interval 20. It is labeled engineering-only and is not reused as a formal pre-training configuration.

- [ ] **Step 4: Verify locally, then execute on the allocated H20 node**

Local: `python -m pytest tests/test_distributed.py -q`

Server:

```bash
torchrun --standalone --nproc_per_node=16 scripts/train.py run --model-config configs/model/timer_300m.json --training-config configs/training/h20_300m_preflight.json --manifest "$TSFM_DATA_ROOT/processed/manifest.jsonl" --output-dir "$TSFM_CHECKPOINT_ROOT/h20-300m-preflight"
```

Expected: exactly 16 NCCL ranks; 20 finite BF16 steps; no duplicated sample indices; one rank-zero checkpoint with 307,146,240 model parameters; valid resume metadata; per-rank and aggregate throughput/memory report; no model-size branch.

- [ ] **Step 5: Run final gate and commit after Git exists**

```bash
python -m pytest -q
python scripts/audit_source_tree.py --root .
git add src/tsfm/trainer.py scripts/train.py configs/training/h20_300m_preflight.json tests/test_distributed.py docs/server-validation-runbook.md
git commit -m "feat: package 307M H20 DDP preflight"
```

Expected: local suite/audit PASS; runbook records 26M, 95M, and H20 preflight evidence and explicitly withholds approval for formal 307M training until all three gates pass.

## Final Acceptance Checklist

- [ ] All local tests pass without warnings and use only synthetic `tmp_path` fixtures.
- [ ] Source audit finds no local datasets, weights, generated outputs, unsafe paths, or runtime `third_party` imports.
- [ ] Manifest and every segment checksum validate; train, heldout, and temporal windows respect boundaries.
- [ ] Requested and achieved source proportions are recorded and approximately equal by source.
- [ ] 26M completes 2,000 finite steps and exact resume check; 95M completes 500 finite steps through the same class.
- [ ] Checkpoint sets contain best plus two latest and include optimizer, scheduler, RNG, sampler, configs, manifest checksum, and environment.
- [ ] Persistent disk retains at least 20 GiB; raw and processed validation corpora each remain at or below 20 GB.
- [ ] 307M completes a 20-step 16-card H20 DDP preflight before any formal allocation is approved.

## Scope Boundary

This plan does not include formal 307M pre-training, 1B/3B scaling, Timer-XL cross-variable attention, downstream fine-tuning, anomaly detection, imputation, or full-corpus download. Those decisions require measured evidence from this plan.
