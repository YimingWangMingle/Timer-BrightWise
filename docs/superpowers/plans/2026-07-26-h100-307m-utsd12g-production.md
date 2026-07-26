# H100 307M UTSD-12G Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one offline, resumable command that verifies a four-H100 node, converts a pinned local UTSD-12G Arrow snapshot, gates the exact 307M model, and starts three deterministic production coverage cycles.

**Architecture:** Keep the existing Timer model and S3 storage contract. Add isolated modules for external-artifact verification, local Arrow input, finite window indexing, resolved training plans, distributed lifecycle, and phase-aware orchestration; the top-level pipeline composes those modules and binds every phase to immutable digests. Preserve the existing online bounded validation path and small-model tests while selecting the new finite/offline path only from production configuration.

**Tech Stack:** CPython 3.11, PyTorch 2.7.1 CUDA 12.6, NCCL/DDP, Hugging Face `datasets`/Arrow, NumPy, psutil, pytest, Bash, JSON, SHA-256.

## Global Constraints

- Windows remains code-only: never download or materialize UTSD, LOTSA, model weights, S3 shards, or production checkpoints locally.
- The target is one Ubuntu 22.04.3 node with exactly four NVIDIA H100 80GB GPUs, driver 575.57.08, and persistent root `/root/work`.
- Use CPython `>=3.11,<3.12` and PyTorch `2.7.1` from the CUDA 12.6 wheel index; no custom CUDA extension is introduced.
- UTSD-12G is pinned to revision `7326ff5f4578da73d843fd675d760c6c6054017f`, 82 selected files, and 3,892,126,910 bytes.
- The production model must remain exactly 307,146,240 parameters and use `configs/model/timer_300m.json` without a size-specific implementation branch.
- Production uses four-rank DDP, BF16 autocast, global batch 4,096, three finite coverage cycles, peak LR `5e-5`, 2 percent warmup, cosine floor `5e-6`, AdamW betas `(0.9, 0.95)`, weight decay `0.1`, clip `1.0`, and seed `2026`.
- The successful gate is 20 four-rank optimizer steps plus a two-step resume and is reusable only while hardware, package, source, processed-data, model, and resolved-plan digests match.
- LOTSA code remains supported but LOTSA is not selected, downloaded, converted, or trained in this production path.
- The runtime trees `D:\学习\TimeSeriesFoundationModel` and `D:\学习\TimeSeriesFoundationModel-ServerUpload` must be byte-identical after every completed task; only the ServerUpload tree is committed.
- Use `apply_patch` for manual edits, keep data/runtime artifacts ignored, and run local tests only with synthetic temporary fixtures.

## File Structure

| File | Responsibility |
| --- | --- |
| `src/tsfm/artifacts.py` | Canonical SHA-256 manifest creation and strict verification for transferred bundles. |
| `scripts/download_utsd12g_snapshot.py` | Networked-server download of the pinned partial Hugging Face snapshot. |
| `configs/data/utsd12g_production.json` | Offline-only UTSD-12G source policy and conversion limits. |
| `src/tsfm/s3/adapters.py` | Online UTSD/LOTSA adapters plus exact local Arrow-file loading. |
| `scripts/prepare_data.py` | Discover, convert, and validate online validation data or the offline production snapshot. |
| `src/tsfm/conversion_state.py` | Incomplete conversion workspace, atomic completion marker, and digest validation. |
| `src/tsfm/s3/finite_sampling.py` | Prefix-sum finite window index and O(1)-memory affine epoch permutation. |
| `src/tsfm/s3/dataset.py` | Existing stochastic dataset plus finite canonical-window dataset. |
| `src/tsfm/runtime.py` | Stable builders for stochastic validation and finite production datasets. |
| `src/tsfm/production_plan.py` | Production template parsing and immutable resolved-plan calculation. |
| `configs/training/h100_307m_production.json` | Stable production hyperparameters before batch resolution. |
| `configs/training/h100_307m_preflight.json` | Fixed 20-step gate and two-step resume settings. |
| `src/tsfm/distributed.py` | DDP setup/teardown, model unwrapping, reductions, object gathering, and `no_sync` support. |
| `src/tsfm/checkpoint.py` | Portable unwrapped weights, per-rank RNG, sampler/plan state, retention, and export. |
| `src/tsfm/trainer.py` | Finite-plan training loop, accumulation, aggregate telemetry, failure diagnostics, and resume. |
| `src/tsfm/validation.py` | Rank-partitioned validation and all-reduced normalized/raw metrics. |
| `src/tsfm/validated_training.py` | Interval execution and normalized-MSE best-checkpoint selection. |
| `src/tsfm/preflight.py` | Four-H100 hardware, memory, disk, BF16, and SDPA checks. |
| `scripts/nccl_probe.py` | Four-rank NCCL collective and disjoint-index server probe. |
| `scripts/batch_probe.py` | Isolated real-data micro-batch candidate probe. |
| `src/tsfm/pipeline.py` | Phase records, digest binding, fail-closed restart decisions, and command assembly. |
| `scripts/h100_pipeline.py` | One foreground orchestrator used under `nohup`. |
| `scripts/launch_h100_307m.sh` | One-command shell entry point with persistent paths and logs. |
| `requirements/h100-py311-cu126.txt` | Exact direct dependency inputs for the external wheelhouse. |
| `scripts/build_h100_offline_bundle.sh` | Networked-server runtime/wheelhouse packager and manifest writer. |
| `scripts/install_h100_offline.sh` | H100-side manifest verification and no-index venv installation. |
| `docs/h100-307m-runbook.md` | Operator commands, expected reports, failure handling, and resume procedure. |

---

### Task 1: Strict External Artifact Manifests and Pinned Snapshot Builder

**Files:**
- Create: `src/tsfm/artifacts.py`
- Create: `scripts/download_utsd12g_snapshot.py`
- Create: `tests/test_artifact_manifest.py`
- Modify: `.gitignore`
- Modify: `.uploadignore`

**Interfaces:**
- Produces: `ArtifactEntry(path: str, size: int, sha256: str)`.
- Produces: `write_artifact_manifest(root: Path, destination: Path) -> str` returning the canonical manifest digest.
- Produces: `verify_artifact_manifest(root: Path, manifest: Path, *, expected_files: int | None = None, expected_bytes: int | None = None) -> str`.
- Produces: snapshot-builder CLI options `--destination`, `--manifest`, and optional injected `--endpoint`; it always selects the fixed revision and `UTSD-12G/*`.

- [ ] **Step 1: Write manifest and snapshot-policy failures**

```python
# tests/test_artifact_manifest.py
from pathlib import Path

import pytest

from tsfm.artifacts import verify_artifact_manifest, write_artifact_manifest


def test_manifest_round_trip_rejects_modified_missing_and_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "UTSD-12G"
    root.mkdir()
    (root / "a.arrow").write_bytes(b"arrow-a")
    (root / "dataset_info.json").write_text("{}\n", encoding="utf-8")
    manifest = tmp_path / "UTSD-12G.sha256.json"
    digest = write_artifact_manifest(root, manifest)
    assert verify_artifact_manifest(root, manifest, expected_files=2, expected_bytes=10) == digest

    (root / "a.arrow").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch: a.arrow"):
        verify_artifact_manifest(root, manifest)

    (root / "a.arrow").write_bytes(b"arrow-a")
    (root / "extra.arrow").write_bytes(b"x")
    with pytest.raises(ValueError, match="unlisted file: extra.arrow"):
        verify_artifact_manifest(root, manifest)


def test_snapshot_builder_pins_revision_and_only_utsd12g(monkeypatch, tmp_path: Path) -> None:
    import scripts.download_utsd12g_snapshot as command

    calls = []
    monkeypatch.setattr(command, "snapshot_download", lambda **kwargs: calls.append(kwargs))
    command.download_snapshot(tmp_path / "raw")
    assert calls == [{
        "repo_id": "thuml/UTSD",
        "repo_type": "dataset",
        "revision": "7326ff5f4578da73d843fd675d760c6c6054017f",
        "allow_patterns": ["UTSD-12G/*"],
        "local_dir": tmp_path / "raw",
    }]
```

- [ ] **Step 2: Run the focused tests and confirm missing modules**

Run: `python -m pytest -q tests/test_artifact_manifest.py`

Expected: FAIL during collection because `tsfm.artifacts` and `scripts.download_utsd12g_snapshot` do not exist.

- [ ] **Step 3: Implement canonical manifests and the fixed downloader**

```python
# src/tsfm/artifacts.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ArtifactEntry:
    path: str
    size: int
    sha256: str


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _entries(root: Path) -> list[ArtifactEntry]:
    if not root.is_dir():
        raise ValueError(f"artifact root is not a directory: {root}")
    entries = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact symlink is forbidden: {path.relative_to(root).as_posix()}")
        if path.is_file():
            entries.append(ArtifactEntry(path.relative_to(root).as_posix(), path.stat().st_size, _digest(path)))
    return entries


def _payload(entries: list[ArtifactEntry]) -> bytes:
    return (json.dumps({"format_version": 1, "files": [asdict(item) for item in entries]}, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def write_artifact_manifest(root: Path, destination: Path) -> str:
    payload = _payload(_entries(root.resolve()))
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, destination)
    return hashlib.sha256(payload).hexdigest()


def verify_artifact_manifest(root: Path, manifest: Path, *, expected_files: int | None = None, expected_bytes: int | None = None) -> str:
    payload = manifest.read_bytes()
    document = json.loads(payload)
    declared = [ArtifactEntry(**item) for item in document["files"]]
    actual = _entries(root.resolve())
    by_path = {item.path: item for item in actual}
    declared_paths = {item.path for item in declared}
    for item in declared:
        found = by_path.get(item.path)
        if found is None:
            raise ValueError(f"missing file: {item.path}")
        if found.size != item.size:
            raise ValueError(f"size mismatch: {item.path}")
        if found.sha256 != item.sha256:
            raise ValueError(f"checksum mismatch: {item.path}")
    extras = sorted(set(by_path) - declared_paths)
    if extras:
        raise ValueError(f"unlisted file: {extras[0]}")
    if expected_files is not None and len(actual) != expected_files:
        raise ValueError(f"expected {expected_files} files, found {len(actual)}")
    total = sum(item.size for item in actual)
    if expected_bytes is not None and total != expected_bytes:
        raise ValueError(f"expected {expected_bytes} bytes, found {total}")
    return hashlib.sha256(payload).hexdigest()
```

Implement `download_snapshot(destination: Path) -> str` in the script with the exact call asserted above, require the 82-file/3,892,126,910-byte verification, and write the manifest outside the snapshot directory. Add `runtime-bundles/`, `wheelhouse/`, `*.sha256.json`, and `*.tar.zst` to both ignore files.

- [ ] **Step 4: Run artifact tests**

Run: `python -m pytest -q tests/test_artifact_manifest.py`

Expected: `2 passed`.

- [ ] **Step 5: Commit and synchronize**

```bash
git add .gitignore .uploadignore src/tsfm/artifacts.py scripts/download_utsd12g_snapshot.py tests/test_artifact_manifest.py
git commit -m "feat: add strict offline artifact manifests"
```

Copy these exact files to the formal tree and confirm SHA-256 equality before continuing.

---

### Task 2: Offline UTSD-12G Arrow Policy and Adapter

**Files:**
- Create: `configs/data/utsd12g_production.json`
- Create: `tests/test_offline_utsd_snapshot.py`
- Modify: `src/tsfm/s3/adapters.py:16-140`
- Modify: `scripts/prepare_data.py:63-370`

**Interfaces:**
- Extends: `DatasetSpec` with `local_files: tuple[str, ...] = ()` while preserving `data_files` for remote exact files.
- Produces: `UTSDAdapter._rows()` local mode using `load_dataset("arrow", data_files={split: files}, split=split, streaming=True)`.
- Produces: production policy fields `mode="local_arrow"`, `local_path="raw/utsd/UTSD-12G"`, `expected_files=82`, `expected_bytes=3892126910`, and `snapshot_revision`.
- Extends: `prepare_data.py` with `--source-root`; local discovery never imports a Hub client or reads `HF_ENDPOINT`.

- [ ] **Step 1: Write local-mode tests**

```python
# tests/test_offline_utsd_snapshot.py
def test_local_utsd_adapter_uses_exact_arrow_files_without_hub(tmp_path) -> None:
    calls = []
    files = (tmp_path / "a.arrow", tmp_path / "b.arrow")
    for item in files:
        item.write_bytes(b"arrow")

    def loader(*args, **kwargs):
        calls.append((args, kwargs))
        return [{"item_id": "s", "freq": "h", "target": [1.0, 2.0, 3.0]}]

    spec = DatasetSpec("thuml/UTSD", "UTSD-12G", "utsd", "UTSD-12G", file_format="arrow", local_files=tuple(str(item) for item in files))
    rows = list(UTSDAdapter(spec, tmp_path / "cache", loader=loader).iter_series())
    assert rows[0].series_id == "s"
    assert calls == [(('arrow',), {"data_files": {"train": [str(item) for item in files]}, "split": "train", "streaming": True, "cache_dir": tmp_path / "cache"})]


def test_production_policy_is_offline_utsd12g_only() -> None:
    policy = json.loads((ROOT / "configs/data/utsd12g_production.json").read_text())
    assert [(item["source_id"], item["configurations"]) for item in policy["repositories"]] == [("utsd", ["UTSD-12G"])]
    source = policy["repositories"][0]
    assert source["mode"] == "local_arrow"
    assert source["expected_files"] == 82
    assert source["expected_bytes"] == 3_892_126_910
    assert "lotsa" not in json.dumps(policy).lower()
```

- [ ] **Step 2: Verify the new behavior fails**

Run: `python -m pytest -q tests/test_offline_utsd_snapshot.py`

Expected: FAIL because `DatasetSpec.local_files` and the production policy do not exist.

- [ ] **Step 3: Implement local Arrow loading and offline inventory**

Add a shared `_arrow_rows(files)` path in `_HuggingFaceAdapter` that validates every file is under the resolved source root and is named `*.arrow`. Keep current online UTSD and remote exact LOTSA behavior unchanged. In `build_inventory`, branch on `mode`:

```python
if entry.get("mode") == "local_arrow":
    snapshot = (source_root / entry["local_path"]).resolve()
    files = tuple(sorted(snapshot.glob("data-*.arrow")))
    verify_artifact_manifest(snapshot, snapshot.parent / "UTSD-12G.sha256.json", expected_files=entry["expected_files"], expected_bytes=entry["expected_bytes"])
    projected = sum(path.stat().st_size for path in files)
    inventory_entry["local_files"] = [str(path) for path in files]
else:
    # Existing Hub builder or pinned remote Arrow branch.
```

The production policy sets source and processed limits to 20 GB, minimum segment length 2,976, minimum variance `1e-8`, and shard limit 2,000,000,000 bytes. Local `discover`, `convert`, and `validate` accept no missing manifest and never require `HF_HOME`.

- [ ] **Step 4: Run new and regression data tests**

Run: `python -m pytest -q tests/test_offline_utsd_snapshot.py tests/test_exact_arrow_mirror.py tests/test_prepare_data_cli.py tests/test_prepare_data_limits.py`

Expected: all tests pass; online LOTSA exact-file tests remain unchanged.

- [ ] **Step 5: Commit and synchronize**

```bash
git add configs/data/utsd12g_production.json src/tsfm/s3/adapters.py scripts/prepare_data.py tests/test_offline_utsd_snapshot.py
git commit -m "feat: read UTSD-12G from an offline Arrow snapshot"
```

---

### Task 3: Atomic Conversion Publication and Completion Binding

**Files:**
- Create: `src/tsfm/conversion_state.py`
- Create: `tests/test_conversion_state.py`
- Modify: `scripts/prepare_data.py:307-369`
- Modify: `tests/test_prepare_data_cli.py`

**Interfaces:**
- Produces: `ConversionBinding(source_manifest: str, processed_manifest: str, policy_digest: str, records: int, processed_bytes: int)`.
- Produces: `begin_conversion(final_root: Path) -> Path` returning a unique sibling `.utsd-12g.incomplete-<uuid>` directory.
- Produces: `publish_conversion(staging: Path, final_root: Path, binding: ConversionBinding) -> Path` writing `conversion-complete.json` before atomic directory rename.
- Produces: `validate_completed_conversion(final_root: Path, *, source_manifest: str, policy_digest: str) -> ConversionBinding`.

- [ ] **Step 1: Write fail-closed state tests**

```python
def test_conversion_is_accepted_only_after_atomic_completion(tmp_path: Path) -> None:
    final = tmp_path / "processed" / "utsd-12g"
    staging = begin_conversion(final)
    (staging / "manifest.jsonl").write_text("record\n")
    with pytest.raises(ValueError, match="conversion completion marker is missing"):
        validate_completed_conversion(staging, source_manifest="a" * 64, policy_digest="b" * 64)

    binding = ConversionBinding("a" * 64, "c" * 64, "b" * 64, 1, 7)
    publish_conversion(staging, final, binding)
    assert validate_completed_conversion(final, source_manifest="a" * 64, policy_digest="b" * 64) == binding


def test_publish_refuses_nonempty_destination(tmp_path: Path) -> None:
    final = tmp_path / "processed"
    final.mkdir()
    (final / "unknown").write_text("do not overwrite")
    staging = begin_conversion(final)
    with pytest.raises(FileExistsError, match="nonempty production destination"):
        publish_conversion(staging, final, ConversionBinding("a"*64, "b"*64, "c"*64, 0, 0))
```

- [ ] **Step 2: Confirm tests fail before implementation**

Run: `python -m pytest -q tests/test_conversion_state.py`

Expected: FAIL because `tsfm.conversion_state` does not exist.

- [ ] **Step 3: Implement conversion state and integrate it**

Use canonical JSON plus atomic file replacement for the marker, `uuid.uuid4().hex` for staging names, and `os.replace(staging, final_root)` only when the final root is absent or empty. `prepare_data.py convert` packs into staging, writes/loads the processed manifest, computes its checksum and policy-file SHA-256, writes the conversion report, publishes, and prints the final root. `validate` requires the binding to match the source artifact manifest and policy digest.

- [ ] **Step 4: Run conversion and CLI regressions**

Run: `python -m pytest -q tests/test_conversion_state.py tests/test_prepare_data_cli.py tests/test_s3_storage.py`

Expected: all pass and no test writes outside `tmp_path`.

- [ ] **Step 5: Commit and synchronize**

```bash
git add src/tsfm/conversion_state.py scripts/prepare_data.py tests/test_conversion_state.py tests/test_prepare_data_cli.py
git commit -m "feat: publish S3 conversions atomically"
```

---

### Task 4: Finite Window Index and Deterministic Affine Coverage Sampler

**Files:**
- Create: `src/tsfm/s3/finite_sampling.py`
- Create: `tests/test_finite_sampling.py`
- Modify: `src/tsfm/s3/dataset.py:14-62`
- Modify: `src/tsfm/runtime.py:11-31`

**Interfaces:**
- Produces: `FiniteWindowIndex(records, regions, sample_length)` with `window_count`, `sample(canonical_index) -> SampleKey`.
- Produces: `AffineCoverageSampler(window_count, cycles, seed, start_position=0, rank=0, world_size=1, global_batch_size=1)` with `total_real`, `total_padded`, `total_positions`, and deterministic iteration.
- Produces: `build_finite_s3_dataset(manifest_path, split, seed, patch_length, context_patches) -> tuple[S3WindowDataset, str, int]`.
- Preserves: `HierarchicalIndex` and `CounterSampler` behavior for existing engineering configurations.

- [ ] **Step 1: Write finite mapping, permutation, rank, and resume tests**

```python
def test_finite_index_maps_every_stride_one_window_once() -> None:
    records = [_record("a", 5), _record("b", 4)]
    regions = [SplitRegion(0, "train", 0, 5), SplitRegion(1, "train", 0, 4)]
    index = FiniteWindowIndex(records, regions, sample_length=3)
    assert index.window_count == 5
    assert [(index.sample(i).region_index, index.sample(i).window_start) for i in range(5)] == [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]


def test_affine_sampler_is_bijective_partitioned_and_resumable() -> None:
    full = list(AffineCoverageSampler(17, cycles=3, seed=2026, global_batch_size=8))
    assert len(full) == 56
    assert set(full[:17]) == set(range(17))
    rank_parts = [list(AffineCoverageSampler(17, 3, 2026, rank=rank, world_size=4, global_batch_size=8)) for rank in range(4)]
    interleaved = [value for group in zip(*rank_parts) for value in group]
    assert interleaved == full
    assert list(AffineCoverageSampler(17, 3, 2026, start_position=24, global_batch_size=8)) == full[24:]
```

- [ ] **Step 2: Verify finite APIs are absent**

Run: `python -m pytest -q tests/test_finite_sampling.py`

Expected: FAIL because `FiniteWindowIndex` and `AffineCoverageSampler` are undefined.

- [ ] **Step 3: Implement prefix mapping and O(1) permutation**

`FiniteWindowIndex` stores eligible region indices and cumulative counts `max(0, stop-start-sample_length+1)` and uses `bisect_right` to locate a canonical index. For cycle `e`, derive `a` and `b` from SHA-256 of `f"{seed}:{e}"`; increment `a` until `gcd(a, N)==1`; map `i` to `(a*i+b) % N`. Pad only after `cycles*N` to `ceil(cycles*N/global_batch_size)*global_batch_size`, repeating permutation positions from cycle zero and exposing padding through sampler metadata.

Update `S3WindowDataset` to accept either index object without changing the item dictionary. Add the finite runtime builder while keeping `build_s3_dataset` untouched.

- [ ] **Step 4: Run sampling and runtime regressions**

Run: `python -m pytest -q tests/test_finite_sampling.py tests/test_s3_splits_sampling_dataset.py tests/test_runtime_bundle.py`

Expected: all pass, including the existing stochastic balancing test.

- [ ] **Step 5: Commit and synchronize**

```bash
git add src/tsfm/s3/finite_sampling.py src/tsfm/s3/dataset.py src/tsfm/runtime.py tests/test_finite_sampling.py
git commit -m "feat: add finite deterministic S3 coverage"
```

---

### Task 5: Immutable Production Plan Resolution

**Files:**
- Create: `src/tsfm/production_plan.py`
- Create: `configs/training/h100_307m_production.json`
- Create: `configs/training/h100_307m_preflight.json`
- Create: `tests/test_production_plan.py`
- Modify: `src/tsfm/train_config.py:29-52`
- Modify: `tests/test_production_optim.py`

**Interfaces:**
- Produces: `ProductionTemplate.from_json(path)` with the approved stable values.
- Produces: `ResolvedTrainingPlan.resolve(template, *, window_count, micro_batch_size, world_size, digests) -> ResolvedTrainingPlan`.
- Produces: `ResolvedTrainingPlan.training_config() -> TrainingConfig`.
- Changes: `TrainingConfig` requires positive validation/checkpoint intervals but no longer requires either to divide total steps; final-step save remains mandatory.

- [ ] **Step 1: Write exact resolution tests**

```python
def test_production_template_resolves_padding_steps_and_warmup() -> None:
    template = ProductionTemplate.from_json(ROOT / "configs/training/h100_307m_production.json")
    plan = ResolvedTrainingPlan.resolve(
        template,
        window_count=10_001,
        micro_batch_size=256,
        world_size=4,
        digests={"model": "a"*64, "source": "b"*64, "manifest": "c"*64, "packages": "d"*64},
    )
    assert plan.global_batch_size == 4096
    assert plan.gradient_accumulation_steps == 4
    assert plan.total_real_samples == 30_003
    assert plan.total_padded_samples == 2_765
    assert plan.total_steps == 8
    assert plan.warmup_steps == 1
    assert plan.minimum_lr_ratio == pytest.approx(0.1)


def test_training_intervals_need_not_divide_total_steps() -> None:
    config = TrainingConfig(total_steps=2001, warmup_steps=40, peak_lr=5e-5, validation_interval=2000, checkpoint_interval=2000)
    assert config.total_steps == 2001
```

- [ ] **Step 2: Run and observe config failures**

Run: `python -m pytest -q tests/test_production_plan.py tests/test_production_optim.py`

Expected: FAIL because production-plan types/configs do not exist and the old interval validation rejects 2,001 steps.

- [ ] **Step 3: Implement templates, digest binding, and resolution**

The production JSON contains `coverage_cycles: 3`, `global_batch_size: 4096`, candidate micro-batches `[512,256,128,64]`, LR `5e-5`, floor ratio `0.1`, warmup ratio `0.02`, intervals `2000`, logging interval `10`, loader workers `16`, prefetch `2`, BF16, and the approved optimizer fields. Resolution uses `ceil(total_real/global_batch)`, `max(1, ceil(total_steps*0.02))`, validates every digest as 64 lowercase hex characters, and serializes canonical sorted JSON atomically.

The preflight JSON contains 20 steps, checkpoint/validation at step 20, the same optimizer family, and `resume_steps: 2`; micro-batch and accumulation are injected from the resolved candidate rather than duplicated.

- [ ] **Step 4: Run production plan and optimizer tests**

Run: `python -m pytest -q tests/test_production_plan.py tests/test_production_optim.py`

Expected: all pass, and old RTX 5090 configuration assertions remain unchanged.

- [ ] **Step 5: Commit and synchronize**

```bash
git add src/tsfm/production_plan.py src/tsfm/train_config.py configs/training/h100_307m_production.json configs/training/h100_307m_preflight.json tests/test_production_plan.py tests/test_production_optim.py
git commit -m "feat: resolve immutable 307M production plans"
```

---

### Task 6: Portable Multi-Rank Checkpoints and Efficient DDP Training

**Files:**
- Modify: `src/tsfm/distributed.py:10-58`
- Modify: `src/tsfm/checkpoint.py:16-143`
- Modify: `src/tsfm/optim.py:25-33`
- Modify: `src/tsfm/trainer.py:34-112`
- Create: `tests/test_ddp_production_contract.py`
- Modify: `tests/test_full_checkpoint.py`
- Modify: `tests/test_production_trainer.py`

**Interfaces:**
- Produces: `unwrap_model(model) -> torch.nn.Module`, `gather_objects(value, context) -> list[object] | None`, and `destroy_distributed(context) -> None`.
- Produces: DDP wrapping with `gradient_as_bucket_view=True` and a 10-minute process-group timeout.
- Changes: `save_training_checkpoint(..., rank_rng_states: dict[int, dict[str, object]], resolved_plan: dict[str, object])` always saves unwrapped keys.
- Changes: `load_training_checkpoint(..., rank: int, expected_world_size: int)` restores that rank's RNG and rejects world-size/plan mismatches.
- Changes: `run_training` uses `model.no_sync()` on all non-final accumulation micro-batches and records real/padded sampler positions.

- [ ] **Step 1: Write DDP contract tests without requiring GPUs**

```python
def test_checkpoint_uses_unwrapped_keys_and_rank_specific_rng(tmp_path, monkeypatch) -> None:
    model = FakeDDP(torch.nn.Linear(2, 2))
    states = {0: _rng_fixture(1), 1: _rng_fixture(2), 2: _rng_fixture(3), 3: _rng_fixture(4)}
    path = tmp_path / "step.pt"
    save_training_checkpoint(path, model=model, optimizer=_optimizer(model), scheduler=_scheduler(), state=_state(), config_snapshots=_snapshots(), manifest_checksum="a"*64, environment={}, rank_rng_states=states, resolved_plan={"world_size": 4})
    payload = torch.load(path, weights_only=True)
    assert all(not key.startswith("module.") for key in payload["model"])
    assert set(payload["rank_rng_states"]) == {0, 1, 2, 3}


def test_accumulation_uses_no_sync_except_final_microbatch(tmp_path) -> None:
    model = CountingNoSyncModel()
    run_training(model=model, dataset=Windows(), training_config=_config(accumulation=4), output_dir=tmp_path, manifest_checksum="a"*64, config_snapshots={}, device=torch.device("cpu"))
    assert model.no_sync_entries == 3
```

- [ ] **Step 2: Confirm current checkpoint/trainer fails the contract**

Run: `python -m pytest -q tests/test_ddp_production_contract.py tests/test_full_checkpoint.py tests/test_production_trainer.py`

Expected: FAIL because wrapper keys, per-rank RNG, and `no_sync` are unsupported.

- [ ] **Step 3: Implement distributed lifecycle, per-rank state, and fused optimizer**

Use `contextlib.nullcontext()` for the final micro-batch and non-DDP models; use `model.no_sync()` only when `world_size>1`. Gather rank RNG with `torch.distributed.gather_object` before rank-zero save. Create AdamW with `fused=True` on CUDA and retry without `fused` only for `RuntimeError`/`TypeError` from unsupported fused construction. Set `persistent_workers=config.num_workers>0`, keep pinned/non-blocking transfers, aggregate per-rank source/sample/memory reports, and always destroy the process group in the train CLI `finally` block.

- [ ] **Step 4: Run checkpoint, trainer, rank-zero, and distributed tests**

Run: `python -m pytest -q tests/test_ddp_production_contract.py tests/test_full_checkpoint.py tests/test_production_trainer.py tests/test_rank_zero_artifacts.py tests/test_distributed.py`

Expected: all pass. Existing single-rank resume reproduces its next loss; multi-rank tests validate metadata/state without initializing NCCL locally.

- [ ] **Step 5: Commit and synchronize**

```bash
git add src/tsfm/distributed.py src/tsfm/checkpoint.py src/tsfm/optim.py src/tsfm/trainer.py tests/test_ddp_production_contract.py tests/test_full_checkpoint.py tests/test_production_trainer.py
git commit -m "feat: harden DDP training and portable resume"
```

---

### Task 7: Partitioned Validation and Normalized-MSE Selection

**Files:**
- Modify: `src/tsfm/validation.py:12-57`
- Modify: `src/tsfm/validated_training.py:22-62`
- Modify: `src/tsfm/checkpoint.py:112-143`
- Modify: `tests/test_validation.py`
- Modify: `tests/test_validated_training.py`

**Interfaces:**
- Changes: `evaluate_model(..., rank=0, world_size=1) -> dict[str, float]` evaluates sample positions `rank, rank+world_size, ...` and all-reduces metric sums/counts.
- Produces: `primary_validation_metric(views) -> float`, the arithmetic mean of normalized MSE for `val_heldout` and `val_temporal`.
- Changes: latest checkpoint retention is independent from best-checkpoint promotion; best is promoted only after validation.

- [ ] **Step 1: Write partition and metric-selection tests**

```python
def test_validation_partitions_positions_and_reduces_weighted_totals(monkeypatch) -> None:
    seen = []
    dataset = RecordingWindows(seen)
    evaluate_model(IdentityModel(), dataset, device=torch.device("cpu"), batch_size=2, batches=2, precision="fp32", rank=1, world_size=4)
    assert seen == [1, 5, 9, 13]


def test_best_checkpoint_uses_normalized_not_raw_mse() -> None:
    views = {
        "val_heldout": {"normalized_mse": 0.4, "mse": 1_000_000.0, "mae": 1.0},
        "val_temporal": {"normalized_mse": 0.6, "mse": 1.0, "mae": 1.0},
    }
    assert primary_validation_metric(views) == pytest.approx(0.5)
```

- [ ] **Step 2: Run tests and confirm duplicate validation behavior**

Run: `python -m pytest -q tests/test_validation.py tests/test_validated_training.py`

Expected: FAIL because rank/world-size partition and normalized primary selection are absent.

- [ ] **Step 3: Implement partitioned loader and all-reduced numerators**

Use the existing counter sampler with `start_sample=0, rank, world_size` for bounded validation. Maintain FP64 numerators for `normalized_mse`, `mse`, and `mae`, plus integer sample count; all-reduce all four values before division. Run promotion on rank zero after a barrier and copy the already-atomic latest checkpoint to `best.pt` only when the normalized primary metric improves.

- [ ] **Step 4: Run validation, logging, and checkpoint retention tests**

Run: `python -m pytest -q tests/test_validation.py tests/test_validated_training.py tests/test_metrics_shapes.py tests/test_full_checkpoint.py`

Expected: all pass; raw metrics remain reported but cannot choose `best.pt`.

- [ ] **Step 5: Commit and synchronize**

```bash
git add src/tsfm/validation.py src/tsfm/validated_training.py src/tsfm/checkpoint.py tests/test_validation.py tests/test_validated_training.py tests/test_full_checkpoint.py
git commit -m "feat: distribute validation and select normalized best"
```

---

### Task 8: Four-H100, NCCL, and Isolated Batch Probes

**Files:**
- Modify: `src/tsfm/preflight.py:14-107`
- Modify: `scripts/server_preflight.py:13-20`
- Create: `scripts/nccl_probe.py`
- Create: `scripts/batch_probe.py`
- Create: `tests/test_h100_preflight.py`
- Modify: `tests/test_server_preflight.py`

**Interfaces:**
- Changes: `run_server_preflight(..., expected_gpu, expected_gpu_count, minimum_gpu_memory_bytes)`.
- Produces: per-GPU name, total memory, BF16 result, UUID, driver/runtime versions, CPU memory limit, persistent disk, and SDPA result.
- Produces: `scripts/nccl_probe.py --expected-world-size 4 --report-dir PATH` under `torchrun`.
- Produces: `scripts/batch_probe.py` returning one rank-zero JSON report and nonzero exit for OOM/non-finite/over-80-percent candidates.

- [ ] **Step 1: Write synthetic hardware and CLI tests**

```python
def test_h100_gate_requires_four_matching_80gb_devices(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    report = collect_environment_report(tmp_path)
    errors = validate_server_report(report, expected_gpu="H100", expected_gpu_count=4, minimum_gpu_memory_bytes=80 * 1024**3)
    assert "expected 4 GPUs, found 3" in errors


def test_nccl_probe_rejects_wrong_world_size(monkeypatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert nccl_probe.main(["--expected-world-size", "4", "--report-dir", "."]) == 2
```

- [ ] **Step 2: Verify the H100-specific APIs fail**

Run: `python -m pytest -q tests/test_h100_preflight.py tests/test_server_preflight.py`

Expected: FAIL because current preflight records only GPU zero and has no count/memory contract.

- [ ] **Step 3: Implement hardware reporting and isolated probes**

Enumerate all CUDA devices through PyTorch properties. Record cgroup v2 and v1 memory when available and fall back to psutil available/total memory in the report. Query topology with `nvidia-smi topo -m` using a 10-second timeout. The NCCL probe all-reduces rank values to 6, all-gathers four disjoint sample IDs, writes only on rank zero, and barriers before exit. The batch probe builds the exact model and one real finite-data batch, creates optimizer state through a step, measures `max_memory_reserved` on every rank, reduces the maximum, and accepts only `<0.8 * total_memory`.

- [ ] **Step 4: Run local preflight tests**

Run: `python -m pytest -q tests/test_h100_preflight.py tests/test_server_preflight.py tests/test_distributed.py`

Expected: all pass without requiring CUDA; the actual CUDA/NCCL path remains a server acceptance test.

- [ ] **Step 5: Commit and synchronize**

```bash
git add src/tsfm/preflight.py scripts/server_preflight.py scripts/nccl_probe.py scripts/batch_probe.py tests/test_h100_preflight.py tests/test_server_preflight.py
git commit -m "feat: add isolated four-H100 acceptance probes"
```

---

### Task 9: Phase-Aware One-Command Gate, Resume, and Production Launch

**Files:**
- Create: `src/tsfm/pipeline.py`
- Create: `scripts/h100_pipeline.py`
- Create: `scripts/launch_h100_307m.sh`
- Create: `tests/test_h100_pipeline.py`
- Modify: `scripts/train.py:22-54`

**Interfaces:**
- Produces: `PipelineBinding` with package/source/processed/model/plan/hardware digests.
- Produces: `PipelineState.load(path)`, `record_phase(name, binding, report)`, and `can_reuse_preflight(binding) -> bool`.
- Produces: pipeline phases `runtime`, `source`, `conversion`, `hardware`, `nccl`, `batch`, `preflight20`, `resume2`, `pass`, `production`.
- Extends: train CLI with `--resolved-plan` and finite production dataset selection while preserving old actions.

- [ ] **Step 1: Write phase reuse and fail-closed tests**

```python
def test_pipeline_reuses_pass_only_for_identical_binding(tmp_path) -> None:
    state = PipelineState(tmp_path / "pipeline-state.json")
    binding = _binding("a")
    state.record_phase("pass", binding, {"status": "PASS"})
    assert state.can_reuse_preflight(binding)
    assert not state.can_reuse_preflight(_binding("b"))


def test_nonempty_production_without_valid_checkpoint_fails(tmp_path) -> None:
    output = tmp_path / "production"
    output.mkdir()
    (output / "partial.pt").write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="nonempty production directory has no valid latest checkpoint"):
        choose_production_resume(output)


def test_command_order_stops_before_production_on_failed_resume_gate(tmp_path) -> None:
    runner = RecordingRunner(fail_phase="resume2")
    result = run_pipeline(_options(tmp_path), runner=runner)
    assert result == 1
    assert runner.phases == ["runtime", "source", "conversion", "hardware", "nccl", "batch", "preflight20", "resume2"]
```

- [ ] **Step 2: Verify the pipeline module is missing**

Run: `python -m pytest -q tests/test_h100_pipeline.py tests/test_train_cli.py`

Expected: FAIL because phase state, finite-plan train arguments, and pipeline scripts do not exist.

- [ ] **Step 3: Implement atomic phase records and subprocess orchestration**

Represent each phase as one canonical JSON record containing start/end UTC timestamps, command argv, exit code, report path, and binding. Never invoke a shell string from Python; pass argv lists to `subprocess.run`. Run candidate probes in new `torchrun` subprocesses, select the first PASS, resolve and atomically write the plan, run preflight20, run `resume-check --steps 2`, write `preflight-report.json`, then call production `train.py run --resolved-plan ...`. Existing matching PASS skips phases through `pass`; a production checkpoint resumes automatically. Any mismatched binding removes no file and exits nonzero.

The shell entry point sets `TSFM_PERSISTENT_ROOT=/root/work`, sources the project bootstrap, exports NCCL diagnostics, creates `/root/work/logs`, and `exec`s the Python pipeline. The runbook will invoke it once with `nohup`.

- [ ] **Step 4: Run pipeline and CLI tests**

Run: `python -m pytest -q tests/test_h100_pipeline.py tests/test_train_cli.py tests/test_rank_zero_artifacts.py`

Expected: all pass with fake runners; no local subprocess initializes CUDA or accesses a network.

- [ ] **Step 5: Commit and synchronize**

```bash
git add src/tsfm/pipeline.py scripts/h100_pipeline.py scripts/launch_h100_307m.sh scripts/train.py tests/test_h100_pipeline.py tests/test_train_cli.py
git commit -m "feat: orchestrate one-command H100 production launch"
```

---

### Task 10: Offline CPython and Wheelhouse Packaging

**Files:**
- Create: `requirements/h100-py311-cu126.txt`
- Create: `scripts/build_h100_offline_bundle.sh`
- Create: `scripts/install_h100_offline.sh`
- Create: `tests/test_offline_runtime_scripts.py`
- Modify: `pyproject.toml`

**Interfaces:**
- Build script: `build_h100_offline_bundle.sh PYTHON_RUNTIME_TAR OUTPUT_DIR` on a networked Linux x86_64 server.
- Install script: `install_h100_offline.sh BUNDLE_DIR /root/work/venvs/tsfm-h100 /root/work/TimeSeriesFoundationModel` on H100.
- Runtime report: `/root/work/runtime-bundles/runtime-install-report.json` with Python, torch, CUDA runtime, package-manifest digest, and project version.

- [ ] **Step 1: Write script-contract tests**

```python
def test_runtime_requirements_pin_core_versions() -> None:
    text = (ROOT / "requirements/h100-py311-cu126.txt").read_text()
    assert "torch==2.7.1" in text
    assert "datasets==3.6.0" in text
    assert "numpy==2.2.6" in text


def test_installer_is_offline_and_uses_root_work() -> None:
    text = (ROOT / "scripts/install_h100_offline.sh").read_text()
    assert "--no-index" in text
    assert "--find-links" in text
    assert "verify_artifact_manifest" in text
    assert "/root/work" in text
    assert "curl " not in text and "wget " not in text
```

- [ ] **Step 2: Run and confirm missing runtime assets**

Run: `python -m pytest -q tests/test_offline_runtime_scripts.py`

Expected: FAIL because requirements and scripts do not exist.

- [ ] **Step 3: Implement deterministic build and no-index install scripts**

Pin direct inputs to CPython 3.11-compatible versions: torch 2.7.1, numpy 2.2.6, datasets 3.6.0, pyarrow 20.0.0, psutil 7.0.0, tqdm 4.67.1, pytest 8.4.1, setuptools 80.9.0, and wheel 0.45.1. The build script validates Linux x86_64, validates the supplied runtime reports Python 3.11, downloads PyTorch from `https://download.pytorch.org/whl/cu126`, downloads remaining wheels, copies the Python runtime, writes the canonical artifact manifest using Task 1, and creates no archive inside the bundle root.

The installer verifies the bundle before mutation, extracts the runtime below `/root/work/runtime/cpython-3.11`, creates the venv with `--copies`, installs wheels using `--no-index --find-links`, installs the project editable with `--no-build-isolation`, imports torch/datasets/pyarrow, and atomically writes the report. `pyproject.toml` narrows server-tested dependency ceilings only where required by the pinned file; generic local dependencies remain compatible.

- [ ] **Step 4: Run script, source-audit, and package metadata tests**

Run: `python -m pytest -q tests/test_offline_runtime_scripts.py tests/test_source_audit_boundaries.py`

Run: `python scripts/audit_source_tree.py --root .`

Expected: tests pass and audit prints `source tree audit passed`; no wheel or runtime archive exists locally.

- [ ] **Step 5: Commit and synchronize**

```bash
git add requirements/h100-py311-cu126.txt scripts/build_h100_offline_bundle.sh scripts/install_h100_offline.sh tests/test_offline_runtime_scripts.py pyproject.toml
git commit -m "feat: package an offline H100 Python runtime"
```

---

### Task 11: Operator Runbook, Full Regression Gate, and Byte-Identical Delivery

**Files:**
- Create: `docs/h100-307m-runbook.md`
- Modify: `README.md`
- Modify: `docs/server-validation-runbook.md`
- Create: `tests/test_h100_runbook.py`
- Modify: `tests/test_source_audit_boundaries.py`

**Interfaces:**
- Documents: networked-server snapshot/runtime bundle commands, transfer layout, H100 install, one `nohup` launch, log inspection, PASS interpretation, automatic resume, and fail-closed recovery.
- Preserves: RTX 5090 validation runbook as engineering evidence; clearly labels H20 16-card config legacy/preflight-only.

- [ ] **Step 1: Write runbook acceptance tests**

```python
def test_h100_runbook_contains_one_launch_and_all_persistent_paths() -> None:
    text = (ROOT / "docs/h100-307m-runbook.md").read_text()
    assert text.count("nohup bash scripts/launch_h100_307m.sh") == 1
    for value in ("/root/work/venvs/tsfm-h100", "/root/work/tsfm-data/raw/utsd/UTSD-12G", "/root/work/checkpoints/timer-307m-production", "preflight-report.json", "resolved-training-config.json"):
        assert value in text
    assert "Windows" in text and "must not contain datasets" in text


def test_h100_runbook_documents_exact_hardware_and_resume_gate() -> None:
    text = (ROOT / "docs/h100-307m-runbook.md").read_text()
    for value in ("4 x NVIDIA H100 80GB", "20", "two-step resume", "307,146,240", "3,892,126,910"):
        assert value in text
```

- [ ] **Step 2: Verify documentation tests fail**

Run: `python -m pytest -q tests/test_h100_runbook.py`

Expected: FAIL because the H100 production runbook does not exist.

- [ ] **Step 3: Write the exact operator workflow and update project status**

The runbook includes separate commands for the networked Linux server and H100 node. The H100 section installs the offline environment, verifies paths, and provides exactly one launch command:

```bash
cd /root/work/TimeSeriesFoundationModel
nohup bash scripts/launch_h100_307m.sh > /root/work/logs/h100-307m-pipeline.log 2>&1 &
```

It documents `tail -f`, `jobs -l`, report inspection, PASS binding, checkpoint names, rerunning the same command for automatic resume, and explicit stop conditions. Update README capabilities without claiming paper-quality reproduction. Keep the old server runbook and point its production readers to the new file.

- [ ] **Step 4: Run the complete local verification gate**

Run: `python -m pytest -q`

Expected: all tests pass. The previous baseline was 78 tests; the exact new count is whatever pytest reports after Tasks 1-11, with zero failures or skips caused by missing local datasets/GPUs.

Run: `python scripts/audit_source_tree.py --root .`

Expected: `source tree audit passed`.

Run: `git status --short`

Expected before the final documentation commit: only Task 11 documentation/test files are modified or untracked.

- [ ] **Step 5: Commit documentation**

```bash
git add README.md docs/h100-307m-runbook.md docs/server-validation-runbook.md tests/test_h100_runbook.py tests/test_source_audit_boundaries.py
git commit -m "docs: add the four-H100 production runbook"
```

- [ ] **Step 6: Synchronize and prove delivery identity**

Apply the completed file set to both `D:\学习\TimeSeriesFoundationModel` and `D:\学习\TimeSeriesFoundationModel-ServerUpload`. Run the source audit in both trees. Generate sorted relative-path SHA-256 inventories excluding `.git`, `.venv`, `.pytest_cache`, `outputs`, `third_party`, and ignored artifacts, then compare them byte-for-byte.

Expected: both audits pass, both inventories are identical, and `git status --short` in ServerUpload is clean.

- [ ] **Step 7: Record the server-only acceptance commands without running them locally**

The final handoff must explicitly state that the following remain unverified until the H100 allocation exists: four visible GPUs, topology, NCCL, batch candidate, 20+2 resume, and production transition. The operator runs the one launch command from the runbook; local execution must not simulate or claim these results.

---

## Plan Self-Review

- Spec coverage: Tasks 1-3 cover transfer integrity and offline conversion; Tasks 4-7 cover finite training, immutable planning, DDP/checkpoints, and validation; Tasks 8-9 cover H100 acceptance and one-command orchestration; Task 10 covers the offline runtime; Task 11 covers operation, audit, synchronization, and honest server-only acceptance.
- Scope: every task contributes directly to the approved single-node 307M UTSD-12G production path; LOTSA expansion, FSDP, 1B/3B models, downstream benchmarks, and publication evaluation remain excluded.
- Type consistency: `ArtifactEntry`, `ConversionBinding`, `FiniteWindowIndex`, `AffineCoverageSampler`, `ProductionTemplate`, `ResolvedTrainingPlan`, `PipelineBinding`, and `PipelineState` are each introduced before their consumers.
- Failure behavior: data, conversion, hardware, NCCL, batch sizing, preflight resume, binding mismatch, invalid output state, and non-finite training all fail closed without deletion or overwrite.
- Local safety: all tests use synthetic `tmp_path` data and mocked hardware/process runners; no plan step downloads datasets or runtime artifacts on Windows.
