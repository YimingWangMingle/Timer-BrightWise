# Bounded Data Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace mirror-incompatible LOTSA repository scanning with explicit bounded dataset discovery and accurate server cache validation.

**Architecture:** Keep `prepare_data.py` as the guarded command boundary, but extract pure policy/discovery helpers that accept an injected builder loader for offline tests. The JSON policy declares four exact groups and per-source quotas; validation accounts for the Hugging Face cache and proves every selected group produced records.

**Tech Stack:** Python 3.11+, pytest, Hugging Face `datasets`, JSON policy, pathlib, existing S3 mmap/manifest pipeline.

## Global Constraints

- Never download UTSD, LOTSA, converted shards, checkpoints, or model weights locally.
- Server mutations remain below `/root/autodl-tmp` and require `--execute-server`.
- The corpus is `UTSD-1G`, `traffic_hourly`, `beijing_air_quality`, and `weather`.
- Projected/cache source bytes and processed bytes are each capped at 2,000,000,000.
- Formal source and upload-clone runtime files must match byte-for-byte by SHA-256.
- The repository has no Git metadata, so commit steps are recorded but cannot be executed.

---

### Task 1: Explicit Policy And Discovery

**Files:**
- Modify: `configs/data/server_validation.json`
- Modify: `scripts/prepare_data.py`
- Test: `tests/test_prepare_data_cli.py`

**Interfaces:**
- Consumes: policy `repositories` entries with explicit `configurations`, per-configuration `domains`, and `minimum_dataset_groups_by_source`.
- Produces: `build_inventory(policy, load_builder, cache_dir, endpoint, progress)` returning the complete inventory dictionary.

- [ ] **Step 1: Write failing tests for explicit discovery**

Add tests that load `prepare_data.py` as a module, inject a builder returning `download_size`, and assert the four configuration calls occur in declaration order. Add a loader that raises `OSError` and assert the message includes repository, configuration, and endpoint. Assert the checked-in policy has no string-valued `"discover"` configuration and contains exactly the approved groups.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python -m pytest -q tests/test_prepare_data_cli.py`

Expected: FAIL because `build_inventory` and the new policy fields do not exist.

- [ ] **Step 3: Implement minimal explicit discovery**

Implement pure helpers in `prepare_data.py` that reject duplicate groups, print `discovering source/repository/configuration`, call only the injected builder loader, wrap exceptions with endpoint context, enforce exact per-source/domain/projected-byte quotas, and return a selected inventory containing every declared entry. Remove `get_dataset_config_names` and smallest-two selection.

Update the policy to one UTSD group and three LOTSA groups with explicit domains and 2 GB limits.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest -q tests/test_prepare_data_cli.py`

Expected: all tests pass with no network access.

### Task 2: Cache Accounting And Empty-Group Validation

**Files:**
- Modify: `scripts/prepare_data.py`
- Test: `tests/test_prepare_data_cli.py`

**Interfaces:**
- Consumes: inventory selected entries, manifest records, `HF_HOME`, processed root, source and processed limits.
- Produces: validation report fields `projected_source_bytes`, `hf_cache_bytes`, `processed_bytes`, and exact dataset groups.

- [ ] **Step 1: Write failing tests for byte accounting and group coverage**

Create temporary Hugging Face cache and processed files with known sizes; call a pure `build_validation_report` helper and assert all three byte totals. Add a manifest-record fixture missing one selected dataset group and assert `ValueError` names that group.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `python -m pytest -q tests/test_prepare_data_cli.py`

Expected: FAIL because cache accounting still reads `data_root/raw` and missing selected groups are not identified.

- [ ] **Step 3: Implement minimal validation report helper**

Measure actual files below `HF_HOME`, measure processed files, carry projected bytes from inventory, compare each independently to its policy limit, compare selected `(source_id, dataset_id)` pairs with manifest pairs, and return the report. Wire `validate` to the helper after checksum verification.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run: `python -m pytest -q tests/test_prepare_data_cli.py`

Expected: all focused tests pass.

### Task 3: Server Runbook And Regression Gate

**Files:**
- Modify: `docs/server-validation-runbook.md`
- Test: `tests/test_prepare_data_cli.py`
- Test: `tests/test_safety_and_data_cli.py`

**Interfaces:**
- Consumes: working `HF_ENDPOINT`, venv activation, existing guarded CLI.
- Produces: exact mirror-aware discover instructions and 2 GB evidence gates.

- [ ] **Step 1: Add a failing documentation contract test**

Assert the runbook names `HF_ENDPOINT`, the four explicit configurations, 2 GB cache/processed limits, and the inventory review gate before conversion.

- [ ] **Step 2: Run the contract test and verify RED**

Run: `python -m pytest -q tests/test_prepare_data_cli.py`

Expected: FAIL because the runbook still describes two groups per source and 20 GB limits.

- [ ] **Step 3: Update the runbook**

Document venv activation, mirror exports, dry-run, nohup-safe discovery, inventory inspection, and the prohibition on conversion until the four groups and byte limits are verified.

- [ ] **Step 4: Run local regression gates**

Run: `python -m pytest -q`

Run: `python scripts/audit_source_tree.py --root .`

Expected: all tests pass and source audit succeeds.

### Task 4: Formal And Upload-Clone Synchronization

**Files:**
- Sync runtime files to: `D:\学习\TimeSeriesFoundationModel`
- Sync runtime files to: `D:\学习\TimeSeriesFoundationModel-ServerUpload`
- Sync development docs only to: `D:\学习\TimeSeriesFoundationModel\docs\superpowers`

**Interfaces:**
- Consumes: verified staging files.
- Produces: formal and upload-clone runtime files with matching SHA-256 hashes.

- [ ] **Step 1: Copy the verified runtime files**

Copy `configs/data/server_validation.json`, `scripts/prepare_data.py`, `tests/test_prepare_data_cli.py`, and `docs/server-validation-runbook.md` to both destinations without touching excluded directories.

- [ ] **Step 2: Copy design and plan documents to the formal project only**

Copy the bounded-discovery spec and this plan below the formal project's `docs/superpowers`; these are intentionally outside the server-upload whitelist.

- [ ] **Step 3: Verify hashes and tests in both trees**

Compare SHA-256 for every runtime file across staging, formal, and upload clone. Run the formal project full suite and the upload clone full suite with bytecode/cache creation disabled.

Expected: zero hash mismatches, all tests pass in both trees, and no forbidden generated/data/weight files appear in the upload clone.
