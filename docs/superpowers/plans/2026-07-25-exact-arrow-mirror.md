# Exact Arrow Mirror Access Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate all implicit LOTSA repository scans from discovery and conversion.

**Architecture:** The data policy pins exact Arrow files and expected sizes. Pure discovery probes exact resolve URLs, while the LOTSA adapter downloads those files to `HF_HOME` and passes local paths to the packaged Arrow loader.

**Tech Stack:** Python 3.11+, pytest, urllib HEAD requests, huggingface_hub, datasets Arrow builder, existing S3 pipeline.

## Global Constraints

- Never download dataset payloads, weights, or checkpoints locally.
- Keep server mutations below `/root/autodl-tmp`.
- Pin LOTSA revision `8191fd29eb5cf906ec55effca44d8059888b615d`.
- Keep source cache and processed data independently below 2,000,000,000 bytes.
- Keep formal and upload-clone runtime files byte-for-byte synchronized.

---

### Task 1: Exact LOTSA Discovery

**Files:**
- Modify: `configs/data/server_validation.json`
- Modify: `scripts/prepare_data.py`
- Modify: `tests/test_prepare_data_cli.py`

**Interfaces:**
- Consumes: policy `revision`, `format`, and per-configuration `data_files` entries containing `path` and `size`.
- Produces: `build_inventory(..., probe_file=...)` entries preserving exact file evidence.

- [ ] Write failing tests asserting the exact checked-in revision, files, and sizes.
- [ ] Write a failing test where the LOTSA builder raises if called and the injected exact-file probe succeeds.
- [ ] Run `python -m pytest -q tests/test_prepare_data_cli.py` and confirm RED.
- [ ] Implement exact resolve-URL HEAD probing and pinned-size validation for LOTSA while retaining the UTSD builder path.
- [ ] Run the focused tests and confirm GREEN.

### Task 2: Exact LOTSA Conversion

**Files:**
- Modify: `src/tsfm/s3/adapters.py`
- Modify: `scripts/prepare_data.py`
- Modify: `tests/test_s3_segments_adapters.py`

**Interfaces:**
- Consumes: `DatasetSpec.data_files`, `DatasetSpec.revision`, and `DatasetSpec.file_format`.
- Produces: LOTSA rows loaded from exact cached Arrow files without a repository/configuration scan.

- [ ] Write a failing adapter test asserting exact `hf_hub_download` calls and local Arrow loader arguments.
- [ ] Run `python -m pytest -q tests/test_s3_segments_adapters.py` and confirm RED.
- [ ] Extend `DatasetSpec`, inject the downloader boundary, and override LOTSA row loading through local Arrow files.
- [ ] Wire inventory evidence into `convert`.
- [ ] Run focused adapter and CLI tests and confirm GREEN.

### Task 3: Documentation And Synchronization

**Files:**
- Modify: `docs/server-validation-runbook.md`
- Modify: `tests/test_prepare_data_cli.py`

**Interfaces:**
- Consumes: the exact-file discovery and conversion commands.
- Produces: a server gate that identifies the fixed revision and prohibits conversion after any discovery failure.

- [ ] Add a failing runbook contract assertion for the pinned revision and exact-file behavior.
- [ ] Update the runbook and run the complete test suite plus source audit.
- [ ] Synchronize runtime files to the formal project and upload clone.
- [ ] Verify SHA-256 equality and run complete tests in both trees.
