# Tiny Timer Smoke Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify a CPU-only Timer-style end-to-end training smoke test that shares one code path with future 0.3B, 1B, and 3B configurations.

**Architecture:** A standalone PyTorch package implements continuous patch embedding, RoPE causal SDPA, gated decoder blocks, next-patch MSE, deterministic synthetic data, rolling generation, and checkpoint resume. Model size is controlled entirely by validated JSON configuration.

**Tech Stack:** Python 3.12, PyTorch 2.x CPU, pytest, standard-library JSON and argparse.

## Global Constraints

- Local execution must require no real datasets or downloaded model weights.
- Tiny and large model variants must use the same production classes.
- Patch size defaults to 96 for parity with Timer.
- Production code is written only after its test has failed for the expected reason.
- `third_party/` remains reference-only and is not imported at runtime.

---

### Task 1: Project package and configuration contract

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `src/tsfm/__init__.py`
- Create: `tests/test_config.py`
- Create: `configs/model/tiny.json`
- Create: `configs/model/timer_300m.json`

**Interfaces:**
- Produces: `TimerConfig.from_json(path) -> TimerConfig`
- Produces: `estimate_parameter_count(config) -> int`

- [ ] Write tests for validation, tiny size, and the 300M analytical range.
- [ ] Run `pytest tests/test_config.py -v` and confirm import failure.
- [ ] Implement the minimal dataclass and analytical count.
- [ ] Re-run the test and confirm PASS.

### Task 2: Model forward, loss, and causality

**Files:**
- Create: `tests/test_model.py`
- Create: `src/tsfm/model.py`

**Interfaces:**
- Produces: `TimerModel(config)`
- Produces: `TimerOutput(predictions, loss)`
- Produces: `TimerModel.generate(input_values, prediction_length)`

- [ ] Write failing tests for shape, finite loss/backward, causality, and exact generation length.
- [ ] Run `pytest tests/test_model.py -v` and confirm missing-module failure.
- [ ] Implement patch embedding, RoPE, causal SDPA, gated MLP, decoder, loss, and generation.
- [ ] Re-run model tests and confirm PASS.

### Task 3: Deterministic synthetic data and normalization

**Files:**
- Create: `tests/test_data.py`
- Create: `src/tsfm/data.py`

**Interfaces:**
- Produces: `SyntheticTimeSeriesDataset(num_samples, total_length, seed)`
- Produces: `normalize_context_target(context, target) -> NormalizedBatch`

- [ ] Write failing tests for deterministic samples, lengths, and context-only normalization.
- [ ] Run `pytest tests/test_data.py -v` and confirm failure.
- [ ] Implement sine, trend, and AR mixtures with deterministic per-index generators.
- [ ] Re-run data tests and confirm PASS.

### Task 4: Checkpoint round trip

**Files:**
- Create: `tests/test_checkpoint.py`
- Create: `src/tsfm/checkpoint.py`

**Interfaces:**
- Produces: `save_checkpoint(path, model, optimizer, step) -> None`
- Produces: `load_checkpoint(path, model, optimizer) -> int`

- [ ] Write a failing prediction round-trip test.
- [ ] Run `pytest tests/test_checkpoint.py -v` and confirm failure.
- [ ] Implement atomic checkpoint save and CPU-safe restore.
- [ ] Re-run checkpoint tests and confirm PASS.

### Task 5: Training loop and smoke entry point

**Files:**
- Create: `tests/test_training.py`
- Create: `src/tsfm/training.py`
- Create: `scripts/run_smoke.py`
- Create: `configs/training/smoke.json`
- Create: `README.md`

**Interfaces:**
- Produces: `run_smoke(model_config, training_config, output_dir) -> SmokeResult`

- [ ] Write a failing test that requires final loss below initial loss on a fixed synthetic corpus.
- [ ] Run `pytest tests/test_training.py -v` and confirm failure.
- [ ] Implement deterministic AdamW training, clipping, logging, and final checkpoint.
- [ ] Re-run the training test and confirm PASS.
- [ ] Run the complete suite with `pytest -q`.
- [ ] Run `python scripts/run_smoke.py --steps 100` and record initial/final loss and checkpoint path.

### Task 6: Final verification

**Files:**
- Modify: `README.md`

- [ ] Confirm no code imports from `third_party/`.
- [ ] Confirm no data or weights were downloaded.
- [ ] Confirm all tests pass without warnings.
- [ ] Document which checks remain server-only.
