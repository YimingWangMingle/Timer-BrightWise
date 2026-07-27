# Batch Probe Entrypoint Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure `torchrun scripts/batch_probe.py` executes the probe and writes its batch report instead of exiting without running `main()`.

**Architecture:** Preserve the existing probe and pipeline behavior. Add only the conventional Python script entrypoint and cover it with a real subprocess CLI test that does not require CUDA.

**Tech Stack:** CPython 3.11, argparse, subprocess, pytest, PyTorch torchrun.

## Global Constraints

- Do not change the 307,146,240-parameter model or any data/training configuration.
- Do not delete or rewrite server data, reports, processed artifacts, or checkpoints.
- Keep both local project directories byte-identical for the modified production and test files.

---

### Task 1: Make the Batch Probe Executable

**Files:**
- Modify: `tests/test_h100_pipeline.py`
- Modify: `scripts/batch_probe.py`

**Interfaces:**
- Consumes: `scripts.batch_probe.main(argv: list[str] | None = None) -> int`
- Produces: executable `scripts/batch_probe.py` behavior under CPython and torchrun

- [ ] **Step 1: Write the failing test**

```python
def test_batch_probe_script_exposes_cli_help() -> None:
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, str(root / "scripts" / "batch_probe.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "Probe a real four-rank H100 batch" in result.stdout
    assert "--micro-batch-size" in result.stdout
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest -q tests/test_h100_pipeline.py::test_batch_probe_script_exposes_cli_help`

Expected: FAIL because `result.stdout` is empty when the script never invokes `main()`.

- [ ] **Step 3: Write the minimal implementation**

```python
if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run focused and full tests**

Run: `python -m pytest -q tests/test_h100_pipeline.py::test_batch_probe_script_exposes_cli_help`

Expected: PASS.

Run: `python -m pytest -q`

Expected: 127 passed.

- [ ] **Step 5: Synchronize and commit**

Copy the two modified files to `D:\学习\TimeSeriesFoundationModel`, verify their SHA-256 hashes match, then commit them in the upload repository with message `fix: execute the H100 batch probe` and push `main`.
