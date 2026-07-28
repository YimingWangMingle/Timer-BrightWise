# DDP RNG Gather Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make four-rank checkpoint RNG collection safe under PyTorch 2.7.1/NCCL while preserving the existing checkpoint schema and exact per-rank resume state.

**Architecture:** Serialize each rank's tensor-containing RNG dictionary into a `torch.save` byte payload before the object collective. Gather only `(rank, bytes)`, then deserialize on rank 0 before invoking the existing checkpoint manager.

**Tech Stack:** Python 3.11, PyTorch 2.7.1, `torch.distributed`, pytest, Git.

## Global Constraints

- Do not modify model, dataset, optimizer, precision, batch size, or production-plan configuration.
- Preserve all four ranks' Python, NumPy, CPU Torch, and CUDA RNG states.
- Preserve the current on-disk checkpoint schema and resume compatibility.
- Keep `D:\学习\TimeSeriesFoundationModel` and `D:\学习\TimeSeriesFoundationModel-ServerUpload` synchronized.
- Push all pending local commits to `https://github.com/YimingWangMingle/Timer-BrightWise.git`.

---

### Task 1: Make RNG Gather Tensor-Free On The Collective Boundary

**Files:**
- Modify: `tests/test_rank_rng_gather.py`
- Modify: `src/tsfm/trainer.py`

**Interfaces:**
- Consumes: `capture_rng_state() -> dict[str, object]` and initialized `torch.distributed` state.
- Produces: `_gather_rank_rng_states(*, rank: int, world_size: int) -> dict[int, dict[str, object]] | None` with unchanged return semantics.

- [ ] **Step 1: Write the failing regression test**

```python
import torch

from tsfm.trainer import _gather_rank_rng_states


def test_distributed_rng_gather_transports_serialized_bytes(monkeypatch) -> None:
    known_state = {"torch_cpu": torch.arange(8, dtype=torch.uint8)}

    monkeypatch.setattr("tsfm.trainer.capture_rng_state", lambda: known_state)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def gather_object(value, gathered, dst):
        rank, payload = value
        assert rank == 0
        assert isinstance(payload, bytes)
        assert dst == 0
        for item_rank in range(4):
            gathered[item_rank] = (item_rank, payload)

    monkeypatch.setattr(torch.distributed, "gather_object", gather_object)

    states = _gather_rank_rng_states(rank=0, world_size=4)

    assert states is not None
    assert set(states) == {0, 1, 2, 3}
    for state in states.values():
        assert torch.equal(state["torch_cpu"], known_state["torch_cpu"])
```

- [ ] **Step 2: Run the regression test and verify RED**

Run from the repository root:

```powershell
D:\学习\TimeSeriesFoundationModel\.venv\Scripts\python.exe -m pytest -q tests/test_rank_rng_gather.py
```

Expected: the new test fails because the current collective payload is a dictionary rather than `bytes`.

- [ ] **Step 3: Implement the minimal serialization boundary**

Add `io` to the imports in `src/tsfm/trainer.py`, then change the distributed branch:

```python
def _serialize_rng_state(state: dict[str, object]) -> bytes:
    buffer = io.BytesIO()
    torch.save(state, buffer)
    return buffer.getvalue()


def _deserialize_rng_state(payload: bytes) -> dict[str, object]:
    state = torch.load(
        io.BytesIO(payload), map_location="cpu", weights_only=True
    )
    if not isinstance(state, dict):
        raise TypeError("serialized RNG state must contain a dictionary")
    return state


def _gather_rank_rng_states(
    *, rank: int, world_size: int
) -> dict[int, dict[str, object]] | None:
    local = capture_rng_state()
    if world_size == 1 or not torch.distributed.is_initialized():
        return {rank: local}
    payload = _serialize_rng_state(local)
    gathered = [None] * world_size if rank == 0 else None
    torch.distributed.gather_object((rank, payload), gathered, dst=0)
    if rank != 0:
        return None
    assert gathered is not None
    return {
        item_rank: _deserialize_rng_state(item_payload)
        for item_rank, item_payload in gathered
    }
```

- [ ] **Step 4: Run focused and full tests and verify GREEN**

```powershell
D:\学习\TimeSeriesFoundationModel\.venv\Scripts\python.exe -m pytest -q tests/test_rank_rng_gather.py tests/test_ddp_production_contract.py
D:\学习\TimeSeriesFoundationModel\.venv\Scripts\python.exe -m pytest -q
```

Expected: all tests pass; the focused regression confirms bytes cross the collective boundary and tensors are restored on rank 0.

- [ ] **Step 5: Commit the implementation**

```powershell
git add src/tsfm/trainer.py tests/test_rank_rng_gather.py
git commit -m "fix: serialize DDP RNG gather payloads"
```

### Task 2: Synchronize, Verify, And Publish

**Files:**
- Create: `docs/superpowers/plans/2026-07-28-ddp-rng-gather-fix.md`
- Synchronize: the design, plan, implementation, and regression test to both local project folders.

**Interfaces:**
- Consumes: the tested commit from Task 1.
- Produces: byte-identical project files in both local folders and a GitHub `main` branch containing every pending commit.

- [ ] **Step 1: Copy the four changed files to both local folders**

Copy these exact relative paths from the tested work copy:

```text
src/tsfm/trainer.py
tests/test_rank_rng_gather.py
docs/superpowers/specs/2026-07-28-ddp-rng-gather-fix-design.md
docs/superpowers/plans/2026-07-28-ddp-rng-gather-fix.md
```

- [ ] **Step 2: Verify synchronization with SHA-256**

Compute SHA-256 for each relative path in both local folders and require every pair to match.

- [ ] **Step 3: Commit the implementation plan**

```powershell
git add docs/superpowers/plans/2026-07-28-ddp-rng-gather-fix.md
git commit -m "docs: plan DDP RNG gather fix"
```

- [ ] **Step 4: Push all pending commits**

```powershell
git push https://github.com/YimingWangMingle/Timer-BrightWise.git main:main
```

Expected: GitHub `main` advances from `fab6da2` through the batch-probe fix, this design, the implementation, and this plan.

- [ ] **Step 5: Report the server synchronization set**

The H100 server must replace only:

```text
/root/work/TimeSeriesFoundationModel/src/tsfm/trainer.py
/root/work/TimeSeriesFoundationModel/tests/test_rank_rng_gather.py
```

Documentation files may also be synchronized for repository completeness but are not required by the runtime.
