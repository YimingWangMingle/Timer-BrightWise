# LOTSA Target Layout Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Normalize LOTSA multivariate targets from channel-first to time-first so every Beijing air-quality channel reaches S3 segmentation.

**Architecture:** Add a target-normalization hook at the Hugging Face adapter boundary. Preserve the base/UTSD behavior and override only LOTSA's handling of two-dimensional targets.

**Tech Stack:** Python 3.11, NumPy, pytest, Hugging Face datasets

## Global Constraints

- Never download dataset payloads, weights, or checkpoints on the local machine.
- Do not change the model, trainer, manifest format, S3 segment representation, minimum segment length, or minimum variance.
- Keep the formal project and server-upload clone runtime files byte-identical.
- The project trees are not Git repositories; use test, audit, and SHA-256 checkpoints instead of commits.

---

### Task 1: Normalize LOTSA Targets

**Files:**
- Modify: `src/tsfm/s3/adapters.py`
- Modify: `tests/test_s3_segments_adapters.py`

**Interfaces:**
- Consumes: `_HuggingFaceAdapter.iter_series()`, `LOTSAAdapter`, and `finite_univariate_segments()`.
- Produces: `_HuggingFaceAdapter._target_values(target: object) -> np.ndarray`, with a LOTSA override returning `values.T` only when `values.ndim == 2`.

- [ ] **Step 1: Write the failing regression tests**

Add tests with a LOTSA row whose target is channel-first:

```python
def test_lotsa_adapter_normalizes_channel_first_targets(tmp_path) -> None:
    loader = FakeLoader(
        [{"item_id": "station", "target": [[1, 2, 3, 4], [5, 6, 7, 8]]}]
    )
    spec = DatasetSpec("Salesforce/lotsa_data", "air", "lotsa", "air")

    row = next(LOTSAAdapter(spec, tmp_path, loader=loader).iter_series())
    segments = list(finite_univariate_segments(row, min_length=4))

    assert row.values.shape == (4, 2)
    assert [segment.values.tolist() for segment in segments] == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]


def test_lotsa_adapter_preserves_univariate_targets(tmp_path) -> None:
    loader = FakeLoader([{"item_id": "station", "target": [1, 2, 3, 4]}])
    spec = DatasetSpec("Salesforce/lotsa_data", "traffic", "lotsa", "traffic")

    row = next(LOTSAAdapter(spec, tmp_path, loader=loader).iter_series())

    assert row.values.shape == (4,)
```

- [ ] **Step 2: Verify the regression test is red**

Run:

```powershell
& 'D:\学习\TimeSeriesFoundationModel\.venv\Scripts\python.exe' -m pytest -q tests/test_s3_segments_adapters.py
```

Expected: the channel-first test fails because the shape is `(2, 4)` instead of `(4, 2)`.

- [ ] **Step 3: Implement the adapter-boundary normalization**

In `_HuggingFaceAdapter`:

```python
def _target_values(self, target: object) -> np.ndarray:
    return np.asarray(target)
```

Call this hook when constructing `RawSeries`. In `LOTSAAdapter`:

```python
def _target_values(self, target: object) -> np.ndarray:
    values = np.asarray(target)
    if values.ndim == 2:
        return values.T
    return values
```

- [ ] **Step 4: Verify focused and complete local tests**

Run focused tests, then the full suite and audit:

```powershell
& 'D:\学习\TimeSeriesFoundationModel\.venv\Scripts\python.exe' -m pytest -q tests/test_s3_segments_adapters.py tests/test_exact_arrow_mirror.py
& 'D:\学习\TimeSeriesFoundationModel\.venv\Scripts\python.exe' -m pytest -q
& 'D:\学习\TimeSeriesFoundationModel\.venv\Scripts\python.exe' scripts/audit_source_tree.py --root .
```

Expected: all tests pass and the source-tree audit passes in the clean staging and upload trees.

- [ ] **Step 5: Synchronize and verify artifacts**

Synchronize `src/tsfm/s3/adapters.py` and `tests/test_s3_segments_adapters.py` to the formal project and server-upload clone. Compute SHA-256 for both files across staging, formal, and clone; each file must have one identical hash across all three trees.

- [ ] **Step 6: Reconvert and validate on the server**

Upload the two runtime files, reuse `/root/autodl-tmp/cache/huggingface`, rerun `convert`, and then run `validate`. Expected: the validation report includes `lotsa/beijing_air_quality` and all four selected groups.
