# DDP RNG Gather Fix Design

## Problem

The four-H100 preflight completes all 20 optimizer steps, then fails before
writing `step-000020.pt`. Each rank passes the dictionary returned by
`capture_rng_state()` directly to `torch.distributed.gather_object()`. That
dictionary contains PyTorch tensors. With PyTorch 2.7.1 and NCCL, rank 0 fails
while unpickling the nested tensor storage with:

```text
AttributeError: type object 'torch.storage.UntypedStorage' has no attribute 'dtype'
```

The batch probe, model, data loader, BF16 computation, optimizer update, and
four-rank NCCL communication have already passed. This fix is limited to the
checkpoint RNG collection boundary.

## Chosen Design

Serialize each rank's complete RNG state into an opaque byte payload with
`torch.save()` before calling `gather_object()`. The collective therefore
transports only `(rank, bytes)`. Rank 0 deserializes every payload with
`torch.load(..., map_location="cpu", weights_only=True)` before handing the
same tensor-based state dictionaries to `CheckpointManager`.

This keeps the on-disk checkpoint schema unchanged. Existing restore code and
existing checkpoints remain compatible, and every rank retains independent
Python, NumPy, CPU Torch, and CUDA RNG state.

## Rejected Alternatives

- Converting every tensor to nested integer lists adds schema maintenance and
  unnecessary conversion code.
- Replacing `gather_object()` with `all_gather_object()` retains the same
  tensor-pickling failure mode.
- Saving only rank 0 RNG state breaks exact four-rank resume determinism.

## Tests

Add a regression test that simulates the distributed gather boundary and
requires the transported RNG value to be bytes, then verifies rank-indexed
states are restored as tensors. Run that test first against the old code to
observe the expected failure, then run the full suite after the minimal fix.

## Scope And Rollout

Modify only `src/tsfm/trainer.py` and `tests/test_rank_rng_gather.py`, plus this
design and its implementation plan. Synchronize those files to both local
project folders, push all pending commits to GitHub, and replace the same
runtime files on the H100 server. No dataset, model, batch-size, precision,
optimizer, or production-plan setting changes.
