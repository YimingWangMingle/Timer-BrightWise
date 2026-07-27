# Batch Probe Entrypoint Fix Design

## Problem

The four-rank batch probe exits successfully without generating its required
`batch-probe-<size>.json` report. `scripts/batch_probe.py` defines `main()` but
does not invoke it when executed by `torchrun`, so the pipeline fails while
opening a report that was never written.

## Design

Add the standard Python script guard to `scripts/batch_probe.py` and return
`main()` through `SystemExit`. Keep the probe, pipeline, model, data, and
training configuration unchanged. Add a subprocess regression test that runs
the script with `--help` and requires the batch-probe CLI help text; this
fails when the entrypoint is absent without requiring CUDA or distributed
workers.

## Recovery

Synchronize the fixed script to the H100 server and rerun the existing
one-command launcher. The pipeline preserves the verified source snapshot and
completed atomic conversion, then repeats the gate and continues through batch
selection, the 20-step preflight, the two-step resume check, and production.
No dataset, environment, processed artifact, or checkpoint is deleted.
