# Exact Arrow Mirror Access Design

**Status:** Approved on 2026-07-25

## Problem

Explicit LOTSA configuration names do not prevent Hugging Face `datasets`
from recursively globbing the complete repository. The mirror returns a
pagination link on `huggingface.co`, which is unreachable from AutoDL. The
same implicit repository scan occurs in both builder discovery and adapter
conversion.

## Design

The validation policy pins LOTSA revision
`8191fd29eb5cf906ec55effca44d8059888b615d` and declares one exact Arrow file
for each selected LOTSA group. Discovery sends HEAD requests only to those
resolve URLs and records their content lengths. It never calls a LOTSA
dataset builder or repository tree API.

Conversion downloads each declared file with `hf_hub_download`, using the
pinned revision and `HF_HOME`, then reads the local Arrow files with the
packaged `arrow` builder. UTSD-1G keeps its existing builder path because it
already succeeds through the mirror. Inventory entries carry revision,
format, and data-file paths so conversion uses only reviewed discovery
evidence.

## Selected LOTSA Files

- `traffic_hourly/data-00000-of-00001.arrow`: 59,934,920 bytes
- `beijing_air_quality/data-00000-of-00001.arrow`: 18,516,128 bytes
- `weather/data-00000-of-00001.arrow`: 171,846,176 bytes

Together with UTSD-1G, projected source data remains below the independent
2,000,000,000-byte source-cache limit.

## Error Handling And Tests

HEAD failures identify repository, revision, file, and endpoint. A size
mismatch against the pinned policy is fatal. Tests use injected HEAD,
download, and dataset-loader boundaries and perform no network access. They
prove LOTSA discovery never calls its builder, conversion downloads only
declared files, Arrow loading uses local paths, and the checked-in policy
contains the exact revision, paths, and sizes.

The model, S3 segmentation, trainer, and model-scale configurations are
unchanged.
