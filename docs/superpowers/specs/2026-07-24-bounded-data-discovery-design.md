# Bounded Server Data Discovery Design

**Status:** Approved on 2026-07-24

## Objective

Replace repository-wide LOTSA configuration discovery with a deterministic,
bounded validation corpus that works through a Hugging Face mirror and fits the
100 GB AutoDL data disk. The corpus is an engineering gate for the 26.35M model,
not a claim of foundation-model pretraining coverage.

No UTSD, LOTSA, converted shards, checkpoints, or model weights may be written
locally. Local work is limited to source code, tests, configuration, and public
repository metadata.

## Selected Corpus

The server-validation corpus contains exactly four explicit groups:

- `thuml/UTSD`, configuration `UTSD-1G`, domain `mixed-utsd`;
- `Salesforce/lotsa_data`, configuration `traffic_hourly`, domain `traffic`;
- `Salesforce/lotsa_data`, configuration `beijing_air_quality`, domain
  `air-quality`;
- `Salesforce/lotsa_data`, configuration `weather`, domain `weather`.

`UTSD-1G` is used once. `UTSD-2G`, `UTSD-4G`, and `UTSD-12G` are scale variants
and are not counted as independent domains. The policy requires one UTSD group,
three LOTSA groups, and four domain labels.

Public metadata indicates approximately 495 MB for UTSD-1G and about 73 MiB of
published Parquet projections for the three LOTSA configurations. The server
cache target is therefore 0.6-1.0 GB. The hard limits are 2,000,000,000 bytes
for projected/cache input and 2,000,000,000 bytes for processed shards.

## Discovery Flow

The policy contains only explicit configuration lists. `discover` never calls
`get_dataset_config_names` and never recursively scans a repository. It handles
each configured group in declaration order:

1. Print the source, repository, and configuration before network access.
2. Load only that configuration's builder metadata using the configured
   `HF_HOME` and the process-wide `HF_ENDPOINT`.
3. Record `download_size`, falling back to `dataset_size`, as the projected
   source bytes.
4. Preserve the explicit domain and configuration as the dataset identity.
5. Reject duplicate `(repository, configuration)` entries.
6. Reject the inventory before conversion if any per-source count, domain
   count, or byte limit is violated.
7. Write the complete inventory atomically only after every group succeeds.

Any network or builder exception is re-raised with the repository,
configuration, and effective endpoint in the message. No partial inventory is
accepted as successful discovery.

## Conversion And Validation

Conversion keeps the existing lazy streaming adapters, finite-value splitting,
multivariate channel separation, short/constant filtering, float32 mmap shard
packing, and versioned JSONL manifest. Sampling remains equal by source, so the
larger UTSD group cannot dominate the three LOTSA groups.

Validation reports both:

- the inventory's projected source bytes; and
- actual bytes below `HF_HOME` plus actual processed-shard bytes.

The old `TSFM_DATA_ROOT/raw` measurement is removed because Hugging Face assets
are stored below `HF_HOME`. Validation fails when the cache or processed limit
is exceeded, when a selected group yields no valid segment, or when group/domain
quotas are not met.

## Error Handling

Dry-run mode remains non-mutating and performs no network access. Server actions
retain the persistent-root, data-root, free-disk, and `--execute-server` guards.
Discovery progress is line-buffered so nohup logs identify the active group.
Authentication warnings for public data are non-fatal. Network, mirror,
configuration, checksum, quota, and empty-group failures are fatal and preserve
the last complete evidence artifact without treating a partial result as valid.

## Tests

Tests inject fake builder metadata and never contact Hugging Face. They prove:

- explicit discovery never invokes repository-wide configuration listing;
- all four declared groups are preserved in declaration order;
- duplicate groups and per-source/domain quota violations fail;
- errors name the repository, configuration, and endpoint;
- projected, Hugging Face cache, and processed byte limits are independent;
- validation rejects a configured group that produces no manifest record;
- dry-run and local mutation guards remain intact.

The full local suite and source-tree audit must pass before syncing the formal
project and upload clone. Corresponding upload files must match the formal
project byte-for-byte by SHA-256.

## Model And Data Scale Ladder

The completed server smoke used the 354,304-parameter tiny model, 64 synthetic
series, and 20 CPU training steps. After this change, the first real-data gate
uses the exact 26,349,568-parameter model for 2,000 BF16 steps with micro-batch
32, gradient accumulation 8, and 30 context patches. The same bounded corpus may
then be reused for the 94,635,008-parameter 500-step gate and the
307,146,240-parameter 20-step 16-H20 engineering preflight.

Model width, depth, heads, MLP size, and context limits are JSON-driven and use
one model class, so 95M and 307M expansion requires no architecture fork. New
UTSD/LOTSA groups with the same `target` schema are added as policy entries, not
new adapters.

This convenience has explicit limits. A formal 307M run needs a larger training
recipe and much more data than this engineering corpus. Moving to 1B or 3B keeps
the decoder-only model contract but may require activation checkpointing,
FSDP/ZeRO or distributed optimizer state, and distributed checkpoint writing.
Moving from a few GB to tens or hundreds of GB requires parallel conversion,
source deduplication, explicit source weights, and scalable shard staging.
Datasets without the Hugging Face `target` contract require a new adapter. These
are infrastructure extensions around stable model/data interfaces, not a model
rewrite.

## Rollout

1. Run the new tests red, implement the minimal behavior, and run them green.
2. Run the complete local test suite and source audit.
3. Sync the formal project and `TimeSeriesFoundationModel-ServerUpload` clone.
4. Upload only the changed source/config/test/runbook files.
5. On the server, export the working mirror endpoint and run `discover`.
6. Inspect `inventory.json` and actual cache bytes before authorizing `convert`.
7. Run `convert`, `validate`, and only then the 26.35M gate.
