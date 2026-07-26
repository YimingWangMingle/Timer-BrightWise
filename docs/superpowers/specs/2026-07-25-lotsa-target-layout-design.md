# LOTSA Target Layout Normalization

## Problem

LOTSA stores multivariate `target` arrays as channel-first values with shape
`(channels, time)`. The internal S3 segmenter expects time-first arrays with
shape `(time, channels)`. The Beijing air-quality rows observed on the server
have shape `(11, 35064)`, so the current code treats each time point as an
11-value channel and rejects every channel against the minimum length of 2976.

## Design

Normalize data at the adapter boundary. `_HuggingFaceAdapter` will expose a
target-normalization hook whose default preserves the current UTSD behavior.
`LOTSAAdapter` will transpose every two-dimensional target before constructing
`RawSeries`; one-dimensional targets remain unchanged. Arrays with other ranks
remain unchanged so the existing segment validator continues to reject them
with its established error message.

This is a source-format conversion, not a heuristic. It does not lower quality
thresholds, infer layout from dimension lengths, or change the model, trainer,
manifest format, or S3 segment representation.

## Verification

Add regression tests proving that:

- a LOTSA channel-first target becomes time-first;
- the normalized target produces one full-length segment per channel;
- one-dimensional LOTSA targets remain one-dimensional;
- UTSD target handling remains unchanged.

Run the focused adapter and segment tests, then the full offline test suite and
source-tree audit. Synchronize the adapter and regression test to the formal
project and server-upload clone and verify byte-identical hashes.

On the server, rerun `convert` using the existing Hugging Face cache, then run
`validate`. Validation must report manifest records for UTSD-1G and all three
LOTSA groups, including `beijing_air_quality`.
