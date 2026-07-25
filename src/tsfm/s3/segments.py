from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from tsfm.s3.records import PreparedSegment, RawSeries


def finite_univariate_segments(
    series: RawSeries,
    min_length: int = 2_976,
    min_variance: float = 1e-8,
) -> Iterator[PreparedSegment]:
    if min_length <= 0:
        raise ValueError("min_length must be positive")
    if min_variance < 0:
        raise ValueError("min_variance must be non-negative")

    values = np.asarray(series.values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2:
        raise ValueError(
            f"{series.source_id}/{series.series_id}: "
            "values must be one- or two-dimensional"
        )

    for channel in range(values.shape[1]):
        channel_values = np.asarray(values[:, channel], dtype=np.float32)
        finite = np.isfinite(channel_values)
        boundaries = np.flatnonzero(
            np.diff(np.concatenate(([False], finite, [False])))
        ).reshape(-1, 2)
        segment_index = 0
        for start, stop in boundaries:
            segment = np.ascontiguousarray(channel_values[start:stop])
            if len(segment) < min_length:
                continue
            if float(np.var(segment, dtype=np.float64)) <= min_variance:
                continue
            yield PreparedSegment(
                source_id=series.source_id,
                dataset_id=series.dataset_id,
                series_id=series.series_id,
                channel_id=str(channel),
                segment_index=segment_index,
                values=segment,
                frequency=series.frequency,
            )
            segment_index += 1
