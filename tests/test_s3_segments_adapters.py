import sys

import numpy as np
import torch

from tsfm.data import normalize_context_target
from tsfm.s3.adapters import DatasetSpec, LOTSAAdapter, UTSDAdapter
from tsfm.s3.records import RawSeries
from tsfm.s3.segments import finite_univariate_segments


class FakeLoader:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    def __call__(self, repository, configuration, *, split, streaming, cache_dir):
        self.calls.append((repository, configuration, split, streaming, cache_dir))
        return self.rows


def test_multivariate_series_becomes_finite_channel_segments() -> None:
    values = np.column_stack(
        (np.arange(8, dtype=np.float64), np.arange(8, dtype=np.float64) + 10)
    )
    values[3, 0] = np.nan
    raw = RawSeries("utsd", "weather", "station-1", values, "H")

    segments = list(
        finite_univariate_segments(raw, min_length=3, min_variance=1e-8)
    )

    assert [
        (segment.channel_id, segment.segment_index, segment.values.tolist())
        for segment in segments
    ] == [
        ("0", 0, [0.0, 1.0, 2.0]),
        ("0", 1, [4.0, 5.0, 6.0, 7.0]),
        ("1", 0, [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0]),
    ]
    assert all(segment.values.dtype == np.float32 for segment in segments)


def test_short_constant_and_nonfinite_segments_are_rejected() -> None:
    raw = RawSeries(
        "lotsa",
        "energy",
        "meter-1",
        np.array([1, 1, 1, np.inf, 0, 1], dtype=np.float64),
        None,
    )

    assert list(
        finite_univariate_segments(raw, min_length=3, min_variance=1e-8)
    ) == []


def test_utsd_adapter_maps_release_fields_without_eager_import(tmp_path) -> None:
    sys.modules.pop("datasets", None)
    loader = FakeLoader(
        [{"item_id": "station-7", "target": [1, 2, 3], "freq": "H"}]
    )
    spec = DatasetSpec("thuml/UTSD", "UTSD-1G", "utsd", "weather")

    rows = list(UTSDAdapter(spec, tmp_path, loader=loader).iter_series())

    assert rows[0].series_id == "station-7"
    np.testing.assert_array_equal(rows[0].values, np.array([1, 2, 3]))
    assert loader.calls == [
        ("thuml/UTSD", "UTSD-1G", "train", True, tmp_path)
    ]
    assert "datasets" not in sys.modules


def test_lotsa_adapter_normalizes_channel_first_target(tmp_path) -> None:
    loader = FakeLoader(
        [
            {
                "id": "station-2",
                "target": [[1, 2, 3, 4], [5, 6, 7, 8]],
                "frequency": "H",
            }
        ]
    )
    spec = DatasetSpec(
        "Salesforce/lotsa_data", "air", "lotsa", "air"
    )

    row = next(LOTSAAdapter(spec, tmp_path, loader=loader).iter_series())
    segments = list(finite_univariate_segments(row, min_length=4))

    assert row.series_id == "station-2"
    assert row.values.shape == (4, 2)
    assert row.frequency == "H"
    assert [segment.values.tolist() for segment in segments] == [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
    ]


def test_lotsa_adapter_preserves_univariate_target(tmp_path) -> None:
    loader = FakeLoader([{"item_id": "road-1", "target": [1, 2, 3, 4]}])
    spec = DatasetSpec(
        "Salesforce/lotsa_data", "traffic", "lotsa", "traffic"
    )

    row = next(LOTSAAdapter(spec, tmp_path, loader=loader).iter_series())

    assert row.values.shape == (4,)
    np.testing.assert_array_equal(row.values, np.array([1, 2, 3, 4]))


def test_normalization_uses_only_context_population_variance() -> None:
    context = torch.tensor([[1.0, 3.0, 5.0]])
    first = normalize_context_target(context, torch.tensor([[7.0, 9.0]]))
    second = normalize_context_target(
        context, torch.tensor([[7000.0, -9000.0]])
    )
    expected_scale = torch.sqrt(
        context.var(dim=-1, unbiased=False, keepdim=True) + 1e-5
    )

    torch.testing.assert_close(first.mean, torch.tensor([[3.0]]))
    torch.testing.assert_close(first.scale, expected_scale)
    torch.testing.assert_close(second.mean, first.mean)
    torch.testing.assert_close(second.scale, first.scale)
