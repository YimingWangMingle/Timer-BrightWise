from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tsfm.s3.records import RawSeries

DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]
FileDownloader = Callable[..., str | Path]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    repository: str
    configuration: str
    source_id: str
    dataset_id: str
    split: str = "train"
    revision: str | None = None
    file_format: str | None = None
    data_files: tuple[str, ...] = ()


class _HuggingFaceAdapter:
    id_fields = ("item_id", "id")
    frequency_fields = ("freq", "frequency")

    def __init__(
        self,
        spec: DatasetSpec,
        cache_dir: str | Path,
        loader: DatasetLoader | None = None,
        downloader: FileDownloader | None = None,
    ) -> None:
        self.spec = spec
        self.cache_dir = Path(cache_dir)
        self.loader = loader
        self.downloader = downloader

    def _dataset_loader(self) -> DatasetLoader:
        if self.loader is not None:
            return self.loader
        from datasets import load_dataset

        return load_dataset

    def _rows(self) -> Iterable[Mapping[str, Any]]:
        return self._dataset_loader()(
            self.spec.repository,
            self.spec.configuration,
            split=self.spec.split,
            streaming=True,
            cache_dir=self.cache_dir,
        )

    def _target_values(self, target: object) -> np.ndarray:
        return np.asarray(target)

    def iter_series(self) -> Iterator[RawSeries]:
        for index, row in enumerate(self._rows()):
            series_id = next(
                (
                    str(row[field])
                    for field in self.id_fields
                    if row.get(field) is not None
                ),
                str(index),
            )
            frequency = next(
                (
                    str(row[field])
                    for field in self.frequency_fields
                    if row.get(field) is not None
                ),
                None,
            )
            if "target" not in row:
                raise ValueError(
                    f"{self.spec.source_id}/{self.spec.dataset_id}/{series_id}: "
                    "missing target"
                )
            yield RawSeries(
                source_id=self.spec.source_id,
                dataset_id=self.spec.dataset_id,
                series_id=series_id,
                values=self._target_values(row["target"]),
                frequency=frequency,
            )


class UTSDAdapter(_HuggingFaceAdapter):
    pass


class LOTSAAdapter(_HuggingFaceAdapter):
    def _target_values(self, target: object) -> np.ndarray:
        values = np.asarray(target)
        if values.ndim == 2:
            return values.T
        return values

    def _file_downloader(self) -> FileDownloader:
        if self.downloader is not None:
            return self.downloader
        from huggingface_hub import hf_hub_download

        return hf_hub_download

    def _rows(self) -> Iterable[Mapping[str, Any]]:
        if not self.spec.data_files:
            return super()._rows()
        if self.spec.revision is None or self.spec.file_format is None:
            raise ValueError(
                f"{self.spec.source_id}/{self.spec.dataset_id}: "
                "exact files require revision and file_format"
            )

        downloader = self._file_downloader()
        local_files = [
            str(
                downloader(
                    repo_id=self.spec.repository,
                    filename=filename,
                    repo_type="dataset",
                    revision=self.spec.revision,
                    cache_dir=self.cache_dir,
                )
            )
            for filename in self.spec.data_files
        ]
        return self._dataset_loader()(
            self.spec.file_format,
            data_files={self.spec.split: local_files},
            split=self.spec.split,
            streaming=True,
            cache_dir=self.cache_dir,
        )
