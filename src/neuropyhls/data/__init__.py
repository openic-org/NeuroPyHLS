"""Explore registered recordings from Python, IPython or a notebook.

```python
from neuropyhls import data

data.datasets()                                   # what is registered, and what is downloaded
data.items("quiroga-simulator")                   # recordings inside a dataset
rec = data.load("quiroga-simulator", "C_Easy1_noise005")
rec.recording, rec.truth                          # SpikeInterface recording and ground truth
rec.traces, rec.fs                                # numpy samples x channels, Hz
data.plot(rec, t0=1.0, dur=0.05)                  # trace with ground-truth spikes (matplotlib)
```

Datasets are listed in ``examples/datasets.toml`` and downloaded into ``data/`` at the project
root, found by walking up from the current folder. Set ``NEUROPYHLS_ROOT`` to point elsewhere,
or ``NEUROPYHLS_DATA`` to keep downloads in another folder.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from neuropyhls.data.plot import plot
from neuropyhls.data.readers import READERS
from neuropyhls.datasets import Dataset, fetch_dataset, load_datasets, sha256_of

MANIFEST = Path("examples") / "datasets.toml"

__all__ = ["DatasetInfo", "Recording", "datasets", "fetch", "info", "items", "load", "plot"]


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    license: str
    downloaded: bool
    path: Path
    reader: str
    note: str
    url: str


@dataclass(frozen=True)
class Recording:
    """One recording: the SpikeInterface objects plus where they came from."""

    dataset: str
    item: str
    recording: Any  # spikeinterface BaseRecording
    truth: Any  # spikeinterface BaseSorting, or None without ground truth
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def fs(self) -> float:
        return float(self.recording.get_sampling_frequency())

    @property
    def traces(self) -> npt.NDArray[np.float64]:
        """All samples as a (samples, channels) array, in the recording's units."""
        out: npt.NDArray[np.float64] = self.recording.get_traces(return_in_uV=False)
        return out

    @property
    def duration(self) -> float:
        return float(self.recording.get_total_duration())

    def __repr__(self) -> str:
        n_units = 0 if self.truth is None else len(self.truth.get_unit_ids())
        return (
            f"Recording({self.dataset}/{self.item}: {self.recording.get_num_channels()} ch, "
            f"{self.fs:g} Hz, {self.duration:g} s, ground truth: {n_units} units)"
        )


def project_root() -> Path:
    """The folder holding ``examples/datasets.toml``: ``NEUROPYHLS_ROOT`` or the nearest parent."""
    env = os.environ.get("NEUROPYHLS_ROOT")
    if env:
        return Path(env)
    here = Path.cwd().resolve()
    for folder in (here, *here.parents):
        if (folder / MANIFEST).is_file():
            return folder
    raise FileNotFoundError(
        f"no {MANIFEST} found in {here} or its parents; run inside the NeuroPyHLS repository "
        "or set NEUROPYHLS_ROOT"
    )


def _data_root() -> Path:
    env = os.environ.get("NEUROPYHLS_DATA")
    return Path(env) if env else project_root() / "data"


def _registry() -> dict[str, Dataset]:
    return {ds.name: ds for ds in load_datasets(project_root() / MANIFEST)}


def _get(name: str) -> Dataset:
    reg = _registry()
    if name not in reg:
        raise KeyError(f"unknown dataset {name!r}; registered: {', '.join(reg) or 'none'}")
    return reg[name]


def _local(ds: Dataset) -> Path:
    return _data_root() / ds.name / ds.filename


def _info(ds: Dataset) -> DatasetInfo:
    path = _local(ds)
    return DatasetInfo(ds.name, ds.license, path.is_file(), path, ds.reader, ds.note, ds.url)


def datasets() -> list[DatasetInfo]:
    """All registered datasets and whether each one is downloaded."""
    return [_info(ds) for ds in _registry().values()]


def info(name: str) -> DatasetInfo:
    """Details of one registered dataset."""
    return _info(_get(name))


def fetch(name: str) -> Path:
    """Download a dataset (if needed), check its SHA-256 and return the local file."""
    return fetch_dataset(_get(name), _data_root())


def _ensure(ds: Dataset, download: bool) -> Path:
    path = _local(ds)
    if path.is_file():
        return path
    if not download:
        raise FileNotFoundError(f"{ds.name} is not downloaded; run data.fetch({ds.name!r})")
    print(f"downloading {ds.name} from {ds.url} ...")
    return fetch_dataset(ds, _data_root())


def _reader(ds: Dataset) -> Any:
    if ds.reader not in READERS:
        raise ValueError(f"{ds.name}: no reader {ds.reader!r}; known: {', '.join(READERS)}")
    return READERS[ds.reader]


def items(name: str, *, download: bool = True) -> list[str]:
    """Recordings available inside a dataset."""
    ds = _get(name)
    result: list[str] = _reader(ds).items(_ensure(ds, download))
    return result


def load(name: str, item: str, *, download: bool = True, verify: bool = False) -> Recording:
    """Load one recording, downloading the dataset on first use.

    ``verify=True`` re-checks the SHA-256 of an existing download (slow for large files).
    """
    ds = _get(name)
    path = _ensure(ds, download)
    if verify and sha256_of(path) != ds.sha256:
        raise ValueError(f"{name}: {path} does not match the registered SHA-256")
    rec, truth, meta = _reader(ds).read(path, item)
    meta = {"source": ds.url, "license": ds.license, **meta}
    return Recording(name, item, rec, truth, meta)
