"""Format readers: turn a downloaded dataset file into SpikeInterface objects.

A reader lists the items (recordings) inside a downloaded file and reads one of them as a
SpikeInterface recording plus, when the dataset has one, its ground-truth sorting.
Register a new format by adding a ``Reader`` to ``READERS`` and naming it in the dataset's
``reader`` field in ``examples/datasets.toml``.
"""

import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import scipy.io as sio
from spikeinterface.core import NumpyRecording, NumpySorting

ReadResult = tuple[Any, Any, dict[str, Any]]  # (BaseRecording, BaseSorting | None, metadata)


@dataclass(frozen=True)
class Reader:
    items: Callable[[Path], list[str]]
    read: Callable[[Path, str], ReadResult]


# --- Quiroga simulator (Martinez et al. 2009): a zip of wave_clus simulator .mat files ---


def _quiroga_members(archive: Path) -> dict[str, str]:
    """Map item name (file stem) to zip member, skipping wave_clus outputs (``times_*``)."""
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
    out = {}
    for name in names:
        p = PurePosixPath(name)
        if p.suffix == ".mat" and p.name.startswith("C_") and "__MACOSX" not in p.parts:
            out[p.stem] = name
    return dict(sorted(out.items()))


def _quiroga_items(archive: Path) -> list[str]:
    return list(_quiroga_members(archive))


def _quiroga_read(archive: Path, item: str) -> ReadResult:
    members = _quiroga_members(archive)
    if item not in members:
        raise KeyError(f"{item!r} is not in {archive.name}; available: {', '.join(members)}")
    with zipfile.ZipFile(archive) as z, z.open(members[item]) as f:
        m = sio.loadmat(f)

    fs = 1000.0 / float(np.ravel(m["samplingInterval"])[0])  # samplingInterval is in ms
    x = np.ravel(m["data"]).astype(np.float64)
    rec = NumpyRecording([x[:, np.newaxis]], sampling_frequency=fs, channel_ids=["ch0"])
    # The simulator is in arbitrary units (a.u.); gain 1 keeps them as they are.
    rec.set_channel_gains(1.0)
    rec.set_channel_offsets(0.0)

    samples = np.ravel(m["spike_times"][0, 0]).astype(np.int64)
    labels = np.ravel(m["spike_class"][0, 0]).astype(np.int64)
    truth = NumpySorting.from_samples_and_labels([samples], [labels], fs)

    meta = {
        "member": members[item],
        "units": "a.u.",
        "truth_alignment": "onset",
        "truth_note": "Ground-truth times mark the waveform onset and lead the peak by ~0.9 ms.",
    }
    return rec, truth, meta


READERS: dict[str, Reader] = {
    "quiroga-mat": Reader(items=_quiroga_items, read=_quiroga_read),
}
