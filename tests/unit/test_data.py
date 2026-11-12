import zipfile
from pathlib import Path

import numpy as np
import pytest
import scipy.io as sio

from neuropyhls import data
from neuropyhls.datasets import sha256_of

FS = 24_000.0


def _cells(*arrays: object) -> np.ndarray:
    """A 1 x n MATLAB cell array, as the simulator files store spike_times and spike_class."""
    out = np.empty((1, len(arrays)), dtype=object)
    for i, a in enumerate(arrays):
        out[0, i] = np.asarray(a)
    return out


def _make_project(root: Path) -> np.ndarray:
    """A fake project root with a one-dataset manifest and a downloaded Quiroga-style zip."""
    rng = np.random.default_rng(0)
    x = rng.normal(0, 0.1, 4_800)
    mat = root / "C_Test_noise01.mat"
    sio.savemat(
        mat,
        {
            "data": x[np.newaxis, :],
            "samplingInterval": np.array([[1000.0 / FS]]),
            "spike_times": _cells([[100, 900, 2000]]),
            "spike_class": _cells([[1, 2, 1]], np.zeros((1, 3)), np.zeros((1, 3))),
        },
    )
    archive = root / "data" / "fake" / "Sim.zip"
    archive.parent.mkdir(parents=True)
    with zipfile.ZipFile(archive, "w") as z:
        z.write(mat, "Simulator/C_Test_noise01.mat")
        z.writestr("Simulator/times_C_Test_noise01.mat", b"")  # wave_clus output: skipped
        z.writestr("__MACOSX/Simulator/C_Test_noise01.mat", b"")  # macOS metadata: skipped
    (root / "examples").mkdir()
    (root / "examples" / "datasets.toml").write_text(
        f"""[[dataset]]
name = "fake"
url = "https://example.org/Sim.zip"
sha256 = "{sha256_of(archive)}"
filename = "Sim.zip"
license = "CC-BY-4.0"
reader = "quiroga-mat"
"""
    )
    return x


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> np.ndarray:
    x = _make_project(tmp_path)
    monkeypatch.delenv("NEUROPYHLS_ROOT", raising=False)
    monkeypatch.delenv("NEUROPYHLS_DATA", raising=False)
    (tmp_path / "notebooks").mkdir()
    monkeypatch.chdir(tmp_path / "notebooks")  # the root is found by walking up
    return x


def test_datasets_and_info(project: np.ndarray) -> None:
    [d] = data.datasets()
    assert d.name == "fake" and d.downloaded and d.reader == "quiroga-mat"
    assert data.info("fake") == d


def test_items_skip_non_recordings(project: np.ndarray) -> None:
    assert data.items("fake") == ["C_Test_noise01"]


def test_load(project: np.ndarray) -> None:
    rec = data.load("fake", "C_Test_noise01", verify=True)
    assert rec.fs == FS and rec.duration == pytest.approx(0.2)
    np.testing.assert_array_equal(rec.traces[:, 0], project)
    assert list(rec.truth.get_unit_ids()) == [1, 2]
    np.testing.assert_array_equal(rec.truth.get_unit_spike_train(1), [100, 2000])
    assert rec.meta["license"] == "CC-BY-4.0" and rec.meta["truth_alignment"] == "onset"
    assert "fake/C_Test_noise01" in repr(rec)


def test_errors(project: np.ndarray, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown dataset"):
        data.load("nope", "x")
    with pytest.raises(KeyError, match="available"):
        data.load("fake", "C_Missing")
    monkeypatch.setenv("NEUROPYHLS_DATA", str(tmp_path / "empty"))
    assert not data.info("fake").downloaded
    with pytest.raises(FileNotFoundError, match="not downloaded"):
        data.load("fake", "C_Test_noise01", download=False)


def test_no_project_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NEUROPYHLS_ROOT", raising=False)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match="NEUROPYHLS_ROOT"):
        data.datasets()
