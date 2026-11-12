# Getting started

## Install

You need [uv](https://docs.astral.sh/uv/) and Git.

```bash
git clone https://github.com/openic-org/NeuroPyHLS.git
cd NeuroPyHLS
uv sync
uv run neuropyhls --help
```

The project uses VS Code as its default editor: open the folder and accept the recommended extensions.

For RTL simulation, install [Verilator](https://www.veripool.org/verilator/). For the baseline flow, also install [Bambu](https://panda.dei.polimi.it/). Vitis HLS is optional. Put them on your `PATH`.

## Reference code and recordings

```bash
uv run neuropyhls fetch-sources   # clone pinned reference repos (examples/sources.toml) into external/
uv run neuropyhls fetch           # download recordings (examples/datasets.toml) into data/, checking SHA-256
```

## Explore a dataset

From Python, IPython or a notebook cell in VS Code:

```python
from neuropyhls import data

data.datasets()                       # registered datasets, and which are downloaded
data.items("quiroga-simulator")       # recordings inside the dataset (downloads on first use)
rec = data.load("quiroga-simulator", "C_Easy1_noise005")
rec                                   # Recording(quiroga-simulator/C_Easy1_noise005: 1 ch, 24000 Hz, 60 s, ...)
rec.traces, rec.fs                    # numpy (samples, channels) and sampling rate
rec.recording, rec.truth              # SpikeInterface recording and ground-truth sorting
```

`rec.recording` and `rec.truth` are ordinary SpikeInterface objects, so SpikeInterface's filters, peak detection, ground-truth comparison and plots work on them directly. Datasets are found from `examples/datasets.toml` at the project root, from any subfolder; set `NEUROPYHLS_ROOT` or `NEUROPYHLS_DATA` to use other folders.

## Run the tests

```bash
uv run pytest -m unit    # fast, no external tools
uv run pytest -m bambu   # needs Bambu and Verilator
```
