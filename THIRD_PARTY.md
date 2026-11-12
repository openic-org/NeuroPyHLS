# Third-party software and code

## External tools (not included, linked, or redistributed)

NeuroPyHLS runs these as separate programs that users install themselves. Their licences do not extend to NeuroPyHLS or to the output it generates.

| Tool | Licence | Use |
|---|---|---|
| [Bambu (PandA)](https://panda.dei.polimi.it/) | GPL-3.0 | Baseline flow, primary tool: HLS C++ → Verilog |
| [AMD Vitis HLS](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis/vitis-hls.html) | Proprietary (AMD) | Baseline flow, secondary tool |
| [Verilator](https://www.veripool.org/verilator/) | LGPL-3.0 / Artistic-2.0 | RTL simulation of both flows |
| [cocotb](https://www.cocotb.org/) | BSD-3-Clause | Python testbenches for RTL (optional `rtl` dependency group) |

## Reference repositories (cloned into `external/`, not committed)

These are listed in `examples/sources.toml`.

| Repository | Licence | Use |
|---|---|---|
| [SpikeInterface](https://github.com/SpikeInterface/spikeinterface) | MIT | Reference implementations, golden vectors, code-db input |

## Data excerpts committed in this repository

Short excerpts of public recordings, used as golden-vector inputs. Each folder's `README.md` gives the full attribution and the changes made. The excerpts keep their own license.

| Files in this repo | Source | Licence | Changes |
|---|---|---|---|
| `tests/golden/bandpass-threshold/quiroga_*.npz` | Quiroga simulator: Martinez et al. 2009, [figshare 11897595](https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595) | CC-BY-4.0 | 2 s excerpt, quantized to int16 |

## Code ported into this repository

When code is ported or adapted from a permissively licensed source, add a row and keep the original copyright notice in the file header.

| File in this repo | Source (repo, commit, file) | Licence | Notes |
|---|---|---|---|
| none yet | | | |
