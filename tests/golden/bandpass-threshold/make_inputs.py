"""Write the input golden vectors of the bandpass-threshold kernel from config.toml.

Run from the repository root (downloads the dataset on first use):

    uv run python tests/golden/bandpass-threshold/make_inputs.py
"""

import tomllib
from pathlib import Path

import numpy as np

from neuropyhls import data

HERE = Path(__file__).resolve().parent

SOURCE = (
    "Martinez J, Pedreira C, Ison MJ, Quian Quiroga R (2009). Realistic simulation of "
    "extracellular recordings. J Neurosci Methods 184:285-293, "
    "doi:10.1016/j.jneumeth.2009.08.017. Data: University of Leicester, "
    "https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595"
)
LICENSE = "CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)"


def main() -> None:
    cfg = tomllib.loads((HERE / "config.toml").read_text())
    q = cfg["quantization"]
    for inp in cfg["input"]:
        item = Path(inp["member"]).stem
        rec = data.load(inp["dataset"], item)
        fs = rec.fs
        start = round(inp["start_s"] * fs)
        n = round(inp["duration_s"] * fs)
        seg = rec.traces[start : start + n, 0]

        spikes = rec.truth.to_spike_vector()  # sorted by sample; unit_index -> unit_ids
        st = spikes["sample_index"].astype(np.int64)
        sc = np.asarray(rec.truth.get_unit_ids())[spikes["unit_index"]].astype(np.int64)
        keep = (st >= start) & (st < start + n)

        lsb = 2 * q["full_scale"] / 2 ** q["bits"]
        lim = 2 ** (q["bits"] - 1)
        xq = np.clip(np.rint(seg / lsb), -lim, lim - 1).astype(np.int16)

        changes = (
            f"Excerpt of {inp['member']} from {inp['start_s']} s for {inp['duration_s']} s; "
            f"quantized to {q['bits']}-bit signed integers, full scale +/-{q['full_scale']} a.u., "
            f"{q['rounding']} rounding, {q['overflow']} on overflow. Spike times shifted to the "
            "start of the excerpt."
        )
        np.savez_compressed(
            HERE / f"{inp['name']}.npz",
            x=seg,
            x_q=xq,
            lsb=np.float64(lsb),
            fs=np.float64(fs),
            spike_times=st[keep] - start,
            spike_class=sc[keep],
            source=np.str_(SOURCE),
            license=np.str_(LICENSE),
            changes=np.str_(changes),
        )
        print(f"{inp['name']}: {n} samples, {keep.sum()} spikes, LSB {lsb:.3e} a.u.")


if __name__ == "__main__":
    main()
