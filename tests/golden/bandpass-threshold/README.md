# Golden vectors: bandpass + threshold-crossing detection

Settings for this kernel (input excerpts, quantization, tolerances) are in [`config.toml`](config.toml).

## Inputs

| File | Contents |
|---|---|
| `quiroga_easy1_noise01_2s.npz` | 2 s (48 000 samples at 24 kHz) of `C_Easy1_noise01`; 126 ground-truth spikes |

Arrays in each `.npz`:
- `x`: the original samples (float64, a.u.).
- `x_q`: the same samples as int16.
- `lsb`: the value of one LSB in a.u.
- `fs`: the sampling rate in Hz.
- `spike_times`: ground-truth spike times, in samples from the start of the excerpt. They lead the spike peak by about 0.9 ms.
- `spike_class`: the neuron class of each spike (1-3).
- `source`, `license`, `changes`: attribution text.

Load with `np.load(path, allow_pickle=False)`.

To regenerate, from the repository root:

```bash
uv run neuropyhls fetch quiroga-simulator
uv run python tests/golden/bandpass-threshold/make_inputs.py
```

Outputs from the Python reference will be added once the reference is fixed (see `docs/kernels/bandpass-threshold.md`).

## Source and license of the input excerpts

The input excerpts are adapted from the **Quiroga simulator dataset**:

> Martinez J, Pedreira C, Ison MJ, Quian Quiroga R (2009). Realistic simulation of extracellular recordings. *Journal of Neuroscience Methods* 184:285–293. [doi:10.1016/j.jneumeth.2009.08.017](https://doi.org/10.1016/j.jneumeth.2009.08.017).
> Data: University of Leicester, [figshare 11897595](https://figshare.le.ac.uk/articles/dataset/Simulated_dataset/11897595).

They are licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), not under the repository's Apache-2.0 license.

**Changes made:**
- Cut to the time window given in `config.toml`.
- Quantized to 16-bit signed integers (±4 a.u. full scale, nearest rounding, saturation). The original samples are kept in `x`.
- Spike times shifted to the start of the excerpt.

The original data are provided as-is, without warranty.
