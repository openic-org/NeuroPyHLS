# %% [markdown]
# # Exploring the Quiroga simulator
#
# Plot any recording of the Quiroga spike-sorting simulator (Quiroga et al. 2004; Martinez et
# al. 2009; figshare 11897595, CC BY 4.0) together with its ground-truth spikes.
#
# **Before the first run:** `uv sync` (installs matplotlib and ipykernel), then optionally
# `uv run neuropyhls fetch quiroga-simulator`. Without the fetch, the first `data.load` downloads
# the dataset (~0.5 GB) into `data/` and checks its SHA-256.
#
# In VS Code, run the cells with **Shift+Enter**; plots appear in the Interactive Window.

# %%
import matplotlib.pyplot as plt
import numpy as np
import spikeinterface.preprocessing as spre

from neuropyhls import data

data.info("quiroga-simulator")

# %% [markdown]
# ## Which recordings are there?
#
# Names read `C_<difficulty>_noise<level>`: `noise005` is a noise level of 0.05, relative to the
# peak spike amplitude. Each has three units and 60 s at 24 kHz.

# %%
names = data.items("quiroga-simulator")
print(len(names), "recordings:")
print(*names, sep="\n")

# %% [markdown]
# ## Load one and plot it
#
# Change `ITEM` to any name from the list above. `rec.traces` is a numpy array
# (samples, channels); `rec.recording` and `rec.truth` are SpikeInterface objects.

# %%
ITEM = "C_Easy1_noise01"

rec = data.load("quiroga-simulator", ITEM)
print(rec)
print("truth spikes per unit:", rec.truth.count_num_spikes_per_unit())

# %%
ax = data.plot(rec, t0=1.0, dur=0.1)

# %% [markdown]
# The triangles mark ground-truth spike times. The simulator stores the **onset** of each spike,
# so the peak comes about 1 ms after the marker (`rec.meta["truth_alignment"]`).
#
# A longer window, and the full recording:

# %%
fig, axs = plt.subplots(2, 1, figsize=(10, 6), layout="constrained")
data.plot(rec, t0=0, dur=2.0, ax=axs[0])
data.plot(rec, dur=None, truth=False, ax=axs[1])
plt.show()

# %% [markdown]
# ## Same recording, increasing noise
#
# `ax=` puts several recordings on one figure.

# %%
levels = ["C_Easy1_noise005", "C_Easy1_noise01", "C_Easy1_noise02", "C_Easy1_noise04"]
fig, axs = plt.subplots(len(levels), 1, figsize=(10, 9), sharex=True, layout="constrained")
for ax, name in zip(axs, levels, strict=True):
    data.plot(data.load("quiroga-simulator", name), t0=1.0, dur=0.1, ax=ax)
    ax.label_outer()
plt.show()

# %% [markdown]
# ## Spike waveforms per unit
#
# Cut a 2.5 ms snippet around each ground-truth onset and overlay them. This uses only numpy;
# SpikeInterface's `create_sorting_analyzer` does the same with more options.

# %%
x = rec.traces[:, 0]
pre, post = int(0.0005 * rec.fs), int(0.002 * rec.fs)
t_ms = np.arange(-pre, post) / rec.fs * 1e3

units = rec.truth.get_unit_ids()
fig, axs = plt.subplots(1, len(units), figsize=(10, 3), sharey=True, layout="constrained")
for ax, unit in zip(axs, units, strict=True):
    onsets = rec.truth.get_unit_spike_train(unit, segment_index=0)
    onsets = onsets[(onsets >= pre) & (onsets + post <= x.size)]
    snippets = np.stack([x[s - pre : s + post] for s in onsets])
    ax.plot(t_ms, snippets[:200].T, color="0.6", lw=0.3)
    ax.plot(t_ms, snippets.mean(axis=0), color="C0", lw=2)
    ax.set_title(f"unit {unit}  (n = {len(onsets)})")
    ax.set_xlabel("ms from onset")
axs[0].set_ylabel(f"amplitude ({rec.meta['units']})")
plt.show()

# %% [markdown]
# ## SpikeInterface tools work directly
#
# For example, the same bandpass filter used later as the hardware reference:

# %%
filtered = spre.bandpass_filter(rec.recording, freq_min=300, freq_max=3000)
ax = data.plot(rec, t0=1.0, dur=0.1, truth=False, label="raw")
fx = filtered.get_traces(start_frame=24_000, end_frame=26_400, return_in_uV=False)[:, 0]
ax.plot(1.0 + np.arange(fx.size) / rec.fs, fx, lw=0.6, color="C3", label="300-3000 Hz")
ax.legend(loc="upper right", fontsize="small")
plt.show()
