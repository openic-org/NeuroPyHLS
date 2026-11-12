"""Quick-look plot of a loaded recording, with its ground-truth spikes marked.

matplotlib is imported inside :func:`plot`, so ``neuropyhls.data`` works without it. It comes
with the ``dev`` dependency group that ``uv sync`` installs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from neuropyhls.data import Recording


def plot(
    rec: Recording,
    t0: float = 0.0,
    dur: float | None = 0.1,
    *,
    channel: int = 0,
    truth: bool = True,
    ax: Axes | None = None,
    **line_kw: Any,
) -> Axes:
    """Plot ``dur`` seconds of one channel from ``t0`` (``dur=None``: to the end).

    With ``truth=True`` the ground-truth spike times inside the window are marked above the
    trace, one colour per unit. Returns the matplotlib ``Axes`` so the plot can be adjusted.
    Extra keyword arguments go to ``Axes.plot`` for the trace.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover - depends on the environment
        raise ImportError(
            "data.plot needs matplotlib; run `uv sync` (it is in the dev group)"
        ) from e

    si = rec.recording
    n_total = si.get_num_samples()
    start = min(max(round(t0 * rec.fs), 0), n_total)
    stop = n_total if dur is None else min(start + max(round(dur * rec.fs), 1), n_total)
    if stop <= start:
        raise ValueError(f"empty window: t0={t0} s is past the end ({rec.duration:g} s)")
    ch_id = si.get_channel_ids()[channel]
    y = si.get_traces(start_frame=start, end_frame=stop, channel_ids=[ch_id], return_in_uV=False)
    y = np.asarray(y[:, 0], dtype=np.float64)
    t = (start + np.arange(stop - start)) / rec.fs

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3), layout="constrained")
    ax.plot(t, y, lw=line_kw.pop("lw", 0.6), color=line_kw.pop("color", "0.25"), **line_kw)

    if truth and rec.truth is not None:
        span = float(np.ptp(y)) or 1.0
        level = float(y.max()) + 0.08 * span
        colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        for i, unit in enumerate(rec.truth.get_unit_ids()):
            s = np.asarray(rec.truth.get_unit_spike_train(unit, segment_index=0))
            s = s[(s >= start) & (s < stop)]
            if s.size:
                ax.plot(
                    s / rec.fs,
                    np.full(s.size, level),
                    "v",
                    ms=5,
                    color=colours[i % len(colours)],
                    label=f"unit {unit} ({s.size})",
                    linestyle="none",
                )
        if ax.get_legend_handles_labels()[0]:
            align = rec.meta.get("truth_alignment", "")
            ax.legend(
                title=f"ground truth{f' ({align})' if align else ''}",
                loc="lower right",
                bbox_to_anchor=(1.0, 1.0),  # above the axes, beside the title
                fontsize="small",
                title_fontsize="small",
                ncols=4,
                frameon=False,
            )

    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel(f"amplitude ({rec.meta.get('units', '?')})")
    ax.set_title(f"{rec.dataset} / {rec.item}  ·  ch {ch_id}  ·  {rec.fs:g} Hz", loc="left")
    return ax
