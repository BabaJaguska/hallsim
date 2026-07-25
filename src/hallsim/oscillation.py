"""Grid-independent oscillation readouts for solved trajectories.

A period measured as the spacing of save-grid local maxima is quantised to
``save_dt``: with a 0.02 d grid a 6.82 h oscillator reads 6.72 h or 7.20 h
and a *deterministic* cell shows a ~3% period CV that is entirely grid
snapping. Both readouts here refine each extremum by fitting a parabola to
its three samples, so they resolve well below one save step, and both are
vectorised over a trailing batch axis (one population, one call).

Trajectories are time-first, matching ``SolveResult.ys`` and the
:mod:`hallsim.gene_reporters` summaries: ``y`` has shape ``(n_time, ...)``
on a uniform ``ts`` grid.
"""

from __future__ import annotations

import jax.numpy as jnp

__all__ = ["peak_mask", "dominant_period", "coherence_curve"]


def peak_mask(y, height: float = 0.15):
    """Interior local maxima above ``height`` of each series' own range.

    Returns a boolean array of ``y``'s shape (endpoints always False), so it
    broadcasts against ``y`` and sums to a per-series peak count.
    """
    lo, hi = y.min(0), y.max(0)
    interior = (
        (y[1:-1] > y[:-2]) & (y[1:-1] > y[2:]) & (y[1:-1] > lo + height * (hi - lo))
    )
    pad = jnp.zeros_like(y[:1], dtype=bool)
    return jnp.concatenate([pad, interior, pad], 0)


def _parabolic_shift(y, idx):
    """Sub-sample offset (in grid steps) of the true maximum near ``idx``."""
    take = lambda i: jnp.take_along_axis(y, i[None], 0)[0]  # noqa: E731
    a, b, c = take(idx - 1), take(idx), take(idx + 1)
    denom = a - 2.0 * b + c
    return jnp.where(denom == 0.0, 0.0, 0.5 * (a - c) / denom)


def dominant_period(ts, y, height: float = 0.15):
    """Mean peak-to-peak period of each series in ``y``, in ``ts`` units.

    Measured across the whole record — the first and last peak's refined
    times divided by the number of intervals between them — so the estimate
    averages out per-cycle jitter instead of inheriting it. Series with
    fewer than two qualifying peaks return NaN.
    """
    dt = ts[1] - ts[0]
    m = peak_mask(y, height)
    count = m.sum(0)
    first = jnp.argmax(m, 0)
    last = m.shape[0] - 1 - jnp.argmax(m[::-1], 0)
    span = (last - first) * dt + (
        _parabolic_shift(y, last) - _parabolic_shift(y, first)
    ) * dt
    return jnp.where(count > 1, span / jnp.maximum(count - 1, 1), jnp.nan)


def coherence_curve(ts, y, window: float):
    """Bulk-to-single-cell amplitude ratio in sliding windows.

    ``y`` is ``(n_time, n_cells)``. In each window the amplitude is the
    peak-to-trough range: 1 means the population is in phase (the mean
    oscillates as hard as its cells), and it decays to 0 as the cells
    dephase and the mean flattens. Windows are half-overlapping and
    ``window`` wide in ``ts`` units.

    Returns ``(centers, coherence, single_cell_amplitude, bulk_amplitude)``.
    """
    dt = ts[1] - ts[0]
    w = max(3, int(round(window / dt)))
    step = max(1, w // 2)
    n = 1 + (y.shape[0] - w) // step
    if n < 1:
        raise ValueError(
            f"window {window} exceeds the {ts[-1] - ts[0]} span of ts"
        )
    idx = jnp.arange(n)[:, None] * step + jnp.arange(w)[None, :]
    win = y[idx]  # (n_windows, w, n_cells)
    bulk = y.mean(1)[idx]  # (n_windows, w)

    single_amp = (win.max(1) - win.min(1)).mean(-1)
    bulk_amp = bulk.max(1) - bulk.min(1)
    centers = ts[idx].mean(1)
    coherence = jnp.where(single_amp > 0, bulk_amp / single_amp, jnp.nan)
    return centers, coherence, single_amp, bulk_amp
