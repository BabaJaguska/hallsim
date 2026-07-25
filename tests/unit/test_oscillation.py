"""Grid-independence tests for :mod:`hallsim.oscillation`.

The estimators exist because reading peak spacing straight off the save grid
quantises every period to a multiple of ``save_dt``: on a 0.02 d grid a
6.82 h oscillator reads 6.72 or 7.20 h, and a deterministic cell shows a
~3% period CV that is pure grid snapping. These tests pin both the accuracy
and that separation against synthetic populations with known answers.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from hallsim.oscillation import coherence_curve, dominant_period, peak_mask

SAVE_DT = 0.02
T_END = 15.0
TRUE_PERIOD = 0.2842  # the composite's p53 period, 6.82 h


def population(n=200, period_cv=0.03, seed=0):
    """``(ts, y)`` for ``n`` sinusoidal cells with lognormal-free, known
    periods; ``y`` is time-first ``(n_time, n_cells)``."""
    rng = np.random.default_rng(seed)
    ts = np.arange(0.0, T_END, SAVE_DT)
    periods = TRUE_PERIOD * (1 + period_cv * rng.standard_normal(n))
    y = np.sin(2 * np.pi * ts[:, None] / periods[None, :]) + 1.5
    return jnp.asarray(ts), jnp.asarray(y), periods


def grid_snapped(ts, y):
    """The estimator these functions replace: median spacing of save-grid
    maxima, per cell."""
    ts, y = np.asarray(ts), np.asarray(y)
    out = []
    for i in range(y.shape[1]):
        pk = np.flatnonzero(
            (y[1:-1, i] > y[:-2, i]) & (y[1:-1, i] > y[2:, i])
        ) + 1
        out.append(np.median(np.diff(ts[pk])))
    return np.array(out)


def test_period_resolves_below_one_save_step():
    ts, y, periods = population()
    got = np.asarray(dominant_period(ts, y))
    assert np.abs(got - periods).max() < 1e-4 * TRUE_PERIOD


def test_period_cv_is_not_grid_quantised():
    """The load-bearing property: the measured spread is the real spread."""
    ts, y, periods = population()
    got = np.asarray(dominant_period(ts, y))
    true_cv = periods.std() / periods.mean()
    assert got.std() / got.mean() == pytest.approx(true_cv, rel=1e-3)

    snapped = grid_snapped(ts, y)
    assert snapped.std() / snapped.mean() > 1.15 * true_cv


def test_deterministic_population_has_no_period_spread():
    """Identical cells read as identical, at the right period.

    Grid snapping shows up here as *bias* rather than spread: every cell
    lands on the same wrong grid multiple, so the estimator it replaces has
    zero CV and a period a full grid step off.
    """
    ts, y, _ = population(n=32, period_cv=0.0)
    got = np.asarray(dominant_period(ts, y))
    assert got.std() / got.mean() < 1e-6
    assert got.mean() == pytest.approx(TRUE_PERIOD, rel=1e-4)

    snapped = grid_snapped(ts, y)
    assert np.allclose(snapped % SAVE_DT, 0.0)  # pinned to grid multiples
    assert abs(snapped.mean() - TRUE_PERIOD) > 0.01 * TRUE_PERIOD


def test_period_is_independent_of_the_save_grid():
    ts, y, _ = population(n=16)
    coarse = np.asarray(dominant_period(ts[::3], y[::3]))
    fine = np.asarray(dominant_period(ts, y))
    assert np.abs(coarse - fine).max() < 1e-3 * TRUE_PERIOD


def test_coherence_holds_for_identical_cells_and_decays_when_dephasing():
    ts, y, _ = population(n=64, period_cv=0.0)
    coh = np.asarray(coherence_curve(ts, y, window=1.5)[1])
    assert coh.min() > 0.99

    ts, y, _ = population(n=200, period_cv=0.03)
    coh = np.asarray(coherence_curve(ts, y, window=1.5)[1])
    assert coh[0] > 0.95 and coh[-1] < 0.2


def test_coherence_curve_shapes_and_window_guard():
    ts, y, _ = population(n=8)
    centers, coh, single, bulk = coherence_curve(ts, y, window=1.5)
    assert centers.shape == coh.shape == single.shape == bulk.shape
    assert np.all(np.asarray(single) >= np.asarray(bulk) - 1e-9)
    with pytest.raises(ValueError):
        coherence_curve(ts, y, window=2 * T_END)


def test_flat_series_returns_nan_period():
    ts = jnp.arange(0.0, T_END, SAVE_DT)
    flat = jnp.ones((ts.shape[0], 3))
    assert np.isnan(np.asarray(dominant_period(ts, flat))).all()
    assert not np.asarray(peak_mask(flat)).any()


def test_endpoints_are_never_peaks():
    ts, y, _ = population(n=4)
    m = np.asarray(peak_mask(y))
    assert not m[0].any() and not m[-1].any()
