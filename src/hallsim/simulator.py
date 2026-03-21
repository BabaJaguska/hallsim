"""Simulator — Diffrax-based ODE solver wrapper for Composites.

Provides a clean interface for running simulations:

    >>> sim = Simulator()
    >>> composite = Composite(processes=..., topology=...)
    >>> result = sim.run(composite, t_span=(0.0, 100.0), dt=1.0)
    >>> result.ts   # time points
    >>> result.ys   # dict of trajectories
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import diffrax as dfx
import jax.numpy as jnp

from hallsim.composite import Composite


@dataclass
class SimResult:
    """Container for simulation output.

    Attributes
    ----------
    ts:
        Time points (1-D array).
    ys:
        State trajectories — ``{store_path: array}`` where each array
        has shape ``(n_time_points,)`` (or ``(n_time_points, *state_shape)``
        for vector states).
    stats:
        Solver statistics from Diffrax (number of steps, etc.).
    """

    ts: jnp.ndarray
    ys: dict[str, jnp.ndarray]
    stats: dict[str, Any]


class Simulator:
    """Run a Composite through a Diffrax ODE solver.

    Parameters
    ----------
    solver:
        Diffrax solver instance.  Default: ``Tsit5()`` (explicit 5th-order
        Runge-Kutta, good for non-stiff systems).
    rtol, atol:
        Relative and absolute tolerances for adaptive stepping.
    max_steps:
        Safety limit on solver steps.
    dt0:
        Initial step size for adaptive controller.
    """

    def __init__(
        self,
        solver: dfx.AbstractSolver | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_steps: int = 400_000,
        dt0: float = 1e-3,
    ) -> None:
        self.solver = solver or dfx.Tsit5()
        self.controller = dfx.PIDController(rtol=rtol, atol=atol)
        self.max_steps = max_steps
        self.dt0 = dt0

    def run(
        self,
        composite: Composite,
        t_span: tuple[float, float],
        dt: float = 1.0,
        y0: dict[str, jnp.ndarray] | None = None,
        keep_trajectory: bool = True,
    ) -> SimResult:
        """Solve the composite ODE system.

        Parameters
        ----------
        composite:
            Wired bundle of processes.
        t_span:
            ``(t0, t1)`` — start and end times.
        dt:
            Save interval for trajectory output.
        y0:
            Initial state.  If ``None``, uses ``composite.initial_state()``.
        keep_trajectory:
            If ``True``, saves at every ``dt`` step.
            If ``False``, saves only the final state.

        Returns
        -------
        :class:`SimResult`
        """
        t0, t1 = t_span
        rhs = composite.build_rhs()
        if y0 is None:
            y0 = composite.initial_state()

        term = dfx.ODETerm(rhs)

        if keep_trajectory:
            ts = jnp.arange(t0, t1 + dt, dt)
            ts = ts[ts <= t1]  # clip to avoid floating-point overshoot
            saveat = dfx.SaveAt(ts=ts)
        else:
            saveat = dfx.SaveAt(t1=True)

        sol = dfx.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=self.dt0,
            y0=y0,
            saveat=saveat,
            stepsize_controller=self.controller,
            max_steps=self.max_steps,
        )

        return SimResult(
            ts=sol.ts,
            ys=sol.ys,
            stats={
                "num_steps": sol.stats.get("num_steps", None) if hasattr(sol, "stats") else None,
            },
        )

    def run_with_perturbation(
        self,
        composite: Composite,
        t_span: tuple[float, float],
        kick_time: float,
        kick_dict: dict[str, float],
        dt: float = 1.0,
        y0: dict[str, jnp.ndarray] | None = None,
    ) -> SimResult:
        """Two-phase simulation with a mid-run state perturbation.

        Integrates from ``t0`` to ``kick_time``, applies additive kicks
        to specified store paths, then integrates from ``kick_time`` to ``t1``.

        Parameters
        ----------
        composite:
            Wired bundle of processes.
        t_span:
            ``(t0, t1)``
        kick_time:
            Time at which to apply the perturbation.
        kick_dict:
            ``{store_path: delta}`` — additive changes to apply.
        dt:
            Save interval.
        y0:
            Initial state.

        Returns
        -------
        :class:`SimResult` with concatenated trajectories.
        """
        t0, t1 = t_span

        # Phase 1: t0 → kick_time
        res1 = self.run(composite, (t0, kick_time), dt=dt, y0=y0, keep_trajectory=True)

        # Extract final state and apply kick
        kicked = {}
        for k, v in res1.ys.items():
            final_val = v[-1]
            if k in kick_dict:
                final_val = final_val + jnp.asarray(kick_dict[k], dtype=final_val.dtype)
            kicked[k] = final_val

        # Phase 2: kick_time → t1
        res2 = self.run(composite, (kick_time, t1), dt=dt, y0=kicked, keep_trajectory=True)

        # Concatenate (skip duplicate time point)
        ts = jnp.concatenate([res1.ts, res2.ts[1:]])
        ys = {
            k: jnp.concatenate([res1.ys[k], res2.ys[k][1:]])
            for k in res1.ys
        }

        return SimResult(ts=ts, ys=ys, stats={})
