"""Scheduler — multi-rate orchestrator for heterogeneous process composites.

Replaces the monolithic ``Simulator`` for composites that mix continuous,
discrete, and event-driven processes at different timescales.

Architecture::

    Scheduler
      |
      macro_dt (communication interval)
      |
    +------------------+------------------+------------------+
    | Continuous groups | Discrete procs   | Event procs      |
    | (Diffrax ODE)    | (called at       | (conditions      |
    | per group        |  their dt_step)  |  checked at sync)|
    +------------------+------------------+------------------+

Design borrows scheduling concepts from:
- Vivarium's Engine (Agmon et al., 2022)
- Ptolemy II's Directors (UC Berkeley)

Implemented natively on JAX for GPU acceleration and differentiability
within continuous groups.

Example
-------
>>> scheduler = Scheduler()
>>> result = scheduler.run(composite, t_span=(0.0, 1000.0), macro_dt=1.0)
>>> result.ts    # macro step times
>>> result.ys    # state trajectories
>>> result.events  # fired event log
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import diffrax as dfx
import jax.numpy as jnp

from hallsim.composite import Composite
from hallsim.process import ProcessKind
from hallsim.store import extract_port_view, route_derivatives


@dataclass
class EventRecord:
    """Log entry for a fired event."""

    time: float
    process: str
    delta: dict[str, jnp.ndarray]


@dataclass
class SchedulerResult:
    """Container for scheduler output.

    Attributes
    ----------
    ts:
        Macro step time points.
    ys:
        State trajectories — ``{store_path: array}`` where each array
        has shape ``(n_time_points, *state_shape)``.
    events:
        Log of fired events.
    stats:
        Per-group solver statistics.
    """

    ts: jnp.ndarray
    ys: dict[str, jnp.ndarray]
    events: list[EventRecord] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)


class Scheduler:
    """Multi-rate orchestrator for composites with mixed process kinds.

    Uses Lie operator splitting for continuous groups: groups are solved
    sequentially, each seeing the updated state from the previous group.

    For composites with only CONTINUOUS processes and no timescale
    declarations, degrades to a single Diffrax solve (equivalent to
    the old ``Simulator``).

    Parameters
    ----------
    solver:
        Diffrax solver instance.  Default: ``Tsit5()``.
    rtol, atol:
        Relative and absolute tolerances for adaptive stepping.
    max_steps:
        Safety limit on solver steps per macro step.
    dt0:
        Initial step size for adaptive controller.
    groups:
        Manual group assignment ``{group_name: [proc_name, ...]}``.
        If ``None``, uses ``composite.auto_groups()``.
    """

    def __init__(
        self,
        solver: dfx.AbstractSolver | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_steps: int = 400_000,
        dt0: float = 1e-3,
        groups: dict[str, list[str]] | None = None,
    ) -> None:
        self.solver = solver or dfx.Tsit5()
        self.controller = dfx.PIDController(rtol=rtol, atol=atol)
        self.max_steps = max_steps
        self.dt0 = dt0
        self.manual_groups = groups

    def run(
        self,
        composite: Composite,
        t_span: tuple[float, float],
        macro_dt: float = 1.0,
        y0: dict[str, jnp.ndarray] | None = None,
        save_dt: float | None = None,
    ) -> SchedulerResult:
        """Run the composite with multi-rate scheduling.

        Parameters
        ----------
        composite:
            Wired bundle of processes (may include continuous, discrete,
            and event processes).
        t_span:
            ``(t0, t1)`` — start and end times.
        macro_dt:
            Communication interval.  At each macro step:
            1. Continuous groups are solved independently.
            2. Discrete processes fire if due.
            3. Event conditions are checked.
        y0:
            Initial state.  If ``None``, uses ``composite.initial_state()``.
        save_dt:
            Save interval for trajectory output.  If ``None``, saves at
            every macro step.  If larger than ``macro_dt``, saves less
            frequently.

        Returns
        -------
        :class:`SchedulerResult`
        """
        t0, t1 = t_span
        state = dict(y0) if y0 is not None else composite.initial_state()

        # Resolve groups
        groups = self.manual_groups or composite.auto_groups()
        discrete_procs = composite.discrete_processes()
        event_procs = composite.event_processes()

        # If no groups and no discrete/event, single-group fallback
        if not groups and not discrete_procs and not event_procs:
            continuous = composite.continuous_processes()
            if continuous:
                groups = {"default": list(continuous.keys())}

        # Pre-build group RHS functions
        group_rhs = {
            gname: composite.build_group_rhs(proc_names)
            for gname, proc_names in groups.items()
        }

        # Event tracking: was_active[proc_name] -> bool
        was_active: dict[str, bool] = {n: False for n in event_procs}

        # Trajectory recording
        save_dt = save_dt or macro_dt
        trajectory_ts: list[float] = [t0]
        trajectory_snapshots: list[dict[str, jnp.ndarray]] = [
            {k: v.copy() for k, v in state.items()}
        ]
        last_save_t = t0

        # Event log
        events: list[EventRecord] = []

        # Per-group stats
        stats: dict[str, Any] = {gname: {"num_macro_steps": 0} for gname in groups}

        n_macro = int((t1 - t0) / macro_dt) + 1
        try:
            from tqdm import tqdm
            pbar = tqdm(total=n_macro, desc="Scheduler", unit="step", leave=False)
        except ImportError:
            pbar = None

        t = t0
        while t < t1 - 1e-12:
            t_next = min(t + macro_dt, t1)

            # 1. Solve each continuous group (Lie splitting: sequential)
            for gname, rhs_fn in group_rhs.items():
                state = self._solve_group(rhs_fn, state, t, t_next)
                stats[gname]["num_macro_steps"] += 1

            # 2. Fire discrete processes that are due
            for proc_name, proc in discrete_procs.items():
                if self._is_due(t, t_next, proc.dt_step):
                    proc_topo = composite.topology[proc_name]
                    view = extract_port_view(state, proc_topo)
                    delta = proc.update(t_next, view)
                    routed = route_derivatives(delta, proc_topo)
                    for sp, dval in routed.items():
                        state[sp] = state[sp] + dval

            # 3. Check event conditions
            for proc_name, proc in event_procs.items():
                proc_topo = composite.topology[proc_name]
                view = extract_port_view(state, proc_topo)
                cond = bool(proc.condition(t_next, view))
                if cond and not was_active[proc_name]:
                    delta = proc.handler(t_next, view)
                    routed = route_derivatives(delta, proc_topo)
                    for sp, dval in routed.items():
                        state[sp] = state[sp] + dval
                    events.append(EventRecord(
                        time=float(t_next),
                        process=proc_name,
                        delta={sp: dval for sp, dval in routed.items()},
                    ))
                was_active[proc_name] = cond

            t = t_next
            if pbar is not None:
                pbar.update(1)

            # Save snapshot if due
            if t - last_save_t >= save_dt - 1e-12 or t >= t1 - 1e-12:
                trajectory_ts.append(t)
                trajectory_snapshots.append(
                    {k: v.copy() for k, v in state.items()}
                )
                last_save_t = t

        if pbar is not None:
            pbar.close()

        # Assemble result
        ts = jnp.array(trajectory_ts)
        ys: dict[str, jnp.ndarray] = {}
        all_keys = trajectory_snapshots[0].keys()
        for k in all_keys:
            ys[k] = jnp.stack([snap[k] for snap in trajectory_snapshots])

        return SchedulerResult(ts=ts, ys=ys, events=events, stats=stats)

    def _solve_group(
        self,
        rhs_fn,
        state: dict[str, jnp.ndarray],
        t0: float,
        t1: float,
    ) -> dict[str, jnp.ndarray]:
        """Solve one continuous group from t0 to t1."""
        if t1 <= t0:
            return state

        term = dfx.ODETerm(rhs_fn)
        sol = dfx.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=min(self.dt0, t1 - t0),
            y0=state,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=self.controller,
            max_steps=self.max_steps,
        )
        # Extract final state from solution
        return {k: v[-1] for k, v in sol.ys.items()}

    @staticmethod
    def _is_due(t: float, t_next: float, dt_step: float) -> bool:
        """Check if a discrete process should fire in the interval (t, t_next].

        A process with dt_step fires whenever a multiple of dt_step falls
        within the interval.
        """
        if dt_step <= 0:
            return False
        # Number of complete periods at t and t_next
        n_before = int(t / dt_step)
        n_after = int(t_next / dt_step)
        # Also handle exact alignment
        if t_next % dt_step < 1e-12:
            n_after = int(round(t_next / dt_step))
        return n_after > n_before
