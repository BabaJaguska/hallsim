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

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)

import diffrax as dfx
import jax.numpy as jnp

from hallsim.composite import Composite
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
    coupling_mode:
        How inter-group state is communicated during Lie splitting.

        - ``"frozen"`` (default): each group sees the previous group's
          final state — standard Lie splitting.
        - ``"interpolated"``: each group receives a dense interpolant
          from the previous group's Diffrax solution.  The group's RHS
          queries the interpolant at the current time ``t``, so it
          "feels" the previous group's trajectory within the macro step.
          Reduces splitting error from O(macro_dt) to O(macro_dt^p)
          where p depends on the interpolant order.

          Inspired by dense-output multirate methods (cf. SUNDIALS,
          continuous-output Runge-Kutta).  See
          ``docs/crossgen-suggestions.md``, suggestion #1.
    splitting:
        Operator splitting scheme for continuous groups.

        - ``"lie"`` (default): groups solved sequentially in one pass.
          First-order accurate: error ~ O(macro_dt).
        - ``"strang"``: symmetric splitting — groups solved for dt/2
          in forward order, then dt/2 in reverse order.  Cancels the
          leading-order commutator error, giving O(macro_dt²) accuracy.
          Independently recommended by gyrokinetics (Boris push), FSI
          (predictor-corrector), and variational integrators (VPRK).
          See ``docs/crossgen-suggestions.md``, Strang splitting.
    adaptive_dt:
        Enable PLL-inspired adaptive ``macro_dt`` sizing.  After each
        macro step, the relative coupling residual is measured.  If it
        exceeds ``adaptive_dt_rho_max`` the step is shrunk; if it stays
        below ``adaptive_dt_rho_min`` for ``adaptive_dt_grow_wait``
        consecutive steps, the step is grown.  See
        ``docs/crossgen-suggestions.md``, suggestion #2.
    adaptive_dt_rho_max:
        Relative residual threshold for shrinking (default 0.5).
    adaptive_dt_rho_min:
        Relative residual threshold for growing (default 0.01).
    adaptive_dt_grow_wait:
        Consecutive low-residual steps before growing (default 3).
    adaptive_dt_factor:
        Multiplicative factor for shrink/grow (default 2.0).
    adaptive_dt_min:
        Minimum allowed ``macro_dt``.  Default: ``macro_dt / 64``.
    adaptive_dt_max:
        Maximum allowed ``macro_dt``.  Default: ``macro_dt * 4``.
    """

    def __init__(
        self,
        solver: dfx.AbstractSolver | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_steps: int = 400_000,
        dt0: float = 1e-3,
        groups: dict[str, list[str]] | None = None,
        coupling_mode: str = "frozen",
        splitting: str = "lie",
        adaptive_dt: bool = False,
        adaptive_dt_rho_max: float = 0.5,
        adaptive_dt_rho_min: float = 0.01,
        adaptive_dt_grow_wait: int = 3,
        adaptive_dt_factor: float = 2.0,
        adaptive_dt_min: float | None = None,
        adaptive_dt_max: float | None = None,
        debug: bool = False,
    ) -> None:
        if coupling_mode not in ("frozen", "interpolated"):
            raise ValueError(
                f"coupling_mode must be 'frozen' or 'interpolated', "
                f"got {coupling_mode!r}"
            )
        if splitting not in ("lie", "strang"):
            raise ValueError(
                f"splitting must be 'lie' or 'strang', got {splitting!r}"
            )
        self.solver = solver or dfx.Tsit5()
        self.controller = dfx.PIDController(rtol=rtol, atol=atol)
        self.max_steps = max_steps
        self.dt0 = dt0
        self.manual_groups = groups
        self.coupling_mode = coupling_mode
        self.splitting = splitting
        self.adaptive_dt = adaptive_dt
        self.adaptive_dt_rho_max = adaptive_dt_rho_max
        self.adaptive_dt_rho_min = adaptive_dt_rho_min
        self.adaptive_dt_grow_wait = adaptive_dt_grow_wait
        self.adaptive_dt_factor = adaptive_dt_factor
        self.adaptive_dt_min = adaptive_dt_min
        self.adaptive_dt_max = adaptive_dt_max
        self.debug = debug

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
            Communication interval (initial value if ``adaptive_dt``
            is enabled).  At each macro step:
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
        stats: dict[str, Any] = {
            gname: {"num_macro_steps": 0} for gname in groups
        }

        # Adaptive macro_dt state
        current_macro_dt = macro_dt
        if self.adaptive_dt:
            dt_min = self.adaptive_dt_min or macro_dt / 64.0
            dt_max = self.adaptive_dt_max or macro_dt * 4.0
            consecutive_low = 0
            stats["adaptive_dt"] = {
                "shrinks": 0,
                "grows": 0,
                "min_dt": macro_dt,
                "max_dt": macro_dt,
            }

        n_macro = int((t1 - t0) / macro_dt) + 1
        try:
            from tqdm import tqdm

            pbar = tqdm(
                total=n_macro, desc="Scheduler", unit="step", leave=False
            )
        except ImportError:
            pbar = None

        t = t0
        while t < t1 - 1e-12:
            t_next = min(t + current_macro_dt, t1)

            # Snapshot state before splitting (for residual check)
            if self.adaptive_dt:
                state_before = {k: v.copy() for k, v in state.items()}

            # 1. Solve continuous groups
            if self.splitting == "strang" and len(group_rhs) > 1:
                t_mid = t + (t_next - t) / 2.0
                items = list(group_rhs.items())
                for gname, rhs_fn in items:
                    state = self._solve_group(
                        rhs_fn, state, t, t_mid, group_name=gname
                    )
                    stats[gname]["num_macro_steps"] += 1
                for gname, rhs_fn in reversed(items):
                    state = self._solve_group(
                        rhs_fn, state, t_mid, t_next, group_name=gname
                    )
                    stats[gname]["num_macro_steps"] += 1
            else:
                # Lie splitting: sequential, one pass
                prev_interpolant = None
                for gname, rhs_fn in group_rhs.items():
                    if self.coupling_mode == "interpolated":
                        state, prev_interpolant = (
                            self._solve_group_interpolated(
                                rhs_fn,
                                state,
                                t,
                                t_next,
                                prev_interpolant,
                            )
                        )
                    else:
                        state = self._solve_group(
                            rhs_fn, state, t, t_next, group_name=gname
                        )
                    stats[gname]["num_macro_steps"] += 1

            # Adaptive macro_dt: measure coupling residual and adjust
            if self.adaptive_dt:
                num = sum(
                    float(jnp.sum((state[k] - state_before[k]) ** 2))
                    for k in state
                )
                den = (
                    sum(
                        float(jnp.sum(state_before[k] ** 2))
                        for k in state_before
                    )
                    + 1e-30
                )
                rho = (num / den) ** 0.5

                if rho > self.adaptive_dt_rho_max:
                    current_macro_dt = max(
                        dt_min, current_macro_dt / self.adaptive_dt_factor
                    )
                    consecutive_low = 0
                    stats["adaptive_dt"]["shrinks"] += 1
                elif rho < self.adaptive_dt_rho_min:
                    consecutive_low += 1
                    if consecutive_low >= self.adaptive_dt_grow_wait:
                        current_macro_dt = min(
                            dt_max, current_macro_dt * self.adaptive_dt_factor
                        )
                        consecutive_low = 0
                        stats["adaptive_dt"]["grows"] += 1
                else:
                    consecutive_low = 0

                stats["adaptive_dt"]["min_dt"] = min(
                    stats["adaptive_dt"]["min_dt"], current_macro_dt
                )
                stats["adaptive_dt"]["max_dt"] = max(
                    stats["adaptive_dt"]["max_dt"], current_macro_dt
                )

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
                    events.append(
                        EventRecord(
                            time=float(t_next),
                            process=proc_name,
                            delta={sp: dval for sp, dval in routed.items()},
                        )
                    )
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
        group_name: str = "",
    ) -> dict[str, jnp.ndarray]:
        """Solve one continuous group from t0 to t1."""
        if t1 <= t0:
            return state

        state_before = (
            {k: float(v) for k, v in state.items()} if self.debug else None
        )

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

        if self.debug:
            n_steps = int(sol.stats["num_steps"])
            n_rejected = int(sol.stats["num_rejected_steps"])
            # Show max delta (what this group actually changed)
            final = {k: float(v[-1]) for k, v in sol.ys.items()}
            deltas = {k: abs(final[k] - state_before[k]) for k in final}
            max_delta_key = max(deltas, key=deltas.get)
            max_delta = deltas[max_delta_key]
            any_nan = any(jnp.isnan(v[-1]).any() for v in sol.ys.values())
            log.info(
                f"  [{group_name}] [{t0:.1f} → {t1:.1f}]: "
                f"{n_steps} steps ({n_rejected} rej) | "
                f"max Δ: {max_delta_key}={max_delta:.4g}"
                + (" *** NaN ***" if any_nan else "")
            )

        # Extract final state from solution
        return {k: v[-1] for k, v in sol.ys.items()}

    def _solve_group_interpolated(
        self,
        rhs_fn,
        state: dict[str, jnp.ndarray],
        t0: float,
        t1: float,
        prev_interpolant: Any | None = None,
    ) -> tuple[dict[str, jnp.ndarray], Any]:
        """Solve one group with dense output, injecting a previous group's interpolant.

        When ``prev_interpolant`` is provided, the group's RHS receives
        a time-varying view of the previous group's state via the ``args``
        parameter.  The RHS can query ``args["interpolant"].evaluate(t)``
        to get the previous group's state at the current solver time,
        instead of seeing a frozen snapshot.

        Returns
        -------
        (final_state, interpolant)
            The updated state dict and this group's dense interpolant
            (for passing to the next group).
        """
        if t1 <= t0:
            return state, prev_interpolant

        # Wrap the RHS to inject interpolant-based coupling
        if prev_interpolant is not None:
            base_rhs = rhs_fn

            def coupled_rhs(t, y, args=None):
                # Evaluate the previous group's interpolant at current t
                interp_state = prev_interpolant.evaluate(t)
                # Merge: interpolated values for paths owned by the
                # previous group, current y for everything else.
                merged = dict(y)
                for k, v in interp_state.items():
                    if k in merged:
                        merged[k] = v
                return base_rhs(t, merged, args)

            rhs_to_solve = coupled_rhs
        else:
            rhs_to_solve = rhs_fn

        term = dfx.ODETerm(rhs_to_solve)
        sol = dfx.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=min(self.dt0, t1 - t0),
            y0=state,
            saveat=dfx.SaveAt(t1=True, dense=True),
            stepsize_controller=self.controller,
            max_steps=self.max_steps,
        )

        final_state = {k: v[-1] for k, v in sol.ys.items()}
        return final_state, sol.interpolation

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
