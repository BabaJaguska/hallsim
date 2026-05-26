"""Scheduler — multi-rate orchestrator for heterogeneous process composites.

For composites that mix continuous, discrete, and event-driven processes
at different timescales.

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

State is a flat ``jnp.ndarray`` indexed by ``sorted(composite.store_paths())``.
Continuous group solves feed it straight to Diffrax; discrete/event
handlers precompute read/write index arrays once at run-start and apply
batched scatter updates.  Dict shape only appears at the API boundary
(``y0`` in, ``ys`` out) and inside ``Process.derivative/update/handler``
calls where the per-process port view is a small dict of named ports.

Scheduling concepts borrow from Vivarium's Engine (Agmon et al., 2022)
and Ptolemy II's Directors (UC Berkeley).  Implemented natively on JAX
for GPU acceleration and differentiability within continuous groups.

Example
-------
>>> scheduler = Scheduler()
>>> result = scheduler.run(composite, t_span=(0.0, 1000.0), macro_dt=1.0)
>>> result.ts      # macro step times
>>> result.ys      # (n_time, ..., n_vars) state tensor; per-path via .get(key)
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
from hallsim.process import PortRole

# Float epsilon for floating-point time comparisons in the macro-step loop
# (start/end of run, save_dt alignment, dt_step alignment for discrete
# processes). Sized for second-scale ``t_span`` values; runs that span
# ranges < ~1e-9 should pass an explicit save_dt and not rely on this.
_TIME_EPS: float = 1e-12


@dataclass
class EventRecord:
    """Log entry for a fired event."""

    time: float
    process: str
    delta: dict[str, jnp.ndarray]


@dataclass
class SchedulerResult:
    """JAX-native container for scheduler output.

    ``ys`` is the raw stacked tensor — same convention as Diffrax's
    ``sol.ys`` — so it composes cleanly with ``jax.vmap``, ``jax.grad``,
    and downstream JAX-native code without a Python-side dict of arrays
    in the middle.

    Attributes
    ----------
    ts:
        Macro step time points, shape ``(n_time,)``.
    ys:
        State trajectory tensor, shape ``(n_time, ..., n_vars)``.
        For scalar runs this is ``(n_time, n_vars)``; for batched runs
        (population studies, parameter sweeps via batched y0) this is
        ``(n_time, batch, n_vars)``. The trailing axis matches
        ``Composite.flatten/unflatten`` — index it via ``keys``.
    keys:
        Store paths in trailing-axis order — the inverse map for ``ys``.
        Use ``result.get("eriq/p53_activity")`` for ergonomic per-path
        access.
    events:
        Log of fired events.
    stats:
        Per-group solver statistics.
    """

    ts: jnp.ndarray
    ys: jnp.ndarray
    keys: list[str]
    events: list[EventRecord] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str) -> jnp.ndarray:
        """Per-path trajectory: ``ys[..., keys.index(key)]``.

        Returns shape ``(n_time,)`` for scalar runs, ``(n_time, batch)``
        for batched runs.
        """
        return self.ys[..., self.keys.index(key)]

    def __contains__(self, key: str) -> bool:
        return key in self.keys


def _build_proc_index_maps(
    proc, proc_topo: dict[str, str], key_to_idx: dict[str, int]
) -> tuple[tuple, tuple]:
    """Precompute (read_pairs, write_pairs) for a discrete/event process.

    ``read_pairs``  = ((port_name, store_idx), ...) — gather port view from state.
    ``write_pairs`` = ((port_name, store_idx), ...) — scatter LATCHED writes.
    """
    read_pairs = tuple(
        (port, key_to_idx[sp]) for port, sp in proc_topo.items()
    )
    schema = proc.ports_schema()
    write_pairs = tuple(
        (port, key_to_idx[proc_topo[port]])
        for port, p in schema.items()
        if p.role == PortRole.LATCHED
    )
    return read_pairs, write_pairs


def _apply_delta(
    state_vec: jnp.ndarray,
    raw_delta: dict[str, jnp.ndarray],
    write_pairs: tuple,
) -> jnp.ndarray:
    """Scatter-add a process's delta dict into the flat state vector."""
    out = [
        (idx, raw_delta[port])
        for port, idx in write_pairs
        if port in raw_delta
    ]
    if not out:
        return state_vec
    idxs = jnp.array([i for i, _ in out])
    vals = jnp.stack([v for _, v in out])
    return state_vec.at[idxs].add(vals)


class Scheduler:
    """Multi-rate orchestrator for composites with mixed process kinds.

    Uses Lie operator splitting for continuous groups: groups are solved
    sequentially, each seeing the updated state from the previous group.

    For single-group composites with no events / discrete / adaptive_dt /
    Strang / interpolated coupling, takes a fast path that issues one
    ``dfx.diffeqsolve`` over the whole ``t_span`` — no per-macro-step
    restarts.

    Parameters
    ----------
    solver:
        Diffrax solver instance.  Default: ``Tsit5()`` — explicit RK5
        with adaptive stepping, fast and robust for the majority of
        biological composites (signaling cascades, oscillators, mass-
        action kinetics with moderate rate constants). For known-stiff
        systems (genome-scale metabolism, reaction-diffusion with
        widely-separated timescales, problems where step-size is
        stability-limited rather than accuracy-limited) pass an implicit
        solver explicitly, e.g. ``Scheduler(solver=dfx.Kvaerno5())``.
        Empirically Tsit5 outperforms Kvaerno5 by ~4× on the
        ``damage_p53_eriq`` composite (six processes, three timescales,
        SBML-imported p53-Mdm2 oscillator) and ~3× on Kholodenko 2000
        MAPK; the implicit solver's Newton-iteration overhead is not
        recouped by step-count savings on mildly-stiff oscillators.
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

          Requires ``splitting="lie"``. With ``splitting="strang"``, the
          reverse pass needs interpolants over a half-step that have
          not been produced yet, so the constructor rejects that combo.
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
    progress:
        If ``True``, show a tqdm progress bar over macro steps. Default
        ``False`` because the Python-side ``pbar.update(1)`` is a side
        effect that interferes with ``jax.vmap`` over batched runs (e.g.
        population studies, parameter sweeps).
    """

    def __init__(
        self,
        solver: dfx.AbstractSolver | None = None,
        rtol: float = 1e-3,
        atol: float = 1e-6,
        max_steps: int = 4_000_000,
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
        adjoint: dfx.AbstractAdjoint | None = None,
        debug: bool = False,
        progress: bool = False,
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
        if splitting == "strang" and coupling_mode == "interpolated":
            raise ValueError(
                "splitting='strang' is incompatible with "
                "coupling_mode='interpolated'. Strang's reverse pass "
                "needs each prior group's interpolant over the second "
                "half-step, but that interpolant has not been produced "
                "yet — the group runs after, not before. Use "
                "splitting='lie' with coupling_mode='interpolated' "
                "(O(macro_dt^p) splitting error), or splitting='strang' "
                "with coupling_mode='frozen' (O(macro_dt^2))."
            )
        self.solver = solver or dfx.Tsit5()
        self.controller = dfx.PIDController(rtol=rtol, atol=atol)
        # Adjoint method used by every diffeqsolve in this run.
        # Default (None) → diffrax picks RecursiveCheckpointAdjoint, which
        # is memory-cheap but step-expensive. For calibration through
        # stiff/oscillatory composites, pass dfx.BacksolveAdjoint() for
        # near-forward-cost backward passes.
        self.adjoint = adjoint or dfx.RecursiveCheckpointAdjoint()
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
        self.progress = progress

    def run(
        self,
        composite: Composite,
        t_span: tuple[float, float],
        macro_dt: float = 1.0,
        y0: jnp.ndarray | None = None,
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
            Initial state tensor of shape ``(n_vars,)`` (or
            ``(batch, n_vars)`` for batched-population runs). If
            ``None``, uses ``composite.initial_state_vec()``. Override
            values via
            ``y0.at[composite.store_keys().index("path")].set(...)``.
        save_dt:
            Save interval for trajectory output.  If ``None``, saves at
            every macro step.  If larger than ``macro_dt``, saves less
            frequently.

        Returns
        -------
        :class:`SchedulerResult`
        """
        t0, t1 = t_span

        # Flat state layout — pinned for the whole run. y0 is a tensor in
        # trailing-axis convention; for batched-population runs it has
        # shape (batch, n_vars), and the rest of the run propagates that
        # batch axis through every group's Diffrax solve.
        keys = composite.store_keys()
        state = (
            composite.initial_state_vec(keys)
            if y0 is None
            else jnp.asarray(y0)
        )
        key_to_idx = {k: i for i, k in enumerate(keys)}

        # Resolve groups
        groups = self.manual_groups or composite.auto_groups()
        discrete_procs = composite.discrete_processes()
        event_procs = composite.event_processes()

        # Reject batched y0 against features that rely on Python-side
        # branching (event firing, discrete delta scatter, adaptive_dt
        # residual reduction). These paths use ``bool()`` / ``float()``
        # on traced JAX arrays which would either crash under ``vmap`` or
        # silently collapse the batch axis. The Scheduler's batched-y0
        # support is currently the continuous-only path (fast path or
        # multi-group Lie/Strang). A future PR can lift these by masking
        # event firing per-batch-element via ``lax.cond``.
        is_batched = state.ndim > 1
        if is_batched:
            blockers = []
            if event_procs:
                blockers.append(
                    f"EVENT processes {list(event_procs.keys())} "
                    "(condition fires via Python bool — incompatible with vmap)"
                )
            if discrete_procs:
                blockers.append(
                    f"DISCRETE processes {list(discrete_procs.keys())} "
                    "(delta scatter is not batch-axis-aware)"
                )
            if self.adaptive_dt:
                blockers.append(
                    "adaptive_dt=True (coupling residual is a single dt "
                    "for the whole batch and reduces via Python float)"
                )
            if blockers:
                raise ValueError(
                    f"Batched y0 of shape {tuple(state.shape)} is not "
                    "supported with: " + "; ".join(blockers) + ". "
                    "Run unbatched, drop the blocking feature, or vmap "
                    "Scheduler.run from outside."
                )

        # If no groups and no discrete/event, single-group fallback
        if not groups and not discrete_procs and not event_procs:
            continuous = composite.continuous_processes()
            if continuous:
                groups = {"default": list(continuous.keys())}

        # ─── Fast path: single continuous group, no events, no
        # discrete, no adaptive macro_dt ───────────────────────────────
        # Skip the macro-step loop entirely. One ``diffeqsolve`` over
        # the full t_span lets Diffrax's adaptive stepper find the right
        # step sizes once instead of restarting on every macro step.
        # Strang/interpolated only matter for multi-group splitting, so
        # this path also requires the default Lie/frozen settings.
        fast_path_eligible = (
            len(groups) == 1
            and not discrete_procs
            and not event_procs
            and not self.adaptive_dt
            and self.splitting == "lie"
            and self.coupling_mode == "frozen"
        )
        if fast_path_eligible:
            (gname,) = groups.keys()
            (proc_names,) = groups.values()
            rhs_fn, _ = composite.build_rhs(proc_names)
            save_step = save_dt if save_dt is not None else macro_dt
            save_ts = jnp.arange(t0, t1 + save_step / 2, save_step)
            save_ts = save_ts[save_ts <= t1]
            sol = dfx.diffeqsolve(
                dfx.ODETerm(rhs_fn),
                self.solver,
                t0=t0,
                t1=t1,
                dt0=min(self.dt0, t1 - t0),
                y0=state,
                saveat=dfx.SaveAt(ts=save_ts),
                stepsize_controller=self.controller,
                adjoint=self.adjoint,
                max_steps=self.max_steps,
            )
            stats = {
                gname: {
                    "num_macro_steps": 1,
                    "num_solver_steps": (
                        int(sol.stats["num_steps"])
                        if hasattr(sol, "stats") and "num_steps" in sol.stats
                        else None
                    ),
                }
            }
            return SchedulerResult(
                ts=sol.ts,
                ys=sol.ys,
                keys=keys,
                events=[],
                stats=stats,
            )

        # Pre-build group flat RHS functions (each returns (fn, keys);
        # all keys are identical so we discard the per-group copy) and
        # the indices each group writes derivatives to (used by
        # interpolated coupling to splice prev-group state into the
        # next group's RHS input).
        group_rhs: dict[str, Any] = {}
        group_write_idxs: dict[str, jnp.ndarray] = {}
        for gname, proc_names in groups.items():
            fn, _ = composite.build_rhs(proc_names)
            group_rhs[gname] = fn
            written: set[int] = set()
            for pname in proc_names:
                proc = composite.processes[pname]
                proc_topo = composite.topology[pname]
                for port, p in proc.ports_schema().items():
                    if p.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE):
                        written.add(key_to_idx[proc_topo[port]])
            group_write_idxs[gname] = jnp.array(
                sorted(written), dtype=jnp.int32
            )

        # Precompute per-process index maps for discrete/event handlers.
        discrete_idxs = {
            name: _build_proc_index_maps(
                proc, composite.topology[name], key_to_idx
            )
            for name, proc in discrete_procs.items()
        }
        event_idxs = {
            name: _build_proc_index_maps(
                proc, composite.topology[name], key_to_idx
            )
            for name, proc in event_procs.items()
        }

        # Event tracking: was_active[proc_name] -> bool
        was_active: dict[str, bool] = {n: False for n in event_procs}

        # Trajectory recording — list of flat vectors, unflattened at end.
        save_dt = save_dt or macro_dt
        trajectory_ts: list[float] = [t0]
        trajectory_snapshots: list[jnp.ndarray] = [state]
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
        pbar = None
        if self.progress:
            try:
                from tqdm import tqdm

                pbar = tqdm(
                    total=n_macro, desc="Scheduler", unit="step", leave=False
                )
            except ImportError:
                pbar = None

        # Per-group dt0 hint, threaded across macro steps. After each
        # group's diffeqsolve, we record the average step size as the
        # next call's starting hint — so the adaptive controller doesn't
        # restart from self.dt0=1e-3 every macro step, paying repeated
        # warm-up cost. First call falls back to self.dt0.
        group_dt0_hint: dict[str, float] = {}

        t = t0
        while t < t1 - _TIME_EPS:
            t_next = min(t + current_macro_dt, t1)

            # Snapshot state before splitting (for residual check).
            # Flat state is an immutable jnp.ndarray, so a reference
            # alias is a safe "copy".
            state_before = state if self.adaptive_dt else None

            # 1. Solve continuous groups
            if self.splitting == "strang" and len(group_rhs) > 1:
                t_mid = t + (t_next - t) / 2.0
                items = list(group_rhs.items())
                for gname, rhs_fn in items:
                    state, last_dt = self._solve_group(
                        rhs_fn, state, keys, t, t_mid, group_name=gname,
                        dt0_hint=group_dt0_hint.get(gname),
                    )
                    group_dt0_hint[gname] = last_dt
                    stats[gname]["num_macro_steps"] += 1
                for gname, rhs_fn in reversed(items):
                    state, last_dt = self._solve_group(
                        rhs_fn, state, keys, t_mid, t_next, group_name=gname,
                        dt0_hint=group_dt0_hint.get(gname),
                    )
                    group_dt0_hint[gname] = last_dt
                    stats[gname]["num_macro_steps"] += 1
            else:
                # Lie splitting: sequential, one pass
                prev_interpolant = None
                prev_idxs: jnp.ndarray | None = None
                for gname, rhs_fn in group_rhs.items():
                    if self.coupling_mode == "interpolated":
                        state, prev_interpolant = (
                            self._solve_group_interpolated(
                                rhs_fn,
                                state,
                                t,
                                t_next,
                                prev_interpolant,
                                prev_idxs,
                            )
                        )
                        prev_idxs = group_write_idxs[gname]
                    else:
                        state, last_dt = self._solve_group(
                            rhs_fn, state, keys, t, t_next, group_name=gname,
                            dt0_hint=group_dt0_hint.get(gname),
                        )
                        group_dt0_hint[gname] = last_dt
                    stats[gname]["num_macro_steps"] += 1

            # Adaptive macro_dt: measure coupling residual and adjust
            if self.adaptive_dt:
                num = float(jnp.sum((state - state_before) ** 2))
                den = float(jnp.sum(state_before**2)) + 1e-30
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
                    read_pairs, write_pairs = discrete_idxs[proc_name]
                    view = {p: state[i] for p, i in read_pairs}
                    delta = proc.update(t_next, view)
                    state = _apply_delta(state, delta, write_pairs)

            # 3. Check event conditions
            for proc_name, proc in event_procs.items():
                read_pairs, write_pairs = event_idxs[proc_name]
                view = {p: state[i] for p, i in read_pairs}
                cond = bool(proc.condition(t_next, view))
                if cond and not was_active[proc_name]:
                    delta = proc.handler(t_next, view)
                    state = _apply_delta(state, delta, write_pairs)
                    # Record fired delta keyed by store path for the log.
                    routed = {
                        keys[idx]: delta[port]
                        for port, idx in write_pairs
                        if port in delta
                    }
                    events.append(
                        EventRecord(
                            time=float(t_next),
                            process=proc_name,
                            delta=routed,
                        )
                    )
                was_active[proc_name] = cond

            t = t_next
            if pbar is not None:
                pbar.update(1)

            # Save snapshot if due
            if t - last_save_t >= save_dt - _TIME_EPS or t >= t1 - _TIME_EPS:
                trajectory_ts.append(t)
                trajectory_snapshots.append(state)
                last_save_t = t

        if pbar is not None:
            pbar.close()

        # Assemble result — stack flat snapshots, ship the raw tensor.
        # Shape is (n_time, n_vars) for scalar runs and (n_time, batch,
        # n_vars) for batched runs; the trailing axis matches `keys`.
        ts = jnp.array(trajectory_ts)
        ys = jnp.stack(trajectory_snapshots)

        return SchedulerResult(
            ts=ts, ys=ys, keys=keys, events=events, stats=stats
        )

    def _solve_group(
        self,
        rhs_fn,
        state_vec: jnp.ndarray,
        keys: list[str],
        t0: float,
        t1: float,
        group_name: str = "",
        dt0_hint: float | None = None,
    ) -> tuple[jnp.ndarray, float]:
        """Solve one continuous group from t0 to t1 on the flat state.

        Returns ``(final_state_vec, last_dt_estimate)``. The
        ``last_dt_estimate`` is ``(t1-t0)/num_steps`` and is meant to be
        passed back as the next macro-step's ``dt0_hint`` for this
        group, so the adaptive controller doesn't restart from
        ``self.dt0`` every macro step.
        """
        if t1 <= t0:
            return state_vec, (dt0_hint if dt0_hint is not None else self.dt0)

        state_before = jnp.asarray(state_vec) if self.debug else None

        # Use the prior macro-step's average dt as a starting hint when
        # available; clamp to the remaining interval.
        dt0_base = dt0_hint if dt0_hint is not None else self.dt0
        term = dfx.ODETerm(rhs_fn)
        sol = dfx.diffeqsolve(
            term,
            self.solver,
            t0=t0,
            t1=t1,
            dt0=min(dt0_base, t1 - t0),
            y0=state_vec,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=self.controller,
            adjoint=self.adjoint,
            max_steps=self.max_steps,
        )

        final_vec = sol.ys[-1]
        # Estimate "what step size did the controller settle at?" by
        # averaging over the macro interval. This is a robust proxy that
        # doesn't depend on Diffrax's internal controller_state API.
        last_dt = (t1 - t0) / jnp.maximum(sol.stats["num_steps"], 1)

        if self.debug:
            n_steps = int(sol.stats["num_steps"])
            n_rejected = int(sol.stats["num_rejected_steps"])
            deltas = jnp.abs(final_vec - state_before)
            max_idx = int(jnp.argmax(deltas))
            max_delta = float(deltas[max_idx])
            max_delta_key = keys[max_idx]
            any_nan = bool(jnp.isnan(final_vec).any())
            log.info(
                f"  [{group_name}] [{t0:.1f} → {t1:.1f}]: "
                f"{n_steps} steps ({n_rejected} rej) | "
                f"max Δ: {max_delta_key}={max_delta:.4g}"
                + (" *** NaN ***" if any_nan else "")
            )

        return final_vec, last_dt

    def _solve_group_interpolated(
        self,
        rhs_fn,
        state_vec: jnp.ndarray,
        t0: float,
        t1: float,
        prev_interpolant: Any | None = None,
        prev_idxs: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, Any]:
        """Solve one group with dense output, splicing the previous group's
        evolving variables in via interpolation.

        When ``prev_interpolant`` is provided, this group's RHS sees the
        previous group's evolved variables at the current solver time
        ``t`` (queried from the interpolant), while keeping its own
        evolving ``y`` for everything else.  This is dense-output Lie
        splitting: the next group "feels" the previous group's continuous
        trajectory inside the macro step, not just a frozen snapshot.

        Returns
        -------
        (final_state_vec, interpolant)
            The updated flat state and this group's dense interpolant
            (for passing to the next group).
        """
        if t1 <= t0:
            return state_vec, prev_interpolant

        if prev_interpolant is not None and prev_idxs is not None:
            base_rhs = rhs_fn
            idxs = prev_idxs

            def coupled_rhs(t, y, args=None):
                interp_vec = prev_interpolant.evaluate(t)
                merged = y.at[idxs].set(interp_vec[idxs])
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
            y0=state_vec,
            saveat=dfx.SaveAt(t1=True, dense=True),
            stepsize_controller=self.controller,
                adjoint=self.adjoint,
            max_steps=self.max_steps,
        )

        return sol.ys[-1], sol.interpolation

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
        if t_next % dt_step < _TIME_EPS:
            n_after = int(round(t_next / dt_step))
        return n_after > n_before
