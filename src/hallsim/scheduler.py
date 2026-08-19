"""Scheduler — multi-rate orchestrator for heterogeneous process composites.

Continuous groups (Diffrax, one solve per timescale group), discrete
processes (fired at their ``dt_step``), and event processes (conditions
checked at sync points) advance on a shared ``macro_dt`` communication
interval.

State is a flat ``jnp.ndarray`` indexed by ``sorted(composite.store_paths())``.
Dict shape appears only at the API boundary and inside
``Process.derivative/update/handler``. Scheduling concepts borrow from
Vivarium's Engine (Agmon et al., 2022) and Ptolemy II's Directors.

Example
-------
>>> scheduler = Scheduler()
>>> result = scheduler.run(composite, t_span=(0.0, 1000.0), macro_dt=1.0)
>>> result.ts      # macro step times
>>> result.ys      # (n_time, ..., n_vars); per-path via .get(key)
>>> result.events  # fired event log
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

log = logging.getLogger(__name__)

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
from jax.interpreters import ad
import numpy as np
import optimistix as optx

from hallsim.composite import Composite
from hallsim.config import (
    DEFAULT_ATOL,
    DEFAULT_ATOL_SCALE,
    DEFAULT_NEWTON_ATOL,
    DEFAULT_DT0,
    DEFAULT_MAX_EXPLICIT_SUBSTEPS,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
)
from hallsim.process import PortRole
from hallsim.stiffness import (
    GroupStiffness,
    analyze_groups,
)

# Float epsilon for floating-point time comparisons in the macro-step loop
# (start/end of run, save_dt alignment, dt_step alignment for discrete
# processes). Sized for second-scale ``t_span`` values; runs that span
# ranges < ~1e-9 should pass an explicit save_dt and not rely on this.
_TIME_EPS: float = 1e-12


def _worst_state(ys: jnp.ndarray):
    """``(index, magnitude, any_nonfinite)`` for the largest ``|state|``,
    non-finite ranking above everything. Reduces every leading axis, so it
    takes a final vector or a whole trajectory, batched or not."""
    absy = jnp.abs(ys)
    per_state = jnp.max(
        jnp.where(jnp.isfinite(absy), absy, jnp.inf),
        axis=tuple(range(absy.ndim - 1)),
    )
    idx = jnp.argmax(per_state)
    return idx, per_state[idx], jnp.any(~jnp.isfinite(ys))


def _attach_diagnosis(stats: dict, ys: jnp.ndarray) -> dict:
    """Composite-wide worst-state summary as traced data, so a jitted or
    vmapped run stays diagnosable. ``worst_state_index`` indexes
    ``SchedulerResult.keys``."""
    idx, mag, nonfinite = _worst_state(ys)
    stats["diagnosis"] = {
        "worst_state_index": idx,
        "worst_magnitude": mag,
        "any_nonfinite": nonfinite,
    }
    return stats


@dataclass
class EventRecord:
    """Log entry for a fired event."""

    time: float
    process: str
    delta: dict[str, jnp.ndarray]


@dataclass
class GroupIntegrator:
    """Resolved per-group solver + step-size controller, from
    :meth:`Scheduler._resolve_integrators`. Stiff groups get an implicit
    (A-stable) solver, the rest the cheaper explicit one; both carry the
    scalar controller until :meth:`Scheduler._scaled_tolerances` gives the
    stiff ones their magnitude-scaled vector ``atol`` for the state being
    solved. ``info`` is the stiffness verdict (``None`` in manual-solver
    mode, where none runs)."""

    solver: dfx.AbstractSolver
    controller: dfx.AbstractStepSizeController
    stiff: bool
    info: GroupStiffness | None = None


@dataclass
class SchedulerResult:
    """JAX-native scheduler output. ``ys`` is the raw stacked tensor (Diffrax's
    ``sol.ys`` convention), so it composes with ``vmap``/``grad`` without a
    Python-side dict of arrays in the middle.

    ``ts`` is ``(n_time,)``; ``ys`` is ``(n_time, n_vars)`` or
    ``(n_time, batch, n_vars)``, trailing axis indexed by ``keys`` (every
    store path, including materialized ASSIGNED ports). ``stats`` carries
    per-group solver statistics plus a ``"diagnosis"`` entry as traced,
    batch-carrying arrays — pair with :attr:`ok` to diagnose a jitted or
    vmapped run."""

    ts: jnp.ndarray
    ys: jnp.ndarray
    keys: list[str]
    events: list[EventRecord] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def _index(self) -> dict[str, int]:
        """``key -> trailing-axis position``, built once. A list scan per
        readout is O(n_vars), and callers read every path in a loop."""
        return {k: i for i, k in enumerate(self.keys)}

    def get(self, key: str) -> jnp.ndarray:
        """Per-path trajectory — ``(n_time,)``, or ``(n_time, batch)``
        batched."""
        return self.ys[..., self._index[key]]

    @property
    def ok(self) -> jnp.ndarray:
        """Did every group solve? Traced boolean carrying the batch axis under
        ``vmap`` — with ``throw=False``, the thing to branch on."""
        codes = [
            st["result"]
            for st in self.stats.values()
            if isinstance(st, dict) and "result" in st
        ]
        if not codes:
            return jnp.asarray(True)
        return jnp.all(
            jnp.stack(
                [jnp.asarray(c == dfx.RESULTS.successful) for c in codes]
            ),
            axis=0,
        )

    def __contains__(self, key: str) -> bool:
        return key in self._index


def _build_proc_index_maps(
    proc, proc_topo: dict[str, str], key_to_idx: dict[str, int]
) -> tuple[tuple, tuple]:
    """``(read_pairs, write_pairs)`` for a discrete/event process — gather the
    port view from state, scatter LATCHED writes back. Each is
    ``((port_name, store_idx), ...)``."""
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


def _interp_uniform(
    t: jnp.ndarray, t0: jnp.ndarray, t1: jnp.ndarray, ys: jnp.ndarray
) -> jnp.ndarray:
    """Linear interpolation along axis 0 of a uniform ``(K, ...)`` sample over
    ``[t0, t1]``, at scalar ``t``. Static ``K`` keeps the coupling
    representation a constant shape so the loop compiles under ``lax.scan``,
    unlike a dense interpolant sized by the adaptive step count."""
    k = ys.shape[0]
    pos = jnp.clip((t - t0) / (t1 - t0) * (k - 1), 0.0, k - 1.0)
    i0 = jnp.minimum(jnp.floor(pos).astype(jnp.int32), k - 2)
    frac = pos - i0
    return ys[i0] * (1.0 - frac) + ys[i0 + 1] * frac


class _FrozenFill(eqx.Module):
    """Exogenous states held at their macro-step-start values."""

    full: jnp.ndarray

    def __call__(self, t):
        return self.full


class _InterpFill(eqx.Module):
    """Exogenous states with the previous group's own states read from its
    dense output — interpolated coupling's time-varying inputs."""

    full: jnp.ndarray
    t0: jnp.ndarray
    t1: jnp.ndarray
    ys: jnp.ndarray
    idx: jnp.ndarray

    def __call__(self, t):
        return self.full.at[..., self.idx].set(
            _interp_uniform(t, self.t0, self.t1, self.ys)
        )


class _ReducedRHS(eqx.Module):
    """A group's RHS over only the states it evolves.

    The full-width RHS zeroes every state the group does not own, but the
    solver cannot see that: an implicit method forms its stage system over
    whatever it is handed, so a 24-state group inside a 52-state composite
    factorizes a 52x52 Jacobian for 24 unknowns, and the waste grows with
    every model composed. States this group does not own are not unknowns —
    they are inputs, supplied by ``fill`` at the time asked for. Frozen
    coupling makes that a constant, interpolated a function of ``t``; the
    solved dimension is the same either way.

    A Module, not a closure, for the same reason as :class:`_FlatRHS`.
    """

    base: Any
    own: jnp.ndarray
    fill: Any

    def __call__(self, t, y_own, args=None):
        full = self.fill(t).at[..., self.own].set(y_own)
        return self.base(t, full, args)[..., self.own]


class Scheduler:
    """Multi-rate orchestrator for composites with mixed process kinds.

    Continuous groups are solved sequentially by Lie operator splitting, each
    seeing the previous group's updated state. Single-group composites with no
    events / discrete / adaptive_dt / Strang / interpolated coupling take a
    fast path: one ``dfx.diffeqsolve`` over the whole ``t_span``.

    Parameters
    ----------
    solver:
        Pin one Diffrax solver for every group. Pinning switches per-group
        routing off (and warns), so it always means that solver everywhere.
        ``None`` (default) leaves the choice to ``auto_stiffness``.
    auto_stiffness:
        Per-group solver routing, on by default. Each group's local Jacobian
        spectrum is measured once eagerly via
        :func:`hallsim.stiffness.analyze_groups`; stiff groups get
        ``implicit_solver`` with a magnitude-scaled vector ``atol``, the rest
        keep ``explicit_solver`` and the scalar controller. Routing needs a
        concrete Jacobian — under grad/jvp/vmap with a cold cache it warns and
        falls back to ``explicit_solver``, so call :meth:`warm_up` once
        eagerly first (``CalibrationProblem`` does this for you).
    explicit_solver, implicit_solver:
        The two solvers routing chooses between. Default ``Tsit5()`` and
        ``Kvaerno5()`` with an ``optx.Newton`` root finder — diffrax's default
        ``VeryChord`` needs ~18x more steps on real biochemical RHSs
        (see docs/benchmarks.md). ``explicit_solver`` is also the fallback
        whenever routing is unavailable.
    max_explicit_substeps:
        Stiffness threshold: a group is stiff when its fastest decay rate ×
        ``macro_dt`` exceeds this. Default 100.
    rtol, atol:
        Adaptive-stepping tolerances, default ``1e-6`` / ``1e-9``. Oscillatory
        biology is accuracy-limited; loosening this without screening every
        oscillator risks numerical anti-damping (see CLAUDE.md). For stiff
        groups ``atol`` is the *floor* of a vector tolerance
        ``max(atol, atol_scale·|y0|)``.
    atol_scale:
        Relative coefficient of the stiff-group vector ``atol`` (default 1e-6).
    newton_rtol, newton_atol:
        Convergence tolerances of the Newton solve *inside* each implicit
        stage — algebraic, not an accuracy target. ``newton_rtol`` defaults to
        ``rtol``; ``newton_atol`` defaults to ``1e-6`` and is deliberately not
        ``atol``, which on a model whose smallest state is orders below 1 asks
        every stage to converge far past the state itself and exhausts the
        step budget. Ignored when ``implicit_solver`` is passed explicitly.
    max_steps:
        Safety limit on solver steps per macro step.
    dt0:
        Initial step size for the adaptive controller.
    groups:
        Manual group assignment ``{group_name: [proc_name, ...]}``. ``None``
        uses ``composite.auto_groups()``.
    coupling_mode:
        How inter-group state is communicated during Lie splitting.

        - ``"auto"`` (default): ``"interpolated"`` when the composite has a
          forward cross-group edge, else ``"frozen"`` — so the extra cost is
          paid only where it buys accuracy.
        - ``"frozen"``: each group sees the previous group's final state.
        - ``"interpolated"``: each group reads a fixed-size
          (``coupling_interp_points``) sample of the previous group's
          trajectory, interpolated at ``t``, so it feels that trajectory
          within the macro step. Fixed sample size keeps the loop
          ``lax.scan``-compilable. Requires ``splitting="lie"``.
    splitting:
        ``"lie"`` (default, O(macro_dt)) or ``"strang"`` (symmetric
        half-steps, O(macro_dt²)).
    adaptive_dt:
        PLL-inspired adaptive ``macro_dt`` sizing off the coupling residual:
        shrink above ``adaptive_dt_rho_max``, grow after
        ``adaptive_dt_grow_wait`` steps below ``adaptive_dt_rho_min``, by
        ``adaptive_dt_factor``, bounded by ``adaptive_dt_min`` / ``_max``
        (default ``macro_dt/64`` and ``macro_dt*4``).
    throw:
        If ``True`` (default), a group that does not return
        ``RESULTS.successful`` raises a labelled error via ``eqx.error_if``
        (works under JIT/grad). ``False`` records the code in
        ``result.stats[group]['result']`` instead — the path for diagnosing a
        non-converging composite.
    progress:
        tqdm bar over macro steps. Default ``False``: the Python-side update
        is a side effect that interferes with ``vmap`` over batched runs.
    """

    def __init__(
        self,
        solver: dfx.AbstractSolver | None = None,
        rtol: float = DEFAULT_RTOL,
        atol: float = DEFAULT_ATOL,
        max_steps: int = DEFAULT_MAX_STEPS,
        dt0: float = DEFAULT_DT0,
        explicit_solver: dfx.AbstractSolver | None = None,
        implicit_solver: dfx.AbstractSolver | None = None,
        auto_stiffness: bool = True,
        atol_scale: float = DEFAULT_ATOL_SCALE,
        newton_rtol: float | None = None,
        newton_atol: float = DEFAULT_NEWTON_ATOL,
        max_explicit_substeps: float = DEFAULT_MAX_EXPLICIT_SUBSTEPS,
        groups: dict[str, list[str]] | None = None,
        coupling_mode: str = "auto",
        coupling_interp_points: int = 16,
        splitting: str = "lie",
        adaptive_dt: bool = False,
        adaptive_dt_rho_max: float = 0.5,
        adaptive_dt_rho_min: float = 0.01,
        adaptive_dt_grow_wait: int = 3,
        adaptive_dt_factor: float = 2.0,
        adaptive_dt_min: float | None = None,
        adaptive_dt_max: float | None = None,
        adjoint: dfx.AbstractAdjoint | None = None,
        throw: bool = True,
        debug: bool = False,
        progress: bool = False,
    ) -> None:
        if coupling_mode not in ("auto", "frozen", "interpolated"):
            raise ValueError(
                f"coupling_mode must be 'auto', 'frozen', or "
                f"'interpolated', got {coupling_mode!r}"
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
        # Solver selection. Routing is on by default; a pinned `solver` wins
        # over it — "this solver, everywhere" is unambiguous, so honour it
        # and say out loud that routing is off rather than silently ignoring
        # one of the two arguments.
        if auto_stiffness and solver is not None:
            log.warning(
                "solver=%s is pinned, so per-group stiffness routing is off "
                "and every group uses it. Drop solver= to let the Scheduler "
                "route stiff groups to %s with a magnitude-scaled vector "
                "atol.",
                type(solver).__name__,
                type(implicit_solver or dfx.Kvaerno5()).__name__,
            )
            auto_stiffness = False
        self.auto_solver = auto_stiffness
        self.explicit_solver = explicit_solver or dfx.Tsit5()
        # Default stiff solver is Kvaerno5 with a **Newton** root finder.
        # diffrax's default `VeryChord` (stale-Jacobian chord, 10 iters)
        # rejects ~50% of steps on real biochemical RHSs; a true Newton
        # solve (fresh Jacobian — what CVODE does) cuts that to a few %.
        self.implicit_solver = implicit_solver or dfx.Kvaerno5(
            root_finder=optx.Newton(
                rtol=rtol if newton_rtol is None else newton_rtol,
                atol=newton_atol,
            )
        )
        self.solver = solver or self.explicit_solver
        self.atol_scale = atol_scale
        self.max_explicit_substeps = max_explicit_substeps
        self.rtol = rtol
        self.atol = atol
        self.controller = dfx.PIDController(rtol=rtol, atol=atol)
        # Per-(group structure) cache of resolved integrators. Keyed by a
        # structural signature so the eager resolution (concrete params)
        # is reused under later grad/jvp/vmap tracing, where the Jacobian
        # eigenvalues would be tracers.
        self._integrator_cache: dict[Any, dict[str, GroupIntegrator]] = {}
        self._omega_cache: dict[Any, list] = {}  # antialias spectrum cache
        # Compiled continuous cores by structure. Not an optimization: an
        # unjitted run re-traces its scan of N diffeqsolve bodies every call.
        self._core_cache: dict[Any, Any] = {}
        self._warned_save_res = False
        # Adjoint method used by every diffeqsolve in this run.
        # Default (None) → diffrax picks RecursiveCheckpointAdjoint, which
        # is memory-cheap but step-expensive. For calibration through
        # stiff/oscillatory composites, pass dfx.BacksolveAdjoint() for
        # near-forward-cost backward passes.
        self.adjoint = adjoint or dfx.RecursiveCheckpointAdjoint()
        self._adjoint_explicit = adjoint is not None
        self.throw = throw
        self.max_steps = max_steps
        self.dt0 = dt0
        self._jump_ts = None  # per-run forcing/step discontinuity times
        self.manual_groups = groups
        self.coupling_mode = coupling_mode
        self.coupling_interp_points = coupling_interp_points
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
        adjoint: dfx.AbstractAdjoint | None = None,
        antialias: bool = True,
    ) -> SchedulerResult:
        """Run the composite with multi-rate scheduling.

        Parameters
        ----------
        composite:
            Wired bundle of processes.
        t_span:
            ``(t0, t1)``.
        macro_dt:
            Communication interval (initial value under ``adaptive_dt``). Each
            macro step solves the continuous groups, fires any due discrete
            processes, then checks event conditions.
        y0:
            Initial state ``(n_vars,)``, or ``(batch, n_vars)`` for a
            population run. ``None`` uses ``composite.initial_state_vec()``.
        save_dt:
            Output density, decoupled from ``macro_dt`` via dense output — it
            costs memory, not ODE steps. ``None`` uses
            ``_DEFAULT_SAVE_SAMPLES`` points across the span, refined for
            aliasing (see ``antialias``).
        adjoint:
            Per-run override of the differentiation method. ``None`` uses the
            constructor's. Pass ``dfx.ForwardMode()`` for forward-mode
            calibration without changing scheduler identity, so one instance
            (and its stiffness cache) serves both the eager evaluate pass and
            the differentiated loss.
        antialias:
            Nyquist guardrail: refine the save grid finer (never coarser) if
            ``save_dt`` would undersample the fastest oscillation and alias a
            raw readout. ``False`` takes the grid verbatim.

        Returns
        -------
        :class:`SchedulerResult`
        """
        t0, t1 = t_span
        adjoint = (
            adjoint
            if adjoint is not None
            else self._resolve_adjoint(composite, y0)
        )
        # Discontinuities the solver should land on exactly rather than
        # resolve by adaptive step-rejection.
        self._jump_ts = self._collect_jump_ts(composite, t0, t1)

        # Flat state layout, pinned for the whole run. Batched y0 is
        # (batch, n_vars); the batch axis rides through every group's solve.
        keys = composite.store_keys()
        state = (
            composite.initial_state_vec(keys)
            if y0 is None
            else jnp.asarray(y0)
        )
        key_to_idx = {k: i for i, k in enumerate(keys)}

        groups = self.manual_groups or composite.auto_groups()
        discrete_procs = composite.discrete_processes()
        event_procs = composite.event_processes()

        # These paths branch on Python bool()/float() of the state, which
        # would crash under vmap or silently collapse the batch axis.
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

        coupling = self._effective_coupling(composite, groups, keys)
        integrators = self._resolve_integrators(
            composite, groups, state, t0, macro_dt
        )
        requested_save_dt = (
            save_dt
            if save_dt is not None
            else (t1 - t0) / max(1, self._DEFAULT_SAVE_SAMPLES - 1)
        )
        save_dt = (
            self._resolve_save_dt(
                self._group_omegas(
                    composite, groups, integrators, state, t0, macro_dt
                ),
                requested_save_dt,
            )
            if antialias
            else requested_save_dt
        )

        # Fast path: one diffeqsolve over the full span, so the adaptive
        # stepper sizes its steps once instead of restarting per macro step.
        fast_path_eligible = (
            len(groups) == 1
            and not discrete_procs
            and not event_procs
            and not self.adaptive_dt
            and self.splitting == "lie"
            and coupling == "frozen"
        )
        # Scan path: a statically-known macro-step count, so the whole
        # multi-group run is one lax.scan — bounded compile, reverse-mode
        # memory flat in macro-step count. adaptive_dt (data-dependent step
        # count) and debug logging fall through to the eager loop.
        scan_eligible = (
            not discrete_procs
            and not event_procs
            and not self.adaptive_dt
            and coupling in ("frozen", "interpolated")
            and not self.debug
        )
        if fast_path_eligible or scan_eligible:
            core = self._continuous_core(
                composite,
                groups,
                integrators,
                keys,
                t0,
                t1,
                macro_dt,
                save_dt,
                adjoint,
                coupling,
                fast=fast_path_eligible,
                state=state,
            )
            ts, ys, dyn = core(composite, state)
            stats = {
                g: {
                    **dyn[g],
                    "solver": type(integrators[g].solver).__name__,
                    "stiff": integrators[g].stiff,
                }
                for g in groups
            }
            return SchedulerResult(
                ts=ts,
                ys=ys,
                keys=keys,
                events=[],
                stats=_attach_diagnosis(stats, ys),
            )

        # Eager path: no compiled core to carry the tolerance, so scale it here
        # against this run's initial state.
        integrators = self._scaled_tolerances(
            integrators, composite, groups, keys, state
        )

        # Per-group RHS, plus the indices each group writes — interpolated
        # coupling splices the previous group's state at those positions.
        group_rhs: dict[str, Any] = {}
        group_write_idxs: dict[str, jnp.ndarray] = {}
        for gname, proc_names in groups.items():
            fn, _ = composite.build_rhs(proc_names)
            group_rhs[gname] = fn
            group_write_idxs[gname] = composite.evolved_indices(
                proc_names, keys
            )

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

        was_active: dict[str, bool] = {n: False for n in event_procs}

        save_dt = save_dt or macro_dt
        trajectory_ts: list[float] = [t0]
        trajectory_snapshots: list[jnp.ndarray] = [state]
        last_save_t = t0
        events: list[EventRecord] = []

        # Carries the latest RESULTS code so a non-converging composite stays
        # inspectable through the API under throw=False.
        stats: dict[str, Any] = {
            gname: {
                "num_macro_steps": 0,
                "num_solver_steps": 0,
                "num_rejected_steps": 0,
                "result": dfx.RESULTS.successful,
                "solver": type(integrators[gname].solver).__name__,
                "stiff": integrators[gname].stiff,
            }
            for gname in groups
        }

        def _record(gname: str, diag) -> None:
            result, n_steps, n_rej = diag
            st = stats[gname]
            st["num_solver_steps"] = st["num_solver_steps"] + n_steps
            st["num_rejected_steps"] = st["num_rejected_steps"] + n_rej
            st["result"] = result

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

        # Each group's settled step size warm-starts the next macro step, so
        # the controller doesn't restart from self.dt0 every time.
        group_dt0_hint: dict[str, float] = {}

        t = t0
        while t < t1 - _TIME_EPS:
            t_next = min(t + current_macro_dt, t1)
            state_before = state if self.adaptive_dt else None

            if self.splitting == "strang" and len(group_rhs) > 1:
                t_mid = t + (t_next - t) / 2.0
                items = list(group_rhs.items())
                for gname, rhs_fn in items:
                    state, last_dt, diag = self._solve_group(
                        rhs_fn,
                        state,
                        group_write_idxs[gname],
                        keys,
                        t,
                        t_mid,
                        integ=integrators[gname],
                        adjoint=adjoint,
                        group_name=gname,
                        dt0_hint=group_dt0_hint.get(gname),
                    )
                    group_dt0_hint[gname] = last_dt
                    stats[gname]["num_macro_steps"] += 1
                    _record(gname, diag)
                for gname, rhs_fn in reversed(items):
                    state, last_dt, diag = self._solve_group(
                        rhs_fn,
                        state,
                        group_write_idxs[gname],
                        keys,
                        t_mid,
                        t_next,
                        integ=integrators[gname],
                        adjoint=adjoint,
                        group_name=gname,
                        dt0_hint=group_dt0_hint.get(gname),
                    )
                    group_dt0_hint[gname] = last_dt
                    stats[gname]["num_macro_steps"] += 1
                    _record(gname, diag)
            else:  # Lie: sequential, one pass
                prev_interpolant = None
                prev_idxs: jnp.ndarray | None = None
                for gname, rhs_fn in group_rhs.items():
                    if coupling == "interpolated":
                        state, prev_interpolant, last_dt, diag = (
                            self._solve_group_interpolated(
                                rhs_fn,
                                state,
                                group_write_idxs[gname],
                                t,
                                t_next,
                                prev_interpolant,
                                prev_idxs,
                                integ=integrators[gname],
                                adjoint=adjoint,
                                dt0_hint=group_dt0_hint.get(gname),
                            )
                        )
                        group_dt0_hint[gname] = last_dt
                        _record(gname, diag)
                        prev_idxs = group_write_idxs[gname]
                    else:
                        state, last_dt, diag = self._solve_group(
                            rhs_fn,
                            state,
                            group_write_idxs[gname],
                            keys,
                            t,
                            t_next,
                            integ=integrators[gname],
                            adjoint=adjoint,
                            group_name=gname,
                            dt0_hint=group_dt0_hint.get(gname),
                        )
                        group_dt0_hint[gname] = last_dt
                        _record(gname, diag)
                    stats[gname]["num_macro_steps"] += 1

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

            for proc_name, proc in discrete_procs.items():
                if self._is_due(t, t_next, proc.dt_step):
                    read_pairs, write_pairs = discrete_idxs[proc_name]
                    view = {p: state[i] for p, i in read_pairs}
                    delta = proc.update(t_next, view)
                    state = _apply_delta(state, delta, write_pairs)

            for proc_name, proc in event_procs.items():
                read_pairs, write_pairs = event_idxs[proc_name]
                view = {p: state[i] for p, i in read_pairs}
                cond = bool(proc.condition(t_next, view))
                if cond and not was_active[proc_name]:
                    delta = proc.handler(t_next, view)
                    state = _apply_delta(state, delta, write_pairs)
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

            if t - last_save_t >= save_dt - _TIME_EPS or t >= t1 - _TIME_EPS:
                trajectory_ts.append(t)
                trajectory_snapshots.append(state)
                last_save_t = t

        if pbar is not None:
            pbar.close()

        ts = jnp.array(trajectory_ts)
        ys = jnp.stack(trajectory_snapshots)
        ys = composite.materialize_assigned(ts, ys)

        return SchedulerResult(
            ts=ts,
            ys=ys,
            keys=keys,
            events=events,
            stats=_attach_diagnosis(stats, ys),
        )

    def _continuous_core(
        self,
        composite,
        groups,
        integrators,
        keys,
        t0,
        t1,
        macro_dt,
        save_dt,
        adjoint,
        coupling,
        *,
        fast,
        state,
    ):
        """Compiled ``(composite, y0) -> (ts, ys, per_group_stats)`` for the
        continuous paths, cached by structure.

        The caller resolves groups, coupling, integrators and the save grid
        eagerly — stiffness routing needs a concrete Jacobian — and this
        captures them, so only ``composite`` (its parameter arrays) and ``y0``
        cross the trace boundary. Without the cache each call rebuilds a jaxpr
        containing one ``diffeqsolve`` per group per macro step, which costs
        hundreds of ms while reporting zero recompiles.
        """
        sig = (
            tuple(sorted((g, tuple(sorted(p))) for g, p in groups.items())),
            coupling,
            self.splitting,
            bool(fast),
            float(t0),
            float(t1),
            float(macro_dt),
            None if save_dt is None else float(save_dt),
            type(adjoint).__name__,
            tuple(
                type(integrators[g].solver).__name__ for g in sorted(groups)
            ),
            tuple(bool(integrators[g].stiff) for g in sorted(groups)),
            tuple(state.shape),
            str(state.dtype),
            (
                None
                if self._jump_ts is None
                else tuple(float(t) for t in self._jump_ts)
            ),
        )
        cached = self._core_cache.get(sig)
        if cached is not None:
            return cached

        if fast:
            (gname,) = groups
            (proc_names,) = groups.values()
            save_ts = self._save_grid(
                t0, t1, save_dt if save_dt is not None else macro_dt
            )

            own = composite.evolved_indices(proc_names, keys)

            def core(comp, y0):
                integ = self._scaled_tolerances(
                    integrators, composite, groups, keys, y0
                )[gname]
                rhs_fn, _ = comp.build_rhs(proc_names)
                sol, ys = self._reduced_solve(
                    rhs_fn,
                    y0,
                    own,
                    t0,
                    t1,
                    integ,
                    adjoint,
                    dfx.SaveAt(ts=save_ts),
                    group_name=gname,
                    keys=keys,
                )
                # ASSIGNED ports aren't integrated, so their saved columns
                # hold a stale initial value until recomputed per saved state.
                ys = comp.materialize_assigned(sol.ts, ys)
                return (
                    sol.ts,
                    ys,
                    {
                        gname: {
                            "num_macro_steps": 1,
                            "num_solver_steps": sol.stats.get("num_steps"),
                            "num_rejected_steps": sol.stats.get(
                                "num_rejected_steps", 0
                            ),
                            "result": sol.result,
                        }
                    },
                )

        else:

            def core(comp, y0):
                return self._run_scan_continuous(
                    comp,
                    groups,
                    self._scaled_tolerances(
                        integrators, composite, groups, keys, y0
                    ),
                    y0,
                    keys,
                    t0,
                    t1,
                    macro_dt,
                    save_dt,
                    adjoint,
                    coupling,
                )

        if state.ndim > 1:
            core = self._per_member(core)
        fn = eqx.filter_jit(core)
        # Only cache a core built eagerly. Built under an outer trace it can
        # close over that trace's tracers, which would escape it on reuse.
        if not any(
            isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree_util.tree_leaves((composite, state))
        ):
            self._core_cache[sig] = fn
        return fn

    @staticmethod
    def _per_member(core):
        """Map a single-member ``core`` over the leading batch axis.

        Batch members are independent, and the state layout has to say so: as
        one flat ``(batch, n_vars)`` state an implicit solver treats it as a
        single unknown vector and factorizes a dense ``(batch·n_vars)²``
        Jacobian, cubic in population size. Per member the solve is
        ``n_vars``-sized, which is the block structure the Jacobian already
        has. ``ys`` keeps the ``(n_time, batch, n_vars)`` layout; per-group
        stats come back per-member.
        """
        mapped = eqx.filter_vmap(
            core,
            in_axes=(None, eqx.if_array(0)),
            out_axes=(eqx.if_array(0), eqx.if_array(1), eqx.if_array(0)),
        )

        def batched(comp, y0):
            ts, ys, stats = mapped(comp, y0)
            return ts[0], ys, stats

        return batched

    def _effective_coupling(
        self,
        composite: Composite,
        groups: dict[str, list[str]],
        keys: list[str],
    ) -> str:
        """Resolve ``coupling_mode="auto"`` for this run.

        Interpolated only improves accuracy on a *forward* cross-group edge —
        an earlier group's variables read by a later one. Without such an edge
        it does identical work at higher cost, so auto picks frozen (as do
        Strang and single-group runs). Explicit modes pass through.
        """
        if self.coupling_mode != "auto":
            return self.coupling_mode
        if self.splitting == "strang" or len(groups) < 2:
            return "frozen"
        # Static structure only (no jnp), so this stays concrete when run()
        # is traced.
        key_to_idx = {k: i for i, k in enumerate(keys)}
        items = list(groups.items())
        writes: list[set[int]] = []
        reads: list[set[int]] = []
        for _, procs in items:
            w: set[int] = set()
            r: set[int] = set()
            for pname in procs:
                topo_p = composite.topology[pname]
                schema = composite.processes[pname].ports_schema()
                for port, path in topo_p.items():
                    if path not in key_to_idx:
                        continue
                    r.add(key_to_idx[path])
                    p = schema.get(port)
                    if p is not None and p.role == PortRole.EVOLVED:
                        w.add(key_to_idx[path])
            writes.append(w)
            reads.append(r)
        for a in range(len(items)):
            for b in range(a + 1, len(items)):
                if writes[a] & reads[b]:
                    return "interpolated"
        return "frozen"

    def _run_scan_continuous(
        self,
        composite: Composite,
        groups: dict[str, list[str]],
        integrators: dict[str, GroupIntegrator],
        state: jnp.ndarray,
        keys: list[str],
        t0: float,
        t1: float,
        macro_dt: float,
        save_dt: float | None,
        adjoint: dfx.AbstractAdjoint,
        coupling: str = "frozen",
    ):
        """Continuous-only multi-group run as a single ``lax.scan``, returning
        ``(ts, ys, per_group_stats)``.

        One fixed-length scan over ``macro_dt`` windows; each solves every
        group in turn (Lie) or forward-then-reversed (Strang), threading each
        group's settled step size as the next window's warm-start. The whole
        trajectory compiles to one executable, so compile time and
        reverse-mode memory don't scale with the macro-step count.
        """
        group_rhs = [
            (g, composite.build_rhs(procs)[0]) for g, procs in groups.items()
        ]
        n_groups = len(group_rhs)
        strang = self.splitting == "strang" and n_groups > 1
        interp = coupling == "interpolated" and n_groups > 1
        write_idxs = [
            composite.evolved_indices(p, keys) for p in groups.values()
        ]

        n_macro = int(round((t1 - t0) / macro_dt))
        t_starts = t0 + macro_dt * jnp.arange(n_macro)

        # n_save equally-spaced times per macro window, endpoints included so
        # the sample doubles as interpolated coupling's representation.
        # Dropping the left endpoint (it repeats the previous window's end)
        # decouples output resolution from the coupling cadence.
        if strang or not save_dt or save_dt >= macro_dt:
            base_out = 1
        else:
            base_out = max(1, int(round(macro_dt / save_dt)))
        n_save = base_out + 1
        if interp:
            n_save = max(n_save, self.coupling_interp_points)
        n_out = 1 if strang else n_save - 1
        save_frac = jnp.linspace(0.0, 1.0, n_save)

        dt0_init = jnp.full((n_groups,), float(self.dt0))
        steps_init = jnp.zeros((n_groups,), jnp.int64)
        rej_init = jnp.zeros((n_groups,), jnp.int64)
        res_init = tuple(dfx.RESULTS.successful for _ in range(n_groups))

        def solve_dense(gi, st, t_a, t_b, dt0hi, fill):
            g, rhs = group_rhs[gi]
            own = write_idxs[gi]
            grid = t_a + save_frac * (t_b - t_a)
            sol, saved = self._reduced_solve(
                rhs,
                st,
                own,
                t_a,
                t_b,
                integrators[g],
                adjoint,
                dfx.SaveAt(ts=grid),
                fill=fill,
                dt0_hint=dt0hi,
                group_name=g,
                keys=keys,
            )
            final = saved[-1]
            ld = (t_b - t_a) / jnp.maximum(sol.stats["num_steps"], 1)
            return (
                final,
                sol.ys,
                ld,
                sol.stats["num_steps"],
                sol.stats["num_rejected_steps"],
                sol.result,
            )

        def body(carry, t_start):
            st, dt0h, steps, rej, res = carry
            t_next = jnp.minimum(t_start + macro_dt, t1)
            dt0_next = [None] * n_groups

            if strang:
                t_mid = t_start + (t_next - t_start) / 2.0
                for gi in range(n_groups):
                    g, rhs = group_rhs[gi]
                    st, _, r, ns, nr = self._group_step(
                        rhs,
                        st,
                        write_idxs[gi],
                        t_start,
                        t_mid,
                        integrators[g],
                        adjoint,
                        g,
                        dt0h[gi],
                    )
                    steps, rej = steps.at[gi].add(ns), rej.at[gi].add(nr)
                    res = res[:gi] + (r,) + res[gi + 1 :]
                for gi in range(n_groups - 1, -1, -1):
                    g, rhs = group_rhs[gi]
                    st, ld, r, ns, nr = self._group_step(
                        rhs,
                        st,
                        write_idxs[gi],
                        t_mid,
                        t_next,
                        integrators[g],
                        adjoint,
                        g,
                        dt0h[gi],
                    )
                    steps, rej = steps.at[gi].add(ns), rej.at[gi].add(nr)
                    res = res[:gi] + (r,) + res[gi + 1 :]
                    dt0_next[gi] = ld
                traj = st[None]
            else:
                traj = jnp.broadcast_to(st, (n_out,) + st.shape)
                prev = None
                for gi in range(n_groups):
                    if interp and prev is not None:
                        p_t0, p_t1, p_ys = prev
                        fill = _InterpFill(
                            full=st,
                            t0=p_t0,
                            t1=p_t1,
                            ys=p_ys,
                            idx=write_idxs[gi - 1],
                        )
                    else:
                        fill = _FrozenFill(full=st)

                    st, gy, ld, ns, nr, r = solve_dense(
                        gi, st, t_start, t_next, dt0h[gi], fill
                    )
                    w = write_idxs[gi]
                    traj = traj.at[:, w].set(gy[1:])
                    steps, rej = steps.at[gi].add(ns), rej.at[gi].add(nr)
                    res = res[:gi] + (r,) + res[gi + 1 :]
                    dt0_next[gi] = ld
                    prev = (t_start, t_next, gy)

            carry = (st, jnp.stack(dt0_next), steps, rej, res)
            return carry, traj

        init = (state, dt0_init, steps_init, rej_init, res_init)
        (final_state, _, steps_tot, rej_tot, res_final), ys_stack = (
            jax.lax.scan(body, init, t_starts)
        )

        # y0, then each window's n_out fresh samples; flatten window into time.
        step_len = jnp.minimum(t_starts + macro_dt, t1) - t_starts
        out_frac = jnp.asarray([1.0]) if strang else save_frac[1:]
        sub_ts = t_starts[:, None] + out_frac[None, :] * step_len[:, None]
        all_ts = jnp.concatenate([jnp.asarray([t0]), sub_ts.reshape(-1)])
        all_ys = jnp.concatenate(
            [state[None], ys_stack.reshape((n_macro * n_out,) + state.shape)],
            axis=0,
        )
        if n_out == 1 and save_dt and save_dt > macro_dt:
            stride = max(1, int(round(save_dt / macro_dt)))
            keep = list(range(0, n_macro + 1, stride))
            if keep[-1] != n_macro:
                keep.append(n_macro)
            keep = jnp.asarray(keep)
            ts, ys = all_ts[keep], all_ys[keep]
        else:
            ts, ys = all_ts, all_ys

        stats = {
            g: {
                "num_macro_steps": n_macro * (2 if strang else 1),
                "num_solver_steps": steps_tot[gi],
                "num_rejected_steps": rej_tot[gi],
                "result": res_final[gi],
            }
            for gi, (g, _) in enumerate(group_rhs)
        }
        return ts, composite.materialize_assigned(ts, ys), stats

    @staticmethod
    def _integrator_signature(groups, state, macro_dt):
        """Structural key for the integrator cache: group→process structure,
        state width, and ``macro_dt``, never state *values* — so the eager
        resolution is reused under later traced runs of the same composite."""
        gstruct = tuple(
            sorted((g, tuple(sorted(procs))) for g, procs in groups.items())
        )
        return (gstruct, int(state.shape[-1]), float(macro_dt))

    @staticmethod
    def _save_grid(t0: float, t1: float, save_step: float) -> jnp.ndarray:
        """Save times spanning ``[t0, t1]`` inclusive, ~``save_step`` apart.

        Picks the point *count* and places them with ``linspace``, which pins
        both endpoints and never overshoots ``t1`` — a ``SaveAt`` time past
        ``t1`` makes ``diffeqsolve`` raise. ``save_step`` need not divide the
        span. ``n`` is a Python int, so the output shape is static.

        Returns **numpy**, not ``jnp``: the grid is a constant captured by the
        compiled core, and a ``jnp`` array built while an outer trace is live
        would be a tracer that escapes it."""
        span = t1 - t0
        if not save_step or save_step <= 0.0 or save_step >= span:
            return np.asarray([t0, t1], dtype=float)
        n = int(round(span / save_step)) + 1
        return np.linspace(t0, t1, n)

    #: samples per oscillation period the auto-reducer targets (well under the
    #: Nyquist limit of 2, so a raw amplitude/RMS readout is faithful).
    _SAVE_SAMPLES_PER_PERIOD = 10.0
    # Default output density when the caller passes no ``save_dt`` — points
    # across the span, before the antialias refinement.
    _DEFAULT_SAVE_SAMPLES = 201

    def _collect_jump_ts(self, composite, t0, t1):
        """Sorted interior discontinuity times from ``Process.discontinuity_times``
        within ``(t0, t1)`` — forcing-pulse edges, timed steps — for
        :class:`diffrax.ClipStepSizeController`, or ``None``. Endpoints are
        excluded; the solver already lands on them. Numpy for the same reason
        as :meth:`_save_grid`."""
        times = {
            float(t)
            for proc in composite.processes.values()
            for t in proc.discontinuity_times()
            if t0 < float(t) < t1
        }
        return np.asarray(sorted(times)) if times else None

    def _controller_with_jumps(self, controller):
        """Wrap a step-size controller so it steps exactly onto this run's
        ``jump_ts`` (forcing/step discontinuities); pass-through when there
        are none."""
        if self._jump_ts is None:
            return controller
        return dfx.ClipStepSizeController(controller, jump_ts=self._jump_ts)

    def _resolve_adjoint(self, composite, y0):
        """``dfx.ForwardMode()`` when this run is being forward-differentiated,
        else the configured adjoint; an explicit one always wins.

        The default ``RecursiveCheckpointAdjoint`` is a ``custom_vjp``, so
        ``jvp``/``jacfwd`` through it raises an error naming neither the solve
        nor the fix. A ``JVPTracer`` among the leaves is the tell.
        """
        if self._adjoint_explicit:
            return self.adjoint
        leaves = jax.tree_util.tree_leaves((composite, y0))
        if any(isinstance(leaf, ad.JVPTracer) for leaf in leaves):
            log.debug(
                "Forward-mode trace detected; using dfx.ForwardMode() "
                "(the default adjoint is a custom_vjp and cannot be jvp'd)."
            )
            return dfx.ForwardMode()
        return self.adjoint

    def _group_omegas(
        self, composite, groups, integrators, state, t0, macro_dt
    ):
        """Per-group ``max|Im λ|`` for the Nyquist guard. Reuses the stiffness
        verdict's spectrum when routing computed one, else measures it eagerly
        (cached). Empty under tracing with no concrete Jacobian — call
        ``warm_up`` first."""
        omegas = [
            gi.info.max_abs_im
            for gi in integrators.values()
            if gi.info is not None
        ]
        if omegas:
            return omegas
        if isinstance(state, jax.core.Tracer):
            return []
        sig = self._integrator_signature(groups, state, macro_dt)
        if sig in self._omega_cache:
            return self._omega_cache[sig]
        from hallsim.stiffness import analyze_groups

        try:
            report = analyze_groups(
                composite,
                y0=state,
                groups=groups,
                t0=t0,
                dt=macro_dt,
                max_explicit_substeps=self.max_explicit_substeps,
            )
            omegas = [v.max_abs_im for v in report.values()]
        except (RuntimeError, np.linalg.LinAlgError):
            omegas = []
        self._omega_cache[sig] = omegas
        return omegas

    def _resolve_save_dt(self, omegas, save_dt: float) -> float:
        """Return a Nyquist-safe ``save_dt`` for raw-state readouts.

        A group with ``max|Im λ| = ω`` oscillates with period ``T = 2π/ω``;
        sampling it at ``save_dt > T/2`` aliases, corrupting any readout of it.
        Undersampling is reduced to ``T / _SAVE_SAMPLES_PER_PERIOD`` — dense
        output decouples the grid from ``macro_dt``, so this costs memory, not
        ODE steps. Unchanged when already fine or no spectrum is available.
        """
        if not save_dt:
            return save_dt
        import math

        worst = None
        for omega in omegas:
            if omega > 0.0:
                period = 2.0 * math.pi / omega
                worst = period if worst is None else min(worst, period)
        if worst is None:
            return save_dt
        safe = worst / self._SAVE_SAMPLES_PER_PERIOD
        if save_dt > safe:
            if not self._warned_save_res:
                self._warned_save_res = True
                log.info(
                    "auto-reduced save_dt %.4g -> %.4g (fastest oscillation "
                    "period %.4g; ~%d samples/period so raw-state readouts "
                    "don't alias)",
                    save_dt,
                    safe,
                    worst,
                    int(self._SAVE_SAMPLES_PER_PERIOD),
                )
            return safe
        return save_dt

    def _resolve_integrators(
        self,
        composite: Composite,
        groups: dict[str, list[str]],
        state: jnp.ndarray,
        t0: float,
        macro_dt: float,
    ) -> dict[str, GroupIntegrator]:
        """Pick a solver for each group.

        Routing (default): stiff groups get the implicit solver, the rest the
        explicit one. Pinned (``solver=`` or ``auto_stiffness=False``): every
        group uses ``self.solver``. Every group leaves here with the scalar
        controller; :meth:`_scaled_tolerances` sets the stiff ones' vector
        ``atol`` per run, since only the routing verdict is state-independent
        enough to cache.

        Cached by structural signature so the analysis runs once, eagerly.
        Under grad/jvp/vmap the eigenvalues are tracers, so a traced call with
        a cold cache warns its way down to the explicit solver — run
        :meth:`warm_up` first to resolve the verdict outside the trace.
        """
        sig = self._integrator_signature(groups, state, macro_dt)
        cached = self._integrator_cache.get(sig)
        if cached is not None:
            return cached

        if not self.auto_solver:
            integ = {
                g: GroupIntegrator(self.solver, self.controller, stiff=False)
                for g in groups
            }
            self._integrator_cache[sig] = integ
            return integ

        def _all_explicit():
            return {
                g: GroupIntegrator(
                    self.explicit_solver, self.controller, stiff=False
                )
                for g in groups
            }

        state_traced = isinstance(state, jax.core.Tracer)
        try:
            report = (
                None
                if state_traced
                else analyze_groups(
                    composite,
                    y0=state,
                    groups=groups,
                    t0=t0,
                    dt=macro_dt,
                    max_explicit_substeps=self.max_explicit_substeps,
                )
            )
        except RuntimeError:
            report = None  # tracers in the RHS — same cold-trace situation
        except np.linalg.LinAlgError:
            # A deterministic property of the composite, not a trace artifact,
            # so this verdict is safe to cache.
            log.warning(
                "stiffness analysis failed to converge; using the explicit "
                "solver for all groups"
            )
            integ = _all_explicit()
            self._integrator_cache[sig] = integ
            return integ

        if report is None:
            # Not cached, so a later eager call still gets to measure and route.
            log.warning(
                "cannot measure group stiffness under tracing (grad/jvp/vmap) "
                "with a cold cache; falling back to the explicit solver %s "
                "for all groups. A stiff group solved this way can return NaN "
                "sensitivities even where the trajectory is finite — call "
                "warm_up(y0) once eagerly before differentiating so the "
                "per-group verdict is resolved outside the trace "
                "(CalibrationProblem.fit does this for you).",
                type(self.explicit_solver).__name__,
            )
            return _all_explicit()
        integ: dict[str, GroupIntegrator] = {}
        for g, verdict in report.items():
            if verdict.stiff:
                integ[g] = GroupIntegrator(
                    self.implicit_solver,
                    self.controller,
                    stiff=True,
                    info=verdict,
                )
            else:
                integ[g] = GroupIntegrator(
                    self.explicit_solver,
                    self.controller,
                    stiff=False,
                    info=verdict,
                )
            if self.debug:
                log.info("  stiffness: %s", verdict)
        self._integrator_cache[sig] = integ
        return integ

    def _scaled_tolerances(
        self,
        integrators: dict[str, GroupIntegrator],
        composite: Composite,
        groups: dict[str, list[str]],
        keys: list[str],
        state: jnp.ndarray,
    ) -> dict[str, GroupIntegrator]:
        """Give each stiff group a vector ``atol`` scaled to ``state``'s
        magnitudes, sliced to the group's own indices (what ``_ReducedRHS``
        solves). Loosens the tolerance on large-magnitude states, which would
        otherwise force stability-tiny steps, while keeping a tight floor near
        zero.

        Called on the state a run actually starts from, and inside the traced
        core, so the tolerance is a function of ``y0`` rather than a constant.
        Resolving it alongside the routing verdict instead would bake the
        first run's magnitudes into a structurally-cached executable, and
        every later population member and calibration step would inherit them.
        """
        if not any(integ.stiff for integ in integrators.values()):
            return integrators
        atol_vec = jnp.maximum(self.atol, self.atol_scale * jnp.abs(state))
        return {
            g: (
                replace(
                    integ,
                    controller=dfx.PIDController(
                        rtol=self.rtol,
                        atol=atol_vec[
                            ..., composite.evolved_indices(groups[g], keys)
                        ],
                    ),
                )
                if integ.stiff
                else integ
            )
            for g, integ in integrators.items()
        }

    def warm_up(
        self,
        composite: Composite,
        t_span: tuple[float, float],
        macro_dt: float = 1.0,
        y0: jnp.ndarray | None = None,
    ) -> dict[str, GroupIntegrator]:
        """Eagerly resolve and cache this composite's per-group solvers.

        Run once with concrete parameters before differentiating through
        :meth:`run` (forward- or reverse-mode), so the stiffness analysis
        — which needs concrete Jacobian eigenvalues — happens outside the
        trace and the verdict is cached for the traced runs. Returns the
        resolved integrators for inspection.
        """
        keys = composite.store_keys()
        state = (
            composite.initial_state_vec(keys)
            if y0 is None
            else jnp.asarray(y0)
        )
        groups = self.manual_groups or composite.auto_groups()
        if not groups:
            continuous = composite.continuous_processes()
            if continuous:
                groups = {"default": list(continuous.keys())}
        return self._resolve_integrators(
            composite, groups, state, t_span[0], macro_dt
        )

    def _reduced_solve(
        self,
        rhs_fn,
        state_vec: jnp.ndarray,
        own: jnp.ndarray,
        t0,
        t1,
        integ: GroupIntegrator,
        adjoint: dfx.AbstractAdjoint,
        saveat: dfx.SaveAt,
        fill=None,
        dt0_hint=None,
        group_name: str = "",
        keys: list[str] | None = None,
    ):
        """Solve one group over ``own`` and scatter the saved states back to
        full width. Returns ``(sol, saved)`` — ``sol.ys`` is the group's own
        states (what interpolated coupling passes on), ``saved`` the guarded
        full-width trajectory.

        The single solve site: every path (fast, scan, Strang, eager,
        interpolated) differs only in ``saveat`` and ``fill``.
        """
        dt0_base = dt0_hint if dt0_hint is not None else self.dt0
        # throw=False so a failed solve returns its RESULTS code rather than
        # crashing opaquely; _guard_result decides what to do with it.
        sol = dfx.diffeqsolve(
            dfx.ODETerm(
                _ReducedRHS(
                    base=rhs_fn,
                    own=own,
                    fill=fill if fill is not None else _FrozenFill(state_vec),
                )
            ),
            integ.solver,
            t0=t0,
            t1=t1,
            dt0=jnp.minimum(dt0_base, t1 - t0),
            y0=state_vec[..., own],
            saveat=saveat,
            stepsize_controller=self._controller_with_jumps(integ.controller),
            adjoint=adjoint,
            max_steps=self.max_steps,
            throw=False,
        )
        saved = (
            jnp.broadcast_to(
                state_vec, sol.ys.shape[:-1] + state_vec.shape[-1:]
            )
            .at[..., own]
            .set(sol.ys)
        )
        return sol, self._guard_result(
            saved, sol.result, group_name, integ, keys
        )

    def _group_step(
        self,
        rhs_fn,
        state_vec: jnp.ndarray,
        own: jnp.ndarray,
        t0,
        t1,
        integ: GroupIntegrator,
        adjoint: dfx.AbstractAdjoint,
        group_name: str = "",
        dt0_hint=None,
    ):
        """Trace-safe single-group ``diffeqsolve`` over ``[t0, t1]``, solved
        over ``own`` (the states this group evolves) with the rest held frozen.

        No Python-side guards or logging — ``t0``/``t1`` may be tracers, so
        this is the core used both inside ``lax.scan`` and by the eager
        :meth:`_solve_group`. Returns
        ``(final_vec, last_dt, result, num_steps, num_rejected)``.
        """
        sol, saved = self._reduced_solve(
            rhs_fn,
            state_vec,
            own,
            t0,
            t1,
            integ,
            adjoint,
            dfx.SaveAt(t1=True),
            dt0_hint=dt0_hint,
            group_name=group_name,
        )
        final_vec = saved[-1]
        # Average step over the interval — the settled step size, without
        # depending on Diffrax's internal controller_state API.
        last_dt = (t1 - t0) / jnp.maximum(sol.stats["num_steps"], 1)
        return (
            final_vec,
            last_dt,
            sol.result,
            sol.stats["num_steps"],
            sol.stats["num_rejected_steps"],
        )

    def _solve_group(
        self,
        rhs_fn,
        state_vec: jnp.ndarray,
        own: jnp.ndarray,
        keys: list[str],
        t0: float,
        t1: float,
        integ: GroupIntegrator,
        adjoint: dfx.AbstractAdjoint,
        group_name: str = "",
        dt0_hint: float | None = None,
    ) -> tuple[jnp.ndarray, float]:
        """Eager single-group solve — :meth:`_group_step` plus the
        empty-interval guard and optional debug logging. Returns
        ``(final_state_vec, last_dt, diag)``, ``diag`` being the
        ``(result, num_steps, num_rejected)`` triple recorded into stats."""
        if t1 <= t0:
            return (
                state_vec,
                (dt0_hint if dt0_hint is not None else self.dt0),
                (dfx.RESULTS.successful, 0, 0),
            )

        state_before = jnp.asarray(state_vec) if self.debug else None

        final_vec, last_dt, result, n_steps_s, n_rej_s = self._group_step(
            rhs_fn,
            state_vec,
            own,
            t0,
            t1,
            integ,
            adjoint,
            group_name,
            dt0_hint,
        )
        diag = (result, n_steps_s, n_rej_s)

        if self.debug:
            deltas = jnp.abs(final_vec - state_before)
            max_idx = int(jnp.argmax(deltas))
            max_delta = float(deltas[max_idx])
            max_delta_key = keys[max_idx]
            any_nan = bool(jnp.isnan(final_vec).any())
            log.info(
                f"  [{group_name}] [{t0:.1f} → {t1:.1f}]: "
                f"{int(n_steps_s)} steps ({int(n_rej_s)} rej) | "
                f"result={result} | "
                f"max Δ: {max_delta_key}={max_delta:.4g}"
                + (" *** NaN ***" if any_nan else "")
            )

        return final_vec, last_dt, diag

    def _guard_result(self, value, result, group_name: str, integ, keys=None):
        """On a failed solve (when ``self.throw``) raise a labelled error
        naming the failure kind and worst state.

        ``eqx.error_if`` bakes the check into ``value``, so the raise fires
        eagerly and under JIT/grad alike; ``throw=False`` sends the ``RESULTS``
        code to stats instead. The detailed report is eager-path only — an
        in-graph callback cannot pay for itself, since ``vmap`` turns the
        gating ``lax.cond`` into a select that fires once per batch member
        regardless. Traced runs read :attr:`SchedulerResult.ok`.
        """
        if not self.throw:
            return value
        solver = type(integ.solver).__name__
        stiff = bool(integ.stiff)
        failed = result != dfx.RESULTS.successful

        if not isinstance(failed, jax.core.Tracer) and bool(failed):
            idx, mag, nonfinite = _worst_state(value)
            i = int(idx)
            wname = (
                keys[i]
                if (keys is not None and 0 <= i < len(keys))
                else f"state#{i}"
            )
            kind = (
                "max_steps_reached — the stiff solve ran out of steps; raise "
                "Scheduler(max_steps=…) or reduce the group's stiffness"
                if bool(result == dfx.RESULTS.max_steps_reached)
                else "stiff/implicit-Newton divergence — the state got too "
                "stiff for the implicit step (a fast mode; find the parameter "
                "driving it)"
            )
            log.error(
                "Scheduler SOLVE FAILED — group=%r solver=%s stiff=%s: %s. "
                "worst state=%s (|y|=%.3g)  any_nonfinite=%s",
                group_name,
                solver,
                stiff,
                kind,
                wname,
                float(mag),
                bool(nonfinite),
            )

        reason = (
            f"Scheduler: group {group_name!r} did not solve (solver={solver}, "
            f"stiff={stiff}) — inspect result.ok / result.stats['diagnosis'] "
            f"(run with throw=False) for the RESULTS kind and worst state."
        )
        return eqx.error_if(value, failed, reason)

    def _solve_group_interpolated(
        self,
        rhs_fn,
        state_vec: jnp.ndarray,
        own: jnp.ndarray,
        t0: float,
        t1: float,
        prev_interpolant: Any | None = None,
        prev_idxs: jnp.ndarray | None = None,
        *,
        integ: GroupIntegrator,
        adjoint: dfx.AbstractAdjoint,
        dt0_hint: float | None = None,
    ) -> tuple[jnp.ndarray, Any, float, tuple]:
        """Solve one group, sampling its trajectory on a fixed grid so the next
        group can splice in this group's evolving variables.

        Given ``prev_interpolant`` (a ``(t0, t1, ys)`` fixed-size sample), this
        group's RHS sees the previous group's evolved variables interpolated at
        the current solver time, keeping its own ``y`` for everything else — so
        it feels that trajectory inside the macro step rather than a frozen
        snapshot. The static sample size lets the same routine run eagerly and
        inside the compiled ``lax.scan``.

        Returns ``(final_state_vec, sample, last_dt, diag)``.
        """
        k = self.coupling_interp_points
        grid = t0 + jnp.linspace(0.0, 1.0, k) * (t1 - t0)
        own_vec = state_vec[..., own]
        if t1 <= t0:
            ys_deg = jnp.broadcast_to(own_vec, (k,) + own_vec.shape)
            return (
                state_vec,
                (t0, t1, ys_deg),
                (dt0_hint if dt0_hint is not None else self.dt0),
                (dfx.RESULTS.successful, 0, 0),
            )

        if prev_interpolant is not None and prev_idxs is not None:
            p_t0, p_t1, p_ys = prev_interpolant
            fill = _InterpFill(
                full=state_vec, t0=p_t0, t1=p_t1, ys=p_ys, idx=prev_idxs
            )
        else:
            fill = _FrozenFill(full=state_vec)

        sol, saved = self._reduced_solve(
            rhs_fn,
            state_vec,
            own,
            t0,
            t1,
            integ,
            adjoint,
            dfx.SaveAt(ts=grid),
            fill=fill,
            dt0_hint=dt0_hint,
            group_name="interpolated",
        )
        final_vec = saved[-1]
        last_dt = (t1 - t0) / jnp.maximum(sol.stats["num_steps"], 1)
        diag = (
            sol.result,
            sol.stats["num_steps"],
            sol.stats["num_rejected_steps"],
        )
        return final_vec, (t0, t1, sol.ys), last_dt, diag

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
        if t_next % dt_step < _TIME_EPS:  # exact alignment
            n_after = int(round(t_next / dt_step))
        return n_after > n_before
