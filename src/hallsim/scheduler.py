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
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optimistix as optx

from hallsim.composite import Composite
from hallsim.config import (
    DEFAULT_ATOL,
    DEFAULT_ATOL_SCALE,
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


@dataclass
class EventRecord:
    """Log entry for a fired event."""

    time: float
    process: str
    delta: dict[str, jnp.ndarray]


@dataclass
class GroupIntegrator:
    """Resolved per-group solver + step-size controller.

    One of these is produced for each continuous group by
    :meth:`Scheduler._resolve_integrators` and used for that group's
    ``diffeqsolve``. The split is automatic: stiff groups get an implicit
    (A-stable) solver and a magnitude-scaled vector ``atol``; the rest
    keep the cheaper explicit solver and the scalar controller.

    Attributes
    ----------
    solver:
        Diffrax solver for this group.
    controller:
        Step-size controller for this group.
    stiff:
        Whether stiffness analysis flagged this group.
    info:
        The :class:`~hallsim.stiffness.GroupStiffness` verdict (``None``
        in manual-solver mode, where no analysis runs).
    """

    solver: dfx.AbstractSolver
    controller: dfx.AbstractStepSizeController
    stiff: bool
    info: GroupStiffness | None = None


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


def _interp_uniform(
    t: jnp.ndarray, t0: jnp.ndarray, t1: jnp.ndarray, ys: jnp.ndarray
) -> jnp.ndarray:
    """Linear interpolation along axis 0 of a uniform sample at scalar ``t``.

    ``ys`` has shape ``(K, ...)`` sampled at ``K`` equally spaced times over
    ``[t0, t1]``. Fixed ``K`` (static) keeps the coupling representation a
    constant shape, so the interpolated multi-rate loop compiles under
    ``lax.scan`` — unlike a solver-native dense interpolant, whose size is
    the adaptive step count and varies per macro step.
    """
    k = ys.shape[0]
    pos = jnp.clip((t - t0) / (t1 - t0) * (k - 1), 0.0, k - 1.0)
    i0 = jnp.minimum(jnp.floor(pos).astype(jnp.int32), k - 2)
    frac = pos - i0
    return ys[i0] * (1.0 - frac) + ys[i0 + 1] * frac


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
        Diffrax solver instance applied to every group, with the scalar
        controller. Default ``None`` uses ``explicit_solver`` (``Tsit5()``)
        — the right choice for SBML biochemical systems in float64, where
        the explicit solver is both correct and fastest (implicit's cheaper
        step count is eaten by Newton cost). For a genuinely stiff group,
        either pin a stiff ``solver`` (e.g. ``Scheduler(solver=dfx.Kvaerno5())``)
        or set ``auto_stiffness=True`` to route per group.
    auto_stiffness:
        Opt into **per-group solver routing** (default ``False``). When
        ``True``, each continuous group's local Jacobian spectrum is
        measured once (eagerly) via :func:`hallsim.stiffness.analyze_groups`;
        stiff groups get ``implicit_solver`` (A-stable) with a
        magnitude-scaled vector ``atol`` while the rest keep
        ``explicit_solver`` with the scalar controller. Off by default
        because in float64 the explicit solver handles the example
        composites and wins on wall-clock; turn it on for a composite that
        mixes a stiff dissipative subsystem with cheap signaling cascades
        and you'd rather not pick solvers by hand. Mutually exclusive with
        a pinned ``solver``.
    explicit_solver, implicit_solver:
        The two solvers ``auto_stiffness`` routing chooses between. Defaults
        ``Tsit5()`` (explicit) and ``Kvaerno5()`` (implicit, A-stable — the
        diffrax analogue of CVODE's stiff BDF, stable on stiff forward
        sensitivities). ``explicit_solver`` is also the default ``solver``;
        ``implicit_solver`` is unused unless ``auto_stiffness=True``.
    max_explicit_substeps:
        Stiffness threshold forwarded to the analyzer: a group is stiff
        when its fastest decay rate × ``macro_dt`` (stability-limited
        substeps per solve interval) exceeds this. Default 100 — a wide
        margin between mildly-multiscale-but-slow systems (ERiQ, ~3–26)
        and genuinely stiff ones (~1e4+).
    rtol, atol:
        Relative and absolute tolerances for adaptive stepping. Default
        ``rtol=1e-6, atol=1e-9``. Oscillatory subsystems (p53–Mdm2,
        NF-κB, cell cycle) need accuracy-limited stepping: an explicit
        solver at loose tolerance injects energy and a bounded
        oscillation spirals out ("numerical anti-damping"). The
        Geva-Zatorsky 2006 p53 oscillator, for instance, diverges to
        ~300× its amplitude at ``rtol=1e-4`` and is bounded from
        ``rtol=1e-5`` down. ``1e-6`` keeps a safety margin past that
        threshold. For stiff groups ``atol`` is the *floor* of a vector
        tolerance ``max(atol, atol_scale·|y0|)`` so a state at 1e6 isn't
        held to 1e-9 absolute (which would force stability-tiny steps)
        while a state at 1e-4 still gets a tight tolerance.
    atol_scale:
        Relative coefficient of the stiff-group vector ``atol`` (default
        ``1e-6``). Per state, ``atol_i = max(atol, atol_scale·|y0_i|)``.
    max_steps:
        Safety limit on solver steps per macro step.
    dt0:
        Initial step size for adaptive controller.
    groups:
        Manual group assignment ``{group_name: [proc_name, ...]}``.
        If ``None``, uses ``composite.auto_groups()``.
    coupling_mode:
        How inter-group state is communicated during Lie splitting.

        - ``"auto"`` (default): pick per run — ``"interpolated"`` when the
          composite has a forward cross-group edge (an earlier group's
          variables read by a later group), else ``"frozen"``. Interpolated
          only improves accuracy on such an edge, so this pays for it only
          when it helps.
        - ``"frozen"``: each group sees the previous group's final state —
          standard Lie splitting.
        - ``"interpolated"``: each group reads a fixed-size sample
          (``coupling_interp_points`` points) of the previous group's
          trajectory, linearly interpolated at the current time ``t``, so
          it "feels" the previous group's trajectory within the macro step
          instead of a frozen snapshot. Fixed sample size keeps the loop
          compilable under ``lax.scan``.

          Inspired by dense-output multirate methods (cf. SUNDIALS,
          continuous-output Runge-Kutta).

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
    throw:
        If ``True`` (default), a group whose ``diffeqsolve`` does not
        return ``RESULTS.successful`` raises a clear error naming the
        group, its solver, and the failure reason (via ``eqx.error_if``,
        so it works under JIT/grad too). If ``False``, the run continues
        and the per-group ``RESULTS`` code is recorded in
        ``result.stats[group]['result']`` for inspection — the
        transparent path for diagnosing a non-converging composite.
    progress:
        If ``True``, show a tqdm progress bar over macro steps. Default
        ``False`` because the Python-side ``pbar.update(1)`` is a side
        effect that interferes with ``jax.vmap`` over batched runs (e.g.
        population studies, parameter sweeps).
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
        auto_stiffness: bool = False,
        atol_scale: float = DEFAULT_ATOL_SCALE,
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
        # Solver selection. Default: the explicit solver (`explicit_solver`,
        # Tsit5) for every group with the scalar controller. `auto_stiffness`
        # opts into per-group routing — the analyzer measures each group's
        # Jacobian spectrum and sends stiff groups to `implicit_solver` with
        # a magnitude-scaled vector atol. Pinning `solver` uses that one
        # solver for every group.
        if auto_stiffness and solver is not None:
            raise ValueError(
                "auto_stiffness=True selects a solver per group; do not also "
                "pin solver=. Pass one or the other."
            )
        self.auto_solver = auto_stiffness
        self.explicit_solver = explicit_solver or dfx.Tsit5()
        # Default stiff solver is Kvaerno5 with a **Newton** root finder.
        # diffrax's default `VeryChord` (stale-Jacobian chord, 10 iters)
        # rejects ~50% of steps on real biochemical RHSs; a true Newton
        # solve (fresh Jacobian — what CVODE does) cuts that to a few %.
        self.implicit_solver = implicit_solver or dfx.Kvaerno5(
            root_finder=optx.Newton(rtol=rtol, atol=atol)
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
        self._warned_save_res = False
        # Adjoint method used by every diffeqsolve in this run.
        # Default (None) → diffrax picks RecursiveCheckpointAdjoint, which
        # is memory-cheap but step-expensive. For calibration through
        # stiff/oscillatory composites, pass dfx.BacksolveAdjoint() for
        # near-forward-cost backward passes.
        self.adjoint = adjoint or dfx.RecursiveCheckpointAdjoint()
        self.throw = throw
        self.max_steps = max_steps
        self.dt0 = dt0
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
        warn_save_resolution: bool = True,
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
        adjoint:
            Per-run override of the differentiation method. ``None`` uses
            the constructor's ``adjoint``. Pass ``dfx.ForwardMode()`` for
            forward-mode calibration without changing scheduler identity,
            so one Scheduler instance (and its stiffness cache) serves
            both the eager evaluate pass and the differentiated loss.
        warn_save_resolution:
            Emit a once-only Nyquist warning when ``save_dt`` undersamples
            the fastest oscillation (see :meth:`_check_save_resolution`). Set
            ``False`` for a run whose readouts are provably grid-independent
            (e.g. a burn-in read only through a ``RunningIntegral``), so the
            deliberately coarse grid does not raise a false alarm.

        Returns
        -------
        :class:`SchedulerResult`
        """
        t0, t1 = t_span
        adjoint = adjoint if adjoint is not None else self.adjoint

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
        # support is the continuous-only path (fast path or
        # multi-group Lie/Strang).
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

        # Resolve coupling once groups are known. "auto" (the default)
        # picks interpolated only when a forward cross-group edge exists;
        # otherwise frozen. Explicit modes pass through.
        coupling = self._effective_coupling(composite, groups, keys)

        # Resolve the per-group solver + step-size controller. In auto
        # mode this measures each group's Jacobian spectrum once (eagerly)
        # and routes stiff groups to the implicit solver + scaled vector
        # atol; the verdict is cached by structure so later traced runs
        # reuse it.
        integrators = self._resolve_integrators(
            composite, groups, state, t0, macro_dt
        )
        if warn_save_resolution:
            self._check_save_resolution(
                integrators, save_dt if save_dt is not None else macro_dt
            )

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
            and coupling == "frozen"
        )
        if fast_path_eligible:
            (gname,) = groups.keys()
            (proc_names,) = groups.values()
            integ = integrators[gname]
            rhs_fn, _ = composite.build_rhs(proc_names)
            save_step = save_dt if save_dt is not None else macro_dt
            # Fixed-length save grid on [t0, t1] inclusive via linspace: pins
            # both endpoints exactly (never a SaveAt time past t1, which would
            # make diffeqsolve raise) and stays jit-safe (count is a Python int).
            save_ts = self._save_grid(t0, t1, save_step)
            sol = dfx.diffeqsolve(
                dfx.ODETerm(rhs_fn),
                integ.solver,
                t0=t0,
                t1=t1,
                dt0=min(self.dt0, t1 - t0),
                y0=state,
                saveat=dfx.SaveAt(ts=save_ts),
                stepsize_controller=integ.controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
                throw=False,
            )
            ys = self._guard_result(sol.ys, sol.result, gname, integ, keys)
            stats = {
                gname: {
                    "num_macro_steps": 1,
                    "num_solver_steps": sol.stats.get("num_steps"),
                    "num_rejected_steps": sol.stats.get(
                        "num_rejected_steps", 0
                    ),
                    "result": sol.result,
                    "solver": type(integ.solver).__name__,
                    "stiff": integ.stiff,
                }
            }
            return SchedulerResult(
                ts=sol.ts,
                ys=ys,
                keys=keys,
                events=[],
                stats=stats,
            )

        # ─── Scan path: continuous-only, frozen, fixed macro_dt ────────
        # Multiple groups (or a single group under Strang) with no
        # discrete/event processes and no adaptive macro_dt: the macro
        # loop has a statically-known step count, so express it as one
        # ``lax.scan`` — the whole multi-group run compiles to a single
        # executable (bounded compile, reverse-mode memory flat in
        # macro-step count). ``adaptive_dt`` (data-dependent step count)
        # and debug logging keep the eager loop; both frozen and
        # interpolated coupling scan (interpolated threads a fixed-size
        # ``coupling_interp_points`` sample between groups).
        scan_eligible = (
            not discrete_procs
            and not event_procs
            and not self.adaptive_dt
            and coupling in ("frozen", "interpolated")
            and not self.debug
        )
        if scan_eligible:
            return self._run_scan_continuous(
                composite,
                groups,
                integrators,
                state,
                keys,
                t0,
                t1,
                macro_dt,
                save_dt,
                adjoint,
                coupling,
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
            group_write_idxs[gname] = composite.evolved_indices(
                proc_names, keys
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

        # Per-group stats — carries the resolved solver, stiffness verdict,
        # and the latest diffeqsolve RESULTS code so a non-converging
        # composite is inspectable through the API (throw=False).
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
                    state, last_dt, diag = self._solve_group(
                        rhs_fn,
                        state,
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
            else:
                # Lie splitting: sequential, one pass
                prev_interpolant = None
                prev_idxs: jnp.ndarray | None = None
                for gname, rhs_fn in group_rhs.items():
                    if coupling == "interpolated":
                        state, prev_interpolant, last_dt, diag = (
                            self._solve_group_interpolated(
                                rhs_fn,
                                state,
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

    def _effective_coupling(
        self,
        composite: Composite,
        groups: dict[str, list[str]],
        keys: list[str],
    ) -> str:
        """Resolve ``coupling_mode="auto"`` to ``"frozen"`` or
        ``"interpolated"`` for this run.

        Interpolated coupling only improves accuracy on a **forward**
        cross-group edge — a group written earlier in the solve order whose
        variables a later group reads (the later group then sees that
        trajectory evolving via the sample instead of frozen). If no such
        edge exists, interpolated does identical work to frozen at higher
        cost, so ``auto`` picks frozen. Strang and single-group runs always
        resolve to frozen. Explicit ``"frozen"``/``"interpolated"`` pass
        through unchanged.
        """
        if self.coupling_mode != "auto":
            return self.coupling_mode
        if self.splitting == "strang" or len(groups) < 2:
            return "frozen"
        # Static structure only — no jnp — so this stays concrete when run()
        # is traced under jax.jit/vmap (population studies). A group writes a
        # path via its EVOLVED ports; it reads every path any of its ports
        # is wired to.
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
    ) -> SchedulerResult:
        """Continuous-only multi-group run as a single ``lax.scan``.

        One fixed-length scan over ``macro_dt`` communication windows;
        each window solves every group in turn (Lie) or forward-then-
        reversed (Strang) via :meth:`_group_step`, threading each group's
        settled step size as the next window's warm-start. The whole
        multi-group trajectory compiles to one executable, so compile time
        and reverse-mode memory no longer scale with the macro-step count.
        Handles both frozen and interpolated coupling at a fixed
        ``macro_dt`` (guaranteed by the caller's ``scan_eligible`` gate).
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

        # Save each group at ``n_save`` equally-spaced times over the macro
        # window (endpoints included, so the sample doubles as interpolated
        # coupling's fixed-size representation). Output resolution is set by
        # save_dt, INDEPENDENT of the coupling cadence macro_dt: dropping the
        # left endpoint per window (it repeats the previous window's end)
        # gives ``n_out`` fresh points, so fast intra-group dynamics are
        # resolved without shrinking macro_dt. Strang keeps endpoints only.
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

        def solve_dense(gi, st, t_a, t_b, dt0hi, coupled_rhs):
            g, rhs = group_rhs[gi]
            rhs_use = coupled_rhs if coupled_rhs is not None else rhs
            grid = t_a + save_frac * (t_b - t_a)
            sol = dfx.diffeqsolve(
                dfx.ODETerm(rhs_use),
                integrators[g].solver,
                t0=t_a,
                t1=t_b,
                dt0=jnp.minimum(dt0hi, t_b - t_a),
                y0=st,
                saveat=dfx.SaveAt(ts=grid),
                stepsize_controller=integrators[g].controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
                throw=False,
            )
            final = self._guard_result(
                sol.ys[-1], sol.result, g, integrators[g], keys
            )
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
                    coupled = None
                    if interp and prev is not None:
                        p_t0, p_t1, p_ys = prev
                        base = group_rhs[gi][1]
                        widx = write_idxs[gi - 1]

                        def _coupled(
                            t,
                            y,
                            args=None,
                            _b=base,
                            _w=widx,
                            _t0=p_t0,
                            _t1=p_t1,
                            _y=p_ys,
                        ):
                            pv = _interp_uniform(t, _t0, _t1, _y)
                            return _b(t, y.at[_w].set(pv[_w]), args)

                        coupled = _coupled

                    st, gy, ld, ns, nr, r = solve_dense(
                        gi, st, t_start, t_next, dt0h[gi], coupled
                    )
                    w = write_idxs[gi]
                    traj = traj.at[:, w].set(gy[1:][:, w])
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

        # Assemble the dense trajectory: y0, then each window's n_out fresh
        # samples. ys_stack is (n_macro, n_out, n_vars); flatten the window
        # axis into time. save_dt coarser than macro_dt (n_out==1) subsamples.
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
                "solver": type(integrators[g].solver).__name__,
                "stiff": integrators[g].stiff,
            }
            for gi, (g, _) in enumerate(group_rhs)
        }
        return SchedulerResult(ts=ts, ys=ys, keys=keys, events=[], stats=stats)

    @staticmethod
    def _integrator_signature(groups, state, macro_dt):
        """Structural key for the integrator cache.

        Depends only on group→process structure, state width, and
        ``macro_dt`` (which sets the stiffness threshold) — never on
        state *values*. So the eager resolution (concrete params) is
        reused under later traced runs of the same composite, even across
        conditions whose ``y0`` differ slightly.
        """
        gstruct = tuple(
            sorted((g, tuple(sorted(procs))) for g, procs in groups.items())
        )
        return (gstruct, int(state.shape[-1]), float(macro_dt))

    @staticmethod
    def _save_grid(t0: float, t1: float, save_step: float) -> jnp.ndarray:
        """Save times spanning ``[t0, t1]`` inclusive, ~``save_step`` apart.

        The systematic rule so a save grid never errors: pick the point *count*
        ``round((t1−t0)/save_step)+1`` and place them with ``jnp.linspace``,
        which pins both endpoints exactly and never overshoots ``t1`` — a
        ``SaveAt`` time past ``t1`` makes ``diffeqsolve`` raise. ``save_step``
        need not divide ``(t1−t0)``; linspace fits the count to the span, so the
        realized spacing is ``(t1−t0)/(n−1)`` (≈ ``save_step``). JIT-safe:
        ``t0``/``t1``/``save_step`` are concrete floats, so ``n`` is a Python
        int (static output shape)."""
        span = t1 - t0
        if not save_step or save_step <= 0.0 or save_step >= span:
            return jnp.asarray([t0, t1], dtype=float)
        n = int(round(span / save_step)) + 1
        return jnp.linspace(t0, t1, n)

    def _check_save_resolution(
        self, integrators: dict[str, GroupIntegrator], save_dt: float
    ) -> None:
        """Warn (once) if the save grid undersamples the fastest oscillation.

        A group whose Jacobian carries an imaginary part ``max|Im λ| = ω``
        (run-time units) oscillates with period ``T = 2π/ω``. Sampling the
        raw trajectory at ``save_dt > T/2`` violates Nyquist — the oscillation
        aliases, and any readout of a *raw* oscillating state (an amplitude,
        an endpoint, a plain mean of saved points) is corrupted. This is the
        Nyquist theorem, not a tuned threshold. Phase-insensitive readouts
        routed through a :class:`~hallsim.models.running_integral.RunningIntegral`
        are grid-independent and unaffected — the recommended fix, alongside
        simply saving finer (``save_dt`` is decoupled from ``macro_dt`` via
        dense output, so it costs no extra ODE steps).

        Fires only when the Jacobian spectrum is available (the
        ``auto_stiffness`` path, after ``warm_up``); a manual-solver run
        carries no spectrum to check against.
        """
        if not save_dt or self._warned_save_res:
            return
        import math

        offenders = []
        for g, gi in integrators.items():
            omega = getattr(gi.info, "max_abs_im", 0.0) if gi.info else 0.0
            if omega > 0.0:
                period = 2.0 * math.pi / omega
                if save_dt > 0.5 * period:
                    offenders.append((g, period))
        if offenders:
            self._warned_save_res = True
            worst = min(p for _, p in offenders)
            log.warning(
                "save_dt=%.4g undersamples the fastest oscillation "
                "(period %.4g, group(s) %s) below Nyquist — raw-state "
                "readouts on this grid alias. Save finer (save_dt < %.4g, "
                "~%d+ samples/period) or read oscillators through a "
                "RunningIntegral (grid-independent).",
                save_dt,
                worst,
                ", ".join(g for g, _ in offenders),
                0.5 * worst,
                10,
            )

    def _resolve_integrators(
        self,
        composite: Composite,
        groups: dict[str, list[str]],
        state: jnp.ndarray,
        t0: float,
        macro_dt: float,
    ) -> dict[str, GroupIntegrator]:
        """Pick a solver + step-size controller for each group.

        Manual mode (``auto_stiffness=False``, the default): every group
        uses ``self.solver`` with the scalar controller. Auto mode
        (``auto_stiffness=True``): each group's local Jacobian spectrum is
        measured once via
        :func:`hallsim.stiffness.analyze_groups`; stiff groups get the
        implicit solver and a magnitude-scaled vector ``atol``, the rest
        the explicit solver and the scalar controller.

        The result is cached by structural signature so the analysis runs
        once, eagerly. Under grad/jvp/vmap the Jacobian eigenvalues would
        be tracers — so a traced call with a cold cache raises, directing
        the caller to :meth:`warm_up` first (the calibration path does
        this automatically).
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

        # Stiffness analysis needs a concrete Jacobian. Under tracing
        # (grad/jvp/vmap) with no warmed cache the eigenvalues are tracers
        # and it cannot be measured — this raises (below) rather than
        # silently falling back to the explicit solver, which would give
        # wrong (NaN) sensitivities on stiff groups. warm_up() first.
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
            # analyze_groups hit JAX tracers in the RHS (params under
            # differentiation) — same cold-trace situation.
            report = None
        except np.linalg.LinAlgError:
            # Eigenvalue solve didn't converge (degenerate/non-finite
            # Jacobian). Fall back to explicit and cache it — this is a
            # deterministic property of the composite, not a trace artifact.
            log.warning(
                "stiffness analysis failed to converge; using the explicit "
                "solver for all groups"
            )
            integ = _all_explicit()
            self._integrator_cache[sig] = integ
            return integ

        if report is None:
            raise RuntimeError(
                "Scheduler(auto_stiffness=True): cannot measure group "
                "stiffness under tracing (grad/jvp/vmap) with a cold cache "
                "— the Jacobian eigenvalues are tracers. Call warm_up(y0) "
                "once eagerly before differentiating so the per-group "
                "implicit/explicit verdict is cached "
                "(CalibrationProblem.fit does this for you). Falling back "
                "to the explicit solver here would give wrong (NaN) "
                "sensitivities on stiff groups."
            )
        # Vector atol for stiff groups: loosen absolute tolerance on
        # large-magnitude states (which would otherwise force
        # stability-tiny steps) while keeping a tight floor near zero.
        atol_vec = jnp.maximum(self.atol, self.atol_scale * jnp.abs(state))
        integ: dict[str, GroupIntegrator] = {}
        for g, verdict in report.items():
            if verdict.stiff:
                integ[g] = GroupIntegrator(
                    self.implicit_solver,
                    dfx.PIDController(rtol=self.rtol, atol=atol_vec),
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

    def _group_step(
        self,
        rhs_fn,
        state_vec: jnp.ndarray,
        t0,
        t1,
        integ: GroupIntegrator,
        adjoint: dfx.AbstractAdjoint,
        group_name: str = "",
        dt0_hint=None,
    ):
        """Trace-safe single-group ``diffeqsolve`` over ``[t0, t1]``.

        No Python-side guards or logging — ``t0``/``t1`` may be tracers,
        so this is the core used inside ``lax.scan`` by the continuous
        scan path as well as by the eager :meth:`_solve_group`. Returns
        ``(final_vec, last_dt, result, num_steps, num_rejected)``.
        """
        # Prior macro-step's average dt as a warm-start hint, clamped to
        # the remaining interval.
        dt0_base = dt0_hint if dt0_hint is not None else self.dt0
        # throw=False so a failed solve returns its RESULTS code instead
        # of crashing opaquely; _guard_result re-raises a labelled error
        # when self.throw, else the code is surfaced in stats.
        sol = dfx.diffeqsolve(
            dfx.ODETerm(rhs_fn),
            integ.solver,
            t0=t0,
            t1=t1,
            dt0=jnp.minimum(dt0_base, t1 - t0),
            y0=state_vec,
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=integ.controller,
            adjoint=adjoint,
            max_steps=self.max_steps,
            throw=False,
        )
        final_vec = self._guard_result(
            sol.ys[-1], sol.result, group_name, integ
        )
        # "What step size did the controller settle at?" — average over
        # the macro interval, a robust proxy that doesn't depend on
        # Diffrax's internal controller_state API.
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
        keys: list[str],
        t0: float,
        t1: float,
        integ: GroupIntegrator,
        adjoint: dfx.AbstractAdjoint,
        group_name: str = "",
        dt0_hint: float | None = None,
    ) -> tuple[jnp.ndarray, float]:
        """Eager single-group solve — :meth:`_group_step` plus the
        empty-interval guard and optional debug logging.

        Returns ``(final_state_vec, last_dt_estimate, diag)`` where
        ``diag`` is a ``(result, num_steps, num_rejected)`` triple
        recorded into stats. ``last_dt_estimate`` feeds the next
        macro-step's ``dt0_hint`` so the controller doesn't restart from
        ``self.dt0`` every macro step.
        """
        if t1 <= t0:
            return (
                state_vec,
                (dt0_hint if dt0_hint is not None else self.dt0),
                (dfx.RESULTS.successful, 0, 0),
            )

        state_before = jnp.asarray(state_vec) if self.debug else None

        final_vec, last_dt, result, n_steps_s, n_rej_s = self._group_step(
            rhs_fn, state_vec, t0, t1, integ, adjoint, group_name, dt0_hint
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
        """On a failed solve (when ``self.throw``) emit a self-diagnosing
        report, then re-raise.

        ``eqx.error_if`` bakes the check into ``value`` so it fires both
        eagerly and under JIT/grad; with ``throw=False`` it is a no-op and the
        ``RESULTS`` code reaches stats instead. The report is a runtime callback
        gated by ``lax.cond`` on the failure path (no per-solve overhead on
        success), so a crash names the failure *kind* (max_steps vs stiff/Newton
        divergence) and the worst state — no re-run needed to diagnose.
        """
        if not self.throw:
            return value
        solver = type(integ.solver).__name__
        stiff = bool(integ.stiff)
        failed = result != dfx.RESULTS.successful
        is_maxsteps = result == dfx.RESULTS.max_steps_reached

        def _report(is_ms, ys):
            ys = np.asarray(ys)
            per = np.abs(ys).reshape(-1, ys.shape[-1])
            finite = np.isfinite(per)
            score = np.where(finite, per, np.inf).max(axis=0)
            widx = int(np.argmax(score)) if score.size else -1
            wname = (
                keys[widx]
                if (keys is not None and 0 <= widx < len(keys))
                else f"state#{widx}"
            )
            kind = (
                "max_steps_reached — the stiff solve ran out of steps; raise "
                "Scheduler(max_steps=…) or reduce the group's stiffness"
                if bool(is_ms)
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
                float(score[widx]) if widx >= 0 else float("nan"),
                bool((~finite).any()),
            )

        jax.lax.cond(
            failed,
            lambda: jax.debug.callback(_report, is_maxsteps, value),
            lambda: None,
        )
        reason = (
            f"Scheduler: group {group_name!r} did not solve (solver={solver}, "
            f"stiff={stiff}) — see the SOLVE FAILED report logged above for the "
            f"RESULTS kind and worst state."
        )
        return eqx.error_if(value, failed, reason)

    def _solve_group_interpolated(
        self,
        rhs_fn,
        state_vec: jnp.ndarray,
        t0: float,
        t1: float,
        prev_interpolant: Any | None = None,
        prev_idxs: jnp.ndarray | None = None,
        *,
        integ: GroupIntegrator,
        adjoint: dfx.AbstractAdjoint,
        dt0_hint: float | None = None,
    ) -> tuple[jnp.ndarray, Any, float, tuple]:
        """Solve one group, sampling its trajectory at a fixed grid so the
        next group can splice in the previous group's evolving variables.

        When ``prev_interpolant`` (a ``(t0, t1, ys)`` fixed-size sample) is
        provided, this group's RHS sees the previous group's evolved
        variables at the current solver time ``t`` — linearly interpolated
        over that sample — while keeping its own evolving ``y`` for
        everything else. This is fixed-sample Lie splitting: the next group
        "feels" the previous group's continuous trajectory inside the macro
        step, not just a frozen snapshot. The sample size is static
        (``coupling_interp_points``), so the same routine runs in the eager
        loop and, for the common case, inside the compiled ``lax.scan``.

        Returns
        -------
        (final_state_vec, sample, last_dt, diag)
            The updated flat state, this group's ``(t0, t1, ys)`` sample
            (for passing to the next group), the average step size over the
            interval (warm-start hint for the next macro step), and the
            ``(result, num_steps, num_rejected)`` diagnostics triple.
        """
        k = self.coupling_interp_points
        grid = t0 + jnp.linspace(0.0, 1.0, k) * (t1 - t0)
        if t1 <= t0:
            ys_deg = jnp.broadcast_to(state_vec, (k,) + state_vec.shape)
            return (
                state_vec,
                (t0, t1, ys_deg),
                (dt0_hint if dt0_hint is not None else self.dt0),
                (dfx.RESULTS.successful, 0, 0),
            )

        if prev_interpolant is not None and prev_idxs is not None:
            p_t0, p_t1, p_ys = prev_interpolant
            base_rhs = rhs_fn
            idxs = prev_idxs

            def coupled_rhs(t, y, args=None):
                interp_vec = _interp_uniform(t, p_t0, p_t1, p_ys)
                merged = y.at[idxs].set(interp_vec[idxs])
                return base_rhs(t, merged, args)

            rhs_to_solve = coupled_rhs
        else:
            rhs_to_solve = rhs_fn

        term = dfx.ODETerm(rhs_to_solve)
        dt0_base = dt0_hint if dt0_hint is not None else self.dt0
        sol = dfx.diffeqsolve(
            term,
            integ.solver,
            t0=t0,
            t1=t1,
            dt0=jnp.minimum(dt0_base, t1 - t0),
            y0=state_vec,
            saveat=dfx.SaveAt(ts=grid),
            stepsize_controller=integ.controller,
            adjoint=adjoint,
            max_steps=self.max_steps,
            throw=False,
        )

        final_vec = self._guard_result(
            sol.ys[-1], sol.result, "interpolated", integ
        )
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
        # Also handle exact alignment
        if t_next % dt_step < _TIME_EPS:
            n_after = int(round(t_next / dt_step))
        return n_after > n_before
