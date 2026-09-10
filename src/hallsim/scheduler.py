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

import hashlib
import logging
from dataclasses import dataclass, field
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
from hallsim.store import as_paths, read_write_paths
from hallsim.tracing import is_traced
from hallsim.config import (
    DEFAULT_ATOL,
    DEFAULT_NEWTON_ATOL,
    DEFAULT_DT0,
    DEFAULT_MAX_EXPLICIT_SUBSTEPS,
    DEFAULT_MAX_STEPS,
    DEFAULT_RTOL,
)
from hallsim.process import PortRole
from hallsim.stiffness import (
    GroupStiffness,
    StiffnessNotConcrete,
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
    """Log entry for a fired event. ``members`` is the ``(batch,)`` mask of
    the members it fired for, ``None`` when the run is not batched."""

    time: float
    process: str
    delta: dict[str, jnp.ndarray]
    members: np.ndarray | None = None


@dataclass
class GroupIntegrator:
    """Resolved per-group solver + step-size controller, from
    :meth:`Scheduler._resolve_integrators`. Stiff groups get an implicit
    (A-stable) solver, the rest the cheaper explicit one; both carry the
    same scalar controller: ``rtol`` already scales the error allowance by
    the *current* state, so a stiff group needs no tolerance of its own.
    ``info`` is the stiffness verdict (``None`` in manual-solver mode, where
    none runs)."""

    solver: dfx.AbstractSolver
    controller: dfx.AbstractStepSizeController
    stiff: bool
    info: GroupStiffness | None = None


@dataclass(frozen=True)
class RunPlan:
    """Everything a run resolves before it can integrate, as a value.

    Resolving a run means picking groups, deciding the coupling mode,
    measuring each group's stiffness to route a solver, choosing an
    anti-aliased ``save_dt``, collecting the discontinuity times and tracing a
    core. All of it is a function of *(policy, composite, span)* and none of it
    is a function of the runner alone — so it lives here rather than in a cache
    on :class:`Scheduler`, where the key has to reconstruct the composite's
    identity and has to be complete to be sound.

    A plan carries the composite it was built for, so there is no key to get
    wrong. Hold one and reuse it across a parameter sweep and you have said, in
    one place, that the routing verdict is stable over that sweep — which is an
    assumption worth being able to see, and worth being able to check by
    re-planning at the end and comparing.

    ``core`` is ``None`` for composites that take the eager path (DISCRETE or
    EVENT processes, or ``debug``); there is no compiled core to carry.
    """

    composite: Composite
    t_span: tuple[float, float]
    macro_dt: float
    keys: list[str]
    groups: dict[str, list[str]]
    coupling: str
    integrators: dict[str, GroupIntegrator]
    save_dt: float
    requested_save_dt: float
    jump_ts: Any
    adjoint: Any
    fast: bool
    core: Any
    state_shape: tuple[int, ...]
    state_dtype: Any

    def __repr__(self) -> str:
        path = "fast" if self.fast else ("scan" if self.core else "eager")
        sizes = {g: len(p) for g, p in self.groups.items()}
        solvers = {
            g: type(i.solver).__name__ for g, i in self.integrators.items()
        }
        return (
            f"RunPlan({path}, groups={sizes}, "
            f"coupling={self.coupling!r}, solvers={solvers}, "
            f"save_dt={self.save_dt:g}, t_span={self.t_span})"
        )


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


def _varying_paths(state, keys, cols) -> list[str]:
    """The paths among ``cols`` whose batched values differ between members."""
    if not cols:
        return []
    flat = np.asarray(state).reshape(-1, np.shape(state)[-1])[:, cols]
    spread = flat.max(axis=0) - flat.min(axis=0)
    return [keys[cols[j]] for j in range(len(cols)) if spread[j] != 0.0]


def _batch_varies_only_on_assigned(composite, state, keys) -> list[str]:
    """The ASSIGNED paths a batch varies on, when it varies on nothing else.

    A per-member value on an ASSIGNED path never survives the first RHS
    call, so a population that differs *only* there is a population of
    identical members — silently, which is what makes it worth refusing. A
    population that also differs on an integrated state is a real one, and
    the assigned values are simply overwritten as they would be unbatched.
    """
    if is_traced(state):
        return []
    assigned = composite.assigned_paths()
    on_assigned = _varying_paths(
        state, keys, [i for i, k in enumerate(keys) if k in assigned]
    )
    if not on_assigned:
        return []
    on_integrated = _varying_paths(
        state, keys, [i for i, k in enumerate(keys) if k not in assigned]
    )
    return [] if on_integrated else on_assigned


def _build_proc_index_maps(
    proc, proc_topo: dict[str, str], key_to_idx: dict[str, int]
) -> tuple[tuple, tuple]:
    """``(read_pairs, write_pairs)`` for a discrete/event process — gather the
    port view from state, scatter LATCHED writes back. Each is
    ``((port_name, store_idx), ...)``."""
    read_pairs = tuple(
        (port, key_to_idx[sp])
        for port, entry in proc_topo.items()
        for sp in as_paths(entry)
    )
    schema = proc.ports_schema()
    write_pairs = tuple(
        (port, key_to_idx[sp])
        for port, p in schema.items()
        if p.role == PortRole.LATCHED
        for sp in as_paths(proc_topo[port])
    )
    return read_pairs, write_pairs


def _apply_delta(
    state_vec: jnp.ndarray,
    raw_delta: dict[str, jnp.ndarray],
    write_pairs: tuple,
) -> jnp.ndarray:
    """Scatter-add a process's delta dict into the flat state vector.

    ``state_vec`` is ``(n_vars,)`` or ``(batch, n_vars)``; a delta entry is a
    scalar, applied to every member, or ``(batch,)``, one per member."""
    out = [
        (idx, raw_delta[port])
        for port, idx in write_pairs
        if port in raw_delta
    ]
    if not out:
        return state_vec
    idxs = jnp.array([i for i, _ in out])
    lead = state_vec.shape[:-1]
    vals = jnp.stack(
        [jnp.broadcast_to(jnp.asarray(v), lead) for _, v in out], axis=-1
    )
    return state_vec.at[..., idxs].add(vals)


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


def _fingerprint_difference(want, got) -> str:
    """Which part of a structural fingerprint disagrees, for an error message.

    The three parts are store layout, process classes and wiring; naming the
    one that moved is the difference between a usable error and "they differ".
    """
    labels = ("store paths", "process classes", "wiring")
    for label, a, b in zip(labels, want, got):
        if a != b:
            only_a = sorted(set(a) - set(b))[:3]
            only_b = sorted(set(b) - set(a))[:3]
            return (
                f"They disagree on {label}: "
                f"plan-only {only_a}, given-only {only_b}."
            )
    return "They disagree, but not in store paths, classes or wiring."


def _param_digest(composite: Composite) -> str | None:
    """A digest of ``composite``'s concrete parameter values, or ``None`` when
    any of them is a tracer.

    The stiffness verdict is a function of parameter *values* -- a rate
    constant of 1 and one of 1e6 are the same structure and different
    problems -- so a structural key alone lets a sweep's second arm inherit
    the first arm's solver. ``None`` marks "cannot be measured here", which is
    the traced case, and sends the lookup to the last eagerly-resolved verdict
    for this structure.
    """
    h = hashlib.blake2b(digest_size=16)
    for leaf in jax.tree_util.tree_leaves(composite):
        if isinstance(leaf, jax.core.Tracer):
            return None
        arr = np.asarray(leaf)
        h.update(str(arr.dtype).encode())
        h.update(str(arr.shape).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


class Scheduler:
    """Multi-rate orchestrator for composites with mixed process kinds.

    Continuous groups are solved sequentially by Lie operator splitting, each
    seeing the previous group's updated state. Single-group composites with no
    events / discrete / Strang / interpolated coupling take a
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
        oscillator risks numerical anti-damping (see CLAUDE.md). ``atol`` is a
        true floor near zero and applies to every group — scaling it by the
        state duplicates what ``rtol`` already does, and freezing that scale
        at ``|y0|`` broke any state with a large dynamic range.
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
    fixed_dt:
    Lockstep mode for non-stiff groups. A fixed step (one scalar, or
    ``{group: dt}``) replaces error control there, so every member of a
    batched ``y0`` takes the same number of steps. Stiffness-routed
    implicit groups retain adaptive error control for launch transients
    and Newton convergence; this keeps a valid stiff solve from failing
    merely because the population mode was selected. Accuracy for
    lockstep groups is not controlled: compare readouts with an adaptive
    reference first. ``max_steps`` must cover ``macro_dt / fixed_dt``.
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
        fixed_dt: float | dict[str, float] | None = None,
        explicit_solver: dfx.AbstractSolver | None = None,
        implicit_solver: dfx.AbstractSolver | None = None,
        auto_stiffness: bool = True,
        newton_rtol: float | None = None,
        newton_atol: float = DEFAULT_NEWTON_ATOL,
        max_explicit_substeps: float = DEFAULT_MAX_EXPLICIT_SUBSTEPS,
        groups: dict[str, list[str]] | None = None,
        coupling_mode: str = "auto",
        coupling_interp_points: int = 16,
        splitting: str = "lie",
        waveform_sweeps: int = 1,
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
        self.max_explicit_substeps = max_explicit_substeps
        self.rtol = rtol
        self.atol = atol
        if fixed_dt is not None:
            fixed_steps = (
                list(fixed_dt.values())
                if isinstance(fixed_dt, dict)
                else [fixed_dt]
            )
            if not fixed_steps or any(
                not np.isfinite(float(step)) or float(step) <= 0
                for step in fixed_steps
            ):
                raise ValueError(
                    "fixed_dt must be a positive finite step size, or a "
                    "non-empty group-to-step dictionary"
                )
        # Lockstep mode: a fixed step per group instead of error control.
        # Every member of a batch then takes the same number of steps by
        # construction, so one vmapped loop wastes nothing on its slowest
        # member; accuracy is checked against an adaptive run, not
        # controlled. Stiffness routing still picks the solver.
        self.fixed_dt = fixed_dt
        self.controller = dfx.PIDController(rtol=rtol, atol=atol)
        self.lockstep_controller = (
            dfx.ConstantStepSize() if fixed_dt is not None else self.controller
        )
        # Per-(group structure) cache of resolved integrators. Keyed by a
        # structural signature so the eager resolution (concrete params)
        # is reused under later grad/jvp/vmap tracing, where the Jacobian
        # eigenvalues would be tracers.
        # One resolved plan, kept so a sweep loop does not re-trace. Replaced
        # rather than accumulated: a plan carries its own composite, so a stale
        # entry is a miss, never a wrong answer.
        self._last_plan: tuple[Any, RunPlan] | None = None
        self._integrator_cache: dict[Any, dict[str, GroupIntegrator]] = {}
        # The most recent verdict resolved from *concrete* parameters, per
        # structure. A traced run cannot measure its own, so it reads this;
        # an eager run always measures its own and refreshes it.
        self._eager_verdict: dict[Any, dict[str, GroupIntegrator]] = {}
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
        self.manual_groups = groups
        self.coupling_mode = coupling_mode
        self.coupling_interp_points = coupling_interp_points
        if waveform_sweeps < 1:
            raise ValueError(
                f"waveform_sweeps must be >= 1, got {waveform_sweeps}"
            )
        self.splitting = splitting
        self.waveform_sweeps = waveform_sweeps
        self.debug = debug
        self.progress = progress

    def run(
        self,
        composite: Composite | RunPlan,
        t_span: tuple[float, float] | None = None,
        macro_dt: float = 1.0,
        y0: jnp.ndarray | None = None,
        save_dt: float | None = None,
        adjoint: dfx.AbstractAdjoint | None = None,
        antialias: bool = True,
        params_from: Composite | None = None,
        seed: int = 0,
    ) -> SchedulerResult:
        """Run the composite with multi-rate scheduling.

        Parameters
        ----------
        composite:
            Wired bundle of processes, or a :class:`RunPlan` from
            :meth:`plan` — which already fixes the span, ``macro_dt``,
            ``save_dt`` and ``adjoint``, so passing those alongside it is an
            error rather than a silent override. ``y0`` still varies per run.
        params_from:
            With a plan, take parameter *values* from this composite instead of
            the one the plan was built for. Its structure must match the plan's
            — checked, not assumed — since only values may differ. Turns a
            sweep into one compiled call per arm rather than a re-resolution
            per arm, and **asserts the plan's stiffness verdict holds at these
            values**; :meth:`verify_plan` checks that assertion.
        t_span:
            ``(t0, t1)``. Required unless a plan is given.
        macro_dt:
            Fixed communication interval. Each macro step solves the
            continuous groups, fires any due discrete processes, then checks
            event conditions.
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
        seed:
            Seed for the eager direct-SSA lane. It has no effect on purely
            deterministic composites.

        Returns
        -------
        :class:`SchedulerResult`
        """
        if isinstance(composite, RunPlan):
            conflicting = [
                n
                for n, v in (
                    ("t_span", t_span),
                    ("save_dt", save_dt),
                    ("adjoint", adjoint),
                )
                if v is not None
            ]
            if conflicting:
                raise TypeError(
                    f"run() got {conflicting} alongside a RunPlan, which "
                    "already fixes them. Build a new plan instead — the "
                    "resolution they feed (solver routing, save grid, "
                    "discontinuity times) is what a plan *is*."
                )
            return self._execute(
                composite, y0, params_from=params_from, stochastic_seed=seed
            )
        if params_from is not None:
            raise TypeError(
                "run() got params_from= without a RunPlan. It substitutes "
                "values into a plan's resolution; without a plan there is no "
                "resolution to reuse — pass the composite itself."
            )
        if t_span is None:
            raise TypeError("run() requires t_span unless given a RunPlan")
        plan = self._plan_for(
            composite, t_span, macro_dt, y0, save_dt, adjoint, antialias
        )
        return self._execute(plan, y0, stochastic_seed=seed)

    def plan(
        self,
        composite: Composite,
        t_span: tuple[float, float],
        macro_dt: float = 1.0,
        y0: jnp.ndarray | None = None,
        save_dt: float | None = None,
        adjoint: dfx.AbstractAdjoint | None = None,
        antialias: bool = True,
    ) -> RunPlan:
        """Resolve a run without executing it.

        Returns the :class:`RunPlan` :meth:`run` would build — groups, coupling
        mode, per-group solver, anti-aliased ``save_dt``, discontinuity times
        and traced core. Hold one and pass it to :meth:`run` to reuse the
        resolution across many initial conditions or parameter values, which is
        what makes a sweep cheap; ``repr`` it to see what was chosen.

        ``y0`` is the state the stiffness verdict is *measured at*, defaulting
        to the composite's initial state. Reusing the plan across other states
        or parameters asserts the verdict holds there too.
        """
        return self._plan_for(
            composite, t_span, macro_dt, y0, save_dt, adjoint, antialias
        )

    def _plan_for(
        self, composite, t_span, macro_dt, y0, save_dt, adjoint, antialias
    ) -> RunPlan:
        """The plan for this call, reusing the last one when it still applies.

        A one-entry memo, not a dictionary: the key only has to be
        *conservative* -- any doubt re-plans -- where a growing cache's key has
        to be *complete* or it hands one composite's resolution to another.
        That is the whole reason the artefacts moved onto a plan, so the memo
        is deliberately the weakest thing that keeps a sweep loop off the
        re-tracing path.
        """
        key = (
            composite.structural_fingerprint(),
            _param_digest(composite),
            tuple(float(t) for t in t_span),
            float(macro_dt),
            None if save_dt is None else float(save_dt),
            None if adjoint is None else type(adjoint).__name__,
            bool(antialias),
            (
                None
                if y0 is None
                else (tuple(jnp.shape(y0)), str(jnp.asarray(y0).dtype))
            ),
        )
        cached = self._last_plan
        if cached is not None and cached[0] == key:
            return cached[1]
        plan = self._build_plan(
            composite, t_span, macro_dt, y0, save_dt, adjoint, antialias
        )
        # A plan built under an outer trace closes over that trace's tracers,
        # which would escape it on reuse.
        if not any(
            isinstance(leaf, jax.core.Tracer)
            for leaf in jax.tree_util.tree_leaves((composite, y0))
        ):
            self._last_plan = (key, plan)
        return plan

    def _reject_unsupported_batch(self, composite, state, keys) -> None:
        """Refuse a batched ``y0`` on the paths that cannot carry a batch axis.

        These branch on Python ``bool()``/``float()`` of the state, which would
        crash under vmap or silently collapse the batch axis — and a collapsed
        batch reads as a legitimate null result, every member following the
        same trajectory.
        """
        if state.ndim <= 1:
            return
        blockers = []
        if composite.stochastic_processes():
            blockers.append(
                "stochastic reaction processes currently require an "
                "unbatched state"
            )
        only_assigned = _batch_varies_only_on_assigned(composite, state, keys)
        if only_assigned:
            blockers.append(
                f"per-member values only on ASSIGNED paths {only_assigned} "
                "(an assignment is recomputed from its own process "
                "every step, so those values are overwritten before "
                "anything reads them; with every integrated state "
                "identical across members, every member would run "
                "identically)"
            )
        if blockers:
            raise ValueError(
                f"Batched y0 of shape {tuple(state.shape)} is not "
                "supported with: " + "; ".join(blockers) + ". "
                "Run unbatched, drop the blocking feature, or vmap "
                "Scheduler.run from outside."
            )

    def verify_plan(
        self, plan: RunPlan, composite: Composite, y0=None
    ) -> dict[str, tuple[bool, bool]]:
        """Did the plan's routing verdict survive to ``composite``'s values?

        Reusing a plan across a sweep or a fit asserts the stiffness verdict is
        stable over the values walked. This measures it: ``{group: (planned,
        actual)}`` for every group whose verdict moved, empty when none did.
        Run it at the end of a fit against the fitted parameters.
        """
        fresh = self._resolve_integrators(
            composite,
            plan.groups,
            (
                composite.initial_state_vec(plan.keys)
                if y0 is None
                else jnp.asarray(y0)
            ),
            plan.t_span[0],
            plan.macro_dt,
        )
        return {
            g: (plan.integrators[g].stiff, fresh[g].stiff)
            for g in plan.groups
            if plan.integrators[g].stiff != fresh[g].stiff
        }

    def _build_plan(
        self, composite, t_span, macro_dt, y0, save_dt, adjoint, antialias
    ) -> RunPlan:
        t0, t1 = t_span
        adjoint = (
            adjoint
            if adjoint is not None
            else self._resolve_adjoint(composite, y0)
        )
        # Discontinuities the solver should land on exactly rather than
        # resolve by adaptive step-rejection.
        jump_ts = self._collect_jump_ts(composite, t0, t1)
        if self.fixed_dt is not None and jump_ts is not None:
            # A fixed step cannot be clipped onto a jump, so the macro grid
            # has to land on every one of them.
            off = [
                float(j)
                for j in jump_ts
                if abs((j - t0) / macro_dt - round((j - t0) / macro_dt)) > 1e-9
            ]
            if off:
                raise ValueError(
                    f"fixed_dt cannot step onto discontinuities at {off} "
                    f"inside a macro step of {macro_dt}: a fixed step is "
                    "never clipped. Choose macro_dt so every jump falls on "
                    "its grid, or run adaptively."
                )

        # Flat state layout, pinned for the whole run. Batched y0 is
        # (batch, n_vars); the batch axis rides through every group's solve.
        keys = composite.store_keys()
        state = (
            composite.initial_state_vec(keys)
            if y0 is None
            else jnp.asarray(y0)
        )

        groups = self.manual_groups or composite.auto_groups()
        stochastic_procs = composite.stochastic_processes()
        discrete_procs = composite.discrete_processes()
        event_procs = composite.event_processes()

        self._reject_unsupported_batch(composite, state, keys)

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
            and self.splitting == "lie"
            and coupling == "frozen"
            and not stochastic_procs
        )
        # Scan path: a statically-known macro-step count, so the whole
        # multi-group run is one lax.scan — bounded compile, reverse-mode
        # memory flat in macro-step count.
        scan_eligible = (
            not discrete_procs
            and not event_procs
            and coupling in ("frozen", "interpolated")
            and not self.debug
        )
        core = (
            self._continuous_core(
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
                jump_ts=jump_ts,
                stochastic_procs=stochastic_procs,
            )
            if (fast_path_eligible or scan_eligible)
            else None
        )
        return RunPlan(
            composite=composite,
            t_span=(t0, t1),
            macro_dt=macro_dt,
            keys=keys,
            groups=groups,
            coupling=coupling,
            integrators=integrators,
            save_dt=save_dt,
            requested_save_dt=requested_save_dt,
            jump_ts=jump_ts,
            adjoint=adjoint,
            fast=fast_path_eligible,
            core=core,
            state_shape=tuple(state.shape),
            state_dtype=state.dtype,
        )

    def _execute(
        self,
        plan: RunPlan,
        y0,
        params_from: Composite | None = None,
        stochastic_seed: int = 0,
    ) -> SchedulerResult:
        """Run a resolved :class:`RunPlan` from ``y0``."""
        composite = plan.composite
        if params_from is not None:
            want = plan.composite.structural_fingerprint()
            got = params_from.structural_fingerprint()
            if got != want:
                raise ValueError(
                    "params_from= must match the plan's structure — only "
                    "parameter values may differ. "
                    + _fingerprint_difference(want, got)
                    + " Build a plan for this composite instead."
                )
            composite = params_from
        keys, groups = plan.keys, plan.groups
        t0, t1 = plan.t_span
        macro_dt, save_dt = plan.macro_dt, plan.save_dt
        coupling, adjoint = plan.coupling, plan.adjoint
        integrators = plan.integrators
        state = (
            composite.initial_state_vec(keys)
            if y0 is None
            else jnp.asarray(y0)
        )
        self._reject_unsupported_batch(composite, state, keys)
        key_to_idx = {k: i for i, k in enumerate(keys)}
        stochastic_procs = composite.stochastic_processes()
        if len(stochastic_procs) > 1:
            raise NotImplementedError(
                "Scheduler currently supports one stochastic reaction "
                "process per composite; multiple stochastic groups need "
                "explicit coupling semantics."
            )
        discrete_procs = composite.discrete_processes()
        event_procs = composite.event_processes()

        if plan.core is not None:
            ts, ys, dyn = plan.core(
                composite, state, jax.random.PRNGKey(stochastic_seed)
            )
            stats = {
                g: {
                    **values,
                    **(
                        {
                            "solver": type(integrators[g].solver).__name__,
                            "stiff": integrators[g].stiff,
                        }
                        if g in integrators
                        else {}
                    ),
                }
                for g, values in dyn.items()
            }
            return SchedulerResult(
                ts=ts,
                ys=ys,
                keys=keys,
                events=[],
                stats=_attach_diagnosis(stats, ys),
            )

        jump_ts = plan.jump_ts

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
        stochastic_idxs = {
            name: (
                _build_proc_index_maps(
                    proc, composite.topology[name], key_to_idx
                )[0],
                tuple(
                    (port, key_to_idx[path])
                    for port, port_spec in proc.ports_schema().items()
                    if port_spec.role in (PortRole.EVOLVED, PortRole.EXCLUSIVE)
                    for path in as_paths(composite.topology[name][port])
                ),
            )
            for name, proc in stochastic_procs.items()
        }
        stochastic_rng = np.random.default_rng(stochastic_seed)

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

        was_active: dict[str, Any] = {n: False for n in event_procs}

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
        for name in stochastic_procs:
            stats[name] = {
                "num_events": 0,
                "result": "not_started",
                "solver": "GillespieDirect",
                "stochastic": True,
            }

        def _record(gname: str, diag) -> None:
            result, n_steps, n_rej = diag
            st = stats[gname]
            st["num_solver_steps"] = st["num_solver_steps"] + n_steps
            st["num_rejected_steps"] = st["num_rejected_steps"] + n_rej
            st["result"] = result

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
            t_next = min(t + macro_dt, t1)

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
                        jump_ts=jump_ts,
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
                        jump_ts=jump_ts,
                    )
                    group_dt0_hint[gname] = last_dt
                    stats[gname]["num_macro_steps"] += 1
                    _record(gname, diag)
            else:  # Lie: sequential, one pass
                # Every group solved this window, not just the previous one —
                # they share the window and its grid, so their dense outputs
                # concatenate into one interpolant.
                names = list(group_rhs)
                state_in = state
                last_sweep: dict[str, jnp.ndarray] = {}
                for _sweep in range(self.waveform_sweeps):
                    state = state_in  # every sweep re-solves the same window
                    solved_ys: list[jnp.ndarray] = []
                    solved_idx: list[jnp.ndarray] = []
                    for gpos, (gname, rhs_fn) in enumerate(group_rhs.items()):
                        if coupling == "interpolated":
                            # Groups before this one from this sweep, groups
                            # after it from the last -- the second half is the
                            # backward edge a one-pass split never sees.
                            segs_y = list(solved_ys)
                            segs_i = list(solved_idx)
                            for later in names[gpos + 1 :]:
                                if later in last_sweep:
                                    segs_y.append(last_sweep[later])
                                    segs_i.append(group_write_idxs[later])
                            prev = (
                                (t, t_next, jnp.concatenate(segs_y, axis=-1))
                                if segs_y
                                else None
                            )
                            state, interp_out, last_dt, diag = (
                                self._solve_group_interpolated(
                                    rhs_fn,
                                    state,
                                    group_write_idxs[gname],
                                    t,
                                    t_next,
                                    prev,
                                    (
                                        jnp.concatenate(segs_i)
                                        if segs_i
                                        else None
                                    ),
                                    integ=integrators[gname],
                                    adjoint=adjoint,
                                    dt0_hint=group_dt0_hint.get(gname),
                                    jump_ts=jump_ts,
                                )
                            )
                            group_dt0_hint[gname] = last_dt
                            _record(gname, diag)
                            solved_ys.append(interp_out[2])
                            solved_idx.append(group_write_idxs[gname])
                            last_sweep[gname] = interp_out[2]
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
                                jump_ts=jump_ts,
                            )
                            group_dt0_hint[gname] = last_dt
                            _record(gname, diag)
                        stats[gname]["num_macro_steps"] += 1

            if stochastic_procs:
                from hallsim.stochastic import simulate_ssa

                for proc_name, proc in stochastic_procs.items():
                    read_pairs, write_pairs = stochastic_idxs[proc_name]
                    view = {port: state[..., idx] for port, idx in read_pairs}
                    process_y0 = {
                        species: float(view[species])
                        for species in proc._species_names
                    }
                    process_y0.update(
                        {
                            port: float(value)
                            for port, value in view.items()
                            if port not in process_y0
                        }
                    )
                    ssa = simulate_ssa(
                        proc,
                        t_span=(t, t_next),
                        y0=process_y0,
                        seed=int(stochastic_rng.integers(0, 2**63 - 1)),
                        max_events=10_000_000,
                    )
                    for species, value in zip(
                        proc._species_names, ssa.states[-1]
                    ):
                        if species in proc._species_inputs:
                            continue
                        if species not in view:
                            continue
                        path_idx = next(
                            idx for port, idx in write_pairs if port == species
                        )
                        state = state.at[path_idx].set(value)
                    stats[proc_name]["num_events"] += int(
                        ssa.reaction_indices.size
                    )
                    stats[proc_name]["result"] = "successful"

            for proc_name, proc in discrete_procs.items():
                if self._is_due(t, t_next, proc.dt_step):
                    read_pairs, write_pairs = discrete_idxs[proc_name]
                    view = {p: state[..., i] for p, i in read_pairs}
                    delta = proc.update(t_next, view)
                    state = _apply_delta(state, delta, write_pairs)

            for proc_name, proc in event_procs.items():
                read_pairs, write_pairs = event_idxs[proc_name]
                view = {p: state[..., i] for p, i in read_pairs}
                # One verdict per member: an edge fires for the members whose
                # condition just turned True, and only their deltas land.
                cond = jnp.asarray(proc.condition(t_next, view), dtype=bool)
                fire = cond & ~jnp.asarray(was_active[proc_name], dtype=bool)
                if bool(jnp.any(fire)):
                    delta = {
                        port: jnp.where(fire, v, 0.0)
                        for port, v in proc.handler(t_next, view).items()
                    }
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
                            members=np.asarray(fire) if fire.ndim else None,
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
        jump_ts,
        stochastic_procs=None,
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
            composite.structural_fingerprint(),
            tuple(sorted((g, tuple(sorted(p))) for g, p in groups.items())),
            coupling,
            self.waveform_sweeps,
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
            (None if jump_ts is None else tuple(float(t) for t in jump_ts)),
            tuple(sorted(stochastic_procs or {})),
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

            def core(comp, y0, rng_key=None):
                integ = integrators[gname]
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
                    jump_ts=jump_ts,
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

            def core(comp, y0, rng_key=None):
                return self._run_scan_continuous(
                    comp,
                    groups,
                    integrators,
                    y0,
                    keys,
                    t0,
                    t1,
                    macro_dt,
                    save_dt,
                    adjoint,
                    coupling,
                    jump_ts,
                    stochastic_procs or {},
                    rng_key,
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
            in_axes=(None, eqx.if_array(0), None),
            out_axes=(eqx.if_array(0), eqx.if_array(1), eqx.if_array(0)),
        )

        def batched(comp, y0, rng_key):
            ts, ys, stats = mapped(comp, y0, rng_key)
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
        multi = len(groups) > 1
        if self.coupling_mode != "auto":
            if (
                self.waveform_sweeps > 1
                and multi
                and self.coupling_mode == "frozen"
            ):
                # A frozen fill is the same constant on every sweep, so the
                # extra passes would cost k x and change nothing.
                raise ValueError(
                    "waveform_sweeps>1 with coupling_mode='frozen' does "
                    "nothing: each sweep re-reads the same constant. The "
                    "sweeps converge a *trajectory*, so they need "
                    "coupling_mode='interpolated' (or 'auto')."
                )
            return self.coupling_mode
        if self.waveform_sweeps > 1 and multi:
            return "interpolated"
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
                for port, entry in topo_p.items():
                    for path in as_paths(entry):
                        if path not in key_to_idx:
                            raise KeyError(
                                f"{pname}.{port} wired to unknown path "
                                f"{path!r}"
                            )
                p_reads, p_writes = read_write_paths(schema, topo_p)
                r |= {key_to_idx[p] for p in p_reads}
                w |= {key_to_idx[p] for p in p_writes}
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
        jump_ts=None,
        stochastic_procs=None,
        rng_key=None,
    ):
        """Continuous-only multi-group run as a single ``lax.scan``, returning
        ``(ts, ys, per_group_stats)``.

        One fixed-length scan over ``macro_dt`` windows; each solves every
        group in turn (Lie) or forward-then-reversed (Strang), threading each
        group's settled step size as the next window's warm-start. The whole
        trajectory compiles to one executable, so compile time and
        reverse-mode memory don't scale with the macro-step count.
        """
        stochastic_procs = stochastic_procs or {}
        if len(stochastic_procs) > 1:
            raise NotImplementedError(
                "compiled stochastic scheduling supports one reaction process"
            )
        group_rhs = [
            (g, composite.build_rhs(procs)[0]) for g, procs in groups.items()
        ]
        n_groups = len(group_rhs)
        strang = self.splitting == "strang" and n_groups > 1
        interp = coupling == "interpolated" and n_groups > 1
        write_idxs = [
            composite.evolved_indices(p, keys) for p in groups.values()
        ]
        stochastic_info = None
        stochastic_event_count = jnp.asarray(0, dtype=jnp.int32)
        if stochastic_procs:
            proc_name, proc = next(iter(stochastic_procs.items()))
            key_to_idx = {key: i for i, key in enumerate(keys)}
            species_idxs = jnp.asarray(
                [
                    key_to_idx[
                        as_paths(composite.topology[proc_name][species])[0]
                    ]
                    for species in proc._species_names
                ],
                dtype=jnp.int32,
            )
            stochastic_info = (proc_name, proc, species_idxs)

        # Enough windows to reach t1, not the nearest whole number of them:
        # the body clamps the last one to t1, so a span that is not a multiple
        # of macro_dt ends in a short window rather than stopping early.
        n_macro = int(round((t1 - t0) / macro_dt))
        if n_macro * macro_dt < (t1 - t0) - _TIME_EPS:
            n_macro += 1
        n_macro = max(1, n_macro)
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
                jump_ts=jump_ts,
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
            st, dt0h, steps, rej, res, rng, stochastic_event_count = carry
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
                        jump_ts,
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
                        jump_ts,
                    )
                    steps, rej = steps.at[gi].add(ns), rej.at[gi].add(nr)
                    res = res[:gi] + (r,) + res[gi + 1 :]
                    dt0_next[gi] = ld
                traj = st[None]
            else:
                traj = jnp.broadcast_to(st, (n_out,) + st.shape)
                # Every group solved this window, not just the previous one:
                # they share the window and its grid, so their dense outputs
                # concatenate into one interpolant. Carrying only the previous
                # group left every non-adjacent edge frozen.
                st_in = st
                last: list[tuple[jnp.ndarray, jnp.ndarray]] | None = None
                for _sweep in range(self.waveform_sweeps):
                    st = st_in  # every sweep re-solves the same window
                    solved: list[tuple[jnp.ndarray, jnp.ndarray]] = []
                    for gi in range(n_groups):
                        # Gauss-Seidel: groups before gi from this sweep,
                        # groups after it from the last one. The second half is
                        # what a one-pass Lie split never has, and it is the
                        # backward edge of a feedback loop.
                        segs = list(solved)
                        if last is not None:
                            segs += last[gi + 1 :]
                        if interp and segs:
                            fill = _InterpFill(
                                full=st,
                                t0=t_start,
                                t1=t_next,
                                ys=jnp.concatenate(
                                    [gy_s for _, gy_s in segs], axis=-1
                                ),
                                idx=jnp.concatenate([ix for ix, _ in segs]),
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
                        solved.append((w, gy))
                    last = solved

            if stochastic_info is not None:
                from hallsim.stochastic import ssa_step_jax

                proc_name, proc, species_idxs = stochastic_info
                rng, step_key = jax.random.split(rng)
                local = st[species_idxs]
                _, new_local, _, n_events, _ = ssa_step_jax(
                    proc,
                    t_span=(t_start, t_next),
                    state=local,
                    key=step_key,
                    max_events=100_000,
                )
                st = st.at[species_idxs].set(new_local)
                stochastic_event_count = stochastic_event_count + n_events
                if n_out:
                    traj = traj.at[-1, species_idxs].set(new_local)
            carry = (
                st,
                (
                    jnp.stack(dt0_next)
                    if n_groups
                    else jnp.zeros((0,), dtype=st.dtype)
                ),
                steps,
                rej,
                res,
                rng,
                stochastic_event_count,
            )
            return carry, traj

        init = (
            state,
            dt0_init,
            steps_init,
            rej_init,
            res_init,
            jax.random.PRNGKey(0) if rng_key is None else rng_key,
            stochastic_event_count,
        )
        (
            final_state,
            _,
            steps_tot,
            rej_tot,
            res_final,
            _,
            event_count,
        ), ys_stack = jax.lax.scan(body, init, t_starts)

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
        if stochastic_info is not None:
            stats[stochastic_info[0]] = {
                "num_events": event_count,
                "result": "successful",
                "solver": "GillespieDirect",
                "stochastic": True,
            }
        return ts, composite.materialize_assigned(ts, ys), stats

    def _remember_verdict(self, sig, base, digest, integ):
        """Cache a routing verdict under its exact key, and — when it was
        measured from concrete parameters — as *the* verdict a later traced
        run of this structure should inherit."""
        self._integrator_cache[sig] = integ
        if digest is not None:
            self._eager_verdict[base] = integ

    @staticmethod
    def _integrator_signature(composite, groups, state, macro_dt):
        """Structural key for the integrator cache: the composite's own
        identity, group→process structure, state width, and ``macro_dt``,
        never state *values* — so the eager resolution is reused under later
        traced runs of the same composite.

        The fingerprint is what makes "the same composite" mean it. Without it
        the key is satisfied by any composite of the same width whose
        processes happen to share names, and the artefacts cached against the
        first one — column indices, a traced core — are handed to the second.
        """
        gstruct = tuple(
            sorted((g, tuple(sorted(procs))) for g, procs in groups.items())
        )
        return (
            composite.structural_fingerprint(),
            gstruct,
            int(state.shape[-1]),
            float(macro_dt),
        )

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

    @staticmethod
    def _controller_with_jumps(controller, jump_ts):
        """Wrap a step-size controller so it steps exactly onto ``jump_ts``
        (forcing/step discontinuities); pass-through when there are none.

        The times come from the plan rather than the runner: they are a
        property of *(composite, span)*, and holding them on the Scheduler made
        it non-reentrant while feeding a cache key set by a previous run.
        """
        if jump_ts is None or isinstance(controller, dfx.ConstantStepSize):
            # A fixed step is never clipped; plan() has already required the
            # jumps to lie on the macro grid, where every solve restarts.
            return controller
        return dfx.ClipStepSizeController(controller, jump_ts=jump_ts)

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
        sig = (
            self._integrator_signature(composite, groups, state, macro_dt),
            _param_digest(composite),
        )
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
        controller; every group shares it (the stiff ones' vector
        ``atol`` per run, since only the routing verdict is state-independent
        enough to cache.

        Cached by structural signature so the analysis runs once, eagerly.
        Under grad/jvp/vmap the eigenvalues are tracers, so a traced call with
        a cold cache warns its way down to the explicit solver — run
        :meth:`warm_up` first to resolve the verdict outside the trace.
        """
        base = self._integrator_signature(composite, groups, state, macro_dt)
        digest = _param_digest(composite)
        sig = (base, digest)
        cached = self._integrator_cache.get(sig)
        if cached is None and digest is None:
            # Traced: the values needed to measure a verdict are tracers, so
            # reuse the one resolved eagerly for this structure. That is the
            # warm_up-then-differentiate contract, and it assumes the verdict
            # holds across the parameter search space.
            cached = self._eager_verdict.get(base)
        if cached is not None:
            return cached

        if not self.auto_solver:
            integ = {
                g: GroupIntegrator(
                    self.solver, self.lockstep_controller, stiff=False
                )
                for g in groups
            }
            self._remember_verdict(sig, base, digest, integ)
            return integ

        def _all_explicit():
            return {
                g: GroupIntegrator(
                    self.explicit_solver,
                    self.lockstep_controller,
                    stiff=False,
                )
                for g in groups
            }

        def _all_implicit():
            return {
                g: GroupIntegrator(
                    self.implicit_solver, self.controller, stiff=True
                )
                for g in groups
            }

        # The cache signature excludes state values, so the verdict is already
        # declared independent of them: a traced state can be analysed at the
        # composite's concrete initial state instead of abandoning the
        # analysis. Only a traced *composite* (parameters under grad) leaves
        # nothing concrete to measure.
        probe = (
            composite.initial_state_vec(composite.store_keys())
            if isinstance(state, jax.core.Tracer)
            else state
        )
        try:
            report = analyze_groups(
                composite,
                y0=probe,
                groups=groups,
                t0=t0,
                dt=macro_dt,
                max_explicit_substeps=self.max_explicit_substeps,
            )
        except StiffnessNotConcrete:
            report = None  # tracers in the RHS — same cold-trace situation
        except np.linalg.LinAlgError:
            # A deterministic property of the composite, not a trace artifact,
            # so this verdict is safe to cache.
            log.warning(
                "stiffness analysis failed to converge; using the explicit "
                "solver for all groups"
            )
            integ = _all_explicit()
            self._remember_verdict(sig, base, digest, integ)
            return integ

        if report is None:
            # Not cached, so a later eager call still gets to measure and route.
            # Degrade to the *implicit* solver, not the explicit one: an
            # implicit solve of a non-stiff group is slower but correct, while
            # an explicit solve of a stiff one returns finite, plausible and
            # wrong sensitivities. Fall back toward correctness and let
            # warm_up buy the speed back.
            log.warning(
                "cannot measure group stiffness under tracing (grad/jvp/vmap) "
                "with a cold cache; using the implicit solver %s for all "
                "groups. That is correct either way, but a non-stiff group "
                "pays for it — call warm_up(y0) once eagerly before "
                "differentiating so the per-group verdict is resolved outside "
                "the trace (CalibrationProblem.fit does this for you).",
                type(self.implicit_solver).__name__,
            )
            return _all_implicit()
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
                    self.lockstep_controller,
                    stiff=False,
                    info=verdict,
                )
            if self.debug:
                log.info("  stiffness: %s", verdict)
        self._remember_verdict(sig, base, digest, integ)
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
        jump_ts=None,
    ):
        """Solve one group over ``own`` and scatter the saved states back to
        full width. Returns ``(sol, saved)`` — ``sol.ys`` is the group's own
        states (what interpolated coupling passes on), ``saved`` the guarded
        full-width trajectory.

        The single solve site: every path (fast, scan, Strang, eager,
        interpolated) differs only in ``saveat`` and ``fill``.
        """
        if isinstance(integ.controller, dfx.ConstantStepSize):
            dt0_base = self._fixed_step(group_name)
        else:
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
            stepsize_controller=self._controller_with_jumps(
                integ.controller, jump_ts
            ),
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

    def _fixed_step(self, group_name: str) -> float:
        """The lockstep step for ``group_name``: the scalar ``fixed_dt``, or
        its entry in the per-group dict. The interpolated-coupling pass has
        no group of its own and takes the finest step named."""
        fd = self.fixed_dt
        if not isinstance(fd, dict):
            return float(fd)
        if group_name in fd:
            return float(fd[group_name])
        if group_name == "interpolated":
            return float(min(fd.values()))
        raise KeyError(
            f"fixed_dt names {sorted(fd)} but group {group_name!r} has no "
            "step; name every group, or pass one scalar for all of them."
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
        jump_ts=None,
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
            jump_ts=jump_ts,
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
        jump_ts=None,
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
            jump_ts,
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
        jump_ts=None,
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
            jump_ts=jump_ts,
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
