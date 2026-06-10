"""Differentiable calibration of HallSim composites.

Two layers:

- :class:`Calibrator` — the low-level optimization loop. Takes a
  user-supplied ``loss_fn(params) -> scalar`` and a parameter pytree;
  handles autodiff mode, optax setup, clamping, history logging.

- :class:`CalibrationProblem` (+ :class:`Condition`, :class:`ParameterRef`) —
  the high-level framework. Wires a composite, a set of experimental
  conditions (hallmark severity profiles), gene-reporter Δ_data, and a
  set of fittable mechanism parameters into one declarative object
  that builds the loss function for you and runs concordance on
  arbitrary held-out arms. Reuse this for any composite + dataset.

Choose the autodiff mode by parameter count, not by habit:

- **Forward-mode** (``mode="forward"``): cost is ``(1 + n_params) × forward``.
  Best when ``n_params`` is small (≤ ~10). Required when differentiating
  through Diffrax solves where you'd otherwise hit recursive-checkpoint
  step-budget issues. Wires ``dfx.ForwardMode()`` into the diffeqsolve
  adjoint automatically.

- **Reverse-mode** (``mode="reverse"``): standard ``jax.value_and_grad``.
  Best when ``n_params`` is large (e.g. NeuralODE weights). Uses
  Diffrax's default ``RecursiveCheckpointAdjoint``.

For most mechanism-parameter calibration (a handful of hallmark knobs,
rate constants), forward-mode is the right call.

Low-level usage::

    from hallsim.calibration import Calibrator

    def loss_fn(params):
        composite = build_my_composite(**params)
        result = Scheduler(adjoint=...).run(composite, ...)
        return compute_loss_against_data(result)

    cal = Calibrator(loss_fn=loss_fn, init_params={...}, mode="forward")
    history = cal.fit(steps=50)

High-level usage::

    from hallsim.calibration import (
        CalibrationProblem, Condition, ParameterRef,
    )
    problem = CalibrationProblem(
        composite=my_composite,
        reporters=MULTI_HALLMARK_REPORTERS,
        conditions={"ctrl": Condition(...), "DDIS": Condition(...)},
        data={"DDIS_vs_ctrl": ds.delta(...)},
        params={"rate": ParameterRef("dp14", "parameters.k", init=1.0)},
        fit_arms=["DDIS_vs_ctrl"],
        held_out_arms=[],
    )
    history = problem.fit(steps=40)
    results = problem.evaluate(history.final_params)
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal

import diffrax as dfx
import equinox as eqx
import jax
import jax.flatten_util  # noqa: F401  # for ravel_pytree
import jax.numpy as jnp
import optax

if TYPE_CHECKING:
    import pandas as pd

    from hallsim.composite import Composite
    from hallsim.gene_reporters import GeneReporter


log = logging.getLogger(__name__)

ParamPytree = Any  # PyTree of jnp.ndarrays / scalars


@dataclass
class CalibrationHistory:
    """Per-step record of a Calibrator.fit() run."""

    losses: list[float] = field(default_factory=list)
    param_history: list[ParamPytree] = field(default_factory=list)
    final_params: ParamPytree | None = None
    wall_time_s: float = 0.0

    def __str__(self) -> str:
        if not self.losses:
            return "CalibrationHistory: empty"
        return (
            f"CalibrationHistory: {len(self.losses)} steps, "
            f"loss {self.losses[0]:.4g} → {self.losses[-1]:.4g}, "
            f"{self.wall_time_s:.1f}s"
        )


class Calibrator:
    """Differentiable mechanism-parameter calibration loop.

    Parameters
    ----------
    loss_fn:
        Scalar-valued, JAX-traceable function of the parameter pytree.
        Must return a single scalar (use ``jnp.sum``/``jnp.mean`` to
        reduce vector outputs). The simulator runs inside this function.
    init_params:
        Initial parameter pytree. Leaves are scalars or ``jnp.ndarray``.
        For a flat dict ``{"alpha": 0.05, "k": 0.1}``, leaves are
        scalars; for vector-valued params, use ``jnp.asarray([...])``.
    clamps:
        Optional ``{leaf_name: (lo, hi)}`` for per-leaf box-clamping
        after each optimizer step. Use ``None`` (the default) to skip.
        Only top-level dict keys are inspected — nested pytrees not
        currently clamped.
    mode:
        ``"forward"`` (default) or ``"reverse"``. See module docstring.
    optimizer:
        Pass a custom ``optax.GradientTransformation`` to override the
        default ``optax.adam(learning_rate)``.
    learning_rate:
        Used only if ``optimizer`` is None.
    verbose:
        Print per-step loss every ``log_every`` steps.
    log_every:
        Logging interval.
    """

    def __init__(
        self,
        *,
        loss_fn: Callable[[ParamPytree], jnp.ndarray],
        init_params: ParamPytree,
        clamps: dict[str, tuple[float, float]] | None = None,
        mode: Literal["forward", "reverse"] = "forward",
        optimizer: optax.GradientTransformation | None = None,
        learning_rate: float = 0.05,
        verbose: bool = True,
        log_every: int = 1,
    ) -> None:
        if mode not in ("forward", "reverse"):
            raise ValueError(
                f"mode must be 'forward' or 'reverse', got {mode!r}"
            )
        self.loss_fn = loss_fn
        self.init_params = init_params
        self.clamps = clamps or {}
        self.mode = mode
        self.optimizer = optimizer or optax.adam(learning_rate)
        self.verbose = verbose
        self.log_every = max(1, log_every)

    # ── Autodiff: forward or reverse ───────────────────────────

    def _value_and_grad(self, params: ParamPytree):
        """Compute (loss, grad) using the configured autodiff mode.

        Forward mode flattens the pytree via ``ravel_pytree``, JVPs along
        each basis direction, then repacks to the original pytree shape.
        """
        if self.mode == "reverse":
            return jax.value_and_grad(self.loss_fn)(params)

        flat, unravel = jax.flatten_util.ravel_pytree(params)

        def f_flat(flat_x):
            return self.loss_fn(unravel(flat_x))

        primal = f_flat(flat)
        eye = jnp.eye(flat.shape[0], dtype=flat.dtype)
        _, grad_flat = jax.vmap(lambda v: jax.jvp(f_flat, (flat,), (v,)))(eye)
        return primal, unravel(grad_flat)

    # ── Clamping ───────────────────────────────────────────────

    def _clamp(self, params: ParamPytree) -> ParamPytree:
        if not self.clamps or not isinstance(params, dict):
            return params
        out = {}
        for k, v in params.items():
            if k in self.clamps:
                lo, hi = self.clamps[k]
                out[k] = jnp.clip(v, lo, hi)
            else:
                out[k] = v
        return out

    # ── Fit loop ───────────────────────────────────────────────

    def fit(self, steps: int) -> CalibrationHistory:
        """Run ``steps`` Adam (or supplied-optimizer) iterations.

        Returns a :class:`CalibrationHistory` with per-step loss and
        parameter snapshots.
        """
        params = self.init_params
        opt_state = self.optimizer.init(params)
        history = CalibrationHistory()
        t0 = time.time()
        for s in range(steps):
            loss, grad = self._value_and_grad(params)
            updates, opt_state = self.optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = self._clamp(params)
            history.losses.append(float(loss))
            history.param_history.append(params)
            if self.verbose and (s % self.log_every == 0 or s == steps - 1):
                msg = f"  [{s+1:3d}/{steps}] loss = {float(loss):.4g}"
                if isinstance(params, dict):
                    # Show the first 4 scalar keys for compactness.
                    pieces = [
                        f"{k}={float(v):.3g}"
                        for k, v in list(params.items())[:4]
                        if jnp.ndim(v) == 0
                    ]
                    if pieces:
                        msg += "  " + "  ".join(pieces)
                print(msg)
        history.final_params = params
        history.wall_time_s = time.time() - t0
        return history


# ═══════════════════════════════════════════════════════════════════════════
# High-level framework: Condition, ParameterRef, CalibrationProblem
# ═══════════════════════════════════════════════════════════════════════════


def _normalized(v: jnp.ndarray) -> jnp.ndarray:
    """Zero-mean, unit-norm. Robust to scale mismatch between sim and data.

    Regularization goes inside the sqrt so the gradient stays finite at
    the all-zero input (``sqrt(x)`` has divergent derivative at x=0).
    """
    v = v - jnp.mean(v)
    return v / jnp.sqrt(jnp.sum(v**2) + 1e-12)


@dataclass(frozen=True)
class CalibratableParam:
    """A mechanism parameter exposed by a Process as fittable.

    Returned by ``Process.calibratable_params`` (one per scalar the
    Process considers a reasonable calibration target) and aggregated
    by :meth:`Composite.calibration_targets`. Convert to a
    :class:`ParameterRef` by passing it through ``ParameterRef.from_target``.

    Attributes
    ----------
    process_name:
        Set by the Composite-level aggregator (default empty here so
        Processes can return self-contained descriptions; aggregator
        fills in the namespace).
    field:
        Same dotted convention as :class:`ParameterRef` /
        :class:`hallsim.hallmarks.ParameterMapping.param_name` —
        either a plain attribute or ``"parameters.<key>"``.
    default:
        The Process's current value at this field — typically the
        published rate constant from the source paper / SBML file.
    clamp:
        ``(lo, hi)`` suggested box for Calibrator. Defaults to a
        two-order-of-magnitude span around the default value
        (``(default/100, default*100)``) when not supplied by the
        Process — wider span is harmless because Calibrator's step
        size is what controls actual exploration.
    description:
        What the parameter biologically represents, ideally with the
        source paper. Surfaced in ``simulate info`` and listings.
    """

    process_name: str
    field: str
    default: float
    clamp: tuple[float, float] | None = None
    description: str = ""


def default_clamp(value: float) -> tuple[float, float]:
    """Two-order-of-magnitude box around a parameter's current value.

    The convention used when a Process doesn't supply an explicit clamp:
    a wide span is harmless because Calibrator's step size controls
    actual exploration. Handles positive, negative, and zero values.
    """
    v = float(value)
    if v > 0:
        return (v / 100.0, v * 100.0)
    if v < 0:
        return (-abs(v) * 100.0 - 10.0, abs(v) * 100.0 + 10.0)
    return (0.0, 10.0)


@dataclass(frozen=True)
class Condition:
    """A named experimental arm — a hallmark severity profile.

    A ``Condition`` represents one experimental setup (untreated DDIS,
    control, rapamycin rescue, etc.) by the severities at which each
    hallmark is applied. The same ``Condition`` is reused across
    iterations of a calibration loop.

    Attributes
    ----------
    name:
        Human-readable label, e.g. ``"DDIS"``.
    hallmarks:
        ``{hallmark_name: severity}`` passed to :func:`apply_hallmarks`.
    description:
        Optional notes for the report.
    """

    name: str
    hallmarks: dict[str, float]
    description: str = ""


@dataclass(frozen=True)
class ParameterRef:
    """Declarative pointer to a fittable parameter inside a composite.

    The ``field`` follows the same dotted convention as
    :attr:`hallsim.hallmarks.ParameterMapping.param_name`:

    - Plain attribute: ``field="alpha"`` targets ``proc.alpha``.
    - Dotted: ``field="parameters.<key>"`` targets a single
      entry inside an SBMLProcess's parameters dict.

    Calibrator substitutes ``init`` (or the latest iterate) into this
    location via ``eqx.tree_at`` before each loss evaluation.

    Attributes
    ----------
    process_name:
        Key into ``composite.processes``.
    field:
        Attribute name or dotted path (see above).
    init:
        Initial value for the optimizer.
    clamp:
        Optional ``(lo, hi)`` box-clamp applied after each step.
    description:
        Optional notes.
    """

    process_name: str
    field: str
    init: float
    clamp: tuple[float, float] | None = None
    description: str = ""


def _substitute_param(proc, field: str, value: Any):
    """Return a new Process with ``field`` set to ``value`` via eqx.tree_at.

    Handles both plain and dotted (``parameters.<key>``) forms,
    mirroring :meth:`hallsim.hallmarks.HallmarkHandle.apply` so the
    calibration substitution path and the hallmark substitution path
    use the same convention.
    """
    if "." in field:
        field_name, key = field.split(".", 1)
        current = getattr(proc, field_name)
        if not isinstance(current, dict):
            raise TypeError(
                f"Dotted field {field!r} requires {field_name!r} to be "
                f"a dict on {type(proc).__name__}; got "
                f"{type(current).__name__}"
            )
        if key not in current:
            raise KeyError(
                f"Key {key!r} not in {field_name}; "
                f"available: {sorted(current.keys())}"
            )
        return eqx.tree_at(
            lambda p, fn=field_name, k=key: getattr(p, fn)[k],
            proc,
            value,
        )
    return eqx.tree_at(lambda p, pn=field: getattr(p, pn), proc, value)


class CalibrationProblem:
    """Wire a composite + conditions + data + params into a calibration.

    Encapsulates the standard pattern for fitting mechanism parameters
    against gene-reporter Δ_data with one or more experimental arms.
    Each Δ_arm is a ``(condition, baseline)`` pair; for each arm in
    ``fit_arms``, the loss is a normalized cosine distance between
    sign-adjusted Δ_sim and Δ_data on the reporters. Held-out arms are
    evaluated in :meth:`evaluate` but not included in the fit.

    Parameters
    ----------
    composite:
        The base composite (its ``processes`` dict gets per-iteration
        substitution; topology is reused unchanged).
    reporters:
        List of :class:`GeneReporter` instances. Each reporter's
        ``observable`` is a store path; each reporter's ``summary``
        collapses the per-path trajectory to a scalar.
    conditions:
        ``{arm_name: Condition}``. Severities applied to the
        per-iteration substituted processes.
    data:
        ``{arm_pair_name: pd.Series}``. Each Series is Δ_data indexed
        by gene symbol; the per-arm-pair name (e.g.
        ``"DDIS_vs_ctrl"``) is what ``fit_arms`` / ``held_out_arms``
        select.
    arm_pairs:
        ``{arm_pair_name: (condition_name, baseline_name)}``. Names
        must be keys in ``conditions``.
    params:
        ``{param_name: ParameterRef}``. Each is fit independently.
    fit_arms:
        Subset of ``arm_pairs`` keys included in the loss.
    held_out_arms:
        Subset of ``arm_pairs`` keys excluded from the loss but
        evaluated in ``evaluate`` for held-out concordance.
    t_end, macro_dt, scheduler_kwargs:
        Forwarded to ``Scheduler.run``.
    n_save:
        Number of save points retained for trajectory summaries
        (``save_dt = (t_end - 0) / (n_save - 1)``). The trailing
        portion is what each reporter's ``summary`` operates on.
    """

    def __init__(
        self,
        *,
        composite: "Composite",
        reporters: list["GeneReporter"],
        conditions: dict[str, Condition],
        data: dict[str, "pd.Series"],
        arm_pairs: dict[str, tuple[str, str]],
        params: dict[str, ParameterRef],
        fit_arms: list[str],
        held_out_arms: list[str] | None = None,
        t_end: float = 25.0,
        macro_dt: float = 5.0,
        n_save: int = 6,
        scheduler_kwargs: dict | None = None,
        hallmark_registry: dict | None = None,
    ) -> None:
        from hallsim.composite import Composite  # local import — avoid cycle
        from hallsim.hallmarks import HALLMARK_REGISTRY

        # Validation pass over the wiring — catch typos early so the
        # JIT trace doesn't fail with a confusing message later.
        for arm, (c, b) in arm_pairs.items():
            if c not in conditions:
                raise KeyError(
                    f"arm_pairs[{arm!r}] references unknown condition {c!r}"
                )
            if b not in conditions:
                raise KeyError(
                    f"arm_pairs[{arm!r}] references unknown condition {b!r}"
                )
        for arm in fit_arms:
            if arm not in arm_pairs:
                raise KeyError(f"fit_arms entry {arm!r} not in arm_pairs")
            if arm not in data:
                raise KeyError(f"fit arm {arm!r} has no Δ_data entry")
        for arm in held_out_arms or []:
            if arm not in arm_pairs:
                raise KeyError(f"held_out_arms entry {arm!r} not in arm_pairs")
        for pname, pref in params.items():
            if pref.process_name not in composite.processes:
                raise KeyError(
                    f"params[{pname!r}].process_name={pref.process_name!r} "
                    f"not in composite.processes "
                    f"(have {sorted(composite.processes.keys())})"
                )

        # Guard rail: protect severity *dials*, not the magnitudes they
        # scale. A hallmark transform is `value = transform(severity, base)`
        # where `base` is the parameter's current value. If the transform
        # depends on `base` (e.g. `base * f(severity)`), fitting that
        # parameter calibrates the magnitude full severity maps to —
        # legitimate, and severity keeps its 0→1 meaning. If the transform
        # *ignores* `base` (severity replaces the value — the parameter IS
        # the experimenter's dial, e.g. an exposure level set directly to
        # the severity), fitting it is degenerate: severity would overwrite
        # the fitted value. Block only that case. The test probes each
        # transform with two distinct bases — no hardcoded parameter names,
        # so it generalises to any hallmark.
        reg = (
            HALLMARK_REGISTRY
            if hallmark_registry is None
            else hallmark_registry
        )
        hallmark_targets: dict[tuple[str, str], list[tuple[str, Any]]] = {}
        for hname, handle in reg.items():
            for mapping in handle.mappings:
                key = (mapping.process_name, mapping.param_name)
                hallmark_targets.setdefault(key, []).append(
                    (hname, mapping.transform)
                )

        def _ignores_base(transform) -> bool:
            try:
                return float(transform(0.5, 1.0)) == float(transform(0.5, 2.0))
            except Exception:
                return False  # can't probe → don't block

        offenders = []
        for pname, pref in params.items():
            entries = hallmark_targets.get((pref.process_name, pref.field))
            if not entries:
                continue
            if all(_ignores_base(t) for _, t in entries):
                offenders.append((pname, pref, [h for h, _ in entries]))
            else:
                log.info(
                    "Calibration param %r (%s.%s) is the magnitude scaled "
                    "by the %s hallmark(s); severity multiplies the fitted "
                    "value per arm (dial semantics preserved).",
                    pname,
                    pref.process_name,
                    pref.field,
                    ", ".join(h for h, _ in entries),
                )
        if offenders:
            msgs = []
            for pname, pref, hmarks in offenders:
                msgs.append(
                    f"  params[{pname!r}] targets {pref.process_name}."
                    f"{pref.field}, a pure severity dial set by the "
                    f"{', '.join(repr(h) for h in hmarks)} hallmark(s)."
                )
            raise ValueError(
                "Severity dials are not valid Calibrator inputs:\n"
                + "\n".join(msgs)
                + "\n\nThese parameters are set directly by "
                "Condition.hallmarks severity (the transform discards the "
                "parameter's own value), so fitting them is degenerate — "
                "severity would overwrite the fit. Fit the mechanism "
                "magnitude the dial scales (e.g. a per-exposure potency) "
                "instead."
            )

        self.composite = composite
        self.reporters = reporters
        self.conditions = conditions
        self.data = data
        self.arm_pairs = arm_pairs
        self.params = params
        self.fit_arms = fit_arms
        self.held_out_arms = held_out_arms or []
        self.t_end = t_end
        self.macro_dt = macro_dt
        self.n_save = n_save
        self.scheduler_kwargs = scheduler_kwargs or {}

        # Precompute store-path → trailing-axis index for fast lookup.
        self._store_idx = {k: i for i, k in enumerate(composite.store_keys())}
        # Precompute reporter target indices.
        self._reporter_indices = tuple(
            self._store_idx[r.observable] for r in reporters
        )
        # Reuse the Composite type for re-construction in the loss.
        self._Composite = Composite
        # One persistent Scheduler across the eager evaluate pass and the
        # differentiated loss, so its per-group stiffness verdict (which
        # must be measured with concrete params) is resolved once and
        # reused under tracing. Adjoint is chosen per run, not per
        # instance.
        from hallsim.scheduler import Scheduler

        self._scheduler = Scheduler(**self.scheduler_kwargs)
        self._warmed_up = False
        # Autodiff direction the loss is currently being differentiated in;
        # set by fit(), read by _simulate_condition to pick the matching
        # diffeqsolve adjoint. evaluate() runs eagerly (no autodiff).
        self._fit_mode = "forward"

    # ── Internal: per-condition simulation ────────────────────────

    def _substitute(self, processes: dict, param_values: dict) -> dict:
        new = dict(processes)
        for pname, pref in self.params.items():
            new[pref.process_name] = _substitute_param(
                new[pref.process_name],
                pref.field,
                param_values[pname],
            )
        return new

    def _condition_composite(self, processes: dict, condition: Condition):
        """Apply a condition's hallmark severities and wire a composite."""
        from hallsim.hallmarks import apply_hallmarks

        procs = apply_hallmarks(processes, condition.hallmarks)
        return self._Composite(
            processes=procs,
            topology=self.composite.topology,
            validate=False,
            semantic_validation=False,
        )

    def _simulate_condition(self, processes: dict, condition: Condition):
        """Apply hallmarks + run Scheduler for one condition. Returns the
        per-reporter ``summary``-collapsed observable values (a jnp array
        of length ``len(reporters)``)."""
        comp = self._condition_composite(processes, condition)
        y0 = comp.initial_state_vec()
        save_dt = max(1e-6, self.t_end / max(1, self.n_save - 1))
        # The diffeqsolve adjoint must match the outer autodiff: ForwardMode
        # under a JVP (forward fit), the default recursive-checkpoint reverse
        # adjoint under a VJP (reverse fit). A ForwardMode solve cannot be
        # reverse-differentiated (its inner while-loop has dynamic bounds).
        adjoint = dfx.ForwardMode() if self._fit_mode == "forward" else None
        res = self._scheduler.run(
            comp,
            t_span=(0.0, self.t_end),
            macro_dt=self.macro_dt,
            y0=y0,
            save_dt=save_dt,
            adjoint=adjoint,
        )
        # res.ys is (n_save, ..., n_vars). Trailing-axis convention.
        out = []
        for rep, idx in zip(self.reporters, self._reporter_indices):
            traj = res.ys[..., idx]
            out.append(rep.summary(res.ts, traj))
        return jnp.stack([jnp.asarray(o) for o in out])

    # ── Loss / fit / evaluate ─────────────────────────────────────

    def loss(self, param_values: dict[str, jnp.ndarray]) -> jnp.ndarray:
        substituted = self._substitute(self.composite.processes, param_values)
        # Cache per-condition observables so each appears in the loss
        # only once even if it's both a condition and a baseline in
        # different arm_pairs.
        obs_cache: dict[str, jnp.ndarray] = {}

        def obs_for(cond_name: str) -> jnp.ndarray:
            if cond_name not in obs_cache:
                obs_cache[cond_name] = self._simulate_condition(
                    substituted, self.conditions[cond_name]
                )
            return obs_cache[cond_name]

        signs = jnp.asarray([r.sign for r in self.reporters], dtype=float)
        per_arm_losses = []
        for arm in self.fit_arms:
            cond, base = self.arm_pairs[arm]
            delta_sim = signs * (obs_for(cond) - obs_for(base))
            # Extract Δ_data for the reporters' genes (drop NaN/missing).
            delta_data = jnp.asarray(
                [
                    float(self.data[arm].get(r.gene_symbol, jnp.nan))
                    for r in self.reporters
                ]
            )
            # Same-shape comparison; missing genes get a NaN that the
            # mask removes.
            mask = jnp.isfinite(delta_data)
            sim_norm = _normalized(jnp.where(mask, delta_sim, 0.0))
            data_norm = _normalized(jnp.where(mask, delta_data, 0.0))
            per_arm_losses.append(jnp.mean((sim_norm - data_norm) ** 2))
        return jnp.mean(jnp.stack(per_arm_losses))

    def warm_up(self, param_values: dict[str, jnp.ndarray]) -> None:
        """Resolve the Scheduler's per-group solvers eagerly.

        The auto-solver split measures each group's Jacobian spectrum,
        which needs concrete parameters — impossible once the loss is
        under forward-mode JVP. Running one representative condition's
        composite through ``Scheduler.warm_up`` at ``param_values`` caches
        the verdict (keyed by structure) so every traced loss evaluation
        reuses it. Idempotent.
        """
        if self._warmed_up:
            return
        substituted = self._substitute(self.composite.processes, param_values)
        any_cond = next(iter(self.conditions.values()))
        comp = self._condition_composite(substituted, any_cond)
        self._scheduler.warm_up(
            comp, (0.0, self.t_end), macro_dt=self.macro_dt
        )
        self._warmed_up = True

    def fit(
        self,
        *,
        steps: int = 40,
        mode: str = "forward",
        **calibrator_kwargs,
    ) -> CalibrationHistory:
        """Fit the mechanism parameters.

        ``mode`` selects the autodiff direction. ``"forward"`` JVPs along
        each parameter basis (cost scales with parameter count) and is
        robust through the multi-rate macro loop. ``"reverse"`` is a
        single VJP (cost independent of parameter count, the way neural
        networks train) — much cheaper for several parameters, now that
        the phase-insensitive ``window_mean`` summaries make the reverse
        pass through the oscillators well-behaved.
        """
        init = {k: jnp.asarray(p.init) for k, p in self.params.items()}
        clamps = {
            k: p.clamp for k, p in self.params.items() if p.clamp is not None
        }
        self._fit_mode = mode
        # Measure stiffness with concrete params before the loss goes
        # under autodiff — the per-group solver verdict can't be computed
        # from tracers.
        self.warm_up(init)
        cal = Calibrator(
            loss_fn=self.loss,
            init_params=init,
            clamps=clamps or None,
            mode=mode,
            **calibrator_kwargs,
        )
        return cal.fit(steps=steps)

    def evaluate(
        self,
        param_values: dict[str, jnp.ndarray],
    ):
        """Run all arms (fit + held-out), return per-arm
        :class:`ConcordanceResult` via :func:`compute_concordance`.

        Bypasses the JAX/jvp path used inside ``loss`` — runs each
        condition once with the standard Scheduler default adjoint so
        wall-time is faster (no forward-mode unfold).
        """
        from hallsim.gene_reporters import compute_concordance
        import pandas as pd

        substituted = self._substitute(self.composite.processes, param_values)

        # Per-condition simulation cache.
        obs_cache: dict[str, dict[str, float]] = {}

        def obs_for(cond_name: str) -> dict[str, float]:
            if cond_name in obs_cache:
                return obs_cache[cond_name]
            comp = self._condition_composite(
                substituted, self.conditions[cond_name]
            )
            y0 = comp.initial_state_vec()
            save_dt = max(1e-6, self.t_end / max(1, self.n_save - 1))
            res = self._scheduler.run(
                comp,
                t_span=(0.0, self.t_end),
                macro_dt=self.macro_dt,
                y0=y0,
                save_dt=save_dt,
            )
            out: dict[str, float] = {}
            for rep, idx in zip(self.reporters, self._reporter_indices):
                traj = res.ys[..., idx]
                out[rep.observable] = float(rep.summary(res.ts, traj))
            obs_cache[cond_name] = out
            return out

        results = {}
        all_arms = list(self.fit_arms) + list(self.held_out_arms)
        for arm in all_arms:
            cond, base = self.arm_pairs[arm]
            c_obs = obs_for(cond)
            b_obs = obs_for(base)
            delta_sim_named = {
                r.observable: c_obs[r.observable] - b_obs[r.observable]
                for r in self.reporters
            }
            results[arm] = compute_concordance(
                delta_observables=delta_sim_named,
                delta_gene_expression=self.data.get(
                    arm, pd.Series(dtype=float)
                ),
                condition_name=arm,
                reporters=self.reporters,
            )
        return results

    # ── Output bundle: trajectories + topology + concordance JSON ──

    def _simulate_all_conditions(
        self,
        param_values: dict,
        n_save: int | None = None,
    ) -> dict:
        """Run each condition once at ``param_values``; return
        ``{cond_name: SchedulerResult}``. Uses the Scheduler's default
        adjoint (no forward-mode JVP), so wall-time is fast — meant
        for post-fit visualisation, not loss evaluation."""
        substituted = self._substitute(self.composite.processes, param_values)
        n = n_save if n_save is not None else self.n_save
        save_dt = max(1e-6, self.t_end / max(1, n - 1))
        results: dict = {}
        for cond_name, cond in self.conditions.items():
            comp = self._condition_composite(substituted, cond)
            y0 = comp.initial_state_vec()
            results[cond_name] = self._scheduler.run(
                comp,
                t_span=(0.0, self.t_end),
                macro_dt=self.macro_dt,
                y0=y0,
                save_dt=save_dt,
            )
        return results

    def save_outputs(
        self,
        out_dir: str,
        history: "CalibrationHistory",
        *,
        n_save_plot: int = 50,
    ) -> dict:
        """Produce the post-fit artifact bundle in ``out_dir``.

        Generates:

        - ``graph.png`` — composite topology rendered via networkx.
        - ``trajectories_<cond>_pre_vs_post.png`` — one figure per
          condition overlaying pre-fit and post-fit reporter
          trajectories.
        - ``trajectories_post_all_arms.png`` — all conditions
          overlaid at post-fit params.
        - ``trajectories.json`` — per-condition reporter-path
          trajectories at post-fit (densely sampled, ``n_save_plot``
          points).
        - ``summary.json`` — fitted params, init params, loss history,
          per-arm concordance (pre and post), conditions, params.

        Re-samples each condition at ``n_save_plot`` points so the
        trajectory plots are smooth (the loss path uses ``self.n_save``
        which is intentionally low for jvp tractability).

        Returns a dict describing the written artifacts.
        """
        from hallsim.plotting import (
            draw_composite_graph,
            plot_runs_comparison,
            save_run_results,
        )

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        init = {k: jnp.asarray(p.init) for k, p in self.params.items()}
        final = history.final_params or init

        # Densely-sampled trajectories at both ends of the fit.
        pre_runs = self._simulate_all_conditions(init, n_save=n_save_plot)
        post_runs = self._simulate_all_conditions(final, n_save=n_save_plot)

        # Concordance — uses the standard n_save path (matches the
        # numbers the demo prints) so the JSON tallies with stdout.
        results_pre = self.evaluate(init)
        results_post = self.evaluate(final)

        reporter_paths = [r.observable for r in self.reporters]

        # 1. Topology
        draw_composite_graph(
            self.composite,
            save=str(out / "graph.png"),
            title="composite topology",
        )

        # 2. Per-condition pre-vs-post trajectory overlays
        for cond_name in self.conditions:
            plot_runs_comparison(
                {
                    "pre-fit": pre_runs[cond_name],
                    "post-fit": post_runs[cond_name],
                },
                paths=reporter_paths,
                title=f"{cond_name}: pre vs post",
                save=str(out / f"trajectories_{cond_name}_pre_vs_post.png"),
            )

        # 3. All conditions at post-fit
        plot_runs_comparison(
            post_runs,
            paths=reporter_paths,
            title="all conditions at post-fit params",
            save=str(out / "trajectories_post_all_arms.png"),
        )

        # 4. Trajectories JSON (post-fit only — pre-fit is in the plots)
        save_run_results(
            post_runs,
            str(out / "trajectories.json"),
            paths=reporter_paths,
            metadata={
                "fitted_params": {k: float(v) for k, v in final.items()},
                "n_save_plot": n_save_plot,
                "t_end": self.t_end,
                "macro_dt": self.macro_dt,
            },
        )

        # 5. Summary JSON
        def _conc_to_dict(results_dict):
            out = {}
            for arm, r in results_dict.items():
                out[arm] = {
                    "sign_agreement": float(r.sign_agreement),
                    "spearman_r": float(r.spearman_r),
                    "n_compared": r.n_compared,
                    "rows": [
                        {
                            "gene": row.reporter.gene_symbol,
                            "observable": row.reporter.observable,
                            "delta_sim_signed": float(row.delta_sim),
                            "delta_data": float(row.delta_data),
                            "sign_match": bool(row.sign_match),
                        }
                        for row in r.rows
                    ],
                }
            return out

        summary = {
            "params": {
                k: {
                    "process_name": p.process_name,
                    "field": p.field,
                    "init": p.init,
                    "clamp": list(p.clamp) if p.clamp else None,
                    "description": p.description,
                }
                for k, p in self.params.items()
            },
            "init_params": {k: float(v) for k, v in init.items()},
            "fitted_params": {k: float(v) for k, v in final.items()},
            "loss_history": [float(v) for v in history.losses],
            "wall_time_s": float(history.wall_time_s),
            "conditions": {
                name: {
                    "hallmarks": dict(c.hallmarks),
                    "description": c.description,
                }
                for name, c in self.conditions.items()
            },
            "arm_pairs": dict(self.arm_pairs),
            "fit_arms": list(self.fit_arms),
            "held_out_arms": list(self.held_out_arms),
            "t_end": self.t_end,
            "macro_dt": self.macro_dt,
            "concordance_pre": _conc_to_dict(results_pre),
            "concordance_post": _conc_to_dict(results_post),
            "reporters": [
                {
                    "gene_symbol": r.gene_symbol,
                    "observable": r.observable,
                    "sign": r.sign,
                    "summary": (
                        r.summary.__name__
                        if hasattr(r.summary, "__name__")
                        else type(r.summary).__name__
                    ),
                }
                for r in self.reporters
            ],
        }
        with open(out / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        return {
            "out_dir": str(out),
            "files": sorted(p.name for p in out.iterdir()),
        }
