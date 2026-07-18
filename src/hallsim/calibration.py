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
import optax.contrib

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
    val_losses: list[float] = field(default_factory=list)
    param_history: list[ParamPytree] = field(default_factory=list)
    final_params: ParamPytree | None = None
    best_loss: float = float("inf")
    stopped_epoch: int | None = None
    wall_time_s: float = 0.0

    def __str__(self) -> str:
        if not self.losses:
            return "CalibrationHistory: empty"
        return (
            f"CalibrationHistory: {len(self.losses)} steps, "
            f"loss {self.losses[0]:.4g} → {self.losses[-1]:.4g}, "
            f"{self.wall_time_s:.1f}s"
        )


def load_checkpoint(path) -> tuple[dict, dict]:
    """Load a Calibrator checkpoint written via ``checkpoint_path``.

    Returns ``(params, meta)`` — the fitted param dict and ``{"value",
    "epoch"}`` of the best point at write time.
    """
    import numpy as np

    d = np.load(path)
    pre = "param."
    params = {
        k[len(pre) :]: jnp.asarray(d[k]) for k in d.files if k.startswith(pre)
    }
    return params, {"value": float(d["_value"]), "epoch": int(d["_epoch"])}


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
        val_loss_fn: Callable[[ParamPytree], jnp.ndarray] | None = None,
        checkpoint_path: str | Path | None = None,
        mode: Literal["forward", "reverse"] = "forward",
        method: Literal["adam", "lbfgs"] = "adam",
        optimizer: optax.GradientTransformation | None = None,
        learning_rate: float = 0.05,
        reduce_on_plateau: bool = False,
        plateau_patience: int = 3,
        plateau_factor: float = 0.5,
        early_stop_patience: int = 0,
        early_stop_tol: float = 1e-4,
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
        # When set, best-params / early-stop watch this held-out loss instead
        # of the training loss.
        self.val_loss_fn = val_loss_fn
        # When set, the best params so far are written here (atomically) on
        # every improvement, so a killed run keeps its best.
        self.checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path else None
        )
        self.mode = mode
        self.method = method
        # Early stopping: 0 disables. Stop after `early_stop_patience`
        # steps without a > tol loss improvement; return the best params
        # seen, not the last (the loss is bumpy). Set patience above
        # plateau_patience so the LR decays before giving up.
        self.early_stop_patience = early_stop_patience
        self.early_stop_tol = early_stop_tol
        # Default optimizer: Adam, optionally with a reduce-LR-on-plateau
        # tail that halves the step scale after `plateau_patience` steps
        # without ≥rtol improvement — damps the overshoot a fixed LR shows
        # on the flat direction-only loss. The tail reads the loss via
        # update(..., value=loss); `_uses_plateau` gates that call.
        base = optimizer or optax.adam(learning_rate)
        self._uses_plateau = reduce_on_plateau and optimizer is None
        if self._uses_plateau:
            base = optax.chain(
                base,
                optax.contrib.reduce_on_plateau(
                    factor=plateau_factor,
                    patience=plateau_patience,
                    cooldown=1,
                    rtol=1e-3,
                ),
            )
        self.optimizer = base
        self.verbose = verbose
        self.log_every = max(1, log_every)
        # Built once, on first fit() step, then reused: jitting the whole
        # value-and-grad compiles the composite rebuild + solve + adjoint
        # into one cached executable instead of re-tracing every step.
        self._vg = None

    # ── Autodiff: forward or reverse ───────────────────────────

    def _value_and_grad_fn(self):
        """Return the jitted ``params -> (loss, grad)`` transform.

        Built once and cached: jitting compiles the whole loss (composite
        rebuild + solve) and its autodiff into a single reusable
        executable, so each fit step is an execution rather than a
        re-trace. Reverse mode is a plain ``value_and_grad``; forward mode
        flattens the pytree and JVPs along each basis direction (cost
        scales with parameter count).
        """
        if self._vg is not None:
            return self._vg

        if self.mode == "reverse":
            vg = jax.value_and_grad(self.loss_fn)
        else:

            def vg(params):
                flat, unravel = jax.flatten_util.ravel_pytree(params)

                def f_flat(flat_x):
                    return self.loss_fn(unravel(flat_x))

                primal = f_flat(flat)
                eye = jnp.eye(flat.shape[0], dtype=flat.dtype)
                _, grad_flat = jax.vmap(
                    lambda v: jax.jvp(f_flat, (flat,), (v,))
                )(eye)
                return primal, unravel(grad_flat)

        self._vg = jax.jit(vg)
        return self._vg

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

    def _save_checkpoint(self, params, value, epoch) -> None:
        if self.checkpoint_path is None or not isinstance(params, dict):
            return
        import os
        import tempfile

        import numpy as np

        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"param.{k}": np.asarray(v) for k, v in params.items()}
        arrays["_value"] = np.asarray(value)
        arrays["_epoch"] = np.asarray(epoch)
        fd, tmp = tempfile.mkstemp(
            dir=str(self.checkpoint_path.parent), suffix=".tmp"
        )
        try:
            with os.fdopen(fd, "wb") as fh:
                np.savez(fh, **arrays)
            os.replace(tmp, self.checkpoint_path)  # atomic
        except BaseException:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise

    # ── Fit loop ───────────────────────────────────────────────

    def fit(self, steps: int) -> CalibrationHistory:
        """Run ``steps`` optimizer iterations (Adam or L-BFGS).

        Returns a :class:`CalibrationHistory` with per-step loss and
        parameter snapshots.
        """
        if self.method == "lbfgs":
            return self._fit_lbfgs(steps)
        params = self.init_params
        opt_state = self.optimizer.init(params)
        value_and_grad = self._value_and_grad_fn()
        val_fn = jax.jit(self.val_loss_fn) if self.val_loss_fn else None
        history = CalibrationHistory()
        best_loss = float("inf")
        best_params = params
        no_improve = 0
        t0 = time.time()
        for s in range(steps):
            loss, grad = value_and_grad(params)
            lf = float(loss)
            # `loss`/`grad` are evaluated at `params` (before this step's
            # update), so `params` is the point that achieved `lf`. Record and
            # log this evaluated point — loss and params correspond — then take
            # the step. With a validation loss, that held-out score, not the
            # training loss, is the selection criterion.
            monitored = float(val_fn(params)) if val_fn else lf
            if val_fn:
                history.val_losses.append(monitored)
            if monitored < best_loss - self.early_stop_tol:
                best_loss, best_params, no_improve = monitored, params, 0
                self._save_checkpoint(best_params, best_loss, s)
            else:
                no_improve += 1
            history.losses.append(lf)
            history.param_history.append(params)
            if self.verbose and (s % self.log_every == 0 or s == steps - 1):
                msg = f"  [{s+1:3d}/{steps}] loss = {lf:.4g}"
                if val_fn:
                    msg += f"  val = {monitored:.4g}"
                if isinstance(params, dict):
                    # Show the first 4 scalar keys for compactness.
                    pieces = [
                        f"{k}={float(v):.3g}"
                        for k, v in list(params.items())[:4]
                        if jnp.ndim(v) == 0
                    ]
                    if pieces:
                        msg += "  " + "  ".join(pieces)
                log.info(msg)
            if self._uses_plateau:
                updates, opt_state = self.optimizer.update(
                    grad, opt_state, params, value=loss
                )
            else:
                updates, opt_state = self.optimizer.update(grad, opt_state)
            params = optax.apply_updates(params, updates)
            params = self._clamp(params)
            if (
                self.early_stop_patience
                and no_improve >= self.early_stop_patience
            ):
                history.stopped_epoch = s + 1
                if self.verbose:
                    metric = "val" if val_fn else "loss"
                    log.info(
                        "  early stop at %d (best %s %.4g, "
                        "no improvement in %d steps)",
                        s + 1,
                        metric,
                        best_loss,
                        no_improve,
                    )
                break
        history.final_params = best_params
        history.best_loss = best_loss
        history.wall_time_s = time.time() - t0
        return history

    def _fit_lbfgs(self, steps: int) -> CalibrationHistory:
        """L-BFGS with zoom line search — reverse-mode gradients, curvature
        from recent gradient history, and a line search that guarantees each
        step does not increase the loss. Bounds are handled by projecting
        (clamping) after each step. Shares the early-stop / best-params logic.
        """
        loss_fn = self.loss_fn
        opt = optax.lbfgs()
        value_and_grad = optax.value_and_grad_from_state(loss_fn)
        params = self.init_params
        opt_state = opt.init(params)

        @jax.jit
        def value_grad(p, state):
            return value_and_grad(p, state=state)

        @jax.jit
        def apply(p, state, value, grad):
            updates, state = opt.update(
                grad, state, p, value=value, grad=grad, value_fn=loss_fn
            )
            return self._clamp(optax.apply_updates(p, updates)), state

        history = CalibrationHistory()
        best_loss, best_params, no_improve = float("inf"), params, 0
        t0 = time.time()
        for s in range(steps):
            value, grad = value_grad(params, opt_state)
            lf = float(value)  # loss of current params (pre-update)
            if lf < best_loss - self.early_stop_tol:
                best_loss, best_params, no_improve = lf, params, 0
                self._save_checkpoint(best_params, best_loss, s)
            else:
                no_improve += 1
            history.losses.append(lf)
            history.param_history.append(params)
            if self.verbose and (s % self.log_every == 0 or s == steps - 1):
                msg = f"  [{s+1:3d}/{steps}] loss = {lf:.4g}"
                if isinstance(params, dict):
                    pieces = [
                        f"{k}={float(v):.3g}"
                        for k, v in list(params.items())[:4]
                        if jnp.ndim(v) == 0
                    ]
                    if pieces:
                        msg += "  " + "  ".join(pieces)
                log.info(msg)
            if (
                self.early_stop_patience
                and no_improve >= self.early_stop_patience
            ):
                if self.verbose:
                    log.info(
                        "  early stop at %d (best loss %.4g)",
                        s + 1,
                        best_loss,
                    )
                break
            params, opt_state = apply(params, opt_state, value, grad)
        history.final_params = best_params
        history.best_loss = best_loss
        history.wall_time_s = time.time() - t0
        return history


# ═══════════════════════════════════════════════════════════════════════════
# High-level framework: Condition, ParameterRef, CalibrationProblem
# ═══════════════════════════════════════════════════════════════════════════


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
class ParamStep:
    """A timed SBML-constant intervention on one process in a condition.

    Represents a pharmacological intervention delivered partway through the
    trajectory (e.g. rapamycin added at washout): the constant holds
    ``value_before`` until ``t_step`` and its condition-configured (hallmark-
    set) value afterwards. Applied after :func:`apply_hallmarks` via
    :meth:`hallsim.sbml_import.SBMLProcess.with_param_step`, so the severity
    sets the post-intervention level and this supplies the pre-intervention
    level and the switch time.
    """

    process_name: str
    param_name: str
    t_step: float
    value_before: float

    def apply(self, processes: dict) -> dict:
        out = dict(processes)
        out[self.process_name] = out[self.process_name].with_param_step(
            self.param_name, self.t_step, self.value_before
        )
        return out


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
    interventions:
        Timed :class:`ParamStep` interventions applied after the hallmark
        severities — for a pharmacological effect that starts partway through
        the trajectory rather than a severity held for its whole duration.
    description:
        Optional notes for the report.
    """

    name: str
    hallmarks: dict[str, float]
    interventions: tuple = ()
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
    prior:
        Optional log-normal prior *center* for a MAP penalty (see
        :class:`CalibrationProblem`). ``None`` → this parameter is not
        regularized. Set it to the literature / derived value the fit
        should stay near; with few data points this keeps an
        under-constrained parameter from running to an unphysical rail.
    prior_sigma:
        Prior width in **log10** units (0.5 ≈ a factor of ~3, 1.0 ≈ one
        order of magnitude). Only used when ``prior`` is set.
    description:
        Optional notes.
    """

    process_name: str
    field: str
    init: float
    clamp: tuple[float, float] | None = None
    prior: float | None = None
    prior_sigma: float = 0.5
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


def gaussian_nll(
    model: jnp.ndarray, data: jnp.ndarray, weight: jnp.ndarray
) -> jnp.ndarray:
    """Weighted-Gaussian negative log-likelihood of ``data`` given ``model``.

    Both ``(n_reporter, n_timepoint)`` on the same log2-fold-change scale.
    ``weight`` is per-entry precision (1/variance) — replicate precision or a
    DESeq2/edgeR moderated SE for count data — so noisy genes count for less.
    Non-finite ``data`` entries (gene absent at a timepoint) are masked out.
    With ``weight ≡ 1`` this is exactly the masked mean-squared error, so the
    default fit is unchanged.

    This is the right likelihood when the *residual* on the log-ratio is
    Gaussian (microarray, log-CPM). Raw counts are not Gaussian — their noise
    is mean-dependent (Poisson/negative-binomial) — and want either a
    variance-stabilised log2FC + precision weights fed here, or a count-native
    likelihood that predicts absolute expression (not implemented).
    """
    mask = jnp.isfinite(data)
    w = jnp.where(mask, weight, 0.0)
    err2 = jnp.where(mask, w * (model - data) ** 2, 0.0)
    return jnp.sum(err2) / jnp.maximum(jnp.sum(mask), 1.0)


class CalibrationProblem:
    """Wire a composite + conditions + data + params into a calibration.

    Encapsulates the standard pattern for fitting mechanism parameters
    against gene-reporter Δ_data with one or more experimental arms.
    Each Δ_arm is a ``(condition, baseline)`` pair; for each arm in
    ``fit_arms``, the loss is the MSE between the model's sign-aligned
    log2 fold-change ``sign·log2(cond/base)`` and the measured log2
    fold-change on the reporters — commensurable units, so every
    reporter contributes its O(1) fold-change regardless of the
    observable's absolute scale. Held-out arms are evaluated in
    :meth:`evaluate` but not included in the fit.

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
        Trajectory-native Δ_data: ``{arm_pair_name: {timepoint: pd.Series}}``,
        each Series indexed by gene symbol (a measured log2 fold-change at
        that timepoint, in the model's time units). A plain
        ``{arm_pair_name: pd.Series}`` is accepted as the degenerate
        single-timepoint case and normalized to ``{t_end: series}`` — so an
        endpoint fit needs no timepoint bookkeeping. The loss sums one MSE
        term per (arm, timepoint); the model fits the fold-change
        *trajectory*, not just the endpoint. The per-arm-pair name (e.g.
        ``"DDIS_vs_ctrl"``) is what ``fit_arms`` / ``held_out_arms`` select.
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
    normalization:
        What each reporter value is compared against in the loss.
        ``"baseline"`` (default) — the arm's own t=0, i.e. the model
        reproduces X_t/X_0 fold-change-from-day-0 (supply ``data`` as
        log2(X_t/X_0)). ``"paired"`` — the paired baseline condition at
        matched t, a cross-arm contrast (``data`` = log2(X_cond,t/X_base,t)).
        ``"raw"`` — no reference; the loss fits sign·log2(reporter) directly,
        for a target that is an absolute (log) value rather than a comparison.
    likelihood:
        ``(model, data, weight) -> scalar`` NLL over one arm's
        ``(n_reporter, n_timepoint)`` block. Defaults to
        :func:`gaussian_nll` (weighted MSE) — the right choice when the
        residual on the log-ratio is Gaussian (microarray, log-CPM). Swap
        for a count-native likelihood otherwise.
    weights:
        Optional per-reporter precision, same ``{arm: {t: series}}`` shape
        as ``data`` (e.g. ``1/dataset.variance(cond, base)`` or DESeq2 SEs).
        Absent → unit weights, so ``gaussian_nll`` reduces to plain MSE.
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
        normalization: str = "baseline",
        t_end: float = 25.0,
        macro_dt: float = 5.0,
        n_save: int = 6,
        prior_weight: float = 1.0,
        likelihood: Callable | None = None,
        weights: dict | None = None,
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
        from hallsim.reporter_wiring import validate_reporter_mappings

        self.reporter_wiring = validate_reporter_mappings(reporters, composite)
        for r in self.reporter_wiring.warnings:
            log.warning("reporter wiring: %s", r.message)
        self.conditions = conditions
        # Trajectory-native: each arm's Δ_data is a {timepoint: Δseries}
        # map. A plain Series is the degenerate single-timepoint case —
        # normalized to {t_end: series} so endpoint fits keep working.
        self.data = {
            arm: (
                {float(t): s for t, s in d.items()}
                if isinstance(d, dict)
                else {float(t_end): d}
            )
            for arm, d in data.items()
        }
        self.arm_pairs = arm_pairs
        if normalization not in ("baseline", "paired", "raw"):
            raise ValueError(
                "normalization must be 'baseline', 'paired', or 'raw'; "
                f"got {normalization!r}"
            )
        self.normalization = normalization
        self.params = params
        self.prior_weight = prior_weight
        self.fit_arms = fit_arms
        self.held_out_arms = held_out_arms or []
        self.t_end = t_end
        self.macro_dt = macro_dt
        self.n_save = n_save
        # Tuning defaults to the auto-stiffness picker: explicit-solver
        # sensitivities NaN on stiff groups (p53–Mdm2, NF-κB) though the
        # primal stays finite. Caller's scheduler_kwargs wins if given.
        self.scheduler_kwargs = (
            {"auto_stiffness": True}
            if scheduler_kwargs is None
            else scheduler_kwargs
        )

        # Precompute store-path → trailing-axis index for fast lookup.
        self._store_idx = {k: i for i, k in enumerate(composite.store_keys())}
        # Precompute reporter target indices.
        self._reporter_indices = tuple(
            self._store_idx[r.observable] for r in reporters
        )
        # Precompute per-arm query-time and Δ_data matrices once (static).
        # The timepoint axis is *vectorized*, not looped: each summary reads
        # all query times in one interp, so the traced loss is O(n_reporters)
        # per arm regardless of timepoint count — 2 or 200 timepoints trace
        # to the same graph size (only the array length differs). A Python
        # loop here would unroll under JIT into O(n_reporters × n_timepoints)
        # nodes and blow up compile time.
        # The loss is a negative log-likelihood of Δ_data given the model's
        # Δ_sim. Default is a weighted Gaussian (weight = per-reporter
        # precision); with unit weights it is the plain masked MSE. Swap in a
        # different likelihood (e.g. count-based) via `likelihood`.
        self.likelihood = (
            likelihood if likelihood is not None else gaussian_nll
        )
        # Optional per-reporter precision, same {arm: {t: series}} shape as
        # `data` (e.g. 1/variance from replicate spread or DESeq2 SEs). None →
        # unit weights, so every reporter counts equally (the MSE default).
        weights = weights or {}
        weights = {
            arm: (
                {float(t): s for t, s in w.items()}
                if isinstance(w, dict)
                else {float(t_end): w}
            )
            for arm, w in weights.items()
        }
        self._arm_times: dict[str, list[float]] = {}
        self._arm_query_times: dict[str, jnp.ndarray] = {}
        self._arm_data_matrix: dict[str, jnp.ndarray] = {}
        self._arm_weight_matrix: dict[str, jnp.ndarray] = {}
        for arm, per_t in self.data.items():
            times = sorted(per_t.keys())
            self._arm_times[arm] = times
            self._arm_query_times[arm] = jnp.asarray(times, dtype=float)
            # (n_reporters, n_timepoints); NaN marks a gene absent at a time.
            self._arm_data_matrix[arm] = jnp.asarray(
                [
                    [
                        float(per_t[t].get(r.gene_symbol, jnp.nan))
                        for t in times
                    ]
                    for r in reporters
                ]
            )
            per_w = weights.get(arm, {})
            self._arm_weight_matrix[arm] = jnp.asarray(
                [
                    [
                        float(per_w.get(t, {}).get(r.gene_symbol, 1.0))
                        for t in times
                    ]
                    for r in reporters
                ]
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
        for iv in condition.interventions:
            procs = iv.apply(procs)
        return self._Composite(
            processes=procs,
            topology=self.composite.topology,
            validate=False,
            semantic_validation=False,
        )

    def _simulate_condition(self, processes: dict, condition: Condition):
        """Apply hallmarks + run Scheduler for one condition. Returns the
        full ``(ts, reporter_trajectories)`` — ``ts`` shape ``(n_save,)``
        and ``reporter_trajectories`` shape ``(n_reporters, n_save)`` — so
        the loss can read each reporter at arbitrary query times."""
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
        trajs = jnp.stack([res.ys[..., idx] for idx in self._reporter_indices])
        return res.ts, trajs

    def _reporter_summaries(self, ts, trajs, query_times) -> jnp.ndarray:
        """Each reporter's summary at every query time → ``(n_rep, n_t)``.

        Summaries take ``(ts, y, query_times)`` and return one value per
        query time in a single vectorized interp — so this is O(n_reporters)
        traced work, independent of how many timepoints are queried. Only
        the reporter axis is a Python loop (small, fixed); the timepoint
        axis rides inside each summary as an array.
        """
        qt = jnp.atleast_1d(jnp.asarray(query_times))
        return jnp.stack(
            [
                jnp.atleast_1d(rep.summary(ts, trajs[i], qt))
                for i, rep in enumerate(self.reporters)
            ]
        )

    # ── Loss / fit / evaluate ─────────────────────────────────────

    def _arm_reference(self, run_for, arm: str, qt):
        """``(summ_c, summ_b)`` for one arm under the configured
        ``normalization`` — the two reporter summaries whose log2 ratio is the
        arm's fold change. Single source of truth for the normalization,
        shared by :meth:`data_loss`, :meth:`model_lfc`, and :meth:`evaluate`.
        ``run_for(cond_name) -> (ts, reporter_trajs)`` is a (usually caching)
        condition solver.

        What ``summ_b`` (the reference each reporter is divided by) is:
          baseline — the arm's own t=0 (the fold-change-from-day-0 the
                     transcriptomics measures, X_t/X_0);
          paired   — the paired condition at matched t (cross-arm contrast);
          raw      — unity (no reference) → the raw reporter summary.
        """
        cond, base = self.arm_pairs[arm]
        ts_c, trajs_c = run_for(cond)
        summ_c = self._reporter_summaries(ts_c, trajs_c, qt)  # (n_rep, n_t)
        if self.normalization == "baseline":
            summ_b = self._reporter_summaries(
                ts_c, trajs_c, jnp.zeros_like(qt)
            )
        elif self.normalization == "paired":
            ts_b, trajs_b = run_for(base)
            summ_b = self._reporter_summaries(ts_b, trajs_b, qt)
        else:  # raw
            summ_b = jnp.ones_like(summ_c)
        return summ_c, summ_b

    def _arm_lfc(self, run_for, arm: str, qt) -> jnp.ndarray:
        """One arm's sign-aligned model log2 fold-change at ``qt``. Shared by
        :meth:`data_loss` and :meth:`model_lfc` so figures plot exactly what
        the loss fits."""
        summ_c, summ_b = self._arm_reference(run_for, arm, qt)
        return self._log2_fold_change(summ_c, summ_b)

    def model_lfc(
        self, param_values: dict[str, jnp.ndarray], arm: str, query_times
    ) -> jnp.ndarray:
        """The model's sign-aligned log2 fold-change for ``arm`` at
        ``query_times`` — the exact quantity :meth:`data_loss` fits, so a
        trajectory figure never drifts from the loss. Returns ``(n_rep, n_t)``.
        """
        substituted = self._substitute(self.composite.processes, param_values)
        cache: dict[str, tuple] = {}

        def run_for(cond_name: str):
            if cond_name not in cache:
                cache[cond_name] = self._simulate_condition(
                    substituted, self.conditions[cond_name]
                )
            return cache[cond_name]

        return self._arm_lfc(
            run_for, arm, jnp.atleast_1d(jnp.asarray(query_times))
        )

    def data_loss(
        self, param_values: dict[str, jnp.ndarray], arms: list[str]
    ) -> jnp.ndarray:
        """Mean per-arm MSE of the sign-aligned log2 fold-change over ``arms``.

        The pure data-fit term — no prior penalty. Factored out of
        :meth:`loss` so a held-out (validation) arm can be scored at
        arbitrary params, e.g. to trace a train-vs-validation curve for
        early stopping. ``arms`` must be keys of ``arm_pairs``.
        """
        substituted = self._substitute(self.composite.processes, param_values)
        # Each condition is solved once (whole trajectory) and cached, then
        # read at each arm's measured timepoints — so a condition that is
        # both a condition and a baseline in different arm_pairs still runs
        # once, and every timepoint reuses the same solve.
        run_cache: dict[str, tuple] = {}

        def run_for(cond_name: str):
            if cond_name not in run_cache:
                run_cache[cond_name] = self._simulate_condition(
                    substituted, self.conditions[cond_name]
                )
            return run_cache[cond_name]

        # One arm loss = mean squared error over the whole (reporter ×
        # timepoint) block: the model fits the *trajectory* of the log2
        # fold-change, not just the endpoint. The timepoint axis is
        # vectorized (precomputed static arrays + one interp per reporter),
        # so this scales to many timepoints without unrolling. A
        # single-timepoint arm is the degenerate n_t=1 case.
        arm_losses = []
        for arm in arms:
            qt = self._arm_query_times[arm]  # (n_t,)
            lfc_sim = self._arm_lfc(run_for, arm, qt)  # (n_rep, n_t)
            # The loss compares two log2 ratios, so Δ_data must be supplied as
            # a log2 fold-change (the microarray demo already is; count data
            # is log-normalized upstream). MSE then weighs every reporter's
            # O(1) log-ratio comparably regardless of the observable's absolute
            # scale (a 1e-4 pool and a 1e1 pool both contribute their
            # fold-change, not their raw amplitude).
            delta_data = self._arm_data_matrix[arm]  # (n_rep, n_t)
            weight = self._arm_weight_matrix[arm]  # (n_rep, n_t)
            arm_losses.append(self.likelihood(lfc_sim, delta_data, weight))
        return jnp.mean(jnp.stack(arm_losses))

    def loss(self, param_values: dict[str, jnp.ndarray]) -> jnp.ndarray:
        return self.data_loss(
            param_values, self.fit_arms
        ) + self._prior_penalty(param_values)

    def _prior_penalty(self, param_values: dict) -> jnp.ndarray:
        """MAP log-normal prior penalty: prior_weight · Σ ((log10 p −
        log10 prior) / prior_sigma)² over params with a ``prior`` set.

        With few data points the data term alone is under-constrained and a
        parameter can run to an unphysical rail; anchoring it to its
        literature / derived value (in log space, since rate constants span
        orders of magnitude) keeps the fit physical — a maximum-a-posteriori
        estimate treating the prior as a belief.
        """
        terms = []
        for name, pref in self.params.items():
            if pref.prior is None:
                continue
            lp = jnp.log10(jnp.clip(param_values[name], 1e-30, None))
            target = jnp.log10(jnp.asarray(float(pref.prior)))
            terms.append(((lp - target) / pref.prior_sigma) ** 2)
        if not terms:
            return jnp.asarray(0.0)
        return self.prior_weight * jnp.sum(jnp.stack(terms))

    def _log2_fold_change(
        self, cond: jnp.ndarray, base: jnp.ndarray
    ) -> jnp.ndarray:
        """Sign-aligned log2 fold-change per reporter, ``sign·log2(cond/base)``.

        Observables are non-negative pools/means; a small floor keeps the
        log finite. The result is on the same scale as the measured log2
        fold-change, so model and data compare directly. ``cond``/``base``
        are ``(n_rep,)`` or ``(n_rep, n_t)``; the per-reporter sign
        broadcasts over any trailing timepoint axis.
        """
        eps = 1e-12
        signs = jnp.asarray([r.sign for r in self.reporters], dtype=float)
        signs = signs.reshape(signs.shape + (1,) * (jnp.ndim(cond) - 1))
        log_cond = jnp.log2(jnp.clip(cond, eps, None))
        log_base = jnp.log2(jnp.clip(base, eps, None))
        return signs * (log_cond - log_base)

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
        validation_arms: list[str] | None = None,
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
        for arm in validation_arms or []:
            if arm not in self.arm_pairs:
                raise KeyError(
                    f"validation_arms entry {arm!r} not in arm_pairs"
                )
        val_loss_fn = (
            (lambda p: self.data_loss(p, validation_arms))
            if validation_arms
            else None
        )
        cal = Calibrator(
            loss_fn=self.loss,
            init_params=init,
            clamps=clamps or None,
            val_loss_fn=val_loss_fn,
            mode=mode,
            **calibrator_kwargs,
        )
        return cal.fit(steps=steps)

    def evaluate(
        self,
        param_values: dict[str, jnp.ndarray],
    ) -> dict[str, dict[float, Any]]:
        """Run all arms (fit + held-out), return per-arm, per-timepoint
        :class:`ConcordanceResult` via :func:`compute_concordance` —
        ``{arm: {timepoint: result}}``.

        Bypasses the JAX/jvp path used inside ``loss`` — runs each
        condition once with the standard Scheduler default adjoint so
        wall-time is faster (no forward-mode unfold). Each condition is
        solved once and read at every measured timepoint.
        """
        from hallsim.gene_reporters import compute_concordance

        substituted = self._substitute(self.composite.processes, param_values)

        # Per-condition simulation cache: cond_name -> (ts, reporter_trajs).
        run_cache: dict[str, tuple] = {}

        def run_for(cond_name: str):
            if cond_name in run_cache:
                return run_cache[cond_name]
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
            trajs = jnp.stack(
                [res.ys[..., idx] for idx in self._reporter_indices]
            )
            run_cache[cond_name] = (res.ts, trajs)
            return run_cache[cond_name]

        eps = 1e-12
        results: dict[str, dict[float, Any]] = {}
        all_arms = list(self.fit_arms) + list(self.held_out_arms)
        for arm in all_arms:
            times = self._arm_times[arm]
            qt = self._arm_query_times[arm]
            # Vectorized read: (n_rep, n_t) in one interp per condition, then
            # slice per timepoint. Same normalization as the loss.
            summ_c, summ_b = self._arm_reference(run_for, arm, qt)
            lfc = jnp.log2(jnp.maximum(summ_c, eps)) - jnp.log2(
                jnp.maximum(summ_b, eps)
            )  # (n_rep, n_t), unsigned; compute_concordance applies the sign
            per_t: dict[float, Any] = {}
            for j, t in enumerate(times):
                delta_sim_named = {
                    r.observable: float(lfc[i, j])
                    for i, r in enumerate(self.reporters)
                }
                per_t[float(t)] = compute_concordance(
                    delta_observables=delta_sim_named,
                    delta_gene_expression=self.data[arm][t],
                    condition_name=f"{arm}@t{float(t):g}",
                    reporters=self.reporters,
                )
            results[arm] = per_t
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
            for arm, per_t in results_dict.items():
                out[arm] = {}
                for t, r in per_t.items():
                    out[arm][f"{t:g}"] = {
                        "timepoint": float(t),
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
