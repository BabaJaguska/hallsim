"""Differentiable calibration of HallSim composites.

Two layers: :class:`Calibrator` is the optimization loop over a user-supplied
``loss_fn(params) -> scalar`` (autodiff mode, optax setup, clamping, history);
:class:`CalibrationProblem` wires a composite, experimental conditions,
gene-reporter Δ_data and fittable parameters into one declarative object that
builds that loss and scores held-out arms.

Pick the autodiff mode by parameter count: ``"forward"`` costs
``(1 + n_params) × forward`` and suits ``n_params`` ≤ ~10 (most mechanism
calibration); ``"reverse"`` is one VJP and wins for many parameters
(NeuralODE weights).

::

    problem = CalibrationProblem(
        composite=my_composite,
        reporters=MULTI_HALLMARK_REPORTERS,
        conditions={"ctrl": Condition(...), "DDIS": Condition(...)},
        data={"DDIS_vs_ctrl": ds.delta(...)},
        params={"rate": ParameterRef("dp14", "parameters.k")},
        fit_arms=["DDIS_vs_ctrl"],
    )
    history = problem.fit(steps=40)
    results = problem.evaluate(history.best_params)
"""

from __future__ import annotations

import copy
import logging
import math
import time
from dataclasses import dataclass, field, replace as dc_replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Literal, Sequence

import equinox as eqx
import jax
import jax.flatten_util  # noqa: F401  # for ravel_pytree
import jax.numpy as jnp
import numpy as np
import optax
import optax.contrib

from hallsim.process import read_param, split_param_address, write_param

if TYPE_CHECKING:
    import pandas as pd

    from hallsim.composite import Composite
    from hallsim.gene_reporters import GeneReporter


log = logging.getLogger(__name__)

# Share of a parameter's posterior precision its prior must supply to be doing
# anything, and the share above which the fitted value is just the prior again.
_MIN_PRIOR_SHARE = 1e-3
_MAX_PRIOR_SHARE = 0.99

# Fisher condition number above which a fit is refused. float64 carries ~16
# digits, so past this the flat directions are numerical noise.
MAX_FIT_CONDITION_NUMBER = 1e12
#: Leaves larger than this (a learned block's weights) are not recorded in
#: the per-step parameter history; the best iterate is kept in full.
SNAPSHOT_MAX_SIZE = 1024

ParamPytree = Any  # PyTree of jnp.ndarrays / scalars


@dataclass
class CalibrationHistory:
    """Per-step record of a Calibrator.fit() run."""

    losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    param_history: list[ParamPytree] = field(default_factory=list)
    grad_norms: list[float] = field(default_factory=list)
    lr_scales: list[float] = field(default_factory=list)
    lrs: list[float] = field(
        default_factory=list
    )  # effective LR (schedule×scale)
    # The iterate that reached ``best_loss``, not the last step's.
    best_params: ParamPytree | None = None
    best_loss: float = float("inf")
    stopped_epoch: int | None = None
    wall_time_s: float = 0.0
    # Post-fit local identifiability at best_params (None if not requested or
    # if the analysis failed); see hallsim.identifiability.
    identifiability: Any = None
    #: What produced this history — steps, autodiff mode, optimizer and its
    #: settings — so a run folder can say how its numbers were made.
    settings: dict = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.losses:
            return "CalibrationHistory: empty"
        return (
            f"CalibrationHistory: {len(self.losses)} steps, "
            f"loss {self.losses[0]:.4g} → {self.losses[-1]:.4g}, "
            f"{self.wall_time_s:.1f}s"
        )


@dataclass(frozen=True)
class OperatingRange:
    """Min / mean / max of a store path over a run — the signal band a
    coupling edge sees. Use it to place a Hill threshold (``K``) at the real
    operating point that separates conditions, instead of guessing."""

    lo: float
    mean: float
    hi: float

    def __str__(self) -> str:
        return f"[{self.lo:.4g}  μ={self.mean:.4g}  {self.hi:.4g}]"


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
        Scalar-valued, JAX-traceable function of the parameter pytree; the
        simulator runs inside it.
    init_params:
        Initial parameter pytree, leaves scalar or ``jnp.ndarray``.
    clamps:
        Optional ``{leaf_name: (lo, hi)}`` box applied after each step. Only
        top-level dict keys are inspected — nested pytrees aren't clamped.
    mode:
        ``"forward"`` (default) or ``"reverse"``. See module docstring.
    optimizer:
        Custom ``optax.GradientTransformation``, overriding the default
        ``optax.adam(learning_rate)``.
    learning_rate:
        Used only when ``optimizer`` is None.
    verbose:
        Print per-step loss every ``log_every`` steps.
    log_every:
        Logging interval.
    minibatch_seed:
        When set, ``loss_fn(params, key)`` is called with a fresh PRNG key
        each step, for a loss that draws a minibatch from it. The iterates
        are then ranked on ``loss_fn(params)``, the whole objective, since
        a draw cannot rank thousands of them. Adam only.
    eval_every:
        With a minibatch, how often (in steps) the whole objective is
        evaluated for that ranking; the last step always is.
    """

    def __init__(
        self,
        *,
        loss_fn: Callable[[ParamPytree], jnp.ndarray],
        init_params: ParamPytree,
        log_params: "bool | set[str]" = False,
        clamps: dict[str, tuple[float, float]] | None = None,
        val_loss_fn: Callable[[ParamPytree], jnp.ndarray] | None = None,
        checkpoint_path: str | Path | None = None,
        mode: Literal["forward", "reverse"] = "forward",
        method: Literal["adam", "lbfgs"] = "adam",
        optimizer: optax.GradientTransformation | None = None,
        learning_rate: float = 0.05,
        adam_b1: float = 0.9,
        grad_clip: float | None = None,
        reduce_on_plateau: bool = False,
        plateau_patience: int = 3,
        plateau_factor: float = 0.5,
        early_stop_patience: int = 0,
        early_stop_tol: float = 1e-4,
        verbose: bool = True,
        log_every: int = 1,
        minibatch_seed: int | None = None,
        eval_every: int = 1,
    ) -> None:
        if mode not in ("forward", "reverse"):
            raise ValueError(
                f"mode must be 'forward' or 'reverse', got {mode!r}"
            )
        # log10 inside; everything outside this class stays linear — the
        # loss's argument, history, best params, checkpoints.
        self._log_keys = _log_keys(init_params, log_params)
        self.minibatch_seed = minibatch_seed
        self.eval_every = max(1, int(eval_every))
        self.loss_fn = self._in_linear(loss_fn)
        self.init_params = self._to_opt(init_params)
        self.clamps = {
            k: (
                (math.log10(lo), math.log10(hi))
                if k in self._log_keys
                else (lo, hi)
            )
            for k, (lo, hi) in (clamps or {}).items()
        }
        # When set, best-params / early-stop watch this held-out loss instead
        # of the training loss.
        self.val_loss_fn = (
            self._in_linear(val_loss_fn) if val_loss_fn else None
        )
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
        base = optimizer or optax.adam(learning_rate, b1=adam_b1)
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
        # Clip the raw gradient's global norm first (before adam/plateau) — caps
        # the occasional large step that spikes the loss on a stiff ODE surface.
        if grad_clip is not None:
            base = optax.chain(optax.clip_by_global_norm(grad_clip), base)
        self.optimizer = base
        self.learning_rate = learning_rate
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

            def vg(params, *args):
                flat, unravel = jax.flatten_util.ravel_pytree(params)

                def f_flat(flat_x):
                    return self.loss_fn(unravel(flat_x), *args)

                primal = f_flat(flat)
                eye = jnp.eye(flat.shape[0], dtype=flat.dtype)
                _, grad_flat = jax.vmap(
                    lambda v: jax.jvp(f_flat, (flat,), (v,))
                )(eye)
                return primal, unravel(grad_flat)

        self._vg = eqx.filter_jit(vg)
        return self._vg

    # ── Log-space reparameterization ───────────────────────────

    def _to_opt(self, params: ParamPytree) -> ParamPytree:
        """Linear → optimizer space."""
        if not self._log_keys or not isinstance(params, dict):
            return params
        return {
            k: jnp.log10(v) if k in self._log_keys else v
            for k, v in params.items()
        }

    def _snapshot(self, params: ParamPytree) -> ParamPytree:
        """The step's parameters for the history, in model space, with any
        leaf above ``SNAPSHOT_MAX_SIZE`` elements (a learned block) recorded
        as ``None``: a history of every weight vector would hold the whole
        run in memory, and ``best_params`` keeps the one that matters."""
        model = self._to_model(params)
        if not isinstance(model, dict):
            return model
        return {
            k: (None if jnp.size(v) > SNAPSHOT_MAX_SIZE else v)
            for k, v in model.items()
        }

    def _to_model(self, params: ParamPytree) -> ParamPytree:
        """Optimizer space → linear. Everything a caller sees goes through
        here: the loss's argument, history, best params, checkpoints."""
        if not self._log_keys or not isinstance(params, dict):
            return params
        return {
            k: 10.0**v if k in self._log_keys else v for k, v in params.items()
        }

    def _in_linear(self, fn):
        """Wrap a linear-space callable so it can be handed optimizer-space
        params. Gradients flow through the transform."""
        if not self._log_keys:
            return fn
        return lambda p, *args: fn(self._to_model(p), *args)

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

    @staticmethod
    def _plateau_scale(opt_state) -> float:
        """The reduce-on-plateau LR multiplier in ``opt_state`` (1.0 if none) —
        so ``learning_rate * scale`` is the effective LR at this step."""
        stack = [opt_state]
        while stack:
            x = stack.pop()
            if type(x).__name__ == "ReduceLROnPlateauState":
                return float(x.scale)
            if isinstance(x, tuple):
                stack.extend(x)
        return 1.0

    def fit(self, steps: int) -> CalibrationHistory:
        """Run ``steps`` optimizer iterations (Adam or L-BFGS).

        Returns a :class:`CalibrationHistory` with per-step loss and
        parameter snapshots.
        """
        if self.method == "lbfgs":
            if self.minibatch_seed is not None:
                raise ValueError(
                    "L-BFGS needs the whole loss each step; minibatch_seed "
                    "is for adam"
                )
            return self._fit_lbfgs(steps)
        params = self.init_params
        opt_state = self.optimizer.init(params)
        value_and_grad = self._value_and_grad_fn()
        val_fn = eqx.filter_jit(self.val_loss_fn) if self.val_loss_fn else None
        # A minibatched loss ranks its iterates on the whole objective
        # (no key), every `eval_every` steps: a draw cannot rank thousands
        # of iterates, it picks the one that happens to fit the draw.
        key = exact_fn = None
        if self.minibatch_seed is not None:
            key = jax.random.PRNGKey(self.minibatch_seed)
            exact_fn = eqx.filter_jit(lambda p: self.loss_fn(p))
        history = CalibrationHistory()
        best_loss = float("inf")
        best_params = self._to_model(params)
        no_improve = 0
        t0 = time.time()
        for s in range(steps):
            if key is None:
                loss, grad = value_and_grad(params)
            else:
                key, sub = jax.random.split(key)
                loss, grad = value_and_grad(params, sub)
            lf = float(loss)
            # `loss`/`grad` are evaluated at `params` (before this step's
            # update), so `params` is the point that achieved `lf`. Record and
            # log this evaluated point — loss and params correspond — then take
            # the step. With a validation loss, that held-out score, not the
            # training loss, is the selection criterion.
            if val_fn:
                monitored = float(val_fn(params))
                history.val_losses.append(monitored)
            elif exact_fn is not None:
                due = s % self.eval_every == 0 or s == steps - 1
                monitored = float(exact_fn(params)) if due else None
            else:
                monitored = lf
            # Relative improvement: patience accumulates once the loss stops
            # dropping by more than `early_stop_tol` *fraction* per step —
            # scale-invariant, so it can't fire mid-descent regardless of the
            # loss magnitude (an absolute gradient threshold could).
            if monitored is None:
                pass  # between evaluations of the whole objective
            elif monitored < best_loss * (1.0 - self.early_stop_tol):
                best_loss, no_improve = monitored, 0
                best_params = self._to_model(params)
                self._save_checkpoint(best_params, best_loss, s)
            else:
                no_improve += 1
            history.losses.append(lf)
            history.param_history.append(self._snapshot(params))
            gnorm = float(optax.global_norm(grad))
            scale = self._plateau_scale(opt_state)
            lr_base = (
                float(self.learning_rate(s))
                if callable(self.learning_rate)
                else self.learning_rate
            )
            eff_lr = lr_base * scale
            history.grad_norms.append(gnorm)
            history.lr_scales.append(scale)
            history.lrs.append(eff_lr)
            if self.verbose and (s % self.log_every == 0 or s == steps - 1):
                msg = (
                    f"  [{s+1:3d}/{steps}] loss = {lf:.4g}  "
                    f"|grad| = {gnorm:.3g}  lr = {eff_lr:.2g}"
                )
                if val_fn:
                    msg += f"  val = {monitored:.4g}"
                shown = self._to_model(params)
                if isinstance(shown, dict):
                    # Show the first 4 scalar keys for compactness.
                    pieces = [
                        f"{k}={float(v):.3g}"
                        for k, v in list(shown.items())[:4]
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
        history.best_params = best_params
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

        @eqx.filter_jit
        def value_grad(p, state):
            return value_and_grad(p, state=state)

        @eqx.filter_jit
        def apply(p, state, value, grad):
            updates, state = opt.update(
                grad, state, p, value=value, grad=grad, value_fn=loss_fn
            )
            return self._clamp(optax.apply_updates(p, updates)), state

        history = CalibrationHistory()
        best_loss, no_improve = float("inf"), 0
        best_params = self._to_model(params)
        t0 = time.time()
        for s in range(steps):
            value, grad = value_grad(params, opt_state)
            lf = float(value)  # loss of current params (pre-update)
            if lf < best_loss - self.early_stop_tol:
                best_loss, no_improve = lf, 0
                best_params = self._to_model(params)
                self._save_checkpoint(best_params, best_loss, s)
            else:
                no_improve += 1
            history.losses.append(lf)
            history.param_history.append(self._snapshot(params))
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
        history.best_params = best_params
        history.best_loss = best_loss
        history.wall_time_s = time.time() - t0
        return history


# ═══════════════════════════════════════════════════════════════════════════
# High-level framework: Condition, ParameterRef, CalibrationProblem
# ═══════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CalibratableParam:
    """A mechanism parameter a Process exposes as fittable.

    Returned by ``Process.calibratable_params`` and aggregated by
    :meth:`Composite.calibration_targets`.

    Attributes
    ----------
    process_name:
        Filled in by the Composite-level aggregator, so Processes can return
        self-contained descriptions.
    field:
        Plain attribute or ``"parameters.<key>"``, as in :class:`ParameterRef`.
    default:
        The Process's current value — typically the published rate constant.
    clamp:
        Suggested ``(lo, hi)`` box; defaults to two orders of magnitude around
        ``default``. A wide span is harmless — Calibrator's step size is what
        controls exploration.
    description:
        What the parameter represents, ideally with its source paper. Surfaced
        in ``simulate info``.
    """

    process_name: str
    field: str
    default: float
    clamp: tuple[float, float] | None = None
    description: str = ""


def _log_keys(init_params, log_params) -> set:
    """Which parameters to optimize as ``log10(p)``. A non-positive value has
    no log, so it is refused rather than silently fitted in linear space."""
    if log_params is False or not isinstance(init_params, dict):
        return set()
    keys = set(init_params) if log_params is True else set(log_params)
    bad = sorted(k for k in keys if float(init_params[k]) <= 0)
    if bad:
        raise ValueError(
            f"log_params includes non-positive parameters {bad}; log10 is "
            "undefined there. Fit them in linear space instead."
        )
    return keys


@dataclass(frozen=True)
class ParamStep:
    """A timed SBML-constant intervention on one process in a condition.

    Represents a pharmacological intervention delivered partway through the
    trajectory (e.g. rapamycin added at washout): the constant holds
    ``value_before`` until ``t_step`` and its condition-configured (handle-
    set) value afterwards. Applied after :func:`apply_handles` via
    :meth:`hallsim.sbml_import.SBMLProcess.with_param_step`, so the severity
    sets the post-intervention level and this supplies the pre-intervention
    level and the switch time.
    """

    process_name: str
    param_name: str
    t_step: float
    value_before: float | None = None

    def apply(self, processes: dict, reference: dict | None = None) -> dict:
        """Wire the timed step. ``value_before=None`` holds the pre-step level
        at the param's value in ``reference`` (the substituted, pre-handle
        processes) — i.e. the fitted, untreated-severity level — so it tracks a
        fitted rate rather than a frozen constant."""
        vb = self.value_before
        if vb is None:
            src = (reference or processes)[self.process_name]
            vb = src.parameters[self.param_name]
        out = dict(processes)
        out[self.process_name] = out[self.process_name].with_param_step(
            self.param_name, self.t_step, vb
        )
        return out


@dataclass(frozen=True)
class Condition:
    """A named experimental arm — one setup (untreated DDIS, control,
    rapamycin rescue) expressed as the severities each handle is applied at,
    reused across calibration iterations.

    ``hallmarks`` is ``{hallmark_name: severity}`` for :func:`apply_handles`.
    ``interventions`` are timed :class:`ParamStep` effects applied *after* the
    severities — a drug that starts partway through the trajectory rather than
    a severity held for its whole duration.

    ``start`` overrides the shared start state on the store paths it names,
    ``{path: value}``; a value with a leading axis, ``(n_batch,)``, runs the
    condition once per member through the Scheduler's batch axis, and every
    array-valued path must have the same length. Paths it leaves out keep the
    shared start (the equilibrated baseline when equilibrating). ``window``
    is the condition's own ``(t_start, t_end)``, in place of the problem's.
    Together they express a trajectory observed from a measured state, and a
    shooting window is one such condition per segment.
    """

    name: str
    handles: dict[str, float]
    interventions: tuple = ()
    description: str = ""
    start: dict[str, Any] | None = None
    window: tuple[float, float] | None = None


@dataclass(frozen=True)
class Arm:
    """One comparison the loss fits: a condition read against a reference.

    ``reference`` is ``"t0"`` (the default) for the arm's own start, giving
    the fold change ``log2(X_t / X_0)`` a single time course supplies; the
    name of another condition, read at the matched time, for a contrast
    between arms; or ``None`` for no reference, in which case the value
    itself is compared, in the data's units through each reporter's
    ``scale``. A single arm therefore needs no pair.
    """

    condition: str
    reference: str | None = "t0"


def shooting_conditions(
    ts,
    ys,
    paths: Sequence[str],
    *,
    segments: int = 1,
    match: float = 1.0,
    prefix: str = "shoot",
    held: dict[str, Any] | None = None,
) -> tuple[dict[str, Condition], dict[str, dict], dict[str, Arm]]:
    """Multiple shooting as conditions.

    ``ys`` is ``(n_traj, n_t, len(paths))`` sampled at ``ts`` (a single
    trajectory may be ``(n_t, len(paths))``). It is cut into ``segments``
    windows; each is one condition starting from every trajectory's observed
    state at the window's opening, run over the window, with the samples
    along it as one arm with no reference. Consecutive windows share their
    boundary sample, so a window's end is fitted to the next window's start:
    the continuity term, inside the same loss. ``match`` below 1 keeps the
    first fraction of each window's samples, the curriculum's shorter
    prefix; build one set per stage. ``held`` writes constant values into
    every window's start, ``{path: value | (n_traj,)}``: a conditioning
    input each trajectory carries.

    Returns ``(conditions, data, arms)`` keyed ``f"{prefix}{k}"``, to pass to
    :class:`CalibrationProblem`; a subset of the arm names to
    :meth:`CalibrationProblem.data_loss` is the active windows.
    """
    import pandas as pd

    ts = np.asarray(ts, dtype=float)
    ys = np.asarray(ys, dtype=float)
    if ys.ndim == 2:
        ys = ys[None]
    if ys.ndim != 3 or ys.shape[2] != len(paths):
        raise ValueError(
            f"ys must be (n_traj, n_t, {len(paths)}) for paths {list(paths)}; "
            f"got shape {ys.shape}"
        )
    n_t = ys.shape[1]
    if ts.shape != (n_t,):
        raise ValueError(f"ts has {ts.shape}, ys has {n_t} samples")
    if not 0.0 < match <= 1.0:
        raise ValueError(f"match must be in (0, 1], got {match!r}")
    segments = max(1, min(int(segments), n_t - 1))
    seg_len = (n_t - 1) // segments
    conditions: dict[str, Condition] = {}
    data: dict[str, dict] = {}
    arms: dict[str, Arm] = {}
    for k in range(segments):
        s = k * seg_len
        end = n_t - 1 if k == segments - 1 else s + seg_len
        end = s + max(1, int(np.ceil(match * (end - s))))
        name = f"{prefix}{k}"
        conditions[name] = Condition(
            name,
            {},
            start={
                **(held or {}),
                **{p: ys[:, s, j] for j, p in enumerate(paths)},
            },
            window=(float(ts[s]), float(ts[end])),
        )
        data[name] = {
            float(ts[i]): pd.DataFrame(
                {p: ys[:, i, j] for j, p in enumerate(paths)}
            )
            for i in range(s + 1, end + 1)
        }
        arms[name] = Arm(name, reference=None)
    return conditions, data, arms


@dataclass(frozen=True)
class Collocation:
    """Observed states with their finite-difference slopes, for the
    composite's field to match without a solve.

    ``ys`` is ``(n_traj, n_t, len(paths))`` sampled at ``ts`` (a single
    trajectory may be ``(n_t, len(paths))``); the slopes are central
    differences along ``ts``. The field is evaluated under ``condition``
    (the first condition when ``None``) at each sample, the unobserved paths
    held at the shared start, and its components on ``paths`` are compared
    to the slopes, each scaled by that slope's spread. ``weight`` scales the
    term in :meth:`CalibrationProblem.loss`; ``0`` keeps it out of the loss
    while :meth:`CalibrationProblem.collocation_loss` stays available as a
    pretraining stage of its own. ``matched`` narrows the comparison to some
    of ``paths``, the others only setting the state (a constant input beside
    the observed states). ``batch`` draws that many samples per evaluation
    from the PRNG key the loss is given, every sample without one.
    """

    ts: Any
    ys: Any
    paths: tuple[str, ...]
    condition: str | None = None
    weight: float = 1.0
    matched: tuple[str, ...] | None = None
    batch: int | None = None


@dataclass(frozen=True)
class ParameterRef:
    """Declarative pointer to a fittable parameter inside a composite.

    ``field`` follows the dotted convention of
    :attr:`hallsim.handles.ParameterMapping.param_name`: ``"alpha"`` targets
    ``proc.alpha``, ``"parameters.<key>"`` a single entry in an SBMLProcess's
    parameters dict. Calibrator substitutes the current iterate there via
    ``eqx.tree_at`` before each loss evaluation.

    Attributes
    ----------
    process_name, field:
        Where to substitute — a key into ``composite.processes``, and the
        attribute or dotted path on it.
    clamp:
        Optional ``(lo, hi)`` box applied each step. The optimizer *starts*
        at the composite's own value for ``field`` — there is no separate
        starting value to declare, and so none that can disagree with the
        model.
    prior:
        Log-normal prior *center* for a MAP penalty; ``None`` leaves the
        parameter unregularized. Set it to the literature value the fit should
        stay near — with few data points this keeps an under-constrained
        parameter off an unphysical rail.
    prior_sigma:
        Prior width in **log10** units (0.5 ≈ a factor of 3).
    """

    process_name: str
    field: str
    clamp: tuple[float, float] | None = None
    prior: float | None = None
    prior_sigma: float = 0.5
    description: str = ""


@dataclass(frozen=True)
class HandleCoeffRef:
    """Declarative pointer to a fittable coefficient of a handle mapping.

    Points at an affine coefficient (``floor`` or ``slope``) of the
    :class:`hallsim.handles.ParameterMapping` identified by
    ``(handle, param_name)``. The Calibrator fits it exactly like a
    :class:`ParameterRef` (same ``clamp`` / ``prior`` / ``prior_sigma``
    surface, and the same read-from-the-model start), but instead of substituting into a process it
    overrides the coefficient in a per-evaluation handle registry — so the
    severity map ``base * (floor + slope * h)`` calibrates end to end. This is
    the home for a coefficient that has no SBML-parameter host (the mTOR
    rapamycin suppression gain, say): it lives on the handle edge, not a
    process, so it rides the registry rather than ``eqx.tree_at``.

    Attributes mirror :class:`ParameterRef`; ``handle`` + ``param_name``
    together select the mapping, and ``coeff`` names which affine coefficient
    to fit (``"floor"`` or ``"slope"``).
    """

    handle: str
    param_name: str
    clamp: tuple[float, float] | None = None
    prior: float | None = None
    prior_sigma: float = 0.5
    coeff: str = "floor"
    description: str = ""


@dataclass(frozen=True)
class LearnedRef:
    """A learned block's trainable partition as one fittable.

    Every inexact-array leaf of ``composite.processes[process_name]`` whose
    field is not in ``frozen`` (normalisation buffers, by default) is
    flattened into one array: linear space, no prior, no clamp, and outside
    the identifiability report, whose Fisher block over weights says nothing
    a held-out arm does not say better. :meth:`CalibrationProblem.fit` runs
    it in reverse mode, one VJP per step, where forward mode would cost a
    solve per weight.
    """

    process_name: str
    frozen: tuple[str, ...] = ("in_mean", "in_std", "out_mean", "out_std")
    description: str = ""
    prior = None
    prior_sigma = 0.5
    clamp = None


def _learned_partition(proc, frozen: Sequence[str]):
    """``(flat, unravel, fixed)``: the trainable leaves of ``proc`` as one
    vector, the map back to their tree, and the leaves left as they are."""
    from jax.flatten_util import ravel_pytree

    frozen_tree = jax.tree_util.tree_map(lambda _: False, proc)
    present = [f for f in frozen if hasattr(proc, f)]
    if present:
        frozen_tree = eqx.tree_at(
            lambda p: tuple(getattr(p, f) for f in present),
            frozen_tree,
            tuple(True for _ in present),
        )
    spec = jax.tree_util.tree_map(
        lambda leaf, fr: eqx.is_inexact_array(leaf) and not fr,
        proc,
        frozen_tree,
    )
    train, fixed = eqx.partition(proc, spec)
    flat, unravel = ravel_pytree(train)
    return flat, unravel, fixed


def _substitute_param(proc, field: str, value: Any):
    """Deprecated alias for :func:`hallsim.process.write_param`.

    Kept because the substitution path calls it in a hot loop; the
    implementation lives with :func:`hallsim.process.read_param` so the read
    and the write cannot drift apart in their dotted-field convention.
    """
    return write_param(proc, field, value)


def _placement_advice(sug) -> str:
    if sug is None:
        return "Place it with suggest_hill_gate."
    if sug.ok:
        return f"Place it at K={sug.K:.4g}, n={sug.n:g}."
    return f"K={sug.K:.4g} would open it, but {sug.note}"


def _jsonable(value):
    """``value`` as something ``json.dump`` accepts: arrays and scalars to
    numbers, series and mappings to dicts, dataclasses to their fields,
    callables to their names, anything else to its ``repr``."""
    import dataclasses

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (np.generic,)):
        return value.item()
    if isinstance(value, (np.ndarray, jnp.ndarray)):
        arr = np.asarray(value)
        return arr.item() if arr.ndim == 0 else arr.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            f.name: _jsonable(getattr(value, f.name))
            for f in dataclasses.fields(value)
        }
    if callable(value):
        return getattr(value, "__name__", type(value).__name__)
    return repr(value)


def _validate_parameter_ref(pname: str, pref, proc) -> None:
    """The field a :class:`ParameterRef` names must exist on the process,
    must not be static, and must hold a scalar — checked once when the
    problem is wired, where the failure can name the field, instead of
    inside a derivative several frames from anything the user wrote."""
    import dataclasses

    address = f"{pref.process_name}.{pref.field}"
    top = pref.field.split(".")[0]
    static = {
        f.name
        for f in dataclasses.fields(type(proc))
        if f.metadata.get("static")
    }
    if top in static:
        raise ValueError(
            f"params[{pname!r}] fits {address}, a static field: that is "
            "structure, not a value an optimizer can move."
        )
    try:
        value = read_param(proc, pref.field)
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"params[{pname!r}] fits {address}, which {type(proc).__name__} "
            f"does not have ({exc})."
        ) from exc
    # A process that declares a fittable surface is the authority on it: a
    # field outside that surface is not a mechanism parameter, whatever its
    # value looks like. A compartment size is the case everyone meets.
    declared = {item.field for item in proc.calibratable_params()}
    if declared and pref.field not in declared:
        key = pref.field.partition(".")[2] or pref.field
        if key in getattr(proc, "_compartment_names", ()):
            why = "a compartment size: geometry that scales every concentration in the model, not a rate"
        else:
            why = f"not among the parameters {type(proc).__name__} declares fittable"
        raise ValueError(
            f"params[{pname!r}] fits {address}, which is {why}. "
            "The fittable surface is proc.calibratable_params()."
        )
    if np.ndim(value) != 0:
        raise ValueError(
            f"params[{pname!r}] fits {address}, which holds a "
            f"{type(value).__name__} of shape {tuple(np.shape(value))}, not "
            "a scalar. A tuple-valued field such as a Hill edge's K or n is "
            "not fittable through a ParameterRef."
        )


def _reject_overwritten_edit(pname: str, pref, proc, baseline) -> None:
    """Raise if a fitted field changed after the problem was wired.

    Substitution is about to write the current iterate over it, so the edit
    cannot survive — and an ablation that silently does nothing reads as an
    edge with no influence, which is the one failure mode worse than an error.
    Traced values are skipped: nothing concrete to compare.
    """
    current = read_param(proc, pref.field)
    if isinstance(current, jax.core.Tracer) or isinstance(
        baseline, jax.core.Tracer
    ):
        return
    if np.array_equal(np.asarray(current), np.asarray(baseline)):
        return
    address = f"{pref.process_name}.{pref.field}"
    raise ValueError(
        f"{address} was edited after this problem was wired "
        f"({baseline} → {current}), but params[{pname!r}] fits that field, so "
        "the next substitution overwrites the edit and it has no effect. Use "
        f"problem.with_overrides({{{address!r}: value}}) instead — an "
        "override is applied last and wins."
    )


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

    Each arm is a ``(condition, baseline)`` pair; for every arm in
    ``fit_arms`` the loss compares the model's sign-aligned log2 fold-change
    ``sign·log2(cond/base)`` against the measured one on the reporters —
    commensurable units, so every reporter contributes its O(1) fold-change
    regardless of the observable's absolute scale. Held-out arms are scored in
    :meth:`evaluate` but excluded from the fit.

    Parameters
    ----------
    composite:
        Base composite; its ``processes`` get per-iteration substitution,
        topology is reused unchanged.
    reporters:
        :class:`GeneReporter` instances — each maps a store path
        (``observable``) through a ``summary`` to a scalar.
    conditions:
        ``{arm_name: Condition}`` — severities applied per iteration. A
        condition with its own ``start`` or ``window`` runs from that state
        over that span instead of the shared start and ``(t_start, t_end)``.
    data:
        ``{arm_name: {timepoint: pd.Series}}``, each Series indexed by gene
        symbol with the measured value in the arm's terms: a log2 fold change
        against the arm's reference, or the value itself for an arm with no
        reference. A plain Series is accepted as the single-timepoint case.
        The loss sums one term per (arm, timepoint), so the model fits the
        *trajectory*. For a batched condition a timepoint's entry may be a
        DataFrame, one row per member, columns the gene symbols; a Series
        there is the same target for every member.
    arms:
        ``{arm_name: Arm}``; a plain condition name stands for
        ``Arm(name)``, read against its own start.
    params:
        ``{param_name: ParameterRef | HandleCoeffRef}``.
    fit_arms, held_out_arms:
        Subsets of ``arms`` included in / excluded from the loss.
    equilibrate:
        Start every arm from the shared pre-perturbation fixed point rather
        than ``initial_state_vec()``, so a fold-change is measured against the
        biological baseline instead of an arbitrary relaxation transient.
        Solved by Newton — exact and differentiable in the fitted params.
    equilibration_condition:
        Which condition's fixed point is that baseline (an autonomous one,
        timed inputs off). Required when ``equilibrate=True``.
    likelihood:
        ``(model, data, weight) -> scalar`` over one arm's
        ``(n_reporter, n_timepoint)`` block. Default :func:`gaussian_nll`.
    weights:
        Per-reporter precision, same shape as ``data``. Absent → unit weights,
        so ``gaussian_nll`` reduces to plain MSE.
    t_end, macro_dt, scheduler_kwargs:
        Forwarded to ``Scheduler.run``.
    n_save:
        Save points retained for trajectory summaries.
    collocation:
        A :class:`Collocation`: observed states whose finite-difference
        slopes the composite's field is matched to without a solve, added to
        the loss at its ``weight``.
    member_batch:
        Members of a batched condition to run per loss evaluation, drawn
        afresh from the PRNG key the loss is given (see
        ``Calibrator(minibatch_seed=...)``); every member without a key.
        Memory and time of a solve scale with it.
    """

    def __init__(
        self,
        *,
        composite: "Composite",
        reporters: list["GeneReporter"],
        conditions: dict[str, Condition],
        data: dict[str, "pd.Series"],
        arms: dict[str, "Arm | str"],
        params: dict[str, "ParameterRef | HandleCoeffRef"],
        fit_arms: list[str],
        held_out_arms: list[str] | None = None,
        equilibrate: bool = False,
        equilibration_condition: str | None = None,
        t_end: float = 25.0,
        t_start: float = 0.0,
        macro_dt: float = 5.0,
        n_save: int = 6,
        prior_weight: float = 1.0,
        likelihood: Callable | None = None,
        weights: dict | None = None,
        scheduler_kwargs: dict | None = None,
        registry: dict | None = None,
        notes: dict | None = None,
        collocation: "Collocation | None" = None,
        member_batch: int | None = None,
    ) -> None:
        self._ctor_kwargs = {
            k: v for k, v in locals().items() if k not in ("self", "__class__")
        }
        from hallsim.composite import Composite  # local import — avoid cycle

        if registry is None and any(c.handles for c in conditions.values()):
            raise ValueError(
                "the conditions name handles but no registry was given; "
                "pass registry={name: Handle} (the demos' hallmarks are "
                "demos.models.hallmarks.HALLMARK_REGISTRY)"
            )
        # A ParameterRef substitutes into a process (`eqx.tree_at`); a
        # HandleCoeffRef overrides a handle-mapping coefficient in a
        # per-eval registry. Both share the optimizer surface (init / clamp /
        # prior); only the application path differs.
        proc_params = {
            k: v for k, v in params.items() if isinstance(v, ParameterRef)
        }
        coeff_params = {
            k: v for k, v in params.items() if isinstance(v, HandleCoeffRef)
        }

        # Validation pass over the wiring — catch typos early so the
        # JIT trace doesn't fail with a confusing message later.
        self.arms = {
            name: a if isinstance(a, Arm) else Arm(a)
            for name, a in arms.items()
        }
        for name, a in self.arms.items():
            if a.condition not in conditions:
                raise KeyError(
                    f"arms[{name!r}] references unknown condition "
                    f"{a.condition!r}"
                )
            if (
                a.reference not in (None, "t0")
                and a.reference not in conditions
            ):
                raise KeyError(
                    f"arms[{name!r}] references unknown condition "
                    f"{a.reference!r} (a reference is 't0', a condition name, "
                    "or None)"
                )
        store_keys = set(composite.store_keys())
        for cname, cond in conditions.items():
            if cond.window is not None:
                lo, hi = (float(v) for v in cond.window)
                if not hi > lo:
                    raise ValueError(
                        f"conditions[{cname!r}].window={cond.window!r} must "
                        "be (t_start, t_end) with t_end > t_start"
                    )
            lengths = set()
            for path, value in (cond.start or {}).items():
                if path not in store_keys:
                    raise KeyError(
                        f"conditions[{cname!r}].start names {path!r}, not a "
                        "store path of the composite"
                    )
                if np.ndim(value) > 0:
                    lengths.add(int(np.shape(value)[0]))
            if len(lengths) > 1:
                raise ValueError(
                    f"conditions[{cname!r}].start mixes batch lengths "
                    f"{sorted(lengths)}; every array-valued path is one "
                    "value per member"
                )
        for arm in fit_arms:
            if arm not in self.arms:
                raise KeyError(f"fit_arms entry {arm!r} not in arms")
            if arm not in data:
                raise KeyError(f"fit arm {arm!r} has no data entry")
        for arm in held_out_arms or []:
            if arm not in self.arms:
                raise KeyError(f"held_out_arms entry {arm!r} not in arms")
        for pname, pref in proc_params.items():
            if pref.process_name not in composite.processes:
                raise KeyError(
                    f"params[{pname!r}].process_name={pref.process_name!r} "
                    f"not in composite.processes "
                    f"(have {sorted(composite.processes.keys())})"
                )
            _validate_parameter_ref(
                pname, pref, composite.processes[pref.process_name]
            )

        # Block fitting a level a handle sets: severity writes it directly,
        # so a fit would be overwritten per arm. A rate a handle scales stays
        # fittable — it is the magnitude severity multiplies.
        from hallsim.handles import is_level

        reg = {} if registry is None else registry
        handle_targets: dict[tuple[str, str], list[tuple[str, Any]]] = {}
        for hname, handle in reg.items():
            for mapping in handle.mappings:
                key = (mapping.process_name, mapping.param_name)
                handle_targets.setdefault(key, []).append((hname, mapping))

        offenders = []
        for pname, pref in proc_params.items():
            entries = handle_targets.get((pref.process_name, pref.field))
            if not entries:
                continue
            if is_level(composite.processes[pref.process_name], pref.field):
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

        # Validate each HandleCoeffRef resolves to a real affine mapping
        # with a fittable floor — fail early on a typo, not mid-trace — and
        # record the mapping's own value as the optimizer's start.
        coeff_baseline: dict = {}
        for cname, cref in coeff_params.items():
            handle = reg.get(cref.handle)
            if handle is None:
                raise KeyError(
                    f"params[{cname!r}].handle={cref.handle!r} not in "
                    f"registry (have {sorted(reg)})"
                )
            hits = [
                m for m in handle.mappings if m.param_name == cref.param_name
            ]
            if not hits:
                raise KeyError(
                    f"params[{cname!r}] targets {cref.handle!r}."
                    f"{cref.param_name!r}, which no mapping in that handle "
                    "declares."
                )
            if cref.coeff not in ("floor", "slope") or any(
                getattr(m, cref.coeff) is None for m in hits
            ):
                raise ValueError(
                    f"params[{cname!r}] coeff={cref.coeff!r} is not a "
                    f"fittable affine floor/slope on {cref.handle!r}."
                    f"{cref.param_name!r}."
                )
            # `_registry` overrides every hit, so they must already agree —
            # otherwise the start value depends on which one we happen to read.
            values = {float(getattr(m, f"{cref.coeff}_value")) for m in hits}
            if len(values) > 1:
                raise ValueError(
                    f"params[{cname!r}] matches {len(hits)} mappings on "
                    f"{cref.handle!r}.{cref.param_name!r} whose "
                    f"{cref.coeff} values disagree ({sorted(values)}); the "
                    "starting value would depend on which is read."
                )
            coeff_baseline[cname] = values.pop()
        if offenders:
            msgs = []
            for pname, pref, hmarks in offenders:
                msgs.append(
                    f"  params[{pname!r}] targets {pref.process_name}."
                    f"{pref.field}, a pure severity dial set by the "
                    f"{', '.join(repr(h) for h in hmarks)} handle."
                )
            raise ValueError(
                "Severity dials are not valid Calibrator inputs:\n"
                + "\n".join(msgs)
                + "\n\nThese parameters are input levels set directly by "
                "Condition.handles severity, so fitting them is degenerate "
                "— severity would overwrite the fit. Fit the mechanism "
                "magnitude the level drives (e.g. a per-exposure potency) "
                "instead."
            )

        self.composite = composite
        self.reporters = reporters
        from hallsim.reporter_wiring import validate_reporter_mappings

        self.reporter_wiring = validate_reporter_mappings(reporters, composite)
        for r in self.reporter_wiring.warnings:
            log.warning("reporter wiring: %s", r.message)
        self.conditions = conditions
        self.t_end = t_end
        # Negative = pre-roll: sources are off before 0, so query time 0 is
        # read from a settled state rather than the IC.
        self.t_start = t_start
        # Trajectory-native: each arm's Δ_data is a {timepoint: Δseries}
        # map. A plain Series is the degenerate single-timepoint case —
        # normalized to {t_end: series} so endpoint fits keep working, the
        # end being the arm's condition's own when it has a window.
        self.data = {
            arm: (
                {float(t): s for t, s in d.items()}
                if isinstance(d, dict)
                else {self._arm_window(arm)[1]: d}
            )
            for arm, d in data.items()
        }
        for arm, per_t in self.data.items():
            if arm not in self.arms:
                continue
            lo, hi = self._arm_window(arm)
            outside = [t for t in per_t if t < lo - 1e-9 or t > hi + 1e-9]
            if outside:
                log.warning(
                    "data[%r] has timepoints %s outside the condition's "
                    "window (%g, %g); the readout there is the nearest "
                    "simulated value.",
                    arm,
                    outside,
                    lo,
                    hi,
                )
        if equilibrate and equilibration_condition not in conditions:
            raise KeyError(
                "equilibrate=True needs equilibration_condition to name a "
                f"condition; {equilibration_condition!r} not in "
                f"{sorted(conditions)}"
            )
        self.equilibrate = equilibrate
        self.equilibration_condition = equilibration_condition
        self._laws = None  # conservation laws (structural; computed once)
        # `params` (declaration order) is the optimizer surface; `_params`
        # substitutes into processes, `_coeffs` overrides registry floors.
        self.params = proc_params
        self._check_param_targets(proc_params)
        self._warn_structural_redundancy(proc_params)
        self._note_stochastic_members(proc_params)
        # What the fitted fields held when the problem was wired. Every
        # evaluation overwrites them, so a later edit to one is inert; it is
        # checked against this and raised on rather than silently discarded.
        self._param_baseline = {
            pname: read_param(
                composite.processes[pref.process_name], pref.field
            )
            for pname, pref in proc_params.items()
        }
        # Set by with_overrides: fittables pinned by name, and process fields
        # pinned by (process, field). Both applied after the fitted iterate.
        self._override_params: dict = {}
        self._override_fields: dict = {}
        self._coeffs = coeff_params
        self._coeff_baseline = coeff_baseline
        # A learned block: its trainable leaves as one flat vector, the map
        # back, and the leaves that stay fixed.
        self._learned: dict[str, tuple] = {}
        self._learned_baseline: dict[str, jnp.ndarray] = {}
        for lname, lref in params.items():
            if not isinstance(lref, LearnedRef):
                continue
            if lref.process_name not in composite.processes:
                raise KeyError(
                    f"params[{lname!r}].process_name={lref.process_name!r} "
                    f"not in composite.processes "
                    f"(have {sorted(composite.processes)})"
                )
            flat, unravel, fixed = _learned_partition(
                composite.processes[lref.process_name], lref.frozen
            )
            if flat.size == 0:
                raise ValueError(
                    f"params[{lname!r}]: {lref.process_name!r} has no "
                    "trainable array leaf outside its frozen fields"
                )
            self._learned[lname] = (lref.process_name, unravel, fixed)
            self._learned_baseline[lname] = flat
        self._all_refs = params
        self._base_registry = reg
        self.prior_weight = prior_weight
        self.fit_arms = fit_arms
        self.held_out_arms = held_out_arms or []
        self.macro_dt = macro_dt
        self.n_save = n_save
        # Stiffness routing is a Scheduler default, and `warm_up` below
        # resolves it eagerly — explicit-solver sensitivities NaN on stiff
        # groups (p53–Mdm2, NF-κB) though the primal stays finite.
        self.scheduler_kwargs = scheduler_kwargs or {}

        # Precompute store-path → trailing-axis index for fast lookup.
        self._store_idx = {k: i for i, k in enumerate(composite.store_keys())}
        # Precompute reporter target indices.
        self._reporter_indices = tuple(
            self._store_idx[r.observable] for r in reporters
        )
        self.collocation = collocation
        self.member_batch = member_batch
        if collocation is not None:
            missing = [
                p for p in collocation.paths if p not in self._store_idx
            ]
            if missing:
                raise KeyError(
                    f"collocation.paths {missing} are not store paths of the "
                    "composite"
                )
            if (
                collocation.condition is not None
                and collocation.condition not in conditions
            ):
                raise KeyError(
                    f"collocation.condition={collocation.condition!r} not in "
                    f"conditions {sorted(conditions)}"
                )
            c_ts = np.asarray(collocation.ts, dtype=float)
            c_ys = np.asarray(collocation.ys, dtype=float)
            if c_ys.ndim == 2:
                c_ys = c_ys[None]
            if c_ys.ndim != 3 or c_ys.shape[2] != len(collocation.paths):
                raise ValueError(
                    "collocation.ys must be (n_traj, n_t, len(paths)); got "
                    f"shape {c_ys.shape} for {len(collocation.paths)} paths"
                )
            if c_ts.shape != (c_ys.shape[1],):
                raise ValueError(
                    f"collocation.ts has {c_ts.shape}, ys has "
                    f"{c_ys.shape[1]} samples"
                )
            matched = (
                tuple(collocation.paths)
                if collocation.matched is None
                else tuple(collocation.matched)
            )
            unknown = [p for p in matched if p not in collocation.paths]
            if unknown:
                raise KeyError(
                    f"collocation.matched {unknown} are not among "
                    "collocation.paths"
                )
            cols = [list(collocation.paths).index(p) for p in matched]
            slopes = np.gradient(c_ys, c_ts, axis=1)[..., cols]
            n_traj, n_t, n_paths = c_ys.shape
            self._colloc_idx = jnp.asarray(
                [self._store_idx[p] for p in collocation.paths]
            )
            self._colloc_out = jnp.asarray(
                [self._store_idx[p] for p in matched]
            )
            self._colloc_times = jnp.asarray(np.tile(c_ts, n_traj))
            self._colloc_states = jnp.asarray(
                c_ys.reshape(n_traj * n_t, n_paths)
            )
            flat_slopes = slopes.reshape(n_traj * n_t, len(cols))
            self._colloc_slopes = jnp.asarray(flat_slopes)
            self._colloc_scale = jnp.asarray(flat_slopes.std(axis=0) + 1e-8)
        # Per-arm query-time and Δ_data matrices, precomputed once. The
        # timepoint axis is vectorized, not looped: 2 or 200 timepoints trace
        # to the same graph size. A Python loop would unroll under JIT into
        # O(n_reporters × n_timepoints) nodes.
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
            n_batch = (
                self._condition_batch(
                    self.conditions[self.arms[arm].condition]
                )
                if arm in self.arms
                else 0
            )
            # (n_reporters, n_timepoints[, n_batch]); NaN marks a gene
            # absent at a time.
            self._arm_data_matrix[arm] = self._observed_block(
                arm, per_t, times, n_batch, default=float("nan")
            )
            self._arm_weight_matrix[arm] = self._observed_block(
                arm, weights.get(arm, {}), times, n_batch, default=1.0
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
        try:
            self.check_hill_gates()
        except ValueError:
            raise  # a blocking verdict, not a failure to check
        except Exception as exc:
            log.debug("hill-gate check skipped: %s", exc)

    # ── Internal: per-condition simulation ────────────────────────

    @property
    def param_refs(self) -> dict:
        """All fittable references — process params (:class:`ParameterRef`)
        plus hallmark coefficients (:class:`HandleCoeffRef`) — in
        declaration order. The optimizer's full surface; iterate this, not
        ``params`` (process-only), when you need every fitted quantity."""
        return self._all_refs

    def initial_params(self) -> dict:
        """``{name: value}`` for every fittable reference — the optimizer's
        starting vector, exactly what :meth:`fit` packs internally.

        Read from the model, never declared: a :class:`ParameterRef` starts at
        the composite's value for its field, a :class:`HandleCoeffRef` at its
        mapping's coefficient. There is nowhere to write a starting value that
        the model would then contradict.
        """
        return {
            k: jnp.asarray(
                self._param_baseline[k]
                if k in self._param_baseline
                else (
                    self._learned_baseline[k]
                    if k in self._learned_baseline
                    else self._coeff_baseline[k]
                )
            )
            for k in self._all_refs
        }

    @property
    def scalar_refs(self) -> dict:
        """The fittables that are single positive constants, every
        :class:`ParameterRef` and :class:`HandleCoeffRef`: what the
        identifiability report and the log-space transform cover."""
        return {
            k: v
            for k, v in self._all_refs.items()
            if not isinstance(v, LearnedRef)
        }

    @property
    def registry(self) -> dict:
        """The ``{name: Handle}`` this problem's conditions are applied with."""
        return self._base_registry

    def describe(self) -> dict:
        """Everything that defines this problem, as plain JSON: the composite
        and its processes, the reporters, the conditions and arms, the data
        being fitted, every fittable with its prior and clamp, the loss and
        solver settings, the hallmark mappings the conditions exercise, the
        library versions, and any ``notes`` the caller attached (a dataset
        accession, a declared scalar). Written beside every run's summary
        so the numbers in a folder can be traced to what produced them."""
        import importlib.metadata as md
        import subprocess

        kw = self._ctor_kwargs
        comp = self.composite
        registry = kw.get("registry") or {}
        used = {h for c in kw["conditions"].values() for h in c.handles}
        handles = {
            name: [
                _jsonable(m) for m in getattr(registry[name], "mappings", ())
            ]
            for name in sorted(used)
            if name in registry
        }
        params = {}
        for name, ref in self._all_refs.items():
            entry = _jsonable(ref)
            try:
                entry["initial"] = float(self.initial_params()[name])
            except Exception:  # noqa: BLE001 - a traced or absent start
                entry["initial"] = None
            params[name] = entry
        versions = {}
        for dist in ("jax", "diffrax", "equinox", "optax", "hallsim"):
            try:
                versions[dist] = md.version(dist)
            except md.PackageNotFoundError:
                versions[dist] = None
        try:
            root = Path(__file__).resolve().parents[2]
            versions["hallsim_commit"] = (
                subprocess.run(
                    ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                ).stdout.strip()
                or None
            )
        except Exception:  # noqa: BLE001 - no git, no commit
            versions["hallsim_commit"] = None
        return _jsonable(
            {
                "composite": {
                    "processes": {
                        n: type(pr).__name__
                        for n, pr in comp.processes.items()
                    },
                    "imported": {
                        n: pr.provenance()
                        for n, pr in comp.processes.items()
                        if hasattr(pr, "provenance")
                    },
                    "fingerprint": comp.structural_fingerprint(),
                },
                "reporters": [
                    {
                        "gene_symbol": r.gene_symbol,
                        "observable": r.observable,
                        "sign": r.sign,
                        "summary": getattr(
                            r.summary, "__name__", type(r.summary).__name__
                        ),
                        "reference": r.reference,
                    }
                    for r in kw["reporters"]
                ],
                "conditions": {
                    n: {
                        "handles": dict(c.handles),
                        "interventions": c.interventions,
                        "description": c.description,
                        "start": c.start,
                        "window": c.window,
                    }
                    for n, c in kw["conditions"].items()
                },
                "arms": {
                    name: {"condition": a.condition, "reference": a.reference}
                    for name, a in self.arms.items()
                },
                "member_batch": self.member_batch,
                "collocation": (
                    None
                    if self.collocation is None
                    else {
                        "paths": list(self.collocation.paths),
                        "condition": self.collocation.condition,
                        "weight": self.collocation.weight,
                        "n_samples": int(self._colloc_states.shape[0]),
                    }
                ),
                "fit_arms": kw["fit_arms"],
                "held_out_arms": kw.get("held_out_arms") or [],
                "data": kw["data"],
                "params": params,
                "loss": {
                    "likelihood": getattr(
                        kw.get("likelihood"), "__name__", None
                    )
                    or "gaussian_nll",
                    "weights": kw.get("weights"),
                    "prior_weight": kw["prior_weight"],
                    "equilibrate": kw["equilibrate"],
                    "equilibration_condition": kw.get(
                        "equilibration_condition"
                    ),
                    "t_start": kw["t_start"],
                    "t_end": kw["t_end"],
                    "macro_dt": kw["macro_dt"],
                    "n_save": kw["n_save"],
                },
                "scheduler_kwargs": kw.get("scheduler_kwargs") or {},
                "handles": handles,
                # what the reporter guard said about each mapping: a verdict
                # that scrolled past in a log is not a verdict anyone read
                "reporter_wiring": [
                    {
                        "level": str(getattr(r, "level", "")),
                        "verdict": getattr(r, "category", ""),
                        "message": getattr(r, "message", ""),
                    }
                    for r in getattr(
                        getattr(self, "reporter_wiring", None), "results", []
                    )
                ],
                "versions": versions,
                "notes": kw.get("notes") or {},
            }
        )

    def with_params(self, params: dict) -> "CalibrationProblem":
        """The same problem over a different fitted set — same composite,
        reporters, arms, data and settings — for a screen that asks which
        of a wider pool the data can fit, or a fit that then fits them."""
        return CalibrationProblem(**{**self._ctor_kwargs, "params": params})

    def with_overrides(self, overrides: dict) -> "CalibrationProblem":
        """A copy of this problem with parameters pinned to given values.

        The one route for changing a parameter for a run — ablations included —
        whether or not that parameter is fitted, so nobody has to know which it
        is. A key names either a fittable (whatever ``params`` calls it) or a
        process field in dotted form::

            problem.with_overrides({"mtor_to_nfkb": 0.0})    # a fittable
            problem.with_overrides({"mtor_nfkb.k_act": 0.0})  # the same field
            problem.with_overrides({"dp14.parameters.k": 1.0})

        An override is applied last, so it wins over the fitted iterate and
        over the composite's own value alike. Calls compose: each adds to the
        overrides already set. Under :meth:`fit` an override holds its
        parameter fixed and the optimizer sees a zero gradient for it.
        """
        pinned = dict(self._override_params)
        fields = dict(self._override_fields)
        for key, value in overrides.items():
            value = jnp.asarray(value)
            if key in self._all_refs:
                pinned[key] = value
                continue
            try:
                proc_name, field_path = split_param_address(
                    key, self.composite.processes
                )
            except (KeyError, ValueError):
                raise KeyError(
                    f"{key!r} is neither a fittable of this problem "
                    f"({sorted(self._all_refs)}) nor a '<process>.<field>' "
                    f"address into it (processes: "
                    f"{sorted(self.composite.processes)})."
                ) from None
            proc = self.composite.processes[proc_name]
            try:
                read_param(proc, field_path)
            except (AttributeError, KeyError, TypeError) as exc:
                raise KeyError(
                    f"{key!r} names no field on process {proc_name!r} "
                    f"({type(proc).__name__})."
                ) from exc
            fields[(proc_name, field_path)] = value
        clone = copy.copy(self)
        clone._override_params = pinned
        clone._override_fields = fields
        return clone

    def _pinned(self, param_values: dict) -> dict:
        """``param_values`` with every overridden fittable replaced, so the
        fitted-substitution path routes an override to the right home —
        a process field or a hallmark coefficient — without knowing which."""
        if not self._override_params:
            return param_values
        return {**param_values, **self._override_params}

    def processes_at(self, param_values: dict) -> dict:
        """The composite's processes with ``param_values`` written in: every
        fitted scalar and learned block at the given values, plus any
        override. What a fitted block is read back from."""
        return self._substitute(self.composite.processes, param_values)

    def _substitute(self, processes: dict, param_values: dict) -> dict:
        param_values = self._pinned(param_values)
        new = dict(processes)
        for pname, pref in self.params.items():
            _reject_overwritten_edit(
                pname,
                pref,
                new[pref.process_name],
                self._param_baseline[pname],
            )
            new[pref.process_name] = _substitute_param(
                new[pref.process_name],
                pref.field,
                param_values[pname],
            )
        for lname, (proc_name, unravel, fixed) in self._learned.items():
            new[proc_name] = eqx.combine(unravel(param_values[lname]), fixed)
        # Last, so an override outranks the fitted iterate whichever way the
        # caller spelled it.
        for (proc_name, path), value in self._override_fields.items():
            new[proc_name] = _substitute_param(new[proc_name], path, value)
        return new

    def _registry(self, param_values: dict):
        """The hallmark registry for this evaluation, with each fitted affine
        coefficient (:class:`HandleCoeffRef`, ``floor`` or ``slope``)
        overridden by its current value from ``param_values``. Returns the base
        registry unchanged when no coefficients are fitted, so the affine
        coefficients stay at their ``init``."""
        if not self._coeffs:
            return self._base_registry
        param_values = self._pinned(param_values)
        overrides: dict[str, dict[str, dict[str, Any]]] = {}
        for name, cref in self._coeffs.items():
            overrides.setdefault(cref.handle, {}).setdefault(
                cref.param_name, {}
            )[cref.coeff] = param_values[name]
        reg = dict(self._base_registry)
        for hname, by_param in overrides.items():
            handle = reg[hname]
            reg[hname] = dc_replace(
                handle,
                mappings=[
                    (
                        dc_replace(m, **by_param[m.param_name])
                        if m.param_name in by_param
                        else m
                    )
                    for m in handle.mappings
                ],
            )
        return reg

    def _condition_window(self, condition: Condition) -> tuple[float, float]:
        """The condition's own ``(t_start, t_end)``, else the problem's."""
        if condition.window is None:
            return float(self.t_start), float(self.t_end)
        lo, hi = condition.window
        return float(lo), float(hi)

    def _arm_window(self, arm: str) -> tuple[float, float]:
        return self._condition_window(
            self.conditions[self.arms[arm].condition]
        )

    @staticmethod
    def _condition_batch(condition: Condition) -> int:
        """Members in the condition's ``start``, ``0`` when unbatched."""
        for value in (condition.start or {}).values():
            if np.ndim(value) > 0:
                return int(np.shape(value)[0])
        return 0

    def _start_state(
        self, condition: Condition, y0: jnp.ndarray, members=None
    ):
        """``y0`` with the condition's ``start`` written over it, gaining a
        leading batch axis when the start is batched; ``members`` (from
        :meth:`_member_draws`) selects which members run."""
        if not condition.start:
            return y0
        n_batch = self._condition_batch(condition)
        idx = None if not n_batch or not members else members.get(n_batch)
        if n_batch:
            rows = n_batch if idx is None else idx.shape[0]
            y0 = jnp.broadcast_to(y0, (rows,) + tuple(y0.shape))
        for path, value in condition.start.items():
            value = jnp.asarray(value, dtype=y0.dtype)
            if idx is not None and value.ndim > 0:
                value = jnp.take(value, idx, axis=0)
            y0 = y0.at[..., self._store_idx[path]].set(value)
        return y0

    def _member_draws(self, key) -> dict[int, jnp.ndarray] | None:
        """One draw of ``member_batch`` member indices per distinct member
        count among the conditions, so windows cut from the same
        trajectories select the same ones; ``None`` when not minibatching."""
        if self.member_batch is None or key is None:
            return None
        counts = sorted(
            {self._condition_batch(c) for c in self.conditions.values()} - {0}
        )
        draws = {}
        for n, sub in zip(counts, jax.random.split(key, max(1, len(counts)))):
            draws[n] = jax.random.randint(
                sub, (min(int(self.member_batch), n),), 0, n
            )
        return draws

    def _observed_block(
        self, arm: str, per_t: dict, times, n_batch: int, default: float
    ) -> jnp.ndarray:
        """One arm's values as ``(n_rep, n_t)``, or ``(n_rep, n_t, n_batch)``
        for a batched condition: a DataFrame's rows are the members, a Series
        is broadcast over them. ``default`` fills a reporter a timepoint does
        not carry."""
        import pandas as pd

        cols = []
        for t in times:
            entry = per_t.get(t)
            if isinstance(entry, pd.DataFrame):
                if not n_batch:
                    raise ValueError(
                        f"data[{arm!r}][{t!r}] is a frame of members, but the "
                        "arm's condition has no batched start"
                    )
                if len(entry) != n_batch:
                    raise ValueError(
                        f"data[{arm!r}][{t!r}] has {len(entry)} rows; the "
                        f"condition's start has {n_batch} members"
                    )
                col = np.asarray(
                    [
                        (
                            entry[r.gene_symbol].to_numpy(dtype=float)
                            if r.gene_symbol in entry.columns
                            else np.full(n_batch, default)
                        )
                        for r in self.reporters
                    ]
                )
            else:
                vals = np.asarray(
                    [
                        (
                            float(entry.get(r.gene_symbol, default))
                            if entry is not None
                            else default
                        )
                        for r in self.reporters
                    ]
                )
                col = (
                    np.repeat(vals[:, None], n_batch, axis=1)
                    if n_batch
                    else vals
                )
            cols.append(col)
        return jnp.asarray(np.stack(cols, axis=1), dtype=float)

    def _condition_composite(
        self, processes: dict, condition: Condition, registry=None
    ):
        """Apply a condition's hallmark severities and wire a composite."""
        from hallsim.handles import apply_handles

        procs = apply_handles(
            processes,
            condition.handles,
            registry=registry or self._base_registry,
        )
        for iv in condition.interventions:
            procs = iv.apply(procs, reference=processes)
        return self._Composite(
            processes=procs,
            topology=self.composite.topology,
            validate=False,
            semantic_validation=False,
        )

    def _equilibrate(self, processes: dict, registry=None):
        """Shared pre-perturbation baseline ``(y0, ref_readout)`` — the unperturbed
        condition's fixed point.

        A perturbation baseline is the unperturbed steady state; that condition
        sits at a fixed point (any limit cycle belongs to the perturbation), so
        it is found by Newton (:func:`hallsim.steady_state.steady_state`) rather
        than a burn-in — no horizon, no transient phase, and an exact
        implicit-function-theorem gradient in the fitted params. ``y0`` is the
        fixed point (shared t=0 for every arm; accumulators zero); ``ref_readout`` is
        each reporter's source value at ``y0`` (its homeostatic mean), the
        shared healthy day-0 the within-arm fold-change divides by.

        Returns ``(initial_state_vec, None)`` when equilibrate is off, so
        callers fall back to each arm's own t=0 baseline."""
        comp = self._condition_composite(
            processes, next(iter(self.conditions.values())), registry=registry
        )
        if not self.equilibrate:
            return comp.initial_state_vec(), None
        from hallsim.steady_state import conservation_laws, steady_state

        eq_comp = self._condition_composite(
            processes,
            self.conditions[self.equilibration_condition],
            registry=registry,
        )
        # Conservation laws are structural (param-independent) — compute once.
        if self._laws is None:
            self._laws = conservation_laws(
                eq_comp, eq_comp.initial_state_vec()
            )
        y0 = steady_state(eq_comp, laws=self._laws)
        self._require_stable_baseline(eq_comp, y0)
        # Force-linked baseline: apply the SAME reporter summaries the arms use
        # to the settled state (a constant trajectory at y0), so the reference
        # and the arm readout are the identical transform. Any summary constant
        # (e.g. a leaky reporter's √τ) then cancels in the log2 ratio by
        # construction — no "readout == source at the fixed point" assumption to
        # break, whatever the readout.
        const_ts = jnp.linspace(self.t_start, self.t_end, 16)
        obs = jnp.stack(
            [
                jnp.full((const_ts.size,), y0[self._store_idx[r.observable]])
                for r in self.reporters
            ]
        )
        ref_readout = self._reporter_summaries(
            const_ts, obs, jnp.asarray([self.t_end])
        )
        return y0, ref_readout

    def _simulate_condition(
        self,
        processes: dict,
        condition: Condition,
        y0=None,
        registry=None,
        adjoint=None,
        members=None,
    ):
        """Apply hallmarks + run Scheduler for one condition. Returns the
        full ``(ts, reporter_trajectories)`` — ``ts`` shape ``(n_save,)``
        and ``reporter_trajectories`` shape ``(n_reporters, n_save)`` — so
        the loss can read each reporter at arbitrary query times. ``y0``
        overrides the initial state (the shared equilibrated baseline).

        ``adjoint`` must match the outer autodiff: ``dfx.ForwardMode()``
        under a JVP (forward fit), ``None`` (the Scheduler default recursive-
        checkpoint reverse adjoint) under a VJP or a plain post-fit run. A
        ForwardMode solve cannot be reverse-differentiated (its inner
        while-loop has dynamic bounds)."""
        comp = self._condition_composite(
            processes, condition, registry=registry
        )
        if y0 is None:
            y0 = comp.initial_state_vec()
        y0 = self._start_state(condition, y0, members)
        t0, t1 = self._condition_window(condition)
        span = t1 - t0
        save_dt = max(1e-6, span / max(1, self.n_save - 1))
        res = self._scheduler.run(
            comp,
            t_span=(t0, t1),
            macro_dt=min(self.macro_dt, span),
            y0=y0,
            save_dt=save_dt,
            adjoint=adjoint,
        )
        # res.ys is (n_save, ..., n_vars). Trailing-axis convention, so a
        # batched condition reads as (n_reporters, n_save, n_batch).
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

        def one(rep, y):
            if jnp.ndim(y) == 2:  # (n_save, n_batch): a summary per member
                return jax.vmap(
                    lambda yy: jnp.atleast_1d(rep.summary(ts, yy, qt)),
                    in_axes=1,
                    out_axes=-1,
                )(y)
            return jnp.atleast_1d(rep.summary(ts, y, qt))

        return jnp.stack(
            [one(rep, trajs[i]) for i, rep in enumerate(self.reporters)]
        )

    def _run_condition_set(
        self, param_values: dict, *, adjoint=None, members=None
    ):
        """Set up one evaluation over the condition set.

        Substitutes the fitted params, resolves the hallmark registry, and
        equilibrates the shared baseline, then returns
        ``(run_for, y0, baseline)`` where ``run_for(cond_name) -> (ts,
        reporter_trajs)`` solves each condition once and caches it. The one
        prologue shared by :meth:`model_readout`, :meth:`data_loss`,
        :meth:`evaluate`, and :meth:`simulate_reporters`; ``adjoint``
        threads the autodiff mode through to each solve (see
        :meth:`_simulate_condition`)."""
        substituted = self._substitute(self.composite.processes, param_values)
        registry = self._registry(param_values)
        y0, baseline = self._equilibrate(substituted, registry=registry)
        cache: dict[str, tuple] = {}

        def run_for(cond_name: str):
            if cond_name not in cache:
                cache[cond_name] = self._simulate_condition(
                    substituted,
                    self.conditions[cond_name],
                    y0=y0,
                    registry=registry,
                    adjoint=adjoint,
                    members=members,
                )
            return cache[cond_name]

        return run_for, y0, baseline

    def simulate_reporters(
        self, param_values: dict, cond_name: str, query_times=None
    ):
        """Reporter trajectories for one condition at ``param_values``.

        The public post-fit / figure path — returns ``(ts, reporter_trajs)``
        (``reporter_trajs`` shape ``(n_reporters, n_save)``), or, when
        ``query_times`` is given, each reporter's summary at those times
        (``(n_reporters, n_t)``). Uses the Scheduler default adjoint (no
        forward-mode unfold), so callers don't reach into the private
        substitute/run/reporter internals to draw a trajectory."""
        run_for, _, _ = self._run_condition_set(param_values, adjoint=None)
        ts, trajs = run_for(cond_name)
        if query_times is None:
            return ts, trajs
        return self._reporter_summaries(
            ts, trajs, jnp.atleast_1d(jnp.asarray(query_times))
        )

    # ── Loss / fit / evaluate ─────────────────────────────────────

    @property
    def arm_pairs(self) -> dict[str, tuple[str, str]]:
        """``{arm: (condition, reference condition)}``, the reference being
        the condition itself when the arm reads against its own start or
        nothing. For consumers that only need each arm's condition."""
        return {
            name: (
                a.condition,
                (
                    a.reference
                    if a.reference not in (None, "t0")
                    else a.condition
                ),
            )
            for name, a in self.arms.items()
        }

    def _arm_reference(self, run_for, arm: str, qt, baseline=None):
        """``(arm_readout, ref_readout)`` for one arm: the reporter summaries
        of its condition at ``qt`` and of its reference, ``None`` when the
        arm has none. The one place the reference is resolved, shared by
        :meth:`data_loss`, :meth:`model_readout` and :meth:`evaluate`.
        ``run_for(cond_name) -> (ts, reporter_trajs)`` is a (usually caching)
        condition solver.

        A ``"t0"`` reference is the shared homeostatic value from the
        equilibration burn-in (``baseline``) when equilibrating, else the
        arm's own t=0; a condition name is that condition at the matched
        time."""
        a = self.arms[arm]
        ts_c, trajs_c = run_for(a.condition)
        arm_readout = self._reporter_summaries(
            ts_c, trajs_c, qt
        )  # (n_rep, n_t)
        if a.reference is None:
            return arm_readout, None
        if a.reference == "t0":
            cond = self.conditions[a.condition]
            if baseline is not None and not cond.start:
                return arm_readout, jnp.broadcast_to(
                    baseline, arm_readout.shape
                )
            # A condition with its own start is read against that start,
            # at its window's opening; otherwise at time 0, after any
            # pre-roll.
            t0 = self._condition_window(cond)[0] if cond.window else 0.0
            return arm_readout, self._reporter_summaries(
                ts_c, trajs_c, jnp.full_like(qt, t0)
            )
        ts_b, trajs_b = run_for(a.reference)
        return arm_readout, self._reporter_summaries(ts_b, trajs_b, qt)

    def _arm_readout(
        self, run_for, arm: str, qt, baseline=None
    ) -> jnp.ndarray:
        """One arm's model readout at ``qt`` in the data's terms: the
        sign-aligned log2 fold change against its reference, or, with no
        reference, the summary itself times each reporter's ``scale``.
        Shared by :meth:`data_loss` and :meth:`model_readout` so figures plot
        exactly what the loss fits."""
        arm_readout, ref_readout = self._arm_reference(
            run_for, arm, qt, baseline
        )
        if ref_readout is None:
            scales = jnp.asarray(
                [getattr(r, "scale", 1.0) for r in self.reporters], dtype=float
            )
            return arm_readout * scales.reshape(
                scales.shape + (1,) * (jnp.ndim(arm_readout) - 1)
            )
        return self._log2_fold_change(arm_readout, ref_readout)

    def model_readout(
        self, param_values: dict[str, jnp.ndarray], arm: str, query_times
    ) -> jnp.ndarray:
        """The model's readout for ``arm`` at ``query_times`` in the data's
        terms — the exact quantity :meth:`data_loss` fits, so a trajectory
        figure never drifts from the loss. Returns ``(n_rep, n_t)``."""
        run_for, _, baseline = self._run_condition_set(param_values)
        return self._arm_readout(
            run_for, arm, jnp.atleast_1d(jnp.asarray(query_times)), baseline
        )

    def data_loss(
        self, param_values: dict[str, jnp.ndarray], arms: list[str], key=None
    ) -> jnp.ndarray:
        """Mean per-arm MSE of the sign-aligned log2 fold-change over ``arms``.

        The pure data-fit term — no prior penalty. Factored out of
        :meth:`loss` so a held-out (validation) arm can be scored at
        arbitrary params, e.g. to trace a train-vs-validation curve for
        early stopping. ``arms`` must be keys of ``self.arms``.
        """
        # Each condition is solved once (whole trajectory) and cached, then
        # read at each arm's measured timepoints — so a condition that is
        # both a condition and a reference in different arms still runs
        # once, and every timepoint reuses the same solve.
        if not arms:
            return jnp.asarray(0.0)
        members = self._member_draws(key)
        run_for, _, baseline = self._run_condition_set(
            param_values, members=members
        )

        # One arm loss = mean squared error over the whole (reporter ×
        # timepoint) block: the model fits the *trajectory* of the log2
        # fold-change, not just the endpoint. The timepoint axis is
        # vectorized (precomputed static arrays + one interp per reporter),
        # so this scales to many timepoints without unrolling. A
        # single-timepoint arm is the degenerate n_t=1 case.
        arm_losses = []
        for arm in arms:
            qt = self._arm_query_times[arm]  # (n_t,)
            lfc_sim = self._arm_readout(run_for, arm, qt, baseline)
            # With a reference the loss compares two log2 ratios, so the data
            # must be supplied as a log2 fold change (the microarray demo
            # already is; count data is log-normalized upstream): every
            # reporter then weighs by its O(1) log ratio regardless of the
            # observable's absolute scale. Without one it compares values in
            # the data's units.
            delta_data = self._arm_data_matrix[arm]  # (n_rep, n_t[, n_b])
            weight = self._arm_weight_matrix[arm]
            if members and jnp.ndim(delta_data) == 3:
                idx = members.get(delta_data.shape[-1])
                if idx is not None:
                    delta_data = jnp.take(delta_data, idx, axis=-1)
                    weight = jnp.take(weight, idx, axis=-1)
            arm_losses.append(self.likelihood(lfc_sim, delta_data, weight))
        return jnp.mean(jnp.stack(arm_losses))

    def collocation_loss(
        self, param_values: dict[str, jnp.ndarray], key=None
    ) -> jnp.ndarray:
        """Mean squared residual of the composite's field at the observed
        states against their finite-difference slopes, on the collocation's
        matched paths, each scaled by that slope's spread. No solve, so it is
        cheap and free of phase drift: a pretraining stage on its own, or the
        physics term :meth:`loss` adds at the collocation's ``weight``.
        ``key`` draws the collocation's ``batch`` of samples; without one
        every sample is used."""
        c = self.collocation
        if c is None:
            raise ValueError("the problem was built without a collocation")
        substituted = self._substitute(self.composite.processes, param_values)
        registry = self._registry(param_values)
        y0, _ = self._equilibrate(substituted, registry=registry)
        cond = self.conditions[
            (
                c.condition
                if c.condition is not None
                else next(iter(self.conditions))
            )
        ]
        comp = self._condition_composite(substituted, cond, registry=registry)
        rhs, _ = comp.build_rhs()
        times, obs, slopes = (
            self._colloc_times,
            self._colloc_states,
            self._colloc_slopes,
        )
        if c.batch is not None and key is not None:
            # With replacement: a draw without one permutes every sample
            # (140 ms at 4e5 samples) for a duplicate rate of batch/n.
            idx = jax.random.randint(
                key, (min(int(c.batch), obs.shape[0]),), 0, obs.shape[0]
            )
            times, obs, slopes = times[idx], obs[idx], slopes[idx]
        n = obs.shape[0]
        states = jnp.broadcast_to(y0, (n,) + tuple(y0.shape))
        states = states.at[:, self._colloc_idx].set(obs)
        field = jax.vmap(rhs)(times, states)[:, self._colloc_out]
        resid = (field - slopes) / self._colloc_scale
        return jnp.mean(resid**2)

    def loss(
        self, param_values: dict[str, jnp.ndarray], key=None
    ) -> jnp.ndarray:
        """The objective: the data term over the fit arms, the prior
        penalty, and the collocation term at its weight. ``key`` reaches the
        member and collocation minibatch draws."""
        k_data = k_colloc = None
        if key is not None:
            k_data, k_colloc = jax.random.split(key)
        total = self.data_loss(
            param_values, self.fit_arms, k_data
        ) + self._prior_penalty(param_values)
        if self.collocation is not None and self.collocation.weight:
            total = total + self.collocation.weight * self.collocation_loss(
                param_values, k_colloc
            )
        return total

    def prior_report(self, fisher_diag: dict | None = None) -> list[dict]:
        """How much of each parameter's posterior precision its prior supplies.

        A prior's penalty is ``prior_weight·((log10 p − log10 prior)/sigma)²``,
        so its precision in log10 space is ``2·prior_weight/sigma²`` — and
        whether that constrains anything is only answerable against the
        precision the *data* carries, ``diag(JᵀJ)``. Pass ``fisher_diag`` from
        an :class:`~hallsim.identifiability.IdentifiabilityReport` to get the
        ``share`` each prior holds; without it the precision is still reported
        and ``share``/``operative`` are ``None``.

        Entries carry ``name``, ``prior``, ``prior_sigma``, ``precision``,
        ``fisher``, ``share``, ``operative``.
        """
        out = []
        for name, pref in self._all_refs.items():
            if pref.prior is None:
                continue
            precision = 2.0 * self.prior_weight / float(pref.prior_sigma) ** 2
            fisher = None if fisher_diag is None else fisher_diag.get(name)
            share = (
                None
                if fisher is None
                else precision / (precision + max(fisher, 0.0))
            )
            out.append(
                {
                    "name": name,
                    "prior": float(pref.prior),
                    "prior_sigma": float(pref.prior_sigma),
                    "precision": precision,
                    "fisher": fisher,
                    "share": share,
                    "operative": (
                        None if share is None else share >= _MIN_PRIOR_SHARE
                    ),
                }
            )
        return out

    @staticmethod
    def _require_stable_baseline(comp, y0, tol: float = 1e-6) -> None:
        """Refuse an unstable fixed point as a baseline.

        An autonomous oscillator's Newton solution is the unstable point at the
        centre of its limit cycle; integrating from there diverges, which
        surfaces later as a solver failure rather than a statement about the
        model.
        """
        if isinstance(y0, jax.core.Tracer):
            return
        rhs, _ = comp.build_rhs()
        jac = np.asarray(jax.jacfwd(lambda y: rhs(0.0, y))(y0))
        worst = float(np.max(np.real(np.linalg.eigvals(jac))))
        if worst <= tol:
            return
        raise ValueError(
            f"equilibrate=True found an unstable fixed point (largest "
            f"Re λ = {worst:.3g} > 0), so it is not a baseline: a forward "
            f"solve from it diverges. A composite containing an autonomous "
            f"oscillator has no stable whole-system rest state — equilibrate "
            f"the non-oscillatory part and hold the oscillator at its "
            f"published initial condition, or run with equilibrate=False."
        )

    def _note_stochastic_members(self, refs) -> None:
        """A stochastic member is a sampled forcing in the gradient: its
        jump process has no tangent, so a parameter whose effect reaches the
        reporters only through it gets a zero gradient and shows as
        *structural* in the post-fit identifiability report."""
        members = list(self.composite.stochastic_processes())
        if not members:
            return
        on_member = sorted(
            name for name, r in refs.items() if r.process_name in members
        )
        log.warning(
            "calibration: %s run at reaction level; gradients do not pass "
            "through them, so a fitted parameter reaching the reporters only "
            "through one of them cannot move%s. Fit such parameters on the "
            "mean field.",
            members,
            f" ({on_member} live on one)" if on_member else "",
        )

    def _warn_structural_redundancy(self, refs) -> None:
        """Warn when fitted parameters enter the dynamics only as one
        combination — visible from the declared symbolic forms before any
        data, so it is said here rather than after the fit."""
        if len(refs) < 2:
            return
        from hallsim.identifiability import structural_redundancy

        by_address = {
            f"{r.process_name}.{r.field}": n for n, r in refs.items()
        }
        report = structural_redundancy(self.composite, params=by_address)
        for group in report.groups:
            fitted = [
                by_address[p] for p in group.parameters if p in by_address
            ]
            if len(fitted) > 1:
                log.warning(
                    "structural redundancy: fitted parameters %s enter the "
                    "dynamics only as one combination (%s), so no data can "
                    "separate them and the fit will split it arbitrarily. Fit "
                    "one and fix the rest, or fit the combination.",
                    fitted,
                    group.describe(),
                )

    def _check_param_targets(self, refs) -> None:
        """Reject a ParameterRef whose field cannot carry a fitted scalar."""
        for name, pref in refs.items():
            process_name = pref.process_name
            proc = self.composite.processes.get(process_name)
            if proc is None:
                raise KeyError(
                    f"params[{name!r}] targets process {process_name!r}, "
                    f"which is not in the composite. Available: "
                    f"{sorted(self.composite.processes)}"
                )
            try:
                value = read_param(proc, pref.field)
            except (AttributeError, KeyError) as exc:
                raise AttributeError(
                    f"params[{name!r}] targets "
                    f"{process_name}.{pref.field}, which does not exist "
                    f"({exc})."
                ) from None
            if jnp.asarray(value).ndim:
                raise TypeError(
                    f"params[{name!r}] targets "
                    f"{process_name}.{pref.field}, whose value is "
                    f"{value!r} — shape {jnp.asarray(value).shape}, not a "
                    f"scalar. Substitution writes one number, so a "
                    f"tuple/array field would be overwritten with a scalar "
                    f"and fail inside the process. Fit a scalar field, or "
                    f"split this one."
                )

    def _require_identifiable(self) -> None:
        """Refuse to fit a problem whose Fisher information is singular.

        Also reports each prior against that same Fisher information, which is
        the only thing that says whether a prior is doing anything.
        """
        from hallsim.identifiability import identifiability_report

        if not self.scalar_refs:
            return
        report = identifiability_report(self)
        self._warn_inoperative_priors(report.fisher_diag)
        cond = report.condition_number
        if cond <= MAX_FIT_CONDITION_NUMBER:
            return
        raise ValueError(
            f"Fisher condition number {cond:.2e} exceeds "
            f"{MAX_FIT_CONDITION_NUMBER:.0e}: some combination of these "
            f"parameters changes no reporter, so the fit has no unique "
            f"answer and the iterate will wander along the flat directions. "
            f"Freeze {report.recommended_freeze or '[see the report]'} and "
            f"refit, reparameterize the confounded pairs "
            f"({len(report.confounded)} found) to the combination the data "
            f"sees, or add a condition that separates them. Pass "
            f"allow_unidentifiable=True to fit anyway.\n\n{report}"
        )

    def _warn_inoperative_priors(self, fisher_diag: dict) -> None:
        """Name any prior the data overwhelms, and any that overwhelms it."""
        for r in self.prior_report(fisher_diag):
            if r["operative"] is None:
                continue
            if not r["operative"]:
                log.warning(
                    "prior on %r supplies %.2g%% of its posterior precision, "
                    "so the fit is not seeing it: prior %.3g vs data %.3g. "
                    "prior_sigma is in log10 decades (0.5 ~ a factor of 3); a "
                    "sigma set in the parameter's own units is the usual "
                    "cause.",
                    r["name"],
                    100 * r["share"],
                    r["precision"],
                    r["fisher"],
                )
            elif r["share"] >= _MAX_PRIOR_SHARE:
                log.warning(
                    "prior on %r supplies %.2g%% of its posterior precision, "
                    "so the fitted value is the prior and not a measurement: "
                    "prior %.3g vs data %.3g.",
                    r["name"],
                    100 * r["share"],
                    r["precision"],
                    r["fisher"],
                )

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
        for name, pref in self._all_refs.items():
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
        registry = self._registry(param_values)
        # The verdict is cached per span and macro step, so every distinct
        # window among the conditions is resolved once.
        seen: set = set()
        for cond in self.conditions.values():
            t0, t1 = self._condition_window(cond)
            macro = min(self.macro_dt, t1 - t0)
            if (t0, t1, macro) in seen:
                continue
            seen.add((t0, t1, macro))
            comp = self._condition_composite(
                substituted, cond, registry=registry
            )
            self._scheduler.warm_up(comp, (t0, t1), macro_dt=macro)
        # Conservation laws (structural) are computed eagerly here — they can't
        # be recovered from tracers once the loss is under autodiff.
        if self.equilibrate and self._laws is None:
            self._equilibrate(substituted, registry=registry)
        self._warmed_up = True

    def fit(
        self,
        *,
        steps: int = 40,
        mode: str = "forward",
        validation_arms: list[str] | None = None,
        identifiability: bool = True,
        allow_unidentifiable: bool = False,
        **calibrator_kwargs,
    ) -> CalibrationHistory:
        """Fit the mechanism parameters.

        ``mode`` selects the autodiff direction. ``"forward"`` JVPs along
        each parameter basis (cost scales with parameter count) and is
        robust through the multi-rate macro loop. ``"reverse"`` is a
        single VJP (cost independent of parameter count, the way neural
        networks train) — much cheaper for several parameters; the
        phase-insensitive ``window_mean`` summaries keep the reverse
        pass through the oscillators well-behaved.

        ``identifiability`` (default on) runs a post-fit Fisher-information
        identifiability check at the optimum, logs a one-line summary
        (escalating to a warning if any parameter moves no reporter), and
        attaches it to ``history.identifiability``. Set ``False`` to skip it
        in inner-loop / toy fits. See :mod:`hallsim.identifiability`.

        The same check runs *before* the fit and raises if the Fisher
        condition number exceeds ``MAX_FIT_CONDITION_NUMBER``: descending a
        singular problem moves parameters along directions the data cannot
        see, so the iterate is arbitrary and the loss may rise. Freeze what
        the report recommends, or pass ``allow_unidentifiable=True``.
        """
        if not allow_unidentifiable:
            self._require_identifiable()
        if self._learned and mode != "reverse":
            log.info(
                "mode=%r replaced by 'reverse': a learned block's gradient "
                "is one VJP, where forward mode costs a solve per weight.",
                mode,
            )
            mode = "reverse"
        init = self.initial_params()
        clamps = {
            k: p.clamp
            for k, p in self._all_refs.items()
            if p.clamp is not None
        }
        # Measure stiffness with concrete params before the loss goes
        # under autodiff — the per-group solver verdict can't be computed
        # from tracers.
        self.warm_up(init)
        for arm in validation_arms or []:
            if arm not in self.arms:
                raise KeyError(f"validation_arms entry {arm!r} not in arms")
        val_loss_fn = (
            (lambda p: self.data_loss(p, validation_arms))
            if validation_arms
            else None
        )
        cal = Calibrator(
            loss_fn=self.loss,
            init_params=init,
            # Every scalar fittable is a positive constant; a learned block
            # stays in linear space.
            log_params=[k for k in init if k not in self._learned],
            clamps=clamps or None,
            val_loss_fn=val_loss_fn,
            mode=mode,
            **calibrator_kwargs,
        )
        history = cal.fit(steps=steps)
        history.settings = _jsonable(
            {
                "steps": steps,
                "mode": mode,
                "method": cal.method,
                "learning_rate": cal.learning_rate,
                "early_stop_patience": cal.early_stop_patience,
                "early_stop_tol": cal.early_stop_tol,
                "validation_arms": list(validation_arms or []),
                "identifiability": identifiability,
                "allow_unidentifiable": allow_unidentifiable,
                "log_params": [k for k in init if k not in self._learned],
                "clamps": clamps,
                **calibrator_kwargs,
            }
        )
        if identifiability and self.scalar_refs:
            # Post-fit local identifiability at the optimum — a warn-by-default
            # diagnostic (like the composite's validation layer), never blocks.
            # Lazy import: identifiability imports from this module.
            from hallsim.identifiability import (
                identifiability_report,
                log_summary,
            )

            try:
                report = identifiability_report(
                    self, history.best_params or init
                )
                history.identifiability = report
                log_summary(report, log)
            except Exception as exc:  # noqa: BLE001
                log.warning("identifiability analysis skipped: %s", exc)
        return history

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
        import pandas as pd

        from hallsim.gene_reporters import compute_concordance

        # Default adjoint (no forward-mode unfold): evaluate is not
        # differentiated, so it runs faster than the loss path.
        run_for, _, baseline = self._run_condition_set(
            param_values, adjoint=None
        )

        eps = 1e-12
        results: dict[str, dict[float, Any]] = {}
        all_arms = list(self.fit_arms) + list(self.held_out_arms)
        for arm in all_arms:
            times = self._arm_times[arm]
            qt = self._arm_query_times[arm]
            # Vectorized read: (n_rep, n_t) in one interp per condition, then
            # slice per timepoint. Same reference as the loss.
            arm_readout, ref_readout = self._arm_reference(
                run_for, arm, qt, baseline
            )
            if ref_readout is None:
                lfc = self._arm_readout(run_for, arm, qt, baseline)
            else:
                lfc = jnp.log2(jnp.maximum(arm_readout, eps)) - jnp.log2(
                    jnp.maximum(ref_readout, eps)
                )  # (n_rep, n_t), unsigned; compute_concordance applies sign
            if jnp.ndim(lfc) == 3:
                # A batched condition: the concordance table reads the
                # member mean, model and data alike.
                lfc = jnp.mean(lfc, axis=-1)
            per_t: dict[float, Any] = {}
            for j, t in enumerate(times):
                delta_sim_named = {
                    r.observable: float(lfc[i, j])
                    for i, r in enumerate(self.reporters)
                }
                observed = self.data[arm][t]
                if isinstance(observed, pd.DataFrame):
                    observed = observed.mean(axis=0)
                per_t[float(t)] = compute_concordance(
                    delta_observables=delta_sim_named,
                    delta_gene_expression=observed,
                    condition_name=f"{arm}@t{float(t):g}",
                    reporters=self.reporters,
                )
            results[arm] = per_t
        return results

    # ── Output bundle: trajectories + topology + concordance JSON ──

    def simulate_all_conditions(
        self,
        param_values: dict,
        n_save: int | None = None,
        antialias: bool = True,
    ) -> dict:
        """Run each condition once at ``param_values``; return
        ``{cond_name: SchedulerResult}`` (full state, all species). Uses the
        Scheduler's default adjoint (no forward-mode JVP), so wall-time is
        fast — the public path for post-fit visualisation, not loss
        evaluation. For reporter-only trajectories use
        :meth:`simulate_reporters`."""
        substituted = self._substitute(self.composite.processes, param_values)
        registry = self._registry(param_values)
        y0, _ = self._equilibrate(substituted, registry=registry)
        n = n_save if n_save is not None else self.n_save
        results: dict = {}
        for cond_name, cond in self.conditions.items():
            comp = self._condition_composite(
                substituted, cond, registry=registry
            )
            t0, t1 = self._condition_window(cond)
            results[cond_name] = self._scheduler.run(
                comp,
                t_span=(t0, t1),
                macro_dt=min(self.macro_dt, t1 - t0),
                y0=self._start_state(cond, y0),
                save_dt=max(1e-6, (t1 - t0) / max(1, n - 1)),
                antialias=antialias,
            )
        return results

    def operating_ranges(
        self,
        param_values: dict,
        paths: list[str],
        *,
        n_save: int = 200,
    ) -> dict:
        """Per-condition :class:`OperatingRange` (min/mean/max) of each store
        path, over the *faithful* calibration simulation (equilibrated
        baseline, per-arm interventions) — the band each signal spans across
        conditions. Use it to place a coupling edge's Hill threshold at the
        operating point separating the conditions, rather than guessing.

        Returns ``{cond_name: {path: OperatingRange}}``. A raw composite run
        from the initial state (unequilibrated, no dosing) would report a
        different band — hence this routes through the same simulation the
        loss uses."""
        import numpy as np

        # Operating ranges read slow states (min/mean/max); no reason to pay the
        # oscillator-grade fine grid the auto-reducer would impose.
        sims = self.simulate_all_conditions(
            param_values, n_save=n_save, antialias=False
        )
        out: dict = {}
        for cond, res in sims.items():
            row: dict = {}
            for p in paths:
                v = res.get(p)
                if v is not None:
                    v = np.asarray(v)
                    row[p] = OperatingRange(
                        float(v.min()), float(v.mean()), float(v.max())
                    )
            out[cond] = row
        return out

    def check_hill_gates(
        self,
        param_values: dict | None = None,
        *,
        allow_dead_edges: bool = False,
    ) -> list[dict]:
        """Warn for every Hill gate whose ``K`` lies outside the range its
        driver actually reaches across the conditions.

        A gate placed above its driver's ceiling never opens and the edge is
        dead; one below the floor is saturated and the edge is a constant. Both
        look like a weak coupling downstream, so both **raise** — the answer is
        definite, and a warning about a definite blocker is a warning nobody
        acts on. Pass ``allow_dead_edges=True`` to continue anyway.

        Also raises when an *activating* edge's driver is higher in the
        reference conditions than in every perturbed one: the edge then
        activates its target most where the perturbation is absent, which no
        placement of ``K`` repairs.

        Returns one row per gate, each carrying the ``suggestion`` that would
        place it. One solve, and only for composites that declare a gate.
        """
        gates = self.composite.hill_gates()
        if not gates:
            return []
        params = (
            self.initial_params() if param_values is None else param_values
        )
        paths = sorted({p for srcs, _ in gates.values() for p in srcs})
        rng = self.operating_ranges(params, paths)
        off_conds, on_conds = self._reference_split()

        rows, blockers = [], []
        for name, (srcs, ks) in gates.items():
            for src, k in zip(srcs, ks):
                lo = min(rng[c][src].lo for c in rng)
                hi = max(rng[c][src].hi for c in rng)
                ok = lo <= k <= hi
                sug = None if ok else self._place_from_ranges(rng, src)
                inverted = self._edge_is_inverted(
                    rng, src, off_conds, on_conds
                )
                rows.append(
                    dict(
                        gate=name,
                        source=src,
                        K=k,
                        lo=lo,
                        hi=hi,
                        in_range=ok,
                        inverted=inverted,
                        suggestion=sug,
                    )
                )
                if not ok:
                    blockers.append(
                        f"{name}.K = {k:.4g} on {src!r} is "
                        f"{'above' if k > hi else 'below'} everything its "
                        f"driver reaches ([{lo:.4g}, {hi:.4g}] across "
                        f"conditions), so the edge is "
                        f"{'dead' if k > hi else 'saturated'}. "
                        f"{_placement_advice(sug)}"
                    )
                if inverted:
                    blockers.append(
                        f"{name} activates on {src!r}, but that driver is "
                        f"higher in every reference condition than in any "
                        f"perturbed one, so the edge fires hardest where the "
                        f"perturbation is absent. No K repairs a sign."
                    )
        if blockers and not allow_dead_edges:
            raise ValueError(
                "Coupling edges that cannot carry a signal:\n  - "
                + "\n  - ".join(blockers)
                + "\nEach is a definite defect, not a tuning choice. Fix the "
                "wiring, or pass allow_dead_edges=True to run anyway."
            )
        for b in blockers:
            log.warning("%s", b)
        return rows

    def _reference_split(self) -> tuple[list[str], list[str]]:
        """``(reference, perturbed)`` conditions — the ones only ever used as a
        baseline, and the rest."""
        on = {a.condition for a in self.arms.values()}
        bases = {
            a.reference
            for a in self.arms.values()
            if a.reference not in (None, "t0")
        }
        return sorted(bases - on), sorted(on)

    def _edge_is_inverted(self, rng, src, off_conds, on_conds) -> bool:
        if not off_conds or not on_conds:
            return False

        def seen(cs):
            return [c for c in cs if src in rng.get(c, {})]

        off, on = seen(off_conds), seen(on_conds)
        if not off or not on:
            return False
        return min(rng[c][src].mean for c in off) > max(
            rng[c][src].mean for c in on
        )

    def _place_from_ranges(
        self,
        rng: dict,
        source_path: str,
        off_conditions: list[str] | None = None,
        on_conditions: list[str] | None = None,
        *,
        off_stat: str = "hi",
        on_stat: str = "mean",
        off_occupancy: float = 0.1,
    ):
        """Place a gate on ``source_path`` from operating ranges already
        solved. With no conditions named, the levels are the driver's own
        extremes — the lowest ceiling it settles to and the highest level it
        sustains — so placement needs no experimental design."""
        from hallsim.models.hill_edge import place_hill_gate

        seen = [c for c in rng if source_path in rng[c]]
        if not seen:
            return None
        if off_conditions is None:
            off = min(rng[c][source_path].hi for c in seen)
        else:
            off = max(
                getattr(rng[c][source_path], off_stat) for c in off_conditions
            )
        if on_conditions is None:
            on = max(rng[c][source_path].mean for c in seen)
        else:
            on = max(
                getattr(rng[c][source_path], on_stat) for c in on_conditions
            )
        return place_hill_gate(off, on, off_occupancy=off_occupancy)

    def suggest_hill_gate(
        self,
        param_values: dict,
        source_path: str,
        off_conditions: list[str],
        on_conditions: list[str],
        *,
        off_stat: str = "hi",
        on_stat: str = "mean",
        off_occupancy: float = 0.1,
        n_save: int = 200,
        critical: float | None = None,
        basal: float | None = None,
        hi: float | None = None,
        n: float = 2.0,
    ):
        """Deterministically suggest a Hill gate's ``(K, n)`` for a coupling
        edge driven by ``source_path``, from the *measured* operating point.

        Runs the faithful simulation (:meth:`operating_ranges`), takes the
        ``off_stat`` (default ``hi``, the ceiling) across ``off_conditions`` as
        the level the gate must stay closed at, and the ``on_stat`` (default
        ``mean``) across ``on_conditions`` as the level it should open at, then
        applies :func:`hallsim.models.hill_edge.place_hill_gate`. Returns a
        :class:`hallsim.models.hill_edge.HillGateSuggestion` — ``ok=False`` with
        an explanatory ``note`` when the arms' operating ranges overlap or are
        too close for a clean gate. Deterministic: same params → same suggestion.

        Give ``critical`` (with the edge's ``basal`` and ``hi``) to place the
        gate so the *signal* crosses a named downstream value — a bifurcation
        the coupling is meant to trigger — rather than forming a clean 10/90
        switch. That is the placement to use when the edge exists to flip
        something: it asks only that the crossing fall between the conditions,
        so it succeeds on drivers too weakly separated for a clean gate. See
        :func:`hallsim.models.hill_edge.place_hill_gate_for_crossing`.
        """
        rng = self.operating_ranges(param_values, [source_path], n_save=n_save)
        if critical is None:
            return self._place_from_ranges(
                rng,
                source_path,
                off_conditions,
                on_conditions,
                off_stat=off_stat,
                on_stat=on_stat,
                off_occupancy=off_occupancy,
            )
        if basal is None or hi is None:
            raise ValueError(
                "critical= needs the edge's basal and hi: the driver level "
                "that crosses it depends on the range the signal spans."
            )
        from hallsim.models.hill_edge import place_hill_gate_for_crossing

        return place_hill_gate_for_crossing(
            max(
                getattr(rng[c][source_path], off_stat) for c in off_conditions
            ),
            max(getattr(rng[c][source_path], on_stat) for c in on_conditions),
            basal=basal,
            hi=hi,
            critical=critical,
            n=n,
        )
