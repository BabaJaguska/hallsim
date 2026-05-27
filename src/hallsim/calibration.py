"""Differentiable calibration of HallSim composites.

Wraps the common pattern: a user-supplied ``loss_fn(params) -> scalar``
that runs a composite under the proposed parameters and compares the
output to data, plus a parameter pytree to optimize. The ``Calibrator``
handles autodiff-mode selection, optax setup, parameter clamping,
history logging, and an iteration loop.

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

Usage::

    from hallsim.calibration import Calibrator

    def loss_fn(params):
        composite = build_my_composite(**params)
        result = Scheduler(adjoint=...).run(composite, ...)
        return compute_loss_against_data(result)

    cal = Calibrator(
        loss_fn=loss_fn,
        init_params={"alpha": 0.05, "k": 0.1},
        clamps={"alpha": (0.001, 1.0), "k": (0.0, 1.0)},
        mode="forward",
        learning_rate=0.05,
    )
    history = cal.fit(steps=50)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import jax
import jax.flatten_util  # noqa: F401  # for ravel_pytree
import jax.numpy as jnp
import optax


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
