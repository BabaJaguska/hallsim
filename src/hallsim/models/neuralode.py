"""NeuralODE — learned dynamics as a composable Process.

A Process whose derivative is parameterized by a neural network (MLP).
Can be used as:

1. **Surrogate model** — train on data from an expensive simulation,
   then swap it into a Composite for fast execution.
2. **Unknown dynamics** — train on experimental time-series data
   when the mechanistic equations are unknown.
3. **PINN-style** — combine with physics-based loss terms for
   Physics-Informed Neural ODE training.

The MLP weights are JAX arrays living inside an Equinox module, so
the entire Process is JIT-compilable, vmappable, and differentiable
end-to-end through Diffrax solves.

Usage
-----
>>> proc = NeuralODEProcess(fields=["x", "y"], width=32, depth=2)
>>> comp = Composite(
...     processes={"neural": proc},
...     topology={"neural": {"state": ["pool/x", "pool/y"]}},
... )

One block port, ``state``, carries every field; bind it to one path per field.
>>> # Fit via fit_neuralode_derivative(...) or fit_neuralode_shooting(...)
"""

from __future__ import annotations

import logging
import hashlib
import time
from pathlib import Path
from typing import Any, Sequence

import copy

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from hallsim.process import Port, PortRole, Process
from hallsim.kinetics import hill_gate

log = logging.getLogger(__name__)

#: The single EVOLVED port carrying the whole learned field. One block port
#: rather than one port per field: the MLP already works on the stacked
#: vector, so naming each element separately only costs a slice each way.
STATE_PORT = "state"


class NeuralODEProcess(Process):
    """Process with MLP-parameterized dynamics.

    The MLP maps ``[state_fields, control_inputs] -> d(state)/dt``. Control
    inputs (``input_fields``) let the learned vector field be conditioned on
    external signals, so one network can represent a whole parameter-indexed
    family of dynamics — the mechanism that lets a NeuralODE reproduce a
    bifurcation: train across both sides of the critical value and the
    learned family flips regime as the conditioning input crosses it.

    Each control input is sourced one of three ways at derivative time:

    - **wired** (default) — read straight from an INPUT port of the same
      name;
    - **Hill-driven** (:meth:`with_input_driver`) — read a wired port and
      Hill-interpolate it between a fittable basal (a ``parameters`` entry)
      and ``hi``, the neural analogue of the mechanistic
      :class:`hallsim.models.hill_edge.HillEdge`, so an upstream state
      can drive the block the way it drives the mechanistic model it replaces;
    - **parameter-sourced** (:meth:`with_control_param`) — filled from a
      fittable scalar in ``parameters``, so a bifurcation knob stays a live,
      differentiable parameter of the hybrid rather than being frozen into
      the weights.

    Parameters
    ----------
    fields:
        State variables this process evolves; each an EVOLVED port.
    input_fields:
        Control inputs to the MLP, in the order training data supplies them.
    field_defaults:
        Initial values for the EVOLVED ports (aligned with ``fields``); use
        to start the block at the state its mechanistic counterpart starts
        at. Defaults to zeros.
    width, depth:
        Hidden-layer sizes. ``width`` is either a uniform int (a stack of
        ``depth`` hidden layers of that width) or an explicit per-layer
        sequence like ``(256, 128, 128)``, in which case ``depth`` is unused.
    key:
        PRNG key for weight init.
    """

    fields: tuple[str, ...] = eqx.field(static=True, default=("x", "y"))
    input_fields: tuple[str, ...] = eqx.field(static=True, default=())
    field_defaults: tuple[float, ...] = eqx.field(static=True, default=())
    mlp: eqx.Module = eqx.field(default=None)
    # The (normalised) vector field is ``out_scale ⊙ tanh(in_scale · MLP(ŷ))``
    # — LeCun's scaled tanh with both constants learnable. ``in_scale`` (β, a
    # global scalar inside tanh) is redundant on paper — it folds into the last
    # weights — but acts as a preconditioner that rescales the activation's
    # linear region in one step, smoothing stiff-ODE landscapes (Jagtap 2020
    # adaptive activations). ``out_scale`` (α, per-dim, after tanh) is NOT
    # redundant: tanh saturates at ±1, so without it the field cannot reach the
    # large slopes of a relaxation oscillator's spikes. Both trained by the fits.
    in_scale: jax.Array = eqx.field(default_factory=lambda: jnp.array(1.0))
    out_scale: jax.Array = eqx.field(default_factory=lambda: jnp.array(1.0))
    # Fittable scalar controls (e.g. a bifurcation parameter, a Hill basal).
    # Dynamic (traced) so Calibrator/`jax.grad` reach them via
    # ``parameters.<name>``, mirroring SBMLProcess.
    parameters: dict[str, jax.Array] = eqx.field(default_factory=dict)
    # Fixed I/O normalisation buffers (set from training data). Partitioned
    # out of the optimiser by callers, so they persist unchanged through fits.
    in_mean: jax.Array = eqx.field(default=None)
    in_std: jax.Array = eqx.field(default=None)
    out_mean: jax.Array = eqx.field(default=None)
    out_std: jax.Array = eqx.field(default=None)
    # Hill couplings: each (control_field, port, basal_param, hi, K, n) reads
    # ``port`` and interpolates ``parameters[basal_param]``→``hi``. Static
    # metadata so it round-trips through the ``eqx.tree_at`` substitutions the
    # handle / Calibrator paths apply to ``parameters``.
    _hill_drivers: tuple = eqx.field(static=True, default=())

    def __init__(
        self,
        fields: Sequence[str] = ("x", "y"),
        input_fields: Sequence[str] = (),
        field_defaults: Sequence[float] | None = None,
        width: int | Sequence[int] = 64,
        depth: int = 2,
        key: jax.Array | None = None,
        timescale: float | None = None,
    ):
        self.timescale = timescale
        self.fields = tuple(fields)
        self.input_fields = tuple(input_fields)
        self.field_defaults = (
            tuple(float(v) for v in field_defaults)
            if field_defaults is not None
            else tuple(0.0 for _ in self.fields)
        )
        self.parameters = {}
        self._hill_drivers = ()
        if key is None:
            key = jax.random.PRNGKey(0)
        n_out = len(self.fields)
        n_in = len(self.fields) + len(self.input_fields)
        self.mlp = _build_mlp(n_in, n_out, width, depth, key)
        self.in_scale = jnp.array(1.0)
        self.out_scale = jnp.ones(n_out)
        self.in_mean = jnp.zeros(n_in)
        self.in_std = jnp.ones(n_in)
        self.out_mean = jnp.zeros(n_out)
        self.out_std = jnp.ones(n_out)

    def with_input_driver(
        self,
        field: str,
        *,
        port: str,
        basal_param: str,
        hi: float,
        K: float,
        n: float = 2.0,
        basal: float | None = None,
    ) -> "NeuralODEProcess":
        """Source control input ``field`` from a Hill transform of ``port``.

        Adds an INPUT port ``port`` and, at derivative time, fills the MLP's
        ``field`` column with ``parameters[basal_param] + (hi - basal) *
        hill_gate(port; K, n)``. The basal lives on ``parameters`` so it stays
        fittable; ``hi``/``K``/``n`` are structural. ``basal`` seeds the
        parameter entry if not already present.
        """
        if field not in self.input_fields:
            raise KeyError(
                f"{field!r} is not a control input; have {self.input_fields}"
            )
        new = copy.copy(self)
        params = dict(self.parameters)
        if basal is not None or basal_param not in params:
            params[basal_param] = jnp.asarray(
                float(basal if basal is not None else 0.0)
            )
        object.__setattr__(new, "parameters", params)
        object.__setattr__(
            new,
            "_hill_drivers",
            self._hill_drivers
            + ((field, port, basal_param, float(hi), float(K), float(n)),),
        )
        return new

    def with_control_param(
        self, field: str, value: float
    ) -> "NeuralODEProcess":
        """Source control input ``field`` from a fittable scalar parameter.

        The value stays on ``parameters`` (differentiable, calibratable), so
        a bifurcation knob remains a live parameter of the hybrid instead of
        being baked into the weights.
        """
        if field not in self.input_fields:
            raise KeyError(
                f"{field!r} is not a control input; have {self.input_fields}"
            )
        new = copy.copy(self)
        params = dict(self.parameters)
        params[field] = jnp.asarray(float(value))
        object.__setattr__(new, "parameters", params)
        return new

    def ports_schema(self):
        schema = {
            STATE_PORT: Port(
                role=PortRole.EVOLVED,
                default=self.field_defaults,
                units="dimensionless",
                elements=self.fields,
            )
        }
        hill_fields = {d[0] for d in self._hill_drivers}
        for _, port, *_ in self._hill_drivers:
            schema[port] = Port(
                role=PortRole.INPUT, default=0.0, units="dimensionless"
            )
        param_fields = set(self.parameters)
        for cf in self.input_fields:
            if cf in hill_fields or cf in param_fields:
                continue
            schema[cf] = Port(
                role=PortRole.INPUT, default=0.0, units="dimensionless"
            )
        return schema

    def _control_column(self, cf, state, base_shape):
        for field, port, basal_param, hi, K, n in self._hill_drivers:
            if field == cf:
                basal = self.parameters[basal_param]
                return basal + (hi - basal) * hill_gate(
                    state[port], jnp.asarray(K), jnp.asarray(n)
                )
        if cf in self.parameters:
            return jnp.broadcast_to(
                jnp.asarray(self.parameters[cf]), base_shape
            )
        return state[cf]

    def derivative(self, t, state):
        # Trailing-axis stack so the process is shape-polymorphic: scalar
        # state -> (n_in,), batched -> (..., n_in). The MLP is applied per
        # row via vmap over the flattened leading axes.
        block = state[STATE_PORT]
        base_shape = jnp.shape(block)[:-1]
        controls = [
            self._control_column(cf, state, base_shape)
            for cf in self.input_fields
        ]
        inp = (
            jnp.concatenate(
                [block] + [jnp.asarray(c)[..., None] for c in controls],
                axis=-1,
            )
            if controls
            else block
        )
        inp_n = (inp - self.in_mean) / self.in_std
        flat = inp_n.reshape(-1, inp_n.shape[-1])
        z = jax.vmap(self.mlp)(flat).reshape(
            inp_n.shape[:-1] + (len(self.fields),)
        )
        field = self.out_scale * jnp.tanh(self.in_scale * z)  # normalised
        dy = field * self.out_std + self.out_mean
        return {STATE_PORT: dy}


# ── Input-conditioned templates ─────────────────────────────────────────
# Two interchangeable ways to fit a NeuralODEProcess, both taking the same
# data (ts, ys, us): `us` is one constant input vector per trajectory (None
# for autonomous dynamics). `fit_neuralode_derivative` regresses the vector
# field directly (no solver in the loop — robust for stiff dynamics, needs
# only that the trajectories are dense enough to difference); `fit_neuralode_
# shooting` integrates the learned field and matches trajectories (the
# classic backprop-through-solve). Pass `init=` to warm-start one from the
# other, e.g. derivative-fit then a short shooting fine-tune.


# Fields that must NOT be gradient-trained: the normalisation stats are fixed
# from the data, not learned. Named once here so both fits stay correct.
_FROZEN_FIELDS = ("in_mean", "in_std", "out_mean", "out_std")


def _trainable_partition(proc):
    """Split a NeuralODEProcess into ``(trainable, frozen)`` for fitting.

    Frozen = the fixed normalisation buffers (:data:`_FROZEN_FIELDS`).
    Trainable = every other float-array leaf — MLP weights, ``in_scale``,
    ``out_scale``, the fittable ``parameters`` dict, and any field added later.
    Inverting the logic this way (freeze the known set, train the rest) means
    new learnable parameters are picked up with no change here — the reusable
    analogue of PyTorch's ``Parameter`` vs ``buffer`` split, made explicit.
    """
    frozen = jax.tree_util.tree_map(lambda _: False, proc)
    frozen = eqx.tree_at(
        lambda p: tuple(getattr(p, f) for f in _FROZEN_FIELDS),
        frozen,
        tuple(True for _ in _FROZEN_FIELDS),
    )
    spec = jax.tree_util.tree_map(
        lambda leaf, fr: eqx.is_inexact_array(leaf) and not fr, proc, frozen
    )
    return eqx.partition(proc, spec)


def _build_mlp(n_in, n_out, width, depth, key):
    """Feed-forward net as an ``eqx.nn.Sequential`` of ``Linear`` layers with
    softplus hidden activations and a **raw linear output** (the output tanh +
    learnable scales live in :meth:`NeuralODEProcess.derivative`). ``width`` is
    a uniform int (a stack of ``depth`` hidden layers of that width) or an
    explicit per-hidden-layer sequence like ``(256, 128, 128)``."""
    hidden = (
        [int(width)] * depth
        if isinstance(width, int)
        else [int(w) for w in width]
    )
    sizes = [n_in, *hidden, n_out]
    keys = jax.random.split(key, len(sizes) - 1)
    layers = []
    for i, k in enumerate(keys):
        layers.append(eqx.nn.Linear(sizes[i], sizes[i + 1], key=k))
        if i < len(keys) - 1:
            layers.append(eqx.nn.Lambda(jax.nn.softplus))
    return eqx.nn.Sequential(layers)


_SDTW_INF = 1e10


def _soft_dtw(D: jnp.ndarray, gamma: float) -> jnp.ndarray:
    """Soft-DTW alignment cost of one pairwise cost matrix ``D`` (n×m).

    Cuturi & Blondel 2017: the hard-DTW min over alignment paths relaxed to
    a differentiable ``softmin_γ(a,b,c) = -γ·log Σ exp(-·/γ)``. Row-scan over
    the DP so it stays fixed-shape (JIT/vmap-able); ``γ→0`` recovers hard DTW.
    """
    m = D.shape[1]

    def softmin(vals):
        vmin = jnp.min(vals)
        return vmin - gamma * jnp.log(jnp.sum(jnp.exp(-(vals - vmin) / gamma)))

    def row_step(prev_row, D_row):
        r_up, r_diag = prev_row[1:], prev_row[:-1]

        def col_step(r_left, inp):
            d, up, diag = inp
            r_ij = d + softmin(jnp.stack([up, r_left, diag]))
            return r_ij, r_ij

        _, row = jax.lax.scan(col_step, _SDTW_INF, (D_row, r_up, r_diag))
        return jnp.concatenate([jnp.array([_SDTW_INF]), row]), None

    first_row = jnp.concatenate([jnp.array([0.0]), jnp.full(m, _SDTW_INF)])
    last_row, _ = jax.lax.scan(row_step, first_row, D)
    return last_row[-1]


def _pairwise_sq(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Squared-Euclidean cost matrix between sequences ``a``, ``b`` (L×D)."""
    return jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)


def _soft_dtw_divergence(pred, targ, gamma):
    """Mean soft-DTW **divergence** over a batch of windows.

    ``sdtw(x,y) − ½sdtw(x,x) − ½sdtw(y,y)`` (Blondel et al. 2021): unlike raw
    soft-DTW it is ≥0 and zero iff ``x==y``, removing the self-similarity bias
    a flat prediction could otherwise exploit. Normalised by window length so
    ``dtw_weight`` sits on the same per-step scale as the MSE term.
    """
    L = pred.shape[1]

    def one(x, y):
        return (
            _soft_dtw(_pairwise_sq(x, y), gamma)
            - 0.5 * _soft_dtw(_pairwise_sq(x, x), gamma)
            - 0.5 * _soft_dtw(_pairwise_sq(y, y), gamma)
        )

    return jnp.mean(jax.vmap(one)(pred, targ)) / L


class _RHSProcess(Process):
    """Wraps a bare ``rhs(t, y)`` as a Process, so trajectories generate
    through the Scheduler (auto-stiffness — implicit where stiff, explicit
    where not) rather than a hand-rolled fixed solver."""

    fields: tuple[str, ...] = ()
    rhs: Any = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            STATE_PORT: Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="dimensionless",
                elements=self.fields,
            )
        }

    def derivative(self, t, state):
        d = self.rhs(t, state[STATE_PORT])
        return {STATE_PORT: d}


class _ShootWrap(Process):
    """Wraps a NeuralODEProcess for trajectory shooting through the Scheduler.

    The block's control inputs are exposed as EVOLVED states with zero
    derivative (frozen at their per-trajectory value in ``y0``), so one
    Scheduler run integrates the learned field with its inputs held constant
    — no hand-rolled solver, and gradients reach the block's weights because
    ``block`` is a dynamic child of the composite.
    """

    block: Any = None
    fld: tuple[str, ...] = eqx.field(static=True, default=())
    inp: tuple[str, ...] = eqx.field(static=True, default=())

    def ports_schema(self):
        return {
            f: Port(role=PortRole.EVOLVED, default=0.0, units="dimensionless")
            for f in self.fld + self.inp
        }

    def derivative(self, t, state):
        # One port per field here, one stacked STATE_PORT on the block, so
        # the view is restacked on the way in and split on the way out.
        block_state = {
            STATE_PORT: jnp.stack([state[f] for f in self.fld], axis=-1),
            **{f: state[f] for f in self.inp},
        }
        d = self.block.derivative(t, block_state)[STATE_PORT]
        return {
            **{f: d[..., i] for i, f in enumerate(self.fld)},
            **{f: jnp.zeros_like(state[f]) for f in self.inp},
        }


class _ConditionedRHS(Process):
    """A conditioned RHS whose conditioning rides in the state.

    The control values are EVOLVED ports with zero derivative, frozen at the
    per-trajectory value in ``y0`` — the same device :class:`_ShootWrap` uses.
    That makes the conditioning grid a batch axis over one Composite, rather
    than one Composite and one Scheduler run per point.
    """

    fields: tuple[str, ...] = eqx.field(static=True, default=())
    inputs: tuple[str, ...] = eqx.field(static=True, default=())
    rhs_for_input: Any = eqx.field(static=True, default=None)

    def ports_schema(self):
        return {
            f: Port(role=PortRole.EVOLVED, default=0.0, units="dimensionless")
            for f in self.fields + self.inputs
        }

    def derivative(self, t, state):
        y = jnp.stack([state[f] for f in self.fields], axis=-1)
        u = jnp.stack([state[f] for f in self.inputs], axis=-1)
        d = self.rhs_for_input(u)(t, y)
        return {
            **{f: d[..., i] for i, f in enumerate(self.fields)},
            **{f: jnp.zeros_like(state[f]) for f in self.inputs},
        }


def _conditioned_cache_path(cache_key, ts, inputs, n_ics, y0_range, key):
    """Where a conditioned trajectory set is cached.

    Keyed on the model name the caller supplies plus every number that changes
    the trajectories, so editing a grid produces a different file rather than
    a stale hit.
    """
    h = hashlib.sha256()
    h.update(str(cache_key).encode())
    for arr in (np.asarray(ts), np.asarray(inputs), np.asarray(key)):
        h.update(np.ascontiguousarray(arr).tobytes())
        h.update(str(arr.dtype).encode())
    h.update(repr((int(n_ics), tuple(float(v) for v in y0_range))).encode())
    return (
        Path.home()
        / ".cache"
        / "hallsim"
        / "neuralode_data"
        / f"{h.hexdigest()[:16]}.npz"
    )


def simulate_conditioned(
    rhs_for_input,
    ts: jnp.ndarray,
    inputs: jnp.ndarray,
    n_ics: int = 3,
    y0_range: tuple[float, float] = (0.0, 1.0),
    key: jax.Array | None = None,
    cache_key: str | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Trajectories from a known input-conditioned RHS, for recovery fits.

    Each input's dynamics run through :class:`hallsim.scheduler.Scheduler`
    with ``auto_stiffness`` — so stiff regimes (e.g. near a bifurcation) are
    integrated implicitly instead of blowing past the step budget on an
    explicit solver. Initial conditions are batched per input (one vectorised
    Scheduler run each).

    Parameters
    ----------
    rhs_for_input:
        ``u -> rhs(t, y, args) -> dy/dt`` — the mechanistic RHS at input ``u``.
    ts:
        Uniform time grid, shape ``(T,)``.
    inputs:
        Input values to condition on, shape ``(M, U)``.
    n_ics:
        Initial conditions sampled per input value.
    cache_key:
        Identifies the *model* these trajectories come from. Given one, the
        result is cached under ``~/.cache/hallsim/neuralode_data/`` keyed on it
        together with the grid, and a later call with the same model and grid
        is a local read instead of one Scheduler run per conditioning point.
        Omit it and nothing is cached — the key has to name the model because
        the grid alone cannot tell two models apart.

    Returns
    -------
    ``(ys, us)`` with ``ys`` shape ``(M*n_ics, T, D)`` and ``us`` shape
    ``(M*n_ics, U)`` — the per-trajectory constant input, aligned with ``ys``.
    """
    from hallsim.composite import Composite
    from hallsim.scheduler import Scheduler

    if key is None:
        key = jax.random.PRNGKey(0)
    ts = jnp.asarray(ts)
    cache_path = (
        _conditioned_cache_path(
            cache_key, ts, jnp.asarray(inputs), n_ics, y0_range, key
        )
        if cache_key
        else None
    )
    if cache_path is not None and cache_path.is_file():
        with np.load(cache_path) as z:
            log.info("cached trajectories: %s", cache_path.name)
            return jnp.asarray(z["ys"]), jnp.asarray(z["us"])
    inputs = jnp.asarray(inputs)
    dim = _probe_dim(rhs_for_input(inputs[0]), ts)
    fields = tuple(f"v{i}" for i in range(dim))
    u_dim = int(inputs.shape[-1])
    u_names = tuple(f"u{i}" for i in range(u_dim))
    t0, t1 = float(ts[0]), float(ts[-1])
    save_dt = (t1 - t0) / (len(ts) - 1)

    n_inputs = int(inputs.shape[0])
    log.info(
        "simulate_conditioned: %d conditioning points x %d ICs in one "
        "batched run",
        n_inputs,
        n_ics,
    )

    comp = Composite(
        {
            "m": _ConditionedRHS(
                fields=fields, inputs=u_names, rhs_for_input=rhs_for_input
            )
        },
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    keys_all = comp.store_keys()
    s_idx = jnp.asarray([keys_all.index(f"m/{f}") for f in fields])
    u_idx = jnp.asarray([keys_all.index(f"m/{f}") for f in u_names])

    # Condition-major, matching the per-point loop this replaces: every IC of
    # one conditioning point, then the next point.
    y0v = jax.random.uniform(
        key,
        (n_inputs, n_ics, dim),
        minval=y0_range[0],
        maxval=y0_range[1],
    ).reshape(n_inputs * n_ics, dim)
    us_out = jnp.repeat(inputs, n_ics, axis=0)
    y0 = (
        jnp.broadcast_to(
            comp.initial_state_vec(), (n_inputs * n_ics, len(keys_all))
        )
        .at[:, s_idx]
        .set(y0v)
        .at[:, u_idx]
        .set(us_out)
    )

    _t0 = time.time()
    res = Scheduler(auto_stiffness=True).run(
        comp, t_span=(t0, t1), y0=y0, macro_dt=t1 - t0, save_dt=save_dt
    )
    traj = jnp.stack([res.get(f"m/{f}") for f in fields], axis=-1)
    ys_out = jnp.moveaxis(traj, 0, 1)  # (n_inputs*n_ics, T, dim)
    jax.block_until_ready(ys_out)
    log.info("  trajectories in %.1fs", time.time() - _t0)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = cache_path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, ys=np.asarray(ys_out), us=np.asarray(us_out))
        tmp.replace(cache_path)
        log.info("cached trajectories -> %s", cache_path.name)
    return ys_out, us_out


def _probe_dim(rhs, ts) -> int:
    """Infer state dimension D by probing the RHS at zero vectors."""
    for d in range(1, 64):
        try:
            out = rhs(ts[0], jnp.zeros(d))
            if jnp.size(out) == d:
                return d
        except Exception:
            continue
    raise ValueError("could not infer state dimension from rhs")


def _dataset_arrays(ts, ys, us, n_inputs):
    """Flatten trajectories to (states, inputs, central-difference slopes)."""
    ys = jnp.asarray(ys)
    slopes = jnp.gradient(ys, jnp.asarray(ts), axis=1)
    n, t, d = ys.shape
    states = ys.reshape(n * t, d)
    slopes = slopes.reshape(n * t, d)
    if us is None or n_inputs == 0:
        inputs = jnp.zeros((n * t, 0))
    else:
        us = jnp.asarray(us)
        inputs = jnp.broadcast_to(us[:, None, :], (n, t, us.shape[1])).reshape(
            n * t, us.shape[1]
        )
    return states, inputs, slopes


def _norm_stats(states, inputs, slopes):
    x = jnp.concatenate([states, inputs], axis=-1)
    return (
        x.mean(0),
        x.std(0) + 1e-8,
        slopes.mean(0),
        slopes.std(0) + 1e-8,
    )


def _new_process(fields, input_fields, width, depth, seed, init):
    if init is not None:
        return init
    return NeuralODEProcess(
        fields=fields,
        input_fields=input_fields,
        width=width,
        depth=depth,
        key=jax.random.PRNGKey(seed),
    )


def _wrap_composite(block, fld, inp):
    from hallsim.composite import Composite

    return Composite(
        {"m": _ShootWrap(block=block, fld=fld, inp=inp)},
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )


def _prepared(ts, ys, us, fields, input_fields, width, depth, seed, init):
    """The block with its normalisation set from the data, its field and
    input names, their store paths inside the wrapping composite, the
    trajectories with each input as a constant column beside the states,
    and the inputs themselves, ``(n_traj, n_inputs)``."""
    ts = jnp.asarray(ts)
    ys = jnp.asarray(ys)
    states, inputs, slopes = _dataset_arrays(ts, ys, us, len(input_fields))
    im, isd, om, osd = _norm_stats(states, inputs, slopes)
    proc = _new_process(fields, input_fields, width, depth, seed, init)
    proc = eqx.tree_at(
        lambda p: (p.in_mean, p.in_std, p.out_mean, p.out_std),
        proc,
        (im, isd, om, osd),
    )
    if init is None:
        # A fresh block's output scale covers the normalised slope range so
        # tanh does not start clipped; a warm-started one keeps its own.
        yn = (slopes - om) / osd
        proc = eqx.tree_at(
            lambda p: p.out_scale,
            proc,
            jnp.maximum(1.0, jnp.max(jnp.abs(yn), axis=0) * 1.2),
        )
    fld, inp = tuple(fields), tuple(input_fields)
    paths = tuple(f"m/{f}" for f in fld)
    in_paths = tuple(f"m/{f}" for f in inp)
    n, t = ys.shape[:2]
    us_arr = jnp.zeros((n, 0)) if us is None else jnp.asarray(us)
    full = jnp.concatenate(
        [ys, jnp.broadcast_to(us_arr[:, None, :], (n, t, us_arr.shape[1]))],
        axis=-1,
    )
    return proc, fld, inp, paths, in_paths, full, us_arr


def _problem(block, fld, inp, paths, conditions, data, arms, **kw):
    """The calibration problem whose one fittable is the block."""
    from hallsim.calibration import CalibrationProblem, LearnedRef
    from hallsim.gene_reporters import trajectory_reporters

    return CalibrationProblem(
        composite=_wrap_composite(block, fld, inp),
        reporters=trajectory_reporters(*paths),
        conditions=conditions,
        data=data,
        arms=arms,
        params={"block": LearnedRef("m")},
        fit_arms=list(arms),
        **kw,
    )


def fit_neuralode_derivative(
    ts: jnp.ndarray,
    ys: jnp.ndarray,
    us: jnp.ndarray | None = None,
    *,
    fields: Sequence[str],
    input_fields: Sequence[str] = (),
    width: int = 96,
    depth: int = 3,
    lr: float = 3e-3,
    steps: int = 4000,
    batch_size: int = 512,
    init: NeuralODEProcess | None = None,
    seed: int = 0,
) -> NeuralODEProcess:
    """Fit by **derivative matching**: the block's field against the
    trajectories' central-difference slopes, which is the calibrator's
    collocation term on its own, ``batch_size`` samples a step. No ODE solve
    in the loop, so it is robust for stiff / oscillatory dynamics; the solver
    reappears when the returned block is integrated. Returns the best
    iterate, scored on one fixed draw. Verify by integrating afterward.
    """
    from hallsim.calibration import Collocation, Condition

    block, fld, inp, paths, in_paths, full, _ = _prepared(
        ts, ys, us, fields, input_fields, width, depth, seed, init
    )
    ts = jnp.asarray(ts)
    problem = _problem(
        block,
        fld,
        inp,
        paths,
        {"train": Condition("train", {})},
        {},
        {},
        collocation=Collocation(
            ts, full, paths + in_paths, matched=paths, batch=batch_size
        ),
        t_end=float(ts[-1]),
        macro_dt=float(ts[-1] - ts[0]),
        n_save=2,
    )
    hist = problem.fit(
        steps=steps,
        mode="reverse",
        learning_rate=lr,
        minibatch_seed=seed + 1,
        eval_every=max(1, steps // 30),
        identifiability=False,
        log_every=max(1, steps // 8),
    )
    log.info(
        "derivative fit: loss %.6f -> best %.6f in %.1fs",
        hist.losses[0],
        hist.best_loss,
        hist.wall_time_s,
    )
    return problem.processes_at(hist.best_params)["m"].block


def fit_neuralode_shooting(
    ts: jnp.ndarray,
    ys: jnp.ndarray,
    us: jnp.ndarray | None = None,
    *,
    fields: Sequence[str],
    input_fields: Sequence[str] = (),
    segments: int = 1,
    curriculum: int = 1,
    length_strategy: tuple = (1.0,),
    physics_weight: float = 0.0,
    physics_batch: int = 512,
    batch_size: int = 32,
    dtw_weight: float = 0.0,
    dtw_gamma: float = 0.1,
    width: int = 96,
    depth: int = 3,
    lr: float = 1e-3,
    steps: int = 800,
    init: NeuralODEProcess | None = None,
    seed: int = 0,
    rtol: float | None = None,
    atol: float | None = None,
) -> NeuralODEProcess:
    """Fit by **trajectory shooting** through the calibrator: every window of
    every trajectory is a condition starting from the observed state
    (:func:`hallsim.calibration.shooting_conditions`), matched along the
    window and, through the boundary sample it shares with the next, to that
    window's start; the block is the problem's one fittable, and
    ``batch_size`` trajectories, drawn afresh each step, run in one batched
    solve.

    Plain single shooting over many periods of an oscillator collapses the
    learned trajectory to a fixed point: a small period error drifts the
    prediction out of phase, and full-horizon MSE is then lower for a flat
    line at the mean than for a phase-shifted oscillation, so the optimizer
    damps the amplitude away. The knobs below are independent, combinable
    stabilizers against that:

    - ``segments`` — **multiple shooting**: windows per trajectory, each
      integrated from its own observed start. Shorter windows accrue less
      phase drift, so the flat optimum stops winning.
    - ``curriculum`` — over that many stages match a progressively longer
      prefix of each window, anchoring the field on the near-term dynamics
      before the long-horizon phase.
    - ``length_strategy`` — grow the global horizon: fractions like
      ``(0.1, 1.0)`` train on the first 10% of each trajectory first, then
      the whole, splitting ``steps`` across the stages.
    - ``physics_weight`` — the collocation term
      (:class:`hallsim.calibration.Collocation`) at that weight,
      ``physics_batch`` samples a step: the vector-field magnitude constraint
      pure shooting lacks, the strongest stabilizer, degrading with noisy or
      sparse slopes.
    - ``dtw_weight`` — a soft-DTW divergence (``dtw_gamma`` the softmin
      temperature) added to each window's likelihood, so a prediction with
      the right shape but slightly wrong period is not penalised into
      flatness.

    Each stage warm-starts from the previous stage's best iterate, and the
    last stage's best iterate is returned. ``rtol``/``atol`` reach the
    Scheduler; keep them tight for periodic dynamics. Warm-start via
    ``init=`` from a derivative fit for stiff dynamics. See MPINeuralODE
    (arXiv:2605.13305) for the combined recipe.
    """
    from hallsim.calibration import (
        Collocation,
        gaussian_nll,
        shooting_conditions,
    )

    block, fld, inp, paths, in_paths, full, us_arr = _prepared(
        ts, ys, us, fields, input_fields, width, depth, seed, init
    )
    ts = jnp.asarray(ts)
    ys = jnp.asarray(ys)
    held = {p: us_arr[:, j] for j, p in enumerate(in_paths)}
    sched_kw = {
        k: v for k, v in (("rtol", rtol), ("atol", atol)) if v is not None
    }
    colloc = (
        Collocation(
            ts,
            full,
            paths + in_paths,
            matched=paths,
            weight=physics_weight,
            batch=physics_batch,
        )
        if physics_weight
        else None
    )

    def dtw_likelihood(model, data, weight):
        # (n_fields, n_t, n_traj) -> (n_traj, n_t, n_fields)
        pred = jnp.transpose(model, (2, 1, 0))
        targ = jnp.transpose(data, (2, 1, 0))
        return gaussian_nll(
            model, data, weight
        ) + dtw_weight * _soft_dtw_divergence(pred, targ, dtw_gamma)

    likelihood = dtw_likelihood if dtw_weight else None

    n_t = ys.shape[1]
    stage_steps = max(1, steps // len(length_strategy) // curriculum)
    for frac in length_strategy:
        n_active = min(n_t, max(segments * 2, int(round(frac * n_t))))
        for stage in range(curriculum):
            conds, data, arms = shooting_conditions(
                ts[:n_active],
                ys[:, :n_active],
                paths,
                segments=segments,
                match=(stage + 1) / curriculum,
                held=held,
            )
            problem = _problem(
                block,
                fld,
                inp,
                paths,
                conds,
                data,
                arms,
                collocation=colloc,
                likelihood=likelihood,
                t_end=float(ts[n_active - 1]),
                macro_dt=float(ts[n_active - 1] - ts[0]),
                n_save=max(len(d) for d in data.values()) + 1,
                scheduler_kwargs=sched_kw or None,
                member_batch=batch_size,
            )
            hist = problem.fit(
                steps=stage_steps,
                mode="reverse",
                learning_rate=lr,
                minibatch_seed=seed + 1,
                eval_every=max(1, stage_steps // 10),
                identifiability=False,
                log_every=max(1, stage_steps // 4),
            )
            block = problem.processes_at(hist.best_params)["m"].block
            log.info(
                "shooting length=%.2f stage %d/%d: loss %.6f -> best %.6f "
                "in %.1fs",
                frac,
                stage + 1,
                curriculum,
                hist.losses[0],
                hist.best_loss,
                hist.wall_time_s,
            )
    return block
