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
...     topology={"neural": {"x": "pool/x", "y": "pool/y"}},
... )
>>> # Fit via fit_neuralode_derivative(...) or fit_neuralode_shooting(...)
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import copy

import equinox as eqx
import jax
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process
from hallsim.kinetics import hill_gate

log = logging.getLogger(__name__)


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
      and ``hi``, the neural analogue of
      :meth:`hallsim.sbml_import.SBMLProcess.with_param_driver`, so an
      upstream state can drive the block the way it drives the mechanistic
      model it replaces;
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

    fields: tuple[str, ...] = ("x", "y")
    input_fields: tuple[str, ...] = ()
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
    # hallmark / Calibrator paths apply to ``parameters``.
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
            name: Port(
                role=PortRole.EVOLVED, default=dflt, units="dimensionless"
            )
            for name, dflt in zip(self.fields, self.field_defaults)
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
        base_shape = jnp.shape(state[self.fields[0]])
        cols = [state[f] for f in self.fields] + [
            self._control_column(cf, state, base_shape)
            for cf in self.input_fields
        ]
        inp = jnp.stack(cols, axis=-1)
        inp_n = (inp - self.in_mean) / self.in_std
        flat = inp_n.reshape(-1, inp_n.shape[-1])
        z = jax.vmap(self.mlp)(flat).reshape(
            inp_n.shape[:-1] + (len(self.fields),)
        )
        field = self.out_scale * jnp.tanh(self.in_scale * z)  # normalised
        dy = field * self.out_std + self.out_mean
        return {f: dy[..., i] for i, f in enumerate(self.fields)}


# ── Input-conditioned templates ─────────────────────────────────────────
# Two interchangeable ways to fit a NeuralODEProcess, both taking the same
# data (ts, ys, us): `us` is one constant input vector per trajectory (None
# for autonomous dynamics). `fit_neuralode_derivative` regresses the vector
# field directly (no solver in the loop — robust for stiff dynamics, needs
# only that the trajectories are dense enough to difference); `fit_neuralode_
# shooting` integrates the learned field and matches trajectories (the
# classic backprop-through-solve). Pass `init=` to warm-start one from the
# other, e.g. derivative-fit then a short shooting fine-tune.


def _field_normalized(mlp, in_scale, out_scale, xb):
    """Normalised vector-field prediction ``out_scale ⊙ tanh(in_scale·MLP(xb))``
    for a batch of normalised inputs — the quantity the collocation/derivative
    losses compare against normalised slopes."""
    return out_scale * jnp.tanh(in_scale * jax.vmap(mlp)(xb))


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
            f: Port(role=PortRole.EVOLVED, default=0.0, units="dimensionless")
            for f in self.fields
        }

    def derivative(self, t, state):
        y = jnp.stack([state[f] for f in self.fields])
        d = self.rhs(t, y)
        return {f: d[i] for i, f in enumerate(self.fields)}


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
        d = self.block.derivative(t, state)
        return {**d, **{f: jnp.zeros_like(state[f]) for f in self.inp}}


def simulate_conditioned(
    rhs_for_input,
    ts: jnp.ndarray,
    inputs: jnp.ndarray,
    n_ics: int = 3,
    y0_range: tuple[float, float] = (0.0, 1.0),
    key: jax.Array | None = None,
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
    inputs = jnp.asarray(inputs)
    dim = _probe_dim(rhs_for_input(inputs[0]), ts)
    fields = tuple(f"v{i}" for i in range(dim))
    sched = Scheduler(auto_stiffness=True)
    t0, t1 = float(ts[0]), float(ts[-1])
    save_dt = (t1 - t0) / (len(ts) - 1)
    ys, us = [], []
    for u in inputs:
        comp = Composite(
            {"m": _RHSProcess(fields=fields, rhs=rhs_for_input(u))},
            topology={},
            validate=False,
            semantic_validation={"check_semantics": False},
        )
        idx = jnp.asarray([comp.store_keys().index(f"m/{f}") for f in fields])
        key, k = jax.random.split(key)
        y0v = jax.random.uniform(
            k, (n_ics, dim), minval=y0_range[0], maxval=y0_range[1]
        )
        y0 = (
            jnp.broadcast_to(
                comp.initial_state_vec(), (n_ics, len(comp.store_keys()))
            )
            .at[:, idx]
            .set(y0v)
        )
        res = sched.run(
            comp, t_span=(t0, t1), y0=y0, macro_dt=t1 - t0, save_dt=save_dt
        )
        traj = jnp.stack([res.get(f"m/{f}") for f in fields], axis=-1)
        ys.append(jnp.moveaxis(traj, 0, 1))  # (n_ics, T, dim)
        us.append(jnp.broadcast_to(u, (n_ics,) + u.shape))
    return jnp.concatenate(ys, 0), jnp.concatenate(us, 0)


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
    """Fit by **derivative matching**: regress the network onto the vector
    field (central-difference slopes of the trajectories). No ODE solver in
    the training loop, so it is robust for stiff / oscillatory dynamics; the
    solver reappears only when the returned Process is integrated or
    differentiated inside a Composite. Verify by integrating afterward.
    """
    import optax

    states, inputs, slopes = _dataset_arrays(ts, ys, us, len(input_fields))
    im, isd, om, osd = _norm_stats(states, inputs, slopes)
    proc = _new_process(fields, input_fields, width, depth, seed, init)
    proc = eqx.tree_at(
        lambda p: (p.in_mean, p.in_std, p.out_mean, p.out_std),
        proc,
        (im, isd, om, osd),
    )
    xn = (jnp.concatenate([states, inputs], -1) - im) / isd
    yn = (slopes - om) / osd
    # Seed the output scale to cover the normalised slope range (a fresh block;
    # a warm-started one keeps its learned scale) so tanh doesn't start clipped.
    if init is None:
        proc = eqx.tree_at(
            lambda p: p.out_scale,
            proc,
            jnp.maximum(1.0, jnp.max(jnp.abs(yn), axis=0) * 1.2),
        )

    # Optimise the field (MLP + scales + any fittable parameters); the
    # normalisation buffers stay frozen. See :func:`_trainable_partition`.
    opt = optax.adam(lr)
    train, frozen = _trainable_partition(proc)
    opt_state = opt.init(train)

    @eqx.filter_jit
    def stepf(train, opt_state, xb, yb):
        def loss(tr):
            p = eqx.combine(tr, frozen)
            return jnp.mean(
                (_field_normalized(p.mlp, p.in_scale, p.out_scale, xb) - yb)
                ** 2
            )

        lv, g = eqx.filter_value_and_grad(loss)(train)
        upd, opt_state = opt.update(g, opt_state)
        return eqx.apply_updates(train, upd), opt_state, lv

    key = jax.random.PRNGKey(seed + 1)
    n = xn.shape[0]
    for s in range(steps):
        key, kb = jax.random.split(key)
        idx = jax.random.choice(kb, n, (min(batch_size, n),), replace=False)
        train, opt_state, lv = stepf(train, opt_state, xn[idx], yn[idx])
        if s % max(1, steps // 8) == 0:
            log.info("derivative-fit step %d, loss %.6f", s, float(lv))
    return eqx.combine(train, frozen)


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
    continuity_weight: float = 0.0,
    dtw_weight: float = 0.0,
    dtw_gamma: float = 0.1,
    width: int = 96,
    depth: int = 3,
    lr: float = 1e-3,
    steps: int = 800,
    batch_size: int = 32,
    init: NeuralODEProcess | None = None,
    seed: int = 0,
    rtol: float | None = None,
    atol: float | None = None,
) -> NeuralODEProcess:
    """Fit by **trajectory shooting**: integrate the learned field and match
    the trajectory, backpropagating through the solve.

    Plain single shooting over many periods of an oscillator collapses the
    learned trajectory to a fixed point: a small period error drifts the
    prediction out of phase, and full-horizon MSE is then lower for a flat
    line at the mean than for a phase-shifted oscillation, so the optimizer
    damps the amplitude away. The three knobs below are independent,
    combinable stabilizers against that:

    - ``segments`` — **multiple shooting**: cut each trajectory into
      ``segments`` contiguous windows and integrate each from its own
      observed start state. Shorter windows accrue less phase drift, so the
      flat optimum stops winning. Slope-free — safe for noisy real data.
    - ``curriculum`` — grow the matched horizon *within each window*: over
      ``curriculum`` stages match a progressively longer prefix of each window
      (short→full), so the field is anchored on the easy near-term dynamics
      before the hard long-horizon phase. Slope-free.
    - ``length_strategy`` — grow the *global* horizon (Kidger's Diffrax trick):
      a tuple of fractions like ``(0.1, 1.0)`` trains on the first 10% of each
      trajectory first, then the full length, splitting ``steps`` across the
      stages. Fitting the short-horizon dynamics first is the standard way to
      avoid the flat-line local minimum. Distinct from ``curriculum``: this
      shortens the whole trajectory (fewer, shorter windows), not the matched
      prefix of a fixed window.
    - ``physics_weight`` — **collocation regularizer**
      ``λ·‖f_θ(z) − ż‖²`` on the data's finite-difference slopes (the same
      residual :func:`fit_neuralode_derivative` minimizes). Supplies the
      absolute vector-field magnitude constraint pure shooting lacks — the
      strongest stabilizer, but slope quality degrades with noise/sparsity.
    - ``continuity_weight`` — tie each window's integrated endpoint to the
      next window's observed start, for global consistency across boundaries.
    - ``dtw_weight`` — **soft-DTW divergence** on each window, a
      time-warp-invariant trajectory distance (``dtw_gamma`` sets the
      softmin temperature). Where MSE punishes a phase-shifted-but-correct
      oscillation harder than a flat line — the collapse driver — soft-DTW
      scores the two sequences under their best differentiable alignment, so
      a prediction with the right shape but slightly wrong period is no
      longer penalised into flatness. The divergence form is ≥0 and zero
      only at equality, so it cannot be gamed by damping.

    ``segments=1, curriculum=1, physics_weight=0`` is classic single
    shooting. Warm-start via ``init=`` from a derivative fit for stiff
    dynamics. See MPINeuralODE (arXiv:2605.13305) for the combined recipe.

    Integration runs through the Scheduler; ``rtol``/``atol`` are forwarded to
    it (default = the Scheduler's production tolerances). Loosening them trades
    accuracy for speed — but note that a loose tolerance corrupts oscillator
    integration (numerical anti-damping), so keep it tight for periodic
    dynamics.
    """
    import optax

    from hallsim.composite import Composite
    from hallsim.scheduler import Scheduler

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
    us_arr = jnp.zeros((ys.shape[0], 0)) if us is None else jnp.asarray(us)
    fld, inp = tuple(fields), tuple(input_fields)
    xn = (jnp.concatenate([states, inputs], -1) - im) / isd
    yn = (slopes - om) / osd

    n_t = ys.shape[1]
    dt = float(ts[1] - ts[0])

    # Integrate the block through the Scheduler (its inputs frozen as constant
    # states), so shooting reuses the same runner as deployment — no
    # hand-rolled solver, and auto-stiffness is available if the field is stiff.
    comp0 = Composite(
        {"m": _ShootWrap(block=proc, fld=fld, inp=inp)},
        topology={},
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    _keys = comp0.store_keys()
    _sidx = jnp.asarray([_keys.index(f"m/{f}") for f in fld])
    _iidx = jnp.asarray([_keys.index(f"m/{f}") for f in inp]) if inp else None
    _base = comp0.initial_state_vec()
    _sched_kw = {}
    if rtol is not None:
        _sched_kw["rtol"] = rtol
    if atol is not None:
        _sched_kw["atol"] = atol
    _sched = Scheduler(**_sched_kw)

    # Window start times are concrete; the end is derived from ``dt`` and the
    # point count so the Scheduler's uniform save grid lands exactly on the
    # last sample (no floating-point overshoot past t1).
    def integrate(bk, y0, u, t0, n_pts):
        c = eqx.tree_at(lambda cc: cc.processes["m"].block, comp0, bk)
        y0f = _base.at[_sidx].set(y0)
        if _iidx is not None:
            y0f = y0f.at[_iidx].set(u)
        t1 = t0 + dt * (n_pts - 1)
        res = _sched.run(
            c, t_span=(t0, t1), y0=y0f, macro_dt=t1 - t0, save_dt=dt
        )
        return jnp.stack([res.get(f"m/{f}") for f in fld], axis=-1)

    def make_loss(match_len, starts, seg_len):
        t0s = [float(ts[s]) for s in starts]

        @eqx.filter_value_and_grad
        def loss_fn(train, yb, ub, xb, tb):
            bk = eqx.combine(train, frozen)
            wins, dtws = [], []
            for s, t0 in zip(starts, t0s):
                pred = jax.vmap(
                    lambda a, u: integrate(bk, a, u, t0, match_len)
                )(yb[:, s], ub)
                targ = yb[:, s : s + match_len]
                wins.append(jnp.mean((pred - targ) ** 2))
                if dtw_weight:
                    dtws.append(_soft_dtw_divergence(pred, targ, dtw_gamma))
            total = jnp.mean(jnp.stack(wins))
            if dtw_weight:
                total = total + dtw_weight * jnp.mean(jnp.stack(dtws))
            if physics_weight:
                total = total + physics_weight * jnp.mean(
                    (
                        _field_normalized(
                            bk.mlp, bk.in_scale, bk.out_scale, xb
                        )
                        - tb
                    )
                    ** 2
                )
            if continuity_weight and segments > 1:
                gaps = []
                for s, t0 in zip(starts[:-1], t0s[:-1]):
                    end = jax.vmap(
                        lambda a, u: integrate(bk, a, u, t0, seg_len)[-1]
                    )(yb[:, s], ub)
                    gaps.append(jnp.mean((end - yb[:, s + seg_len]) ** 2))
                total = total + continuity_weight * jnp.mean(jnp.stack(gaps))
            return total

        return loss_fn

    opt = optax.adam(lr)
    train, frozen = _trainable_partition(proc)
    opt_state = opt.init(train)
    key = jax.random.PRNGKey(seed + 1)
    n, m = ys.shape[0], xn.shape[0]
    length_steps = max(1, steps // len(length_strategy))
    stage_steps = max(1, length_steps // curriculum)

    for frac in length_strategy:
        # Restrict to the first `frac` of the horizon, re-segment within it.
        n_active = min(n_t, max(segments * 2, int(round(frac * n_t))))
        seg_l = max(2, n_active // segments)
        seg_starts = [i * seg_l for i in range(segments)]

        for stage in range(curriculum):
            match_len = int(
                min(
                    seg_l,
                    max(2, int(jnp.ceil(seg_l * (stage + 1) / curriculum))),
                )
            )
            loss_fn = make_loss(match_len, seg_starts, seg_l)

            @eqx.filter_jit
            def stepf(train, opt_state, yb, ub, xb, tb, loss_fn=loss_fn):
                lv, g = loss_fn(train, yb, ub, xb, tb)
                upd, opt_state = opt.update(g, opt_state)
                return eqx.apply_updates(train, upd), opt_state, lv

            for s in range(stage_steps):
                key, k1, k2 = jax.random.split(key, 3)
                bi = jax.random.choice(
                    k1, n, (min(batch_size, n),), replace=False
                )
                pi = jax.random.choice(k2, m, (min(512, m),), replace=False)
                train, opt_state, lv = stepf(
                    train, opt_state, ys[bi], us_arr[bi], xn[pi], yn[pi]
                )
                if s % max(1, stage_steps // 4) == 0:
                    log.info(
                        "shooting length=%.2f stage %d/%d (match_len=%d) "
                        "step %d, loss %.6f",
                        frac,
                        stage + 1,
                        curriculum,
                        match_len,
                        s,
                        float(lv),
                    )
    return eqx.combine(train, frozen)
