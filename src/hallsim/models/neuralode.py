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
from typing import Sequence

import copy

import equinox as eqx
import jax
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process
from hallsim.utils import h_act

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
        MLP architecture.
    key:
        PRNG key for weight init.
    """

    fields: tuple[str, ...] = ("x", "y")
    input_fields: tuple[str, ...] = ()
    field_defaults: tuple[float, ...] = eqx.field(static=True, default=())
    mlp: eqx.nn.MLP = eqx.field(default=None)
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
        width: int = 64,
        depth: int = 2,
        key: jax.Array | None = None,
        final_activation=jax.nn.tanh,
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
        self.mlp = eqx.nn.MLP(
            in_size=n_in,
            out_size=n_out,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            final_activation=final_activation,
            key=key,
        )
        self.out_scale = jnp.array(1.0)
        self.in_mean = jnp.zeros(n_in)
        self.in_std = jnp.ones(n_in)
        self.out_mean = jnp.zeros(n_out)
        self.out_std = jnp.ones(n_out)

    def with_input_driver(
        self, field: str, *, port: str, basal_param: str, hi: float,
        K: float, n: float = 2.0, basal: float | None = None,
    ) -> "NeuralODEProcess":
        """Source control input ``field`` from a Hill transform of ``port``.

        Adds an INPUT port ``port`` and, at derivative time, fills the MLP's
        ``field`` column with ``parameters[basal_param] + (hi - basal) *
        h_act(port; K, n)``. The basal lives on ``parameters`` so it stays
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
            new, "_hill_drivers",
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
                return basal + (hi - basal) * h_act(
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
        out = jax.vmap(self.mlp)(flat).reshape(
            inp_n.shape[:-1] + (len(self.fields),)
        )
        dy = self.out_scale * (out * self.out_std + self.out_mean)
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


def _identity(z):
    return z


def simulate_conditioned(
    rhs_for_input,
    ts: jnp.ndarray,
    inputs: jnp.ndarray,
    n_ics: int = 3,
    y0_range: tuple[float, float] = (0.0, 1.0),
    key: jax.Array | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Trajectories from a known input-conditioned RHS, for recovery fits.

    Parameters
    ----------
    rhs_for_input:
        ``u -> rhs(t, y, args) -> dy/dt`` — the mechanistic RHS at input ``u``.
    ts:
        Time grid, shape ``(T,)``.
    inputs:
        Input values to condition on, shape ``(M, U)``.
    n_ics:
        Initial conditions sampled per input value.

    Returns
    -------
    ``(ys, us)`` with ``ys`` shape ``(M*n_ics, T, D)`` and ``us`` shape
    ``(M*n_ics, U)`` — the per-trajectory constant input, aligned with ``ys``.
    """
    import diffrax as dfx

    if key is None:
        key = jax.random.PRNGKey(0)
    inputs = jnp.asarray(inputs)
    dim = _probe_dim(rhs_for_input(inputs[0]), ts)
    ys, us = [], []
    for u in inputs:
        rhs = rhs_for_input(u)
        for _ in range(n_ics):
            key, k = jax.random.split(key)
            y0 = jax.random.uniform(
                k, (dim,), minval=y0_range[0], maxval=y0_range[1]
            )
            sol = dfx.diffeqsolve(
                dfx.ODETerm(rhs),
                dfx.Tsit5(),
                ts[0],
                ts[-1],
                ts[1] - ts[0],
                y0,
                saveat=dfx.SaveAt(ts=ts),
                stepsize_controller=dfx.PIDController(rtol=1e-7, atol=1e-9),
                max_steps=200_000,
            )
            ys.append(sol.ys)
            us.append(u)
    return jnp.stack(ys), jnp.stack(us)


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
        final_activation=_identity,
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

    mlp = proc.mlp
    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(mlp, eqx.is_inexact_array))

    @eqx.filter_jit
    def stepf(mlp, opt_state, xb, yb):
        def loss(m):
            return jnp.mean((jax.vmap(m)(xb) - yb) ** 2)

        lv, g = eqx.filter_value_and_grad(loss)(mlp)
        upd, opt_state = opt.update(g, opt_state)
        return eqx.apply_updates(mlp, upd), opt_state, lv

    key = jax.random.PRNGKey(seed + 1)
    n = xn.shape[0]
    for s in range(steps):
        key, kb = jax.random.split(key)
        idx = jax.random.choice(kb, n, (min(batch_size, n),), replace=False)
        mlp, opt_state, lv = stepf(mlp, opt_state, xn[idx], yn[idx])
        if s % max(1, steps // 8) == 0:
            log.info("derivative-fit step %d, loss %.6f", s, float(lv))
    return eqx.tree_at(lambda p: p.mlp, proc, mlp)


def fit_neuralode_shooting(
    ts: jnp.ndarray,
    ys: jnp.ndarray,
    us: jnp.ndarray | None = None,
    *,
    fields: Sequence[str],
    input_fields: Sequence[str] = (),
    segments: int = 1,
    curriculum: int = 1,
    physics_weight: float = 0.0,
    continuity_weight: float = 0.0,
    width: int = 96,
    depth: int = 3,
    lr: float = 1e-3,
    steps: int = 800,
    batch_size: int = 32,
    init: NeuralODEProcess | None = None,
    seed: int = 0,
    rtol: float = 1e-4,
    atol: float = 1e-7,
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
    - ``curriculum`` — grow the matched horizon: over ``curriculum`` stages
      match a progressively longer prefix of each window (short→full), so
      the field is anchored on the easy near-term dynamics before the hard
      long-horizon phase. Slope-free.
    - ``physics_weight`` — **collocation regularizer**
      ``λ·‖f_θ(z) − ż‖²`` on the data's finite-difference slopes (the same
      residual :func:`fit_neuralode_derivative` minimizes). Supplies the
      absolute vector-field magnitude constraint pure shooting lacks — the
      strongest stabilizer, but slope quality degrades with noise/sparsity.
    - ``continuity_weight`` — tie each window's integrated endpoint to the
      next window's observed start, for global consistency across boundaries.

    ``segments=1, curriculum=1, physics_weight=0`` is classic single
    shooting. Warm-start via ``init=`` from a derivative fit for stiff
    dynamics. See MPINeuralODE (arXiv:2605.13305) for the combined recipe.
    """
    import diffrax as dfx
    import optax

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
    seg_len = max(2, n_t // segments)
    starts = jnp.arange(segments) * seg_len

    def integrate(proc, y0, u, seg_ts):
        def rhs(t, y, args=None):
            port = {f: y[i] for i, f in enumerate(fld)}
            port.update({f: u[j] for j, f in enumerate(inp)})
            d = proc.derivative(t, port)
            return jnp.stack([d[f] for f in fld])

        return dfx.diffeqsolve(
            dfx.ODETerm(rhs), dfx.Tsit5(), seg_ts[0], seg_ts[-1],
            (seg_ts[-1] - seg_ts[0]) / max(seg_ts.shape[0] - 1, 1),
            y0, saveat=dfx.SaveAt(ts=seg_ts),
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            max_steps=100_000,
        ).ys

    def make_loss(match_len):
        def per_window(proc, yb, ub, s):
            seg_ts = jax.lax.dynamic_slice_in_dim(ts, s, match_len)
            y0 = jax.vmap(lambda y: y[s])(yb)
            pred = jax.vmap(lambda a, u: integrate(proc, a, u, seg_ts))(y0, ub)
            tgt = jax.vmap(
                lambda y: jax.lax.dynamic_slice_in_dim(y, s, match_len))(yb)
            return jnp.mean((pred - tgt) ** 2)

        @eqx.filter_value_and_grad
        def loss_fn(proc, yb, ub, xb, tb):
            traj = jnp.mean(jax.vmap(
                lambda s: per_window(proc, yb, ub, s))(starts))
            total = traj
            if physics_weight:
                total = total + physics_weight * jnp.mean(
                    (jax.vmap(proc.mlp)(xb) - tb) ** 2)
            if continuity_weight and segments > 1:
                def gap(s):
                    seg_ts = jax.lax.dynamic_slice_in_dim(ts, s, seg_len)
                    y0 = jax.vmap(lambda y: y[s])(yb)
                    end = jax.vmap(
                        lambda a, u: integrate(proc, a, u, seg_ts)[-1])(y0, ub)
                    nxt = jax.vmap(lambda y: y[s + seg_len])(yb)
                    return jnp.mean((end - nxt) ** 2)
                total = total + continuity_weight * jnp.mean(
                    jax.vmap(gap)(starts[:-1]))
            return total

        return loss_fn

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(proc, eqx.is_inexact_array))
    key = jax.random.PRNGKey(seed + 1)
    n, m = ys.shape[0], xn.shape[0]
    stage_steps = max(1, steps // curriculum)

    for stage in range(curriculum):
        match_len = int(jnp.ceil(seg_len * (stage + 1) / curriculum))
        match_len = int(min(seg_len, max(2, match_len)))
        loss_fn = make_loss(match_len)

        @eqx.filter_jit
        def stepf(proc, opt_state, yb, ub, xb, tb, loss_fn=loss_fn):
            lv, g = loss_fn(proc, yb, ub, xb, tb)
            upd, opt_state = opt.update(g, opt_state)
            return eqx.apply_updates(proc, upd), opt_state, lv

        for s in range(stage_steps):
            key, k1, k2 = jax.random.split(key, 3)
            bi = jax.random.choice(k1, n, (min(batch_size, n),), replace=False)
            pi = jax.random.choice(k2, m, (min(512, m),), replace=False)
            proc, opt_state, lv = stepf(
                proc, opt_state, ys[bi], us_arr[bi], xn[pi], yn[pi])
            if s % max(1, stage_steps // 4) == 0:
                log.info("shooting stage %d/%d (match_len=%d) step %d, "
                         "loss %.6f", stage + 1, curriculum, match_len, s,
                         float(lv))
    return proc
