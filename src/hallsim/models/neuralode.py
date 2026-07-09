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
>>> # Train via hallsim.models.neuralode.train_neuralode(...)
"""

from __future__ import annotations

import logging
from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process

log = logging.getLogger(__name__)


class NeuralODEProcess(Process):
    """Process with MLP-parameterized dynamics.

    Parameters
    ----------
    fields:
        Names of the state variables this process evolves.
        Each becomes an EVOLVED port.
    width:
        Hidden layer width.
    depth:
        Number of hidden layers.
    out_scale:
        Output scaling factor (learned or fixed).
    key:
        JAX PRNG key for weight initialization.
    """

    fields: tuple[str, ...] = ("x", "y")
    input_fields: tuple[str, ...] = ()
    mlp: eqx.nn.MLP = eqx.field(default=None)
    out_scale: jax.Array = eqx.field(default_factory=lambda: jnp.array(1.0))
    # Fixed I/O normalisation buffers (set from training data). Marked as a
    # non-trainable node via ``eqx.field(static=False)`` but partitioned out
    # of the optimiser by callers, so they persist unchanged through fits.
    in_mean: jax.Array = eqx.field(default=None)
    in_std: jax.Array = eqx.field(default=None)
    out_mean: jax.Array = eqx.field(default=None)
    out_std: jax.Array = eqx.field(default=None)

    def __init__(
        self,
        fields: Sequence[str] = ("x", "y"),
        input_fields: Sequence[str] = (),
        width: int = 64,
        depth: int = 2,
        key: jax.Array | None = None,
        final_activation=jax.nn.tanh,
    ):
        self.fields = tuple(fields)
        self.input_fields = tuple(input_fields)
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

    def ports_schema(self):
        schema = {
            name: Port(
                role=PortRole.EVOLVED, default=0.0, units="dimensionless"
            )
            for name in self.fields
        }
        for name in self.input_fields:
            schema[name] = Port(
                role=PortRole.INPUT, default=0.0, units="dimensionless"
            )
        return schema

    def derivative(self, t, state):
        # Trailing-axis stack so the process is shape-polymorphic: scalar
        # state -> (n_in,), batched -> (..., n_in). The MLP is applied per
        # row via vmap over the flattened leading axes.
        cols = [state[f] for f in self.fields] + [
            state[f] for f in self.input_fields
        ]
        inp = jnp.stack(cols, axis=-1)
        inp_n = (inp - self.in_mean) / self.in_std
        flat = inp_n.reshape(-1, inp_n.shape[-1])
        out = jax.vmap(self.mlp)(flat).reshape(
            inp_n.shape[:-1] + (len(self.fields),)
        )
        dy = self.out_scale * (out * self.out_std + self.out_mean)
        return {f: dy[..., i] for i, f in enumerate(self.fields)}


# ── Training infrastructure ─────────────────────────────────────────────


def train_neuralode(
    ts: jnp.ndarray,
    ys: jnp.ndarray,
    fields: Sequence[str] = ("x", "y"),
    width: int = 64,
    depth: int = 2,
    lr: float = 1e-3,
    steps: int = 1000,
    batch_size: int = 64,
    seed: int = 123,
) -> NeuralODEProcess:
    """Train a NeuralODEProcess on time-series data.

    Uses the multiple-shooting approach: for each sample, predict from
    y0 over the time grid and minimize MSE against the target trajectory.

    Parameters
    ----------
    ts:
        Time points, shape ``(T,)``.
    ys:
        Target trajectories, shape ``(N, T, D)`` where N=samples,
        T=time points, D=state dimension.
    fields:
        State variable names (must match dimension D).
    width, depth:
        MLP architecture.
    lr:
        Learning rate (Adam).
    steps:
        Training steps.
    batch_size:
        Mini-batch size.
    seed:
        Random seed.

    Returns
    -------
    Trained NeuralODEProcess.
    """
    import diffrax as dfx
    import optax

    key = jax.random.PRNGKey(seed)
    model_key, loader_key = jax.random.split(key)

    proc = NeuralODEProcess(
        fields=fields, width=width, depth=depth, key=model_key
    )

    # Wrap for array-based solving
    def solve_trajectory(proc, ts, y0):
        def rhs(t, y, args):
            return proc.out_scale * proc.mlp(y)

        sol = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            dfx.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
        )
        return sol.ys

    @eqx.filter_value_and_grad
    def loss_fn(proc, ti, yi):
        y_pred = jax.vmap(lambda y0: solve_trajectory(proc, ti, y0))(yi[:, 0])
        return jnp.mean((y_pred - yi) ** 2)

    optimizer = optax.adam(lr)
    opt_state = optimizer.init(eqx.filter(proc, eqx.is_inexact_array))

    @eqx.filter_jit
    def make_step(ti, yi, proc, opt_state):
        loss, grads = loss_fn(proc, ti, yi)
        updates, opt_state = optimizer.update(grads, opt_state)
        proc = eqx.apply_updates(proc, updates)
        return loss, proc, opt_state

    dataset_size = ys.shape[0]

    for step in range(steps):
        # Simple random batch
        loader_key, batch_key = jax.random.split(loader_key)
        idx = jax.random.choice(
            batch_key,
            dataset_size,
            shape=(min(batch_size, dataset_size),),
            replace=False,
        )
        yi = ys[idx]
        loss, proc, opt_state = make_step(ts, yi, proc, opt_state)
        if step % 100 == 0:
            log.info("Step %d, loss: %.6f", step, float(loss))

    return proc


def generate_training_data(
    model_fn,
    ts: jnp.ndarray,
    n_vars: int,
    dataset_size: int,
    key: jax.Array,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Generate synthetic training data from a known RHS function.

    Parameters
    ----------
    model_fn:
        RHS function ``(t, y, args) -> dy/dt``.
    ts:
        Time grid.
    n_vars:
        Number of state variables.
    dataset_size:
        Number of trajectories to simulate.
    key:
        PRNG key.

    Returns
    -------
    ``(ts, ys)`` where ys has shape ``(dataset_size, len(ts), n_vars)``.
    """
    import diffrax as dfx

    def simulate_single(key):
        y0 = jax.random.uniform(key, (n_vars,), minval=0.1, maxval=1.0)
        sol = dfx.diffeqsolve(
            dfx.ODETerm(model_fn),
            dfx.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=0.1,
            y0=y0,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
        )
        return sol.ys

    keys = jax.random.split(key, dataset_size)
    ys = jax.vmap(simulate_single)(keys)
    return ts, ys


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
    """Fit by **trajectory shooting**: integrate the learned field per
    trajectory (each with its constant input) and match the trajectory,
    backpropagating through the solve. The classic Neural-ODE objective;
    for stiff dynamics warm-start via ``init=`` from a derivative fit so the
    field is already close before any integration.
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

    def solve(proc, y0, u):
        def rhs(t, y, args=None):
            port = {f: y[i] for i, f in enumerate(fld)}
            port.update({f: u[j] for j, f in enumerate(inp)})
            d = proc.derivative(t, port)
            return jnp.stack([d[f] for f in fld])

        sol = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            dfx.Tsit5(),
            ts[0],
            ts[-1],
            ts[1] - ts[0],
            y0,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
            max_steps=100_000,
        )
        return sol.ys

    @eqx.filter_value_and_grad
    def loss_fn(proc, yb, ub):
        pred = jax.vmap(lambda y, u: solve(proc, y[0], u))(yb, ub)
        return jnp.mean((pred - yb) ** 2)

    opt = optax.adam(lr)
    opt_state = opt.init(eqx.filter(proc, eqx.is_inexact_array))

    @eqx.filter_jit
    def stepf(proc, opt_state, yb, ub):
        lv, g = loss_fn(proc, yb, ub)
        upd, opt_state = opt.update(g, opt_state)
        return eqx.apply_updates(proc, upd), opt_state, lv

    key = jax.random.PRNGKey(seed + 1)
    n = ys.shape[0]
    for s in range(steps):
        key, kb = jax.random.split(key)
        idx = jax.random.choice(kb, n, (min(batch_size, n),), replace=False)
        proc, opt_state, lv = stepf(proc, opt_state, ys[idx], us_arr[idx])
        if s % max(1, steps // 8) == 0:
            log.info("shooting-fit step %d, loss %.6f", s, float(lv))
    return proc
