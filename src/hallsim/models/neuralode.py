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

from typing import Sequence

import equinox as eqx
import jax
import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process


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
    mlp: eqx.nn.MLP = eqx.field(default=None)
    out_scale: jax.Array = eqx.field(default_factory=lambda: jnp.array(1.0))

    def __init__(
        self,
        fields: Sequence[str] = ("x", "y"),
        width: int = 64,
        depth: int = 2,
        key: jax.Array | None = None,
    ):
        self.fields = tuple(fields)
        if key is None:
            key = jax.random.PRNGKey(0)
        n = len(fields)
        self.mlp = eqx.nn.MLP(
            in_size=n,
            out_size=n,
            width_size=width,
            depth=depth,
            activation=jax.nn.softplus,
            final_activation=jax.nn.tanh,
            key=key,
        )
        self.out_scale = jnp.array(1.0)

    def ports_schema(self):
        return {
            name: Port(role=PortRole.EVOLVED, default=0.0, units="dimensionless")
            for name in self.fields
        }

    def derivative(self, t, state):
        y_vec = jnp.stack([state[f] for f in self.fields])
        dy_vec = self.out_scale * self.mlp(y_vec)
        return {f: dy_vec[i] for i, f in enumerate(self.fields)}


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

    proc = NeuralODEProcess(fields=fields, width=width, depth=depth, key=model_key)

    # Wrap for array-based solving
    def solve_trajectory(proc, ts, y0):
        def rhs(t, y, args):
            return proc.out_scale * proc.mlp(y)

        sol = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            dfx.Tsit5(),
            t0=ts[0], t1=ts[-1],
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
        idx = jax.random.choice(batch_key, dataset_size, shape=(min(batch_size, dataset_size),), replace=False)
        yi = ys[idx]
        loss, proc, opt_state = make_step(ts, yi, proc, opt_state)
        if step % 100 == 0:
            print(f"Step {step}, loss: {float(loss):.6f}")

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
            t0=ts[0], t1=ts[-1],
            dt0=0.1,
            y0=y0,
            saveat=dfx.SaveAt(ts=ts),
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
        )
        return sol.ys

    keys = jax.random.split(key, dataset_size)
    ys = jax.vmap(simulate_single)(keys)
    return ts, ys
