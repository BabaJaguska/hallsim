# hallsim/training/train_neuralode.py

import jax
import jax.numpy as jnp
import optax
import equinox as eqx
from hallsim.models.neuralode import NeuralODESubModel
import diffrax
from matplotlib import pyplot as plt


class NeuralODEArray(eqx.Module):
    """
    A wrapper around NeuralODESubModel to work with jax arrays directly for training.
    """

    func: eqx.Module

    def __init__(self, baseModel: NeuralODESubModel, **kwargs):
        super().__init__(**kwargs)
        self.func = baseModel.func

    def __call__(self, ts, y0, args=None):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1] - ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys


def generate_training_data(model_fn, ts, n_vars, dataset_size, key):
    """
    model_fn: rhs function (t, y, args) -> dy/dt
    this will be used to generate training data
    in case you want a surrogate model, or just to test the training procedure.
    ts: jnp array of time steps
    dataset_size: how many samples to simulate
    """

    def simulate_single(key):
        y0 = jax.random.uniform(
            key, (n_vars,), minval=0.1, maxval=1.0
        )  # initial condition erm
        solver = diffrax.Tsit5()
        sol = diffrax.diffeqsolve(
            diffrax.ODETerm(model_fn),
            solver,
            t0=ts[0],
            t1=ts[-1],
            dt0=0.1,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return sol.ys

    keys = jax.random.split(key, dataset_size)
    ys = jax.vmap(simulate_single)(keys)
    return ts, ys


def train_neural_ode(
    ts,
    ys,
    fields=["test_field1", "test_field2"],
    width_size=64,
    depth=2,
    lr=1e-3,
    steps_strategy=(500, 1000),
    length_strategy=(0.1, 1),
    batch_size=64,
    seed=123,
):
    key = jax.random.PRNGKey(seed)
    model_key, loader_key = jax.random.split(key)
    model_temp = NeuralODESubModel(
        fields, width_size, depth, key=model_key, load_weights=False
    )
    model = NeuralODEArray(model_temp)
    optimizer = optax.adam(lr)

    @eqx.filter_value_and_grad
    def loss_fn(model, ti, yi):
        y_pred = jax.vmap(lambda y0: model(ti, y0))(yi[:, 0])
        return jnp.mean((y_pred - yi) ** 2)

    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = loss_fn(model, ti, yi)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state

    _, length_size, data_size = ys.shape
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))

    def dataloader(arrays, batch_size, key):
        dataset_size = arrays[0].shape[0]
        indices = jnp.arange(dataset_size)
        while True:
            perm = jax.random.permutation(key, indices)
            (key,) = jax.random.split(key, 1)
            start = 0
            end = batch_size
            while end <= dataset_size:
                batch = tuple(arr[perm[start:end]] for arr in arrays)
                yield batch
                start = end
                end += batch_size

    for steps, length in zip(steps_strategy, length_strategy):
        _ts = ts[: int(length * length_size)]
        _ys = ys[:, : int(length * length_size)]
        loader = dataloader((_ys,), batch_size, loader_key)
        for step in range(steps):
            (yi,) = next(loader)
            loss, model, opt_state = make_step(_ts, yi, model, opt_state)
            if step % 100 == 0:
                print(f"Step {step}, loss: {loss:.5f}")

    return model


def save_model(model, path="trained_neuralode.pkl"):
    eqx.tree_serialise_leaves(path, model.func)


if __name__ == "__main__":
    ts = jnp.linspace(0, 10, 100)
    key = jax.random.PRNGKey(42)

    # Example dummy model: oscillator
    def dummy_model(t, y, args=None):
        x = y / (1 + y**2)
        return jnp.stack([x[1], -x[0]])

    n_vars = 2  # dummy model has 2 variables
    ts, ys = generate_training_data(
        dummy_model, ts, n_vars, dataset_size=256, key=key
    )
    trained_model = train_neural_ode(ts, ys)
    save_model(trained_model)

    # Test it
    ts_longer = jnp.linspace(0, 20, 200)
    ys0 = jnp.array([0.5, 0.5])
    y_pred = trained_model(ts_longer, ys0)
    plt.plot(ts_longer, y_pred)
    plt.title("Trained Neural ODE Prediction")
    plt.xlabel("Time")
    plt.ylabel("Variables")
    plt.legend(["Var 1", "Var 2"])
    plt.show()
