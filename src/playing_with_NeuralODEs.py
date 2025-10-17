import time
import jax
import jax.nn as jnn
import jax.numpy as jnp
import jax.random as jrandom
import optax
from matplotlib import pyplot as plt
import equinox as eqx
import diffrax


class Func(eqx.Module):
    out_scale: jax.Array
    mlp: eqx.nn.MLP

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        # keys = jrandom.split(key, 3)
        self.out_scale = jnp.array(1.0)
        self.mlp = eqx.nn.MLP(
            in_size=data_size,
            out_size=data_size,
            width_size=width_size,
            depth=depth,
            activation=jnn.softplus,
            final_activation=jax.nn.tanh,
            key=key,
        )

    def __call__(self, t, y, args):
        return self.out_scale * self.mlp(y)

class NeuralODE(eqx.Module):
    func: Func

    def __init__(self, data_size, width_size, depth, *, key, **kwargs):
        super().__init__(**kwargs)
        self.func = Func(data_size, width_size, depth, key=key)

    def __call__(self, ts, y0):
        solution = diffrax.diffeqsolve(
            diffrax.ODETerm(self.func),
            diffrax.Tsit5(),
            t0=ts[0],
            t1=ts[-1],
            dt0=ts[1]-ts[0],
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
            saveat=diffrax.SaveAt(ts=ts),
        )
        return solution.ys
    
# toy dataset of nonlinear oscillators
def _get_data(ts, * , key):
    y0 = jrandom.uniform(key, (2,), minval=0.1, maxval=1)

    def f(_t,y,args):
        x = y / (1+y**2)
        # x = jnp.sin(y)
        # x  = 1/(y**2+1) 
        out = jnp.stack([x[1], -x[0]], axis=-1)
        # lotka voltera
        # out1 = 0.5 * y[0] - 0.2 * y[0] * y[1]
        # out2 = -0.5 * y[1] + 0.2 * y[0] * y[1]
        # out = jnp.stack([out1, out2], axis=-1)
        
        return out
    
    solver = diffrax.Tsit5()
    dt0 = 0.1
    saveat = diffrax.SaveAt(ts=ts)
    sol = diffrax.diffeqsolve(
        diffrax.ODETerm(f),
        solver,
        t0=ts[0],
        t1=ts[-1],
        dt0=dt0,
        y0=y0,
        saveat=saveat,
    )
    return sol.ys

def get_data(dataset_size, *, key):
    ts = jnp.linspace(0, 10, 200)
    key = jrandom.split(key, dataset_size)
    ys = jax.vmap(lambda key: _get_data(ts, key=key))(key)
    return ts, ys
    

def dataloader(arrays, batch_size, *, key):
    dataset_size = arrays[0].shape[0]
    assert all(array.shape[0] == dataset_size for array in arrays), "All arrays must have the same first dimension size."
    indices = jnp.arange(dataset_size)
    while True:
        perm = jrandom.permutation(key, indices)
        (key,) = jrandom.split(key, 1)
        start = 1
        end = batch_size
        while end < dataset_size:
            batch_perm = perm[start:end]
            yield tuple(array[batch_perm] for array in arrays)
            start = end
            end = start + batch_size

def main(
        dataset_size=256,
        batch_size=64,
        lr=1e-3,
        steps_strategy=(500, 500),
        length_strategy=(0.1, 1),
        width_size=64,
        depth=2,
        seed=12345,
        plot=True,
        print_every=100,
):
    key = jrandom.PRNGKey(seed)
    data_key, model_key, loader_key = jrandom.split(key, 3)
    ts, ys = get_data(dataset_size, key=data_key)
    _, length_size, data_size = ys.shape
    model = NeuralODE(data_size, width_size, depth, key=model_key)
    optimizer = optax.adam(lr)

    # up until step 500 we train only on the first 10% of each time series
    # this is a standard trick to avoid getting acught in a local minimum
    # CURRICULUM LEARNING

    @eqx.filter_value_and_grad
    def grad_loss(model, ti, yi):
        y_pred = jax.vmap(model, in_axes=(None, 0))(ti, yi[:,0])
        return jnp.mean((y_pred - yi)**2)
    
    @eqx.filter_jit
    def make_step(ti, yi, model, opt_state):
        loss, grads = grad_loss(model, ti, yi)
        updates, opt_state = optimizer.update(grads, opt_state)
        model = eqx.apply_updates(model, updates)
        return loss, model, opt_state
    
    for steps, length in zip(steps_strategy, length_strategy):
        opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
        _ts = ts[: int(length * length_size)]
        _ys = ys[:, : int(length * length_size)]
        for step, (yi,) in zip(
            range(steps),
            dataloader((_ys,), batch_size, key=loader_key),
        ):
            start = time.time()
            loss, model, opt_state = make_step(_ts, yi, model, opt_state)
            end = time.time()
            if step % print_every == 0:
                print(f"step {step}, loss {loss}, time {end-start}")

    if plot:
        plt.plot(ts, ys[0, :, 0], c="dodgerblue", label="Real")
        plt.plot(ts, ys[0, :, 1], c="dodgerblue")
        model_y = model(ts, ys[0, 0])
        plt.plot(ts, model_y[:, 0], "--", c="orangered", label="Model")
        plt.plot(ts, model_y[:, 1], "--", c="orangered")
        plt.legend()
        plt.tight_layout()
        plt.savefig("neural_ode.png", dpi=300)
        plt.show()

    return ts, ys, model



if __name__ == "__main__":
    ts, ys, model = main(steps_strategy=(500, 500), 
                         length_strategy=(0.1, 1),
                         print_every=100)
    

    ts_long = jnp.linspace(0, 20, 200)
    y_hat_long = model(ts_long, ys[0, 0])

    plt.plot(ts_long, y_hat_long[:, 0], "--", c="orangered", label="Model")
    plt.plot(ts_long, y_hat_long[:, 1], "--", c="orangered")
    plt.legend()
    plt.tight_layout()
    plt.savefig("neural_ode_long.png", dpi=300)
    plt.show()
    print("Done")
