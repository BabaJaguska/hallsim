"""An unrolled explicit Runge–Kutta solver is the solver it wraps: the same
steps, the same trajectory, the same interpolated saves."""

import diffrax as dfx
import jax
import jax.numpy as jnp
import pytest

from hallsim.solvers import unrolled


def _solve(solver, controller, dt0, ts):
    # A forced nonlinear oscillator: saves between steps exercise the
    # interpolant, the forcing the stage times.
    def rhs(t, y, args):
        x, v = y
        return jnp.stack([v, -jnp.sin(x) - 0.2 * v + 0.5 * jnp.cos(0.7 * t)])

    sol = dfx.diffeqsolve(
        dfx.ODETerm(rhs),
        solver,
        0.0,
        30.0,
        dt0,
        jnp.array([1.0, 0.0]),
        saveat=dfx.SaveAt(ts=ts),
        stepsize_controller=controller,
        max_steps=4096,
    )
    return sol.ys, sol.stats["num_steps"]


@pytest.mark.parametrize(
    "solver, controller, dt0",
    [
        (dfx.Tsit5(), dfx.PIDController(rtol=1e-6, atol=1e-9), None),
        (dfx.Dopri5(), dfx.PIDController(rtol=1e-6, atol=1e-9), None),
        (dfx.Bosh3(), dfx.PIDController(rtol=1e-6, atol=1e-9), None),
        (dfx.Heun(), dfx.ConstantStepSize(), 0.05),
    ],
    ids=["tsit5", "dopri5", "bosh3", "heun-fixed"],
)
def test_unrolled_matches_the_tableau_solver(solver, controller, dt0):
    ts = jnp.linspace(0.0, 30.0, 23) + 0.137  # off any step grid
    ts = ts.at[-1].set(30.0)
    ys, steps = jax.jit(lambda: _solve(solver, controller, dt0, ts))()
    ys_u, steps_u = jax.jit(
        lambda: _solve(unrolled(solver), controller, dt0, ts)
    )()
    assert int(steps) == int(steps_u)
    assert float(jnp.max(jnp.abs(ys - ys_u))) < 1e-10


def test_unrolled_is_idempotent_and_leaves_other_solvers_alone():
    once = unrolled(dfx.Tsit5())
    assert unrolled(once) is once
    implicit = dfx.Kvaerno5()
    assert unrolled(implicit) is implicit
    assert type(once).__name__ == "UnrolledTsit5"
    assert once.interpolation_cls is dfx.Tsit5.interpolation_cls


def test_unrolled_is_differentiable():
    def loss(scale):
        def rhs(t, y, args):
            return -scale * y

        sol = dfx.diffeqsolve(
            dfx.ODETerm(rhs),
            unrolled(dfx.Tsit5()),
            0.0,
            1.0,
            None,
            jnp.array([1.0]),
            saveat=dfx.SaveAt(t1=True),
            stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-9),
            max_steps=4096,
        )
        return sol.ys[-1, 0]

    grad = jax.grad(loss)(1.5)
    assert abs(float(grad) + jnp.exp(-1.5)) < 1e-5


def test_endpoint_dense_output_is_the_dense_solver_on_its_step_grid():
    """With every save on the step grid the interpolant is read at step
    ends alone, where the endpoints are exact; off the grid the linear
    interpolant is a different one."""
    on_grid = jnp.linspace(0.0, 30.0, 61)  # every ten steps of 0.05
    dense, steps = _solve(
        unrolled(dfx.Heun()), dfx.ConstantStepSize(), 0.05, on_grid
    )
    ends, steps_e = _solve(
        unrolled(dfx.Heun(), dense=False),
        dfx.ConstantStepSize(),
        0.05,
        on_grid,
    )
    assert int(steps) == int(steps_e)
    assert float(jnp.max(jnp.abs(dense - ends))) < 1e-12
    off_grid = on_grid[1:-1] + 0.0137
    dense_off, _ = _solve(
        unrolled(dfx.Heun()), dfx.ConstantStepSize(), 0.05, off_grid
    )
    ends_off, _ = _solve(
        unrolled(dfx.Heun(), dense=False),
        dfx.ConstantStepSize(),
        0.05,
        off_grid,
    )
    assert float(jnp.max(jnp.abs(dense_off - ends_off))) > 1e-6


def test_the_endpoint_variant_is_its_own_class_and_rebuilds_from_the_inner():
    dense = unrolled(dfx.Tsit5())
    ends = unrolled(dense, dense=False)
    assert type(ends).__name__ == "UnrolledTsit5Endpoints"
    assert ends.interpolation_cls is dfx.LocalLinearInterpolation
    assert unrolled(ends, dense=False) is ends
    assert type(unrolled(ends)) is type(dense)
    implicit = dfx.Kvaerno5()
    assert unrolled(implicit, dense=False) is implicit
