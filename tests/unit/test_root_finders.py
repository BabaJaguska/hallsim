"""Accuracy and differentiation of the optional per-step chord solver."""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.root_finders import StepChord
from hallsim.scheduler import Scheduler


class NonlinearDecay(Process):
    rate: float = 100.0

    def ports_schema(self):
        return {
            "x": Port(role=PortRole.EVOLVED, default=1.0, units="uM"),
            "y": Port(role=PortRole.EVOLVED, default=1.0, units="uM"),
        }

    def derivative(self, t, state):
        return {"x": -self.rate * state["x"] ** 2, "y": -state["y"]}


@pytest.mark.parametrize("mode", ["forward", "reverse"])
def test_step_chord_batched_trajectory_and_parameter_gradient(mode):
    comp = Composite(
        processes={"decay": NonlinearDecay()},
        topology={"decay": {"x": "x", "y": "y"}},
        validate=False,
        semantic_validation=False,
    )
    sched = Scheduler(
        solver=dfx.Kvaerno5(root_finder=StepChord(rtol=1e-9, atol=1e-10)),
        rtol=1e-8,
        atol=1e-10,
    )
    population = jnp.array([[0.5, 1.0], [1.0, 2.0], [2.0, 0.5]])
    adjoint = (
        dfx.ForwardMode()
        if mode == "forward"
        else dfx.RecursiveCheckpointAdjoint()
    )
    plan = sched.plan(
        comp, (0.0, 1.0), y0=population, save_dt=0.1, adjoint=adjoint
    )
    result = sched.run(plan, y0=population)
    times = result.ts[:, None]
    expected_x = population[:, 0] / (1 + 100 * population[:, 0] * times)
    expected_y = population[:, 1] * jnp.exp(-times)
    assert jnp.all(result.ok)
    assert jnp.allclose(result.ys[..., 0], expected_x, rtol=2e-6, atol=1e-8)
    assert jnp.allclose(result.ys[..., 1], expected_y, rtol=2e-6, atol=1e-8)

    @eqx.filter_jit
    def objective(rate):
        varied = eqx.tree_at(lambda c: c.processes["decay"].rate, comp, rate)
        return (
            sched.run(plan, y0=population, params_from=varied)
            .ys[-1, :, 0]
            .sum()
        )

    rate = jnp.asarray(100.0)
    if mode == "forward":
        _, derivative = jax.jvp(objective, (rate,), (jnp.ones_like(rate),))
    else:
        derivative = jax.grad(objective)(rate)
    expected_derivative = jnp.sum(
        -(population[:, 0] ** 2) / (1 + rate * population[:, 0]) ** 2
    )
    assert jnp.isfinite(derivative)
    assert jnp.allclose(derivative, expected_derivative, rtol=2e-5, atol=1e-9)
