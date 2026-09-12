"""The Scheduler's chord, and the optional per-step chord solver."""

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx
import pytest

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.root_finders import Chord, StepChord
from hallsim.scheduler import Scheduler


def test_the_scheduler_chord_stops_diverging_with_a_finite_iterate():
    """y² − 4 from 0.1: the chord's Jacobian at the start is 0.2, so each
    update squares the iterate and it overflows well inside ten steps.
    optimistix's chord hands the overflow back; the Scheduler's refuses the
    first update that grows the residual, returns the start, and says so."""

    def fn(y, args):
        return y**2 - 4.0

    y0 = jnp.asarray(0.1)
    theirs = optx.root_find(
        fn, optx.Chord(rtol=1e-6, atol=1e-6), y0, throw=False, max_steps=10
    )
    ours = optx.root_find(
        fn, Chord(rtol=1e-6, atol=1e-6), y0, throw=False, max_steps=10
    )
    assert not jnp.isfinite(theirs.value)
    assert jnp.isfinite(ours.value)
    assert ours.result == optx.RESULTS.nonlinear_divergence
    assert isinstance(Scheduler().implicit_solver.root_finder, Chord)


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
