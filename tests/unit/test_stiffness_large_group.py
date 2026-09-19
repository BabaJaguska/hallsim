"""A group too large for a dense Jacobian still gets a routing verdict."""

import jax.numpy as jnp

from hallsim.composite import Composite
from hallsim.process import Port, PortRole, Process
from hallsim.stiffness import DENSE_JACOBIAN_MAX_DIM, analyze_groups
import equinox as eqx


class Ring(Process):
    """dx_i/dt = (x_{i-1} - x_i)/tau around a ring. Every eigenvalue lies on
    the circle |lambda + 1/tau| = 1/tau, so the largest-magnitude ones sit
    in a cluster near -2/tau that differ in the fourth digit: an Arnoldi
    run at machine-precision tolerance never separates them."""

    n: int = 1024
    tau: float = 1.0
    timescale: float = eqx.field(static=True, default=1.0)
    k: float = 1.0

    def ports_schema(self):
        return {
            f"x{i}": Port(role=PortRole.EVOLVED, default=float(i % 3))
            for i in range(self.n)
        }

    def derivative(self, t, state):
        x = jnp.stack([state[f"x{i}"] for i in range(self.n)])
        dx = self.k * (jnp.roll(x, 1) - x) / self.tau
        return {f"x{i}": dx[i] for i in range(self.n)}


def _verdict(k):
    n = 2 * DENSE_JACOBIAN_MAX_DIM
    comp = Composite(
        processes={"ring": Ring(n=n, k=k)},
        topology={"ring": {f"x{i}": f"x{i}" for i in range(n)}},
        validate=False,
        semantic_validation=False,
    )
    verdict = analyze_groups(comp, groups={"g": ["ring"]}, dt=1.0)["g"]
    assert verdict.dim == n
    return verdict


def test_a_slow_large_group_is_certified_without_an_eigenvalue_estimate():
    verdict = _verdict(k=1.0)
    # Gershgorin on the ring: |J_ii| + sum|J_ij| = 2/tau, exactly the abscissa
    assert verdict.bounded
    assert not verdict.stiff
    assert 1.8 <= verdict.spectral_abscissa <= 2.05


def test_a_stiff_large_group_goes_to_the_estimate():
    verdict = _verdict(k=1e3)
    assert not verdict.bounded
    assert verdict.stiff
    assert 1800 <= verdict.spectral_abscissa <= 2050
