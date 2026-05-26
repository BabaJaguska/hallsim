"""Generic biology / numerical helpers used across HallSim.

Currently exposes Hill activation / inhibition primitives — these come up
inside several Process derivative methods (ERiQ, SASPmTORActivator, future
models). JIT-friendly; differentiable; clamps negative inputs because
Hill kinetics are biophysically defined only for non-negative
concentrations / activities.
"""

from __future__ import annotations

import jax.numpy as jnp


def h_act(x: jnp.ndarray, K: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """Hill activation: ``x^n / (K^n + x^n)``. Bounded in [0, 1].

    Inputs are clamped non-negative before the Hill computation; negative
    arguments return 0.
    """
    x_pos = jnp.maximum(x, 0.0)
    x_n = x_pos**n
    K_n = K**n
    return x_n / (K_n + x_n + 1e-12)


def h_inhib(x: jnp.ndarray, K: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """Hill inhibition: ``K^n / (K^n + x^n) = 1 - H_act(x)``. Bounded
    in [0, 1]."""
    return 1.0 - h_act(x, K, n)
