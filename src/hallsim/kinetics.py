"""Differentiable rate-law primitives for composing Process derivatives.

The standard systems-biology building blocks, as JIT-friendly, end-to-end
differentiable JAX functions with one canonical implementation each — so a
Process author (or an AI agent) reaches for a named primitive instead of
re-deriving `x**n / (K**n + x**n)` inline. Inputs that must be non-negative
(concentrations, activities) are clamped, since these laws are physically
defined only there.
"""

from __future__ import annotations

import jax.numpy as jnp

_EPS = 1e-12


def hill_gate(x: jnp.ndarray, K: jnp.ndarray, n: jnp.ndarray) -> jnp.ndarray:
    """Hill activation ``x^n / (K^n + x^n)`` — a soft switch bounded in [0, 1].

    Half-open at ``x = K``; ``n`` sets the steepness. Negative ``x`` clamps
    to 0. The workhorse gate for cooperative activation and cross-model
    coupling edges.
    """
    x_pos = jnp.maximum(x, 0.0)
    x_n = x_pos**n
    K_n = K**n
    return x_n / (K_n + x_n + _EPS)


def hill_inhibition(
    x: jnp.ndarray, K: jnp.ndarray, n: jnp.ndarray
) -> jnp.ndarray:
    """Hill repression ``K^n / (K^n + x^n) = 1 - hill_gate(x, K, n)``.

    Bounded in [0, 1]; falls from 1 toward 0 as ``x`` rises past ``K``.
    """
    return 1.0 - hill_gate(x, K, n)


def michaelis_menten(
    s: jnp.ndarray, vmax: jnp.ndarray, km: jnp.ndarray
) -> jnp.ndarray:
    """Michaelis-Menten flux ``vmax · s / (Km + s)``.

    Saturating enzyme kinetics: linear in substrate ``s`` at ``s ≪ Km``,
    approaching ``vmax`` at ``s ≫ Km``. Negative ``s`` clamps to 0.
    (``hill_gate`` with ``n = 1`` scaled by ``vmax`` is the same curve; this
    is the enzyme-kinetics spelling with the conventional argument names.)
    """
    s_pos = jnp.maximum(s, 0.0)
    return vmax * s_pos / (km + s_pos + _EPS)


def mass_action(k: jnp.ndarray, *reactants: jnp.ndarray) -> jnp.ndarray:
    """Mass-action rate ``k · ∏ reactants``.

    The reaction flux for elementary kinetics: first-order decay is
    ``mass_action(k, x)``, a bimolecular association ``mass_action(k, a, b)``,
    a zeroth-order source ``mass_action(k)``. Reactants clamp to 0.
    """
    flux = jnp.asarray(k)
    for r in reactants:
        flux = flux * jnp.maximum(r, 0.0)
    return flux
