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
    den = K_n + x_n
    # Guard 0/0 without biasing the curve: an additive floor is only
    # negligible when the denominator is large next to it, so on a species
    # whose K^n is near or below it the gate collapses toward 0 -- at K=1e-8,
    # n=2 the half-saturation point returns 1e-4 instead of 0.5. The doubled
    # `where` keeps the derivative finite at den = 0.
    safe = jnp.where(den > 0.0, den, 1.0)
    return jnp.where(den > 0.0, x_n / safe, 0.0)


def hill_gate_sympy(x, K, n):
    """:func:`hill_gate` as a sympy expression, term for term — the form a
    process declares in :meth:`~hallsim.process.Process.reaction_channels`."""
    import sympy

    x_pos = sympy.Max(x, 0)
    x_n = x_pos**n
    return x_n / (K**n + x_n)


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
    den = km + s_pos
    safe = jnp.where(den > 0.0, den, 1.0)
    return jnp.where(den > 0.0, vmax * s_pos / safe, 0.0)


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
