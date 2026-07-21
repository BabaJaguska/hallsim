"""Exact fixed point of a composite's unperturbed baseline via Newton.

A perturbation experiment's baseline is the unperturbed steady state (cells at
homeostasis before stimulus). That condition sits at a fixed point ``f(y*,θ)=0``
— any limit cycle belongs to the perturbation — so it is found algebraically by
Newton, not by integrating toward it. The gradient ``dy*/dθ`` follows from the
implicit function theorem (one linear solve at ``y*``), wired via
:func:`jax.lax.custom_root` so it stays differentiable without an unrolled
adjoint. Accumulator observer states (RunningIntegral outputs, ``dA/dt≠0``) have
no fixed point; they are held at zero and dropped from the residual — nothing
reads them back, so the real fixed point is unaffected.

Conserved moieties (free+bound totals in the SBML kinetics) make the residual
Jacobian rank-deficient: the level along a conserved direction is fixed by the
initial condition, not by ``f=0``. Borrowing the standard moiety reduction
(Reder 1988 / COPASI), the conservation laws ``L`` (left null space of the
Jacobian, structural so computed once) are re-introduced as constraints
``L·(y−y_ref)=0`` with ``y_ref`` the initial state — pinning the conserved totals
and restoring a full-rank Newton system.
"""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
import numpy as np

log = logging.getLogger(__name__)


def accumulator_mask(composite, keys: list[str]) -> jnp.ndarray:
    """Boolean mask over ``keys`` marking RunningIntegral outputs."""
    from hallsim.models.running_integral import RunningIntegral

    positions = [
        keys.index(composite.topology[name]["integral"])
        for name, proc in composite.processes.items()
        if isinstance(proc, RunningIntegral)
        and composite.topology.get(name, {}).get("integral") in keys
    ]
    mask = jnp.zeros(len(keys), dtype=bool)
    return mask.at[jnp.asarray(positions)].set(True) if positions else mask


def _residual_fn(composite, mask):
    rhs, _ = composite.build_rhs()
    return lambda y: jnp.where(mask, y, rhs(0.0, y))


def warn_if_time_dependent(composite, y, dt: float = 1.0) -> bool:
    """Warn (COPASI-style) if the RHS is explicitly time-dependent at ``y`` —
    a Newton fixed point is meaningless then. Returns True if autonomous."""
    rhs, _ = composite.build_rhs()
    autonomous = float(jnp.max(jnp.abs(rhs(0.0, y) - rhs(dt, y)))) < 1e-9
    if not autonomous:
        log.warning(
            "steady_state: the RHS is explicitly time-dependent at this "
            "condition; a Newton fixed point is not meaningful. Equilibrate an "
            "autonomous condition (e.g. the unperturbed control, where a timed "
            "input vanishes)."
        )
    return autonomous


def conservation_laws(composite, y, mask=None, rcond: float = 1e-9):
    """Conservation-law matrix ``L`` (rows = conserved combinations), the left
    null space of the residual Jacobian at ``y``. Structural, so evaluate once
    at a concrete state and reuse. Returns an ``(n_laws, n_state)`` array."""
    keys = composite.store_keys()
    mask = accumulator_mask(composite, keys) if mask is None else mask
    warn_if_time_dependent(composite, y)
    jac = np.asarray(jax.jacfwd(_residual_fn(composite, mask))(y))
    u, s, _ = np.linalg.svd(jac)
    null = np.argsort(s)[: int(np.sum(s < rcond * s[0]))]
    return jnp.asarray(u[:, null].T)


def steady_state(
    composite,
    y_guess: jnp.ndarray | None = None,
    *,
    laws: jnp.ndarray | None = None,
    y_ref: jnp.ndarray | None = None,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> jnp.ndarray:
    """Fixed point of ``composite``'s dynamics, differentiable in its params.

    ``y_guess`` seeds the damped Newton iteration (default
    ``initial_state_vec``); seed near ``y*`` — e.g. a short forward pre-solve —
    for a stiff baseline. ``laws`` are the conservation laws (from
    :func:`conservation_laws`; computed from ``y_guess`` if omitted) and
    ``y_ref`` (default the initial state) fixes the conserved totals. Returns
    the full state vector (accumulators zero).
    """
    keys = composite.store_keys()
    mask = accumulator_mask(composite, keys)
    residual = _residual_fn(composite, mask)
    ic = composite.initial_state_vec(keys)
    y0 = ic if y_guess is None else y_guess
    y_ref = ic if y_ref is None else y_ref
    laws = conservation_laws(composite, y0, mask) if laws is None else laws

    def g(y):
        return residual(y) + laws.T @ (laws @ (y - y_ref))

    def solve(fn, guess):
        def body(state):
            y, i, _ = state
            f = fn(y)
            dy = jnp.linalg.solve(jax.jacfwd(fn)(y), f)
            f0 = jnp.max(jnp.abs(f))

            def damp(c):
                k, best = c
                ek = jnp.max(jnp.abs(fn(y - dy * (0.5**k))))
                return k + 1, jnp.where((ek < f0) & (best < 0), k, best)

            _, kbest = jax.lax.while_loop(
                lambda c: (c[0] <= 8) & (c[1] < 0), damp, (0, -1)
            )
            y_new = y - dy * jnp.where(kbest < 0, 1.0, 0.5**kbest)
            return y_new, i + 1, jnp.max(jnp.abs(fn(y_new)))

        y, _, _ = jax.lax.while_loop(
            lambda s: (s[1] < max_iter) & (s[2] > tol),
            body,
            (guess, 0, jnp.max(jnp.abs(fn(guess)))),
        )
        return y

    def tangent_solve(gg, b):
        return jnp.linalg.solve(jax.jacfwd(gg)(jnp.zeros_like(b)), b)

    return jax.lax.custom_root(g, y0, solve, tangent_solve)
