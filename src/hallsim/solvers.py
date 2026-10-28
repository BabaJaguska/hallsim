"""Explicit Runge–Kutta solvers with their stages unrolled at trace time.

``diffrax``'s ``AbstractERK`` drives any tableau through one stage loop: a
slope buffer, a dynamic row of the tableau dotted against it, and the
bookkeeping around each derivative call. On a right-hand side of a few
equations those operations outnumber the arithmetic, and under ``vmap``
over a population they scale with it. The tableau is known when the
program is traced, so :func:`unrolled` writes the stages out as straight
code instead — the same slopes, weights, error estimate, interpolation and
first-same-as-last reuse as the solver it wraps, with no loop.

    solver = unrolled(dfx.Tsit5())
"""

from __future__ import annotations

import functools
from typing import ClassVar

import diffrax as dfx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import numpy as np


class _UnrolledERK(dfx.AbstractSolver):
    """The step of :attr:`inner` with its stages unrolled. Concrete classes
    come from :func:`unrolled`, which pins the interpolation to the inner
    solver's."""

    inner: dfx.AbstractERK

    term_structure: ClassVar = dfx.AbstractTerm

    def order(self, terms):
        return self.inner.order(terms)

    def error_order(self, terms):
        return self.inner.error_order(terms)

    def init(self, terms, t0, t1, y0, args):
        if self.inner.tableau.fsal:
            return jnp.array(True), jnp.zeros_like(y0)
        return None

    def func(self, terms, t0, y0, args):
        return terms.vf(t0, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        tab = self.inner.tableau
        h = t1 - t0

        def vf(t, y):
            return terms.vf(t, y, args)

        def combine(weights, slopes):
            terms_ = [
                float(w) * k for w, k in zip(weights, slopes) if w != 0.0
            ]
            return functools.reduce(jnp.add, terms_) if terms_ else None

        if tab.fsal:
            first_step, f0 = solver_state
            k = jax.lax.cond(
                eqxi.unvmap_any(first_step | made_jump),
                lambda: vf(t0, y0),
                lambda: f0,
            )
        else:
            k = vf(t0 + (tab.c1 or 0.0) * h, y0)
        ks = [k]
        for c, a_row in zip(tab.c, tab.a_lower):
            increment = combine(np.asarray(a_row), ks)
            yi = y0 if increment is None else y0 + h * increment
            ks.append(vf(t0 + float(c) * h, yi))
        if tab.ssal:
            y1 = yi
        else:
            y1 = y0 + h * combine(np.asarray(tab.b_sol), ks)
        error = combine(np.asarray(tab.b_error), ks)
        y_error = jnp.zeros_like(y0) if error is None else h * error
        # The interpolants take each slope already scaled by the step.
        dense_info = dict(y0=y0, y1=y1, k=h * jnp.stack(ks))
        new_state = (jnp.array(False), ks[-1]) if tab.fsal else None
        return y1, y_error, dense_info, new_state, dfx.RESULTS.successful


@functools.cache
def _unrolled_class(inner_cls: type) -> type:
    return type(
        f"Unrolled{inner_cls.__name__}",
        (_UnrolledERK,),
        {"interpolation_cls": inner_cls.interpolation_cls},
    )


def unrolled(solver: dfx.AbstractSolver) -> dfx.AbstractSolver:
    """``solver`` with its stages unrolled when it is an explicit
    Runge–Kutta method; any other solver, or one already unrolled, is
    returned as it is."""
    if isinstance(solver, _UnrolledERK) or not isinstance(
        solver, dfx.AbstractERK
    ):
        return solver
    return _unrolled_class(type(solver))(inner=solver)
