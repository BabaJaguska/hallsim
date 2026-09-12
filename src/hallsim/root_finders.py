"""Implicit-stage root finders: the Scheduler's default and the overrides."""

from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import lineax as lx
import optimistix as optx
from jax import lax


def _all_finite(tree) -> jax.Array:
    leaves = [jnp.all(jnp.isfinite(leaf)) for leaf in jtu.tree_leaves(tree)]
    return jnp.all(jnp.asarray(leaves)) if leaves else jnp.asarray(True)


def _select(keep, new, old):
    return jtu.tree_map(lambda a, b: jnp.where(keep, a, b), new, old)


class _ChordState(eqx.Module):
    f: Any  # residual at the current iterate
    f_norm: jax.Array
    aux: Any
    linear_state: tuple
    diff: Any
    diffsize: jax.Array
    result: optx.RESULTS
    step: jax.Array


class Chord(optx.Chord):
    """optimistix's chord — one Jacobian per nonlinear solve, Cauchy
    termination — that keeps the residual of its current iterate and
    refuses an update that makes it non-finite or more than twice as large.

    A chord at a stale Jacobian on a stiff step can grow geometrically
    for its whole iteration budget: on the multi-hallmark launch it reaches
    1e143 in ten iterations. The relative update size saturates at
    ``1/rtol`` while it does, so a divergence test on the update rate sees a
    rate of exactly one. The forward solve only rejects such a step; the
    reverse pass factorises the Jacobian at the returned iterate, and the
    NaN that comes out of an overflowed one times the rejected branch's zero
    cotangent is NaN, so every gradient is NaN. Here every returned iterate
    is one whose residual was evaluated and found finite, and the iteration
    stops with ``nonlinear_divergence`` at the first update that would
    grow it. That costs one residual evaluation per solve, at the start.

    ``growth_limit`` is how much the residual norm may grow in one update
    before the update is refused. A diverging chord grows it by orders of
    magnitude per iteration; a converging one on a curved residual can grow
    it once before it falls, and refusing that costs steps (a limit of 2
    took 11% more steps on the multi-hallmark composite, 10 and above the
    same steps as the unguarded chord).
    """

    growth_limit: float = 10.0

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        del options, f_struct, aux_struct
        jac = lx.linearise(
            lx.JacobianLinearOperator(
                lambda _y, _args: fn(_y, _args)[0], y, args, tags=tags
            )
        )
        linear_state = lax.stop_gradient(
            self.linear_solver.init(jac, options={})
        )
        f, aux = fn(y, args)
        return _ChordState(
            f=f,
            f_norm=self.norm(f),
            aux=aux,
            linear_state=(jac, linear_state),
            diff=jtu.tree_map(lambda leaf: jnp.full_like(leaf, jnp.inf), y),
            diffsize=jnp.asarray(
                jnp.inf, dtype=jnp.result_type(*jtu.tree_leaves(y))
            ),
            result=optx.RESULTS.successful,
            step=jnp.array(0),
        )

    def step(self, fn, y, args, options, state, tags):
        del tags
        jac, linear_state = state.linear_state
        sol = lx.linear_solve(
            jac,
            state.f,
            self.linear_solver,
            state=lax.stop_gradient(linear_state),
            throw=False,
        )
        diff = sol.value
        new_y = jtu.tree_map(lambda a, b: a - b, y, diff)
        lower, upper = options.get("lower"), options.get("upper")
        if lower is not None:
            new_y = jtu.tree_map(lambda a, b: jnp.clip(a, min=b), new_y, lower)
        if upper is not None:
            new_y = jtu.tree_map(lambda a, b: jnp.clip(a, max=b), new_y, upper)
        if lower is not None or upper is not None:
            diff = jtu.tree_map(lambda a, b: a - b, y, new_y)
        f_new, aux_new = fn(new_y, args)
        f_norm_new = self.norm(f_new)
        accept = (
            _all_finite(new_y)
            & _all_finite(f_new)
            & (f_norm_new <= self.growth_limit * state.f_norm)
        )
        scale = jtu.tree_map(
            lambda leaf: self.atol + self.rtol * jnp.abs(leaf), new_y
        )
        diffsize = self.norm(jtu.tree_map(lambda a, s: a / s, diff, scale))
        result = optx.RESULTS.where(
            accept,
            optx.RESULTS.promote(sol.result),
            optx.RESULTS.nonlinear_divergence,
        )
        new_state = _ChordState(
            f=_select(accept, f_new, state.f),
            f_norm=jnp.where(accept, f_norm_new, state.f_norm),
            aux=_select(accept, aux_new, state.aux),
            linear_state=state.linear_state,
            diff=_select(accept, diff, state.diff),
            diffsize=jnp.where(accept, diffsize, state.diffsize),
            result=result,
            step=state.step + 1,
        )
        return _select(accept, new_y, y), new_state, new_state.aux

    def terminate(self, fn, y, args, options, state, tags):
        del fn, args, options, tags
        y_scale = jtu.tree_map(
            lambda leaf: self.atol + self.rtol * jnp.abs(leaf), y
        )
        y_converged = (
            self.norm(
                jtu.tree_map(lambda a, s: jnp.abs(a) / s, state.diff, y_scale)
            )
            < 1
        )
        f_converged = state.f_norm < self.atol
        failed = state.result != optx.RESULTS.successful
        return failed | (y_converged & f_converged), state.result

    def postprocess(self, fn, y, aux, args, options, state, tags, result):
        del fn, args, options, state, tags, result
        return y, aux, {}


class StepChord(optx.Chord):
    """Reuse Diffrax's per-step Jacobian with Chord's convergence criteria.

    Diffrax ESDIRK solvers supply a precomputed root-finder ``init_state``
    to reuse a Jacobian and its factorization across implicit stages.
    Optimistix Chord normally ignores that hint and initializes each stage
    independently. Honoring it here retains Chord's Cauchy termination rule,
    unlike Diffrax VeryChord's convergence-rate-based termination.

    This is opt-in: reuse can reduce large GPU batch costs but increase
    small-batch runtime or rejected steps on nonlinear stiff problems.
    Accuracy and throughput must be checked for the intended workload.
    There is no reuse across integration steps.

    Example::

        Scheduler(implicit_solver=dfx.Kvaerno5(
            root_finder=StepChord(rtol=1e-8, atol=1e-6)))
    """

    def init(self, fn, y, args, options, f_struct, aux_struct, tags):
        if "init_state" in options:
            return options["init_state"]
        return super().init(fn, y, args, options, f_struct, aux_struct, tags)
