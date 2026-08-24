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
(Reder 1988 / COPASI), the conservation laws ``L`` are re-introduced as
constraints ``L·(y−y_ref)=0`` with ``y_ref`` the initial state — pinning the
conserved totals and restoring a full-rank Newton system.

``L`` comes from the stoichiometry ``N``, not from the Jacobian. Both are
rank-deficient along a moiety, but the Jacobian is *also* rank-deficient along
anything merely slow at the state and parameters it was evaluated at — so a
Jacobian-derived ``L`` gains and loses rows as rate constants change, and each
spurious row silently freezes a direction that should have been free to move.
``N`` gives the same integer moieties for every parameter value, which is what
a conservation law means.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from functools import reduce
from math import gcd, lcm

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from hallsim.tracing import is_traced

log = logging.getLogger(__name__)


def accumulator_mask(composite, keys: list[str]) -> jnp.ndarray:
    """Boolean mask over ``keys`` marking *flat* RunningIntegral outputs.

    Only flat (``tau=None``) integrals are unbounded and lack a fixed point, so
    they are masked out of the Newton solve. A *leaky* integral settles to
    ``A=τ·⟨sourceᵖ⟩`` — it has a fixed point and is solved like any state."""
    from hallsim.models.running_integral import RunningIntegral

    positions = [
        keys.index(composite.topology[name]["integral"])
        for name, proc in composite.processes.items()
        if isinstance(proc, RunningIntegral)
        and proc.tau is None
        and composite.topology.get(name, {}).get("integral") in keys
    ]
    mask = jnp.zeros(len(keys), dtype=bool)
    return mask.at[jnp.asarray(positions)].set(True) if positions else mask


def _residual_fn(composite, mask):
    rhs, _ = composite.build_rhs()
    return lambda y: jnp.where(mask, y, rhs(0.0, y))


def _rational_null_space(
    rows: list[list[Fraction]], n_cols: int
) -> list[list[Fraction]]:
    """Exact basis of ``{x : A·x = 0}`` for ``A`` given as ``rows``.

    Gauss-Jordan over the rationals, so the answer is the null space itself
    rather than whatever survived a floating-point threshold.
    """
    a = [list(r) for r in rows]
    pivots: list[int] = []
    r = 0
    for c in range(n_cols):
        piv = next((i for i in range(r, len(a)) if a[i][c] != 0), None)
        if piv is None:
            continue
        a[r], a[piv] = a[piv], a[r]
        inv = Fraction(1, 1) / a[r][c]
        a[r] = [v * inv for v in a[r]]
        for i in range(len(a)):
            if i != r and a[i][c] != 0:
                f = a[i][c]
                a[i] = [v - f * w for v, w in zip(a[i], a[r])]
        pivots.append(c)
        r += 1
        if r == len(a):
            break

    basis = []
    for free_col in (c for c in range(n_cols) if c not in pivots):
        vec = [Fraction(0, 1)] * n_cols
        vec[free_col] = Fraction(1, 1)
        for i, pivot_col in enumerate(pivots):
            vec[pivot_col] = -a[i][free_col]
        basis.append(vec)
    return basis


def _integerize(vec: list[Fraction]) -> list[int]:
    """Scale a rational vector to the smallest integer vector, sign-normalised
    so the first nonzero entry is positive — ``ATP + ADP``, not
    ``(-0.707, -0.707)``."""
    denom = reduce(lcm, (v.denominator for v in vec), 1)
    ints = [int(v * denom) for v in vec]
    common = reduce(gcd, (abs(i) for i in ints if i), 0) or 1
    ints = [i // common for i in ints]
    first = next((i for i in ints if i), 1)
    return [-i for i in ints] if first < 0 else ints


def conserved_moieties(stoichiometry: dict) -> list[dict[str, int]]:
    """Conserved moieties of a stoichiometry matrix, exactly.

    Each moiety is ``{species: integer coefficient}`` — the combinations no
    reaction can change, i.e. the integer basis of ``{L : L·N = 0}``. Depends
    only on ``N``, so it is the same for every parameter value and every
    state, which is what a conservation law means.
    """
    species = stoichiometry["species"]
    matrix = stoichiometry["matrix"]
    n_reactions = len(stoichiometry["reactions"])
    # L·N = 0 is Nᵀ·Lᵀ = 0: one row per reaction, one column per species.
    rows = [
        [
            Fraction(matrix[s][r]).limit_denominator(10**6)
            for s in range(len(species))
        ]
        for r in range(n_reactions)
    ]
    return [
        {species[i]: c for i, c in enumerate(_integerize(vec)) if c}
        for vec in _rational_null_space(rows, len(species))
    ]


def warn_if_time_dependent(composite, y, dt: float = 1.0) -> bool:
    """Warn (COPASI-style) if the RHS is explicitly time-dependent at ``y`` —
    a Newton fixed point is meaningless then. Returns True if autonomous."""
    if is_traced(y) or is_traced(*jax.tree_util.tree_leaves(composite)):
        return True  # diagnostic: nothing concrete to test, so stay quiet
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


def composite_stoichiometry(composite, keys: list[str] | None = None):
    """Composite-level ``N`` over store paths, or ``None``.

    Assembled from each process's :meth:`Process.stoichiometry`, with its
    species mapped through the topology so two models sharing a path share a
    row. Returns ``None`` unless *every* continuous process declares one — a
    single undeclared process (a hand-written coupling edge, a NeuralODE) can
    move any state, so a matrix missing its columns would claim conservation
    that the composite does not have.
    """
    keys = composite.store_keys() if keys is None else keys
    row_of = {k: i for i, k in enumerate(keys)}
    columns: list[list[float]] = []
    covered: set[int] = set()
    for name, proc in composite.continuous_processes().items():
        declared = proc.stoichiometry()
        if declared is None:
            return None
        topo = composite.topology.get(name, {})
        rows = [
            row_of.get(topo.get(species, species))
            for species in declared["species"]
        ]
        covered.update(r for r in rows if r is not None)
        for c in range(len(declared["reactions"])):
            column = [0.0] * len(keys)
            for r, row in enumerate(rows):
                if row is not None:
                    column[row] += declared["matrix"][r][c]
            columns.append(column)
    return columns, sorted(covered)


def _perturbed(composite, key, spread: float):
    """``composite`` with every fitted array leaf scaled by a random factor.

    Multiplicative, so signs and zeros survive — a rate constant moves by up
    to ``10**±spread`` but stays a rate constant. Static leaves (names, index
    maps, port defaults) are structure and are left alone.
    """
    params, static = eqx.partition(composite, eqx.is_inexact_array)
    leaves, treedef = jtu.tree_flatten(params)
    if not leaves:
        return composite
    factors = 10.0 ** jax.random.uniform(
        key, (len(leaves),), minval=-spread, maxval=spread
    )
    scaled = [leaf * factors[i] for i, leaf in enumerate(leaves)]
    return eqx.combine(jtu.tree_unflatten(treedef, scaled), static)


def infer_conservation_laws(
    composite,
    y,
    mask,
    rcond: float = 1e-9,
    n_samples: int = 8,
    spread: float = 0.5,
    seed: int = 0,
):
    """Conservation laws of a composite that declares no stoichiometry.

    ``L`` is conserved exactly when ``L·f(y; θ) = 0`` for *every* state and
    *every* parameter value, so ``L`` must lie in the left null space of the
    residual Jacobian at all of them. Stacking Jacobians sampled over both and
    taking the left null space of the stack imposes all those constraints at
    once.

    Sampling states alone is not enough, and this is the whole point: a
    species decaying at ``k = 1e-12`` has a Jacobian entry of ``-1e-12``
    wherever you evaluate it, so it looks conserved at every state. It stops
    looking conserved as soon as ``k`` is resampled — which is just the
    operational form of "a conservation law cannot depend on a rate constant".

    Non-finite samples (a perturbed parameter that overflows a rate law) are
    skipped rather than allowed to poison the stack.
    """
    key = jax.random.PRNGKey(seed)
    blocks = []
    for _ in range(n_samples):
        key, k_state, k_param = jax.random.split(key, 3)
        factor = 10.0 ** jax.random.uniform(
            k_state, y.shape, minval=-spread, maxval=spread
        )
        y_s = jnp.where(y == 0, factor - 1.0, y * factor)
        jac = np.asarray(
            jax.jacfwd(
                _residual_fn(_perturbed(composite, k_param, spread), mask)
            )(y_s)
        )
        if np.all(np.isfinite(jac)):
            blocks.append(jac)

    if not blocks:
        log.warning(
            "infer_conservation_laws: every sampled Jacobian was non-finite; "
            "falling back to the unperturbed state alone."
        )
        blocks = [np.asarray(jax.jacfwd(_residual_fn(composite, mask))(y))]

    u, s, _ = np.linalg.svd(np.hstack(blocks))
    null = np.argsort(s)[: int(np.sum(s < rcond * s[0]))]
    return _surviving(u[:, null].T, blocks)


def _surviving(candidates, blocks, eps_factor: float = 1e3):
    """Keep the candidates that hold in *every* sample.

    A single stacked SVD is not enough on its own: one inflated rate constant
    raises the largest singular value, and a threshold relative to it then
    readmits the slow direction the sampling was there to exclude. Scoring
    each candidate against each sample's own Jacobian avoids that — a real
    law sits at round-off in all of them, while a slow direction only has to
    betray itself in the one sample where its parameter came out large
    relative to the rest.
    """
    kept = []
    for law in candidates:
        worst = max(
            float(np.abs(law @ jac).max()) / (float(np.abs(jac).max()) or 1.0)
            for jac in blocks
        )
        if worst <= eps_factor * np.finfo(float).eps:
            kept.append(law)
    return np.asarray(kept, dtype=float).reshape(
        len(kept), candidates.shape[1]
    )


def conservation_laws(composite, y, mask=None, rcond: float = 1e-9):
    """Conservation-law matrix ``L`` (rows = conserved combinations) over the
    composite's store paths. Returns an ``(n_laws, n_state)`` array.

    Where every process declares a stoichiometry this is exact: the integer
    left null space of ``N``, identical for every parameter value and every
    state. Otherwise the laws are inferred by
    :func:`infer_conservation_laws`, which samples states *and* parameters —
    a single Jacobian cannot tell a conserved combination from a merely slow
    one, since both are singular at a point.

    Either way each candidate is checked against the composite's own Jacobian
    and dropped if it does not hold.

    Rows also pin any state the composite leaves identically constant. That
    is not physics: those directions are exactly singular, and the Newton
    solve in :func:`steady_state` needs them fixed to have a unique solution.
    """
    if is_traced(y) or is_traced(*jtu.tree_leaves(composite)):
        raise RuntimeError(
            "conservation_laws needs concrete values and this call is traced "
            "(jit/grad/vmap): the moieties come from a Jacobian or a declared "
            "stoichiometry, neither of which can be read off tracers. They are "
            "structural, so resolve them once eagerly and pass them in — "
            "`laws = conservation_laws(composite, y0)` outside the trace, then "
            "`steady_state(composite, laws=laws)` inside it. "
            "CalibrationProblem does this for you."
        )
    keys = composite.store_keys()
    mask = accumulator_mask(composite, keys) if mask is None else mask
    warn_if_time_dependent(composite, y)

    declared = composite_stoichiometry(composite, keys)
    if declared is None:
        laws = infer_conservation_laws(composite, y, mask, rcond)
        jac = np.asarray(jax.jacfwd(_residual_fn(composite, mask))(y))
        kept = _verified(list(laws), jac, keys)
        log.debug(
            "conservation_laws: no declared stoichiometry; inferred %d law(s) "
            "by sampling states and parameters. Declaring "
            "Process.stoichiometry() makes this exact.",
            len(kept),
        )
        return jnp.asarray(kept, dtype=float).reshape(len(kept), len(keys))

    # Solve only over the states N actually describes. A state outside it is
    # driven by something else (an SBML rateRule, a frozen sink), and its
    # empty column would otherwise come back as a free variable — i.e. N
    # would "prove" a state conserved by saying nothing about it.
    columns, covered = declared
    rows = [
        [Fraction(col[i]).limit_denominator(10**6) for i in covered]
        for col in columns
    ]
    laws = []
    for vec in _rational_null_space(rows, len(covered)):
        full = [0] * len(keys)
        for slot, coeff in zip(covered, _integerize(vec)):
            full[slot] = coeff
        laws.append(full)

    # A state the composite never moves is exactly singular for Newton, so it
    # has to be pinned even though it is not a conserved quantity in any
    # physical sense. Tested on an identically-zero Jacobian row rather than a
    # small one: "never moves" is exact, "moves slowly" is the thing this
    # function exists to stop treating as conservation.
    jac = np.asarray(jax.jacfwd(_residual_fn(composite, mask))(y))
    spanned = {i for law in laws for i, c in enumerate(law) if c}
    for i in range(len(keys)):
        if i not in spanned and not np.any(jac[i]):
            unit = [0] * len(keys)
            unit[i] = 1
            laws.append(unit)

    kept = _verified(laws, jac, keys)
    return jnp.asarray(kept, dtype=float).reshape(len(kept), len(keys))


def _verified(laws, jac, keys, rtol: float = 1e-8):
    """Drop candidate laws the composite's own Jacobian contradicts.

    ``L`` is conserved only if ``L·J = 0``. A declared ``N`` can disagree with
    what the composite actually integrates — a frozen sink, a species a rule
    overrides — and a law enforced on a direction that does move biases the
    fixed point silently. Cheaper to check than to debug.
    """
    scale = float(np.abs(jac).max()) or 1.0
    kept = []
    for law in laws:
        residual = float(np.abs(np.asarray(law, dtype=float) @ jac).max())
        if residual <= rtol * scale:
            kept.append(law)
        else:
            log.warning(
                "conservation_laws: dropping %s — L·J = %.3g, not 0, so the "
                "composite does not conserve it. The declared stoichiometry "
                "disagrees with what is being integrated.",
                {keys[i]: c for i, c in enumerate(law) if c},
                residual,
            )
    return kept


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

    y_star = jax.lax.custom_root(g, y0, solve, tangent_solve)
    if not isinstance(y_star, jax.core.Tracer):
        res = float(jnp.max(jnp.abs(g(y_star))))
        if res > tol:
            log.warning(
                "steady_state: Newton stopped at |f| = %.3g, above tol = "
                "%.3g, after at most %d iterations. The returned state is "
                "not a fixed point — seed y_guess closer (e.g. a short "
                "forward pre-solve) or raise max_iter.",
                res,
                tol,
                max_iter,
            )
    return y_star
