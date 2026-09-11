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
import math
from fractions import Fraction

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from hallsim.store import as_paths
from hallsim.structure import (
    JacobianPattern,
    check_pattern,
    composite_stoichiometry,
    compressed_jacobian,
    integerize,
    jacobian_pattern,
    rational_null_space,
)
from hallsim.tracing import is_traced

log = logging.getLogger(__name__)


def accumulator_positions(composite, keys: list[str]) -> list[int]:
    """Indices into ``keys`` of *flat* RunningIntegral outputs — structure,
    readable under a trace."""
    from hallsim.models.running_integral import RunningIntegral

    return [
        keys.index(path)
        for name, proc in composite.processes.items()
        if isinstance(proc, RunningIntegral) and proc.tau is None
        for path in as_paths(
            composite.topology.get(name, {}).get("integral", ())
        )
        if path in keys
    ]


def accumulator_mask(composite, keys: list[str]) -> jnp.ndarray:
    """Boolean mask over ``keys`` marking *flat* RunningIntegral outputs.

    Only flat (``tau=None``) integrals are unbounded and lack a fixed point, so
    they are masked out of the Newton solve. A *leaky* integral settles to
    ``A=τ·⟨sourceᵖ⟩`` — it has a fixed point and is solved like any state."""
    positions = accumulator_positions(composite, keys)
    mask = jnp.zeros(len(keys), dtype=bool)
    return mask.at[jnp.asarray(positions)].set(True) if positions else mask


def _residual_fn(composite, mask):
    rhs, _ = composite.build_rhs()
    return lambda y: jnp.where(mask, y, rhs(0.0, y))


def conserved_moieties(stoichiometry: dict) -> list[dict[str, int]]:
    """Conserved moieties of a stoichiometry matrix, exactly.

    Each moiety is ``{species: integer coefficient}`` — the combinations no
    reaction can change, i.e. the integer basis of ``{L : L·N = 0}``. Depends
    only on ``N``, so it is the same for every parameter value and every
    state, which is what a conservation law means.
    """
    species = stoichiometry["species"]
    matrix = stoichiometry["matrix"]
    # L·N = 0 is Nᵀ·Lᵀ = 0: one row per reaction, one column per species.
    rows = [
        {
            s: Fraction(matrix[s][r]).limit_denominator(10**6)
            for s in range(len(species))
            if matrix[s][r]
        }
        for r in range(len(stoichiometry["reactions"]))
    ]
    return [
        {species[i]: c for i, c in sorted(integerize(vec).items())}
        for vec in rational_null_space(rows, len(species))
    ]


def is_autonomous(composite, y, dt: float = 1.0) -> bool:
    """Whether the RHS at ``y`` is free of explicit time dependence.

    True under trace, where there is nothing concrete to test — callers of this
    are diagnostics, and a diagnostic stays quiet rather than guessing.
    """
    if is_traced(y) or is_traced(*jax.tree_util.tree_leaves(composite)):
        return True
    rhs, _ = composite.build_rhs()
    return float(jnp.max(jnp.abs(rhs(0.0, y) - rhs(dt, y)))) < 1e-9


def warn_if_time_dependent(composite, y, dt: float = 1.0) -> bool:
    """Warn (COPASI-style) if the RHS is explicitly time-dependent at ``y`` —
    a Newton fixed point is meaningless then. Returns True if autonomous."""
    autonomous = is_autonomous(composite, y, dt)
    if not autonomous:
        log.warning(
            "steady_state: the RHS is explicitly time-dependent at this "
            "condition; a Newton fixed point is not meaningful. Equilibrate an "
            "autonomous condition (e.g. the unperturbed control, where a timed "
            "input vanishes)."
        )
    return autonomous


def residual_pattern(composite, keys, laws=None):
    """Sparsity of the pinned residual ``g = f + LᵀL·(y − y_ref)`` over
    ``keys``: the composite's own pattern
    (:func:`hallsim.structure.jacobian_pattern`), a diagonal on the masked
    accumulator rows, and a block on every law's support. ``None`` when
    ``laws`` is traced — its support is then unknown and the Jacobian is
    formed densely."""
    if laws is not None and is_traced(laws):
        return None
    masked = np.asarray(accumulator_positions(composite, keys), dtype=int)
    pattern = jacobian_pattern(composite, keys).with_entries(masked, masked)
    if laws is not None:
        pattern = pattern.with_laws(np.asarray(laws))
    return pattern


def _jacobian(fn, y, pattern: JacobianPattern | None):
    """``∂fn/∂y``, in as many forward passes as ``pattern`` has colours."""
    if pattern is None:
        return jax.jacfwd(fn)(y)
    return compressed_jacobian(fn, y, pattern)


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
    candidates=None,
    pattern: JacobianPattern | None = None,
):
    """Conservation laws that the declared stoichiometry cannot settle.

    ``L`` is conserved exactly when ``L·f(y; θ) = 0`` for *every* state and
    *every* parameter value, so ``L`` must lie in the left null space of the
    residual Jacobian at all of them. Stacking Jacobians sampled over both and
    taking the left null space of the stack imposes all those constraints at
    once — within the span of ``candidates`` (rows over the state; the whole
    space by default), which is what the declared ``N`` already allows.

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
            _jacobian(
                _residual_fn(_perturbed(composite, k_param, spread), mask),
                y_s,
                pattern,
            )
        )
        if np.all(np.isfinite(jac)):
            blocks.append(jac)

    if not blocks:
        log.warning(
            "infer_conservation_laws: every sampled Jacobian was non-finite; "
            "falling back to the unperturbed state alone."
        )
        blocks = [
            np.asarray(_jacobian(_residual_fn(composite, mask), y, pattern))
        ]

    stacked = np.hstack(blocks)
    scale = max(float(np.abs(b).max()) for b in blocks) or 1.0
    if candidates is None:
        u, s, _ = np.linalg.svd(stacked)
        null = s <= rcond * max(float(s[0]), scale)
        laws = u[:, null].T
    else:
        basis = np.asarray(candidates, dtype=float).reshape(-1, y.shape[0])
        u, s, _ = np.linalg.svd(basis @ stacked)
        null = s <= rcond * max(float(s[0]), scale)
        laws = u[:, null].T @ basis
    return _surviving(laws, blocks)


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


def _dense(law: dict, n: int) -> np.ndarray:
    out = np.zeros(n)
    for i, c in law.items():
        out[i] = c
    return out


def conservation_laws(composite, y, mask=None, rcond: float = 1e-9):
    """Conservation-law matrix ``L`` (rows = conserved combinations) over the
    composite's store paths. Returns an ``(n_laws, n_state)`` array whose rows
    are **orthonormal**, so ``L.T @ L`` is the orthogonal projector onto the
    conserved directions and ``I - L.T @ L`` projects onto the leaf's tangent
    space. Use :func:`hallsim.structure.composite_moieties` for the integer
    coefficients that state the conservation as chemistry; these rows span
    the same space.

    The declared stoichiometry settles every law over the paths nothing
    moves outside it: the integer left null space of ``N``, identical for
    every parameter value and every state. A candidate that touches a path
    something else also moves — a process with no symbolic form, a rate rule
    — is only what ``N`` *allows*, and is kept by
    :func:`infer_conservation_laws`, which samples states *and* parameters,
    since a single Jacobian cannot tell a conserved combination from a
    merely slow one. With every process declared nothing is sampled; with
    none declared everything is, which is the old behaviour.

    Either way each law is checked against the composite's own Jacobian and
    dropped if it does not hold, and the Jacobian's own sparsity pattern is
    checked against the composite first (:func:`hallsim.structure.check_pattern`).

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
    y = jnp.asarray(y)  # the RHS scatters assignments; numpy has no .at
    keys = composite.store_keys()
    n = len(keys)
    mask = accumulator_mask(composite, keys) if mask is None else mask
    warn_if_time_dependent(composite, y)

    fn = _residual_fn(composite, mask)
    pattern = residual_pattern(composite, keys)
    jac = np.asarray(check_pattern(fn, y, pattern, keys))

    structure = composite_stoichiometry(composite, keys)
    exact = [_dense(law, n) for law in structure.exact_laws()]
    allowed = structure.null_space() if structure.undescribed else None
    pending = len(allowed) - len(exact) if allowed is not None else 0

    if pending and not _no_law_can_hold(jac, n):
        # Sampling's output is verified against the same Jacobian, so when
        # it admits nothing the samples cannot either — hence the guard.
        candidates = (
            None
            if structure.matrix.shape[1] == 0
            else np.asarray([_dense(law, n) for law in allowed])
        )
        laws = list(
            infer_conservation_laws(
                composite,
                y,
                mask,
                rcond,
                candidates=candidates,
                pattern=pattern,
            )
        )
        log.debug(
            "conservation_laws: %d law(s) exact from declared stoichiometry; "
            "%d candidate(s) touching %s (moved outside it by %s) checked by "
            "sampling states and parameters, %d law(s) kept in all. "
            "Declaring reaction_channels() on every process makes this exact.",
            len(exact),
            pending,
            [keys[i] for i in sorted(structure.undescribed)],
            list(structure.opaque) or "rate rules",
            len(laws),
        )
    else:
        laws = exact
        if pending:
            log.debug(
                "conservation_laws: Jacobian admits no conserved combination "
                "beyond the %d exact law(s); skipped sampling.",
                len(exact),
            )

    kept = _verified(laws, jac, keys)
    # A state the composite never moves is exactly singular for Newton, so it
    # has to be pinned even though it is not a conserved quantity in any
    # physical sense. Tested on an identically-zero Jacobian row rather than a
    # small one: "never moves" is exact, "moves slowly" is the thing this
    # function exists to stop treating as conservation. After verification,
    # so a dropped law leaves nothing unpinned.
    spanned = {
        int(i)
        for law in kept
        for i in np.flatnonzero(np.abs(np.asarray(law, dtype=float)) > 1e-12)
    }
    for i in range(n):
        if i not in spanned and not np.any(jac[i]):
            unit = np.zeros(n)
            unit[i] = 1.0
            kept.append(unit)
    return _orthonormal_rows(kept, n)


def _orthonormal_rows(rows, n_state: int) -> jnp.ndarray:
    """Orthonormal basis of the span of ``rows`` → ``(n_laws, n_state)``.

    ``LᵀL`` is the orthogonal projector onto the conserved directions only when
    ``L`` has orthonormal rows, and a null-space basis has neither unit norm nor
    mutual orthogonality. Orthonormalising preserves the row space — the same
    leaf is pinned, the same totals are fixed — and leaves ``LᵀL`` usable as a
    projector by every caller; dependent rows collapse, so a law listed twice
    counts once. Integer moiety coefficients are the physical statement and
    stay in :func:`hallsim.structure.composite_moieties`.

    Signs are fixed so the first non-zero entry of each row is positive, since
    the SVD's sign convention is otherwise arbitrary.
    """
    if not len(rows):
        return jnp.zeros((0, n_state))
    a = np.asarray(rows, dtype=float).reshape(len(rows), n_state)
    _, s, vt = np.linalg.svd(a, full_matrices=False)
    rank = int(np.sum(s > 1e-10 * max(float(s[0]), 1e-300)))
    q = vt[:rank]
    lead = [np.flatnonzero(np.abs(r) > 1e-12) for r in q]
    sign = np.array(
        [1.0 if not i.size else np.sign(r[i[0]]) for r, i in zip(q, lead)]
    )
    return jnp.asarray(q * sign[:, None])


def pin_conserved(residual, laws, y_ref):
    """``g(y) = f(y) + LᵀL·(y − y_ref)`` — the full-rank Newton residual.

    ``L`` is orthonormal and ``L·f = 0``, so the terms are on complementary
    subspaces: ``g = 0`` iff ``f = 0`` and the conserved totals match
    ``y_ref``. Solving ``f`` alone is singular along every conserved
    direction, for every seed.
    """

    def g(y):
        return residual(y) + laws.T @ (laws @ (y - y_ref))

    return g


def leaf_basis(laws) -> np.ndarray:
    """Orthonormal basis ``V`` (``n × (n − n_laws)``) of the leaf tangent
    space. ``VᵀJV`` holds the real modes; the raw spectrum adds one exact
    zero per conservation law, which is bookkeeping, not marginal stability.
    """
    laws = np.asarray(laws, dtype=float)
    if not laws.size:
        raise ValueError("leaf_basis needs at least one conservation law")
    return np.linalg.svd(laws)[2][len(laws) :].T


#: Shared by :func:`_verified` and :func:`_no_law_can_hold`, so the guard and
#: the check it skips cannot disagree about what counts as conserved.
_VERIFY_RTOL = 1e-8


def _no_law_can_hold(jac, n_state: int, rtol: float = _VERIFY_RTOL) -> bool:
    """Whether ``jac`` rules out every conserved combination at once.

    :func:`_verified` keeps a law only if ``|L·J| ≤ rtol·|J|``, and a unit
    candidate has ``|L·J|_max ≥ σ_min/√n``, so a large enough ``σ_min`` fails
    all of them. One-directional: a small ``σ_min`` may be a slow mode, which
    is what :func:`infer_conservation_laws` samples parameters to tell apart.
    """
    if not jac.size:
        return False
    sigma_min = float(np.linalg.svd(jac, compute_uv=False)[-1])
    scale = float(np.abs(jac).max()) or 1.0
    return sigma_min > rtol * scale * np.sqrt(n_state)


def _verified(laws, jac, keys, rtol: float = _VERIFY_RTOL):
    """Drop candidate laws the composite's own Jacobian contradicts.

    ``L`` is conserved only if ``L·J = 0``. A declared ``N`` can disagree with
    what the composite actually integrates — a species a rule overrides, a
    declaration that is simply wrong — and a law enforced on a direction that
    does move biases the fixed point silently. Cheaper to check than to debug.
    """
    scale = float(np.abs(jac).max()) or 1.0
    kept = []
    for law in laws:
        residual = float(np.abs(np.asarray(law, dtype=float) @ jac).max())
        if residual <= rtol * scale:
            kept.append(np.asarray(law, dtype=float))
        else:
            log.warning(
                "conservation_laws: dropping %s — L·J = %.3g, not 0, so the "
                "composite does not conserve it. The declared stoichiometry "
                "disagrees with what is being integrated.",
                {keys[i]: float(c) for i, c in enumerate(law) if c},
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

    The Newton Jacobian is formed in as many forward passes as the residual's
    sparsity pattern has colours (:func:`residual_pattern`), so a composite of
    declared processes pays for its coupling, not for its size; a composite
    of undeclared ones pays ``jacfwd``'s ``n`` passes, as before.
    """
    keys = composite.store_keys()
    mask = accumulator_mask(composite, keys)
    residual = _residual_fn(composite, mask)
    ic = composite.initial_state_vec(keys)
    y0 = ic if y_guess is None else y_guess
    y_ref = ic if y_ref is None else y_ref
    laws = conservation_laws(composite, y0, mask) if laws is None else laws

    g = pin_conserved(residual, laws, y_ref)
    pattern = residual_pattern(composite, keys, laws)

    def solve(fn, guess):
        def body(state):
            y, i, _ = state
            f = fn(y)
            dy = jnp.linalg.solve(_jacobian(fn, y, pattern), f)
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
        return jnp.linalg.solve(_jacobian(gg, jnp.zeros_like(b), pattern), b)

    y_star = jax.lax.custom_root(g, y0, solve, tangent_solve)
    if not isinstance(y_star, jax.core.Tracer):
        res = float(jnp.max(jnp.abs(g(y_star))))
        # `not (res <= tol)` rather than `res > tol`: a diverged solve gives
        # res = NaN, every comparison with it is False, and the caller would
        # get a state vector of NaNs with no warning at all.
        if not (res <= tol):
            log.warning(
                "steady_state: Newton stopped at |f| = %.3g, above tol = "
                "%.3g, after at most %d iterations. %s",
                res,
                tol,
                max_iter,
                (
                    "A non-finite residual means the solve diverged and no "
                    "fixed point was approached — check the system has one "
                    "(a state with a constant non-zero derivative has none)."
                    if not math.isfinite(res)
                    else "The returned state is not a fixed point — seed "
                    "y_guess closer (e.g. a short forward pre-solve) or "
                    "raise max_iter."
                ),
            )
    return y_star
