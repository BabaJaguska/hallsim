"""Local bifurcation analysis for HallSim vector fields.

Given a vector field ``f: R^n -> R^n`` — typically a Process or Composite
RHS at fixed parameters — this locates equilibria, reads their stability
from the Jacobian spectrum, and finds the codimension-1 bifurcations as a
control parameter is varied. Every derivative is JAX autodiff (exact, no
finite differences), so it composes with the rest of the framework and
works at any state dimension.

Both ways an equilibrium can lose hyperbolicity are covered, because both
matter: a **Hopf** (complex pair crosses the imaginary axis) starts an
oscillation, a **fold** (real eigenvalue crosses zero) creates or destroys
a pair of equilibria and is how a bistable switch is born. Each is
classified by its normal-form coefficient — the first Lyapunov coefficient
``l1`` for a Hopf, the quadratic coefficient ``a`` for a fold — via the
Kuznetsov projection with the multilinear forms of ``f`` taken by autodiff.

    from hallsim.bifurcation import field_from_composite, codim1_scan

    def field_of(alpha_y):
        return field_from_composite(build_gz(alpha_y))[0]

    bifs = codim1_scan(field_of, np.linspace(0.005, 2.0, 200),
                       x0_guess=[0.4, 0.4, 0.4])
    # -> [Hopf(param=0.0195, omega=21.8, l1=-2.95e+03, supercritical), ...]

Detection is by change in the **unstable dimension** — the number of
eigenvalues with ``Re > 0`` — so a crossing is seen wherever it happens in
the spectrum, not only at the leading pair. Pass ``laws`` for any model
with a conserved moiety; see :func:`equilibrium`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from hallsim.steady_state import leaf_basis, pin_conserved

log = logging.getLogger(__name__)

_IMAG_TOL = 1e-9
_DEGEN_TOL = 1e-8


@dataclass(frozen=True)
class Bifurcation:
    """A codimension-1 bifurcation located on a one-parameter sweep.

    ``coefficient`` is the normal-form coefficient for ``kind``: the first
    Lyapunov coefficient ``l1`` for a Hopf, the quadratic coefficient ``a``
    for a fold. ``transversality`` is ``⟨p, ∂f/∂param⟩`` at a fold — nonzero
    is what makes it a genuine saddle-node rather than a transcritical or
    pitchfork crossing.
    """

    kind: str  # "hopf" | "fold"
    param: float
    x: np.ndarray  # equilibrium at the bifurcation
    omega: float  # angular frequency at onset (0.0 for a fold)
    coefficient: float | None  # None if not computed
    transversality: float | None = None

    @property
    def supercritical(self) -> bool | None:
        """Hopf only: True (stable cycle) / False (unstable) / None if
        unclassified or not a Hopf."""
        if self.kind != "hopf" or self.coefficient is None:
            return None
        return self.coefficient < 0

    @property
    def degenerate(self) -> bool | None:
        """Whether the defining coefficient vanishes, so the classification
        does not hold: a Hopf with ``l1 ≈ 0`` (Bautin), a fold with ``a ≈ 0``
        (pitchfork) or with no parameter transversality (transcritical)."""
        if self.coefficient is None:
            return None
        if abs(self.coefficient) < _DEGEN_TOL:
            return True
        t = self.transversality
        return self.kind == "fold" and t is not None and abs(t) < _DEGEN_TOL

    def __str__(self) -> str:
        if self.kind == "hopf":
            tag = (
                "unclassified"
                if self.coefficient is None
                else ("supercritical" if self.supercritical else "subcritical")
            )
            return (
                f"Hopf(param={self.param:.4g}, omega={self.omega:.3g}, "
                f"l1={self.coefficient:+.3g}, {tag})"
                if self.coefficient is not None
                else f"Hopf(param={self.param:.4g}, "
                f"omega={self.omega:.3g}, {tag})"
            )
        tag = "degenerate" if self.degenerate else "saddle-node"
        a = "?" if self.coefficient is None else f"{self.coefficient:+.3g}"
        return f"Fold(param={self.param:.4g}, a={a}, {tag})"


def field_from_composite(composite, proc_names=None):
    """``(f, keys)`` for a composite's autonomous RHS: ``f(y)`` evaluates
    the flat derivative in ``keys`` order (``sorted(store_paths)``). Seed
    equilibrium searches with ``composite.initial_state_vec(keys)``."""
    rhs, keys = composite.build_rhs(proc_names)
    return (lambda y: rhs(0.0, jnp.asarray(y))), keys


def jacobian(f, x):
    """Exact Jacobian of ``f`` at ``x`` (forward-mode autodiff)."""
    return jnp.asarray(jax.jacfwd(f)(jnp.asarray(x, float)))


def spectrum(f, x, laws=None):
    """Jacobian eigenvalues at ``x``, ordered by descending real part.

    With ``laws`` the spectrum is taken on the leaf tangent space, dropping
    the one exact zero each conservation law contributes — read the raw
    spectrum of a conserved model and its leading eigenvalue is a zero that
    says nothing about stability.
    """
    a = np.asarray(jacobian(f, x))
    if laws is not None:
        v = leaf_basis(laws)
        a = v.T @ a @ v
    ev = np.linalg.eigvals(a)
    return ev[np.argsort(-ev.real)]


def critical_eigenvalue(f, x, laws=None) -> complex:
    """The eigenvalue nearest the imaginary axis — the one whose crossing
    defines a bifurcation. Its real part is the bisection target; whether
    its imaginary part is nonzero is what separates a Hopf from a fold."""
    ev = spectrum(f, x, laws)
    return complex(ev[np.argmin(np.abs(ev.real))])


def unstable_dim(f, x, laws=None) -> int:
    """Number of eigenvalues with ``Re > 0``. A change in this along a
    parameter sweep is a bifurcation, wherever in the spectrum it happens —
    which is why it, rather than the leading eigenvalue, drives the scan."""
    return int(np.sum(spectrum(f, x, laws).real > 0.0))


def equilibrium(f, x0, *, laws=None, y_ref=None, tol=1e-11, maxiter=100):
    """Damped Newton to a fixed point ``f(x)=0`` from seed ``x0``; returns
    the equilibrium as a numpy array, or ``None`` if it fails to converge.

    ``laws`` — the orthonormal conservation-law matrix from
    :func:`hallsim.steady_state.conservation_laws` — switches the solve to
    the pinned residual, which is what a model with a conserved moiety needs:
    without it the Jacobian is singular at every state and the search
    reports no equilibrium for a model that has one. The totals are held at
    ``y_ref`` (default ``x0``), so the point returned is the fixed point on
    that leaf.
    """
    x = jnp.asarray(x0, float)
    g = f
    if laws is not None:
        ref = x if y_ref is None else jnp.asarray(y_ref, float)
        g = pin_conserved(f, jnp.asarray(laws, float), ref)
    for _ in range(maxiter):
        fx = g(x)
        if not bool(jnp.all(jnp.isfinite(fx))):
            return None
        nrm = float(jnp.linalg.norm(fx))
        if nrm < tol:
            return np.asarray(x)
        step = jnp.linalg.solve(jax.jacfwd(g)(x), fx)
        if not bool(jnp.all(jnp.isfinite(step))):
            log.warning(
                "equilibrium: singular Jacobian at %s. A conserved moiety "
                "does this at every state — pass laws=conservation_laws("
                "composite, y0) to pin the totals.",
                np.asarray(x),
            )
            return None
        a = 1.0
        for _ in range(20):  # backtrack until the residual decreases
            if float(jnp.linalg.norm(g(x - a * step))) < nrm:
                break
            a *= 0.5
        x = x - a * step
    return np.asarray(x) if float(jnp.linalg.norm(g(x))) < 1e-7 else None


def _quadratic_form(f, x0):
    """``B(u,v) = d²f(x0)[u,v]``."""
    hess = np.asarray(jax.jacfwd(jax.jacfwd(f))(x0))
    return lambda u, v: np.einsum("ijk,j,k->i", hess, u, v)


def _cubic_form(f, x0):
    """``C(u,v,w) = d³f(x0)[u,v,w]``. ``n⁴`` storage — only the Lyapunov
    coefficient needs it, so it is built separately from ``B``."""
    third = np.asarray(jax.jacfwd(jax.jacfwd(jax.jacfwd(f)))(x0))
    return lambda u, v, w: np.einsum("ijkl,j,k,l->i", third, u, v, w)


def _on_leaf(f, x0, laws):
    """``(f_reduced, V)`` — ``f`` restricted to the leaf tangent space at
    ``x0``, in coordinates where the equilibrium is the origin."""
    v = jnp.asarray(leaf_basis(laws), float)
    return (lambda z: v.T @ f(x0 + v @ z)), v


def fold_coefficient(f, x0, laws=None):
    """``(a, q, p)`` for a fold at ``x0``: the quadratic normal-form
    coefficient ``a = ½⟨p, B(q,q)⟩`` with ``Aq = 0``, ``Aᵀp = 0`` and
    ``⟨p,q⟩ = 1``.

    ``a ≠ 0`` is the saddle-node nondegeneracy condition — two equilibria
    collide and vanish. ``a ≈ 0`` means a pitchfork or a higher-order
    degeneracy. Raises ``ValueError`` if the critical eigenvalue is complex.
    """
    x0 = jnp.asarray(x0, float)
    if laws is not None:
        g, v = _on_leaf(f, x0, laws)
        a, q, p = fold_coefficient(g, jnp.zeros(v.shape[1]))
        return a, np.asarray(v) @ q, np.asarray(v) @ p
    A = np.asarray(jax.jacfwd(f)(x0))
    ev, V = np.linalg.eig(A)
    k = int(np.argmin(np.abs(ev)))
    if abs(ev[k].imag) > _IMAG_TOL:
        raise ValueError("critical eigenvalue is complex — not a fold")
    q = np.real(V[:, k])
    q = q / np.linalg.norm(q)
    evL, W = np.linalg.eig(A.T)
    p = np.real(W[:, int(np.argmin(np.abs(evL)))])
    denom = float(np.dot(p, q))
    if abs(denom) < _IMAG_TOL:
        raise ValueError("left and right null vectors are orthogonal")
    p = p / denom
    bl = _quadratic_form(f, x0)
    return 0.5 * float(np.dot(p, bl(q, q))), q, p


def first_lyapunov_coefficient(f, x0, laws=None):
    """First Lyapunov coefficient ``l1`` of a Hopf point at equilibrium
    ``x0`` (Kuznetsov projection). ``l1 < 0`` supercritical, ``l1 > 0``
    subcritical. Requires the Jacobian at ``x0`` to have a complex pair
    near the imaginary axis; raises ``ValueError`` otherwise.

    With ``laws`` the coefficient is computed for the dynamics restricted to
    the leaf — the projection inverts ``A``, which is singular in the full
    space whenever a moiety is conserved."""
    x0 = jnp.asarray(x0, float)
    if laws is not None:
        g, v = _on_leaf(f, x0, laws)
        return first_lyapunov_coefficient(g, jnp.zeros(v.shape[1]))
    A = np.asarray(jax.jacfwd(f)(x0))
    ev, V = np.linalg.eig(A)
    k = int(np.argmax(ev.imag))  # the +i*omega eigenvector
    lam, w0 = ev[k], float(ev[k].imag)
    if w0 <= _IMAG_TOL:
        raise ValueError("no positive-imaginary eigenvalue at this point")
    q = V[:, k]
    q = q / np.sqrt(np.vdot(q, q).real)  # <q,q> = 1
    evL, W = np.linalg.eig(A.T)
    p = W[:, int(np.argmin(np.abs(evL - np.conj(lam))))]
    p = p / np.conj(np.vdot(p, q))  # <p,q> = 1

    bl = _quadratic_form(f, x0)
    tl = _cubic_form(f, x0)

    qb = np.conj(q)
    eye = np.eye(A.shape[0])
    r1 = np.linalg.solve(A, bl(q, qb))
    r2 = np.linalg.solve(2j * w0 * eye - A, bl(q, q))
    l1 = (1.0 / (2 * w0)) * np.real(
        np.vdot(p, tl(q, q, qb))
        - 2 * np.vdot(p, bl(q, r1))
        + np.vdot(p, bl(qb, r2))
    )
    return float(l1)


def _refine_re0(field_of_param, a, b, seed, iters=40, laws=None, y_ref=None):
    """Bisect ``param in [a, b]`` toward the exact Re=0 crossing."""

    def re(p):
        f = field_of_param(p)
        eqp = equilibrium(f, seed, laws=laws, y_ref=y_ref)
        if eqp is None:
            return np.nan
        return critical_eigenvalue(f, eqp, laws).real

    ra = re(a)
    for _ in range(iters):
        m = 0.5 * (a + b)
        rm = re(m)
        if not np.isfinite(rm):
            break
        if ra * rm <= 0:
            b = m
        else:
            a, ra = m, rm
    return 0.5 * (a + b)


def _refine_vanish(field_of_param, a, b, seed, laws, y_ref, iters=40):
    """Bisect ``[a, b]`` — branch exists at ``a``, not at ``b`` — toward the
    parameter where it ceases to. Returns ``(param, x)`` at the last point
    the branch is found, re-seeding as it goes so the continuation stays on
    that branch."""
    x = seed
    for _ in range(iters):
        m = 0.5 * (a + b)
        eqm = equilibrium(field_of_param(m), x, laws=laws, y_ref=y_ref)
        if eqm is None:
            b = m
        else:
            a, x = m, eqm
    return a, x


def _classify(field_of_param, pc, x, laws, h, coefficients):
    """Build the :class:`Bifurcation` at a located crossing."""
    f = field_of_param(pc)
    lam = critical_eigenvalue(f, x, laws)
    if abs(lam.imag) > _IMAG_TOL:
        c = None
        if coefficients:
            try:
                c = first_lyapunov_coefficient(f, x, laws)
            except Exception:
                c = None
        return Bifurcation("hopf", pc, x, abs(lam.imag), c)

    a = s = None
    if coefficients:
        try:
            a, _, p = fold_coefficient(f, x, laws)
            # ∂f/∂param is finite-differenced: the parameter enters through
            # the caller's builder, so there is no autodiff path to it.
            df = np.asarray(
                field_of_param(pc + h)(x) - field_of_param(pc - h)(x)
            ) / (2 * h)
            s = float(np.dot(p, df))
        except Exception:
            a = s = None
    return Bifurcation("fold", pc, x, 0.0, a, s)


def codim1_scan(
    field_of_param,
    params,
    x0_guess,
    *,
    coefficients=True,
    refine=True,
    laws=None,
    y_ref=None,
):
    """Locate codimension-1 bifurcations as ``param`` sweeps ``params``.

    ``field_of_param(p)`` returns the vector field ``f: R^n->R^n`` at
    parameter ``p`` (e.g. via :func:`field_from_composite` on a composite
    rebuilt at ``p``). A bifurcation is flagged wherever the equilibrium's
    unstable dimension changes, or where the branch itself vanishes with a
    near-critical eigenvalue — the signature of a fold. Each is refined by
    bisection (``refine``), classified Hopf or fold from the critical
    eigenvalue, and given its normal-form coefficient (``coefficients``).
    Equilibria are tracked by continuation — each seed is the previous
    converged fixed point — so a single ``x0_guess`` suffices. Returns
    ``list[Bifurcation]`` ordered by param.

    ``laws`` pins the conserved totals at ``y_ref`` (default ``x0_guess``)
    for every equilibrium and spectrum in the scan; without it a model with
    a conserved moiety yields no equilibria and the scan returns empty. The
    laws are structural, so one matrix covers the whole sweep.

    Continuation is plain Newton from the previous point, so a branch is
    followed only until it folds. Past a fold the scan reports the fold and
    stops seeing that branch; it does not turn the corner onto the other
    one. Tracing both sides of a hysteresis loop needs a multi-seed sweep
    per parameter value.
    """
    params = np.asarray(params, float)
    guess = np.asarray(x0_guess, float)
    y_ref = guess if y_ref is None else np.asarray(y_ref, float)
    h = float(np.min(np.abs(np.diff(params)))) / 8 if len(params) > 1 else 1e-6

    xs, dim, crit = [], [], []
    for p in params:
        f = field_of_param(float(p))
        eqp = equilibrium(f, guess, laws=laws, y_ref=y_ref)
        if eqp is None:
            xs.append(None)
            dim.append(-1)
            crit.append(np.nan)
            continue
        guess = eqp  # continuation
        xs.append(eqp)
        dim.append(unstable_dim(f, eqp, laws))
        crit.append(critical_eigenvalue(f, eqp, laws).real)
    crit = np.asarray(crit)

    out: list[Bifurcation] = []
    step = float(np.min(np.abs(np.diff(params)))) if len(params) > 1 else 0.0
    for i in range(1, len(params)):
        prev_x, here_x = xs[i - 1], xs[i]
        if prev_x is None:
            continue

        if here_x is None:
            # The branch ended. Bisect to where, then keep it only if the
            # critical eigenvalue really went to zero there — otherwise this
            # is a Newton failure, and calling it a bifurcation is worse than
            # saying nothing.
            pc, xf = _refine_vanish(
                field_of_param,
                float(params[i - 1]),
                float(params[i]),
                prev_x,
                laws,
                y_ref,
            )
            lam = critical_eigenvalue(field_of_param(pc), xf, laws)
            if abs(lam.real) > 1e-3 or abs(lam.imag) > _IMAG_TOL:
                log.warning(
                    "codim1_scan: lost the branch between %.6g and %.6g with "
                    "critical eigenvalue %.3g — Newton failure, not a fold.",
                    params[i - 1],
                    params[i],
                    lam,
                )
                continue
            if out and abs(out[-1].param - pc) < step:
                continue  # already reported as an unstable-dim change
            out.append(
                _classify(field_of_param, pc, xf, laws, h, coefficients)
            )
            continue

        if dim[i - 1] == dim[i]:
            continue
        pc = float(params[i])
        if refine:
            pc = _refine_re0(
                field_of_param,
                float(params[i - 1]),
                float(params[i]),
                prev_x,
                laws=laws,
                y_ref=y_ref,
            )
        f = field_of_param(pc)
        eqp = equilibrium(f, prev_x, laws=laws, y_ref=y_ref)
        if eqp is None:
            continue
        out.append(_classify(field_of_param, pc, eqp, laws, h, coefficients))
    return out
