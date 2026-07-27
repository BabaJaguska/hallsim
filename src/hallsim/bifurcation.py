"""Local bifurcation analysis for HallSim vector fields.

Given a vector field ``f: R^n -> R^n`` — typically a Process or Composite
RHS at fixed parameters — this locates equilibria, reads their stability
from the Jacobian spectrum, detects Hopf bifurcations as a control
parameter is varied, and classifies each Hopf as super- or subcritical
via the first Lyapunov coefficient. Every derivative is JAX autodiff
(exact, no finite differences), so it composes with the rest of the
framework and works at any state dimension.

    from hallsim.bifurcation import field_from_composite, hopf_scan

    def field_of(alpha_y):
        return field_from_composite(build_gz(alpha_y))[0]

    hopfs = hopf_scan(field_of, np.linspace(0.005, 2.0, 200),
                      x0_guess=[0.4, 0.4, 0.4])
    # -> [Hopf(param=0.0195, omega=21.8, l1=-2.95e+03, supercritical), ...]

The first Lyapunov coefficient uses the Kuznetsov projection with the
2nd/3rd-order multilinear forms of ``f`` taken by autodiff; ``l1 < 0``
means a stable limit cycle is born (supercritical), ``l1 > 0`` an
unstable one (subcritical).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

_IMAG_TOL = 1e-9


@dataclass(frozen=True)
class HopfPoint:
    """A Hopf bifurcation located on a one-parameter sweep."""

    param: float
    x: np.ndarray  # equilibrium at the bifurcation
    omega: float  # angular frequency at onset (|Im lambda|)
    lyapunov1: float | None  # first Lyapunov coefficient (None if skipped)

    @property
    def supercritical(self) -> bool | None:
        """True (stable cycle) / False (unstable) / None if unclassified."""
        if self.lyapunov1 is None:
            return None
        return self.lyapunov1 < 0

    def __str__(self) -> str:
        if self.lyapunov1 is None:
            kind = "unclassified"
        else:
            kind = "supercritical" if self.supercritical else "subcritical"
        return (
            f"Hopf(param={self.param:.4g}, omega={self.omega:.3g}, "
            f"l1={self.lyapunov1:+.3g}, {kind})"
        )


def field_from_composite(composite, proc_names=None):
    """``(f, keys)`` for a composite's autonomous RHS: ``f(y)`` evaluates
    the flat derivative in ``keys`` order (``sorted(store_paths)``). Seed
    equilibrium searches with ``composite.initial_state_vec(keys)``."""
    rhs, keys = composite.build_rhs(proc_names)
    return (lambda y: rhs(0.0, jnp.asarray(y))), keys


def jacobian(f, x):
    """Exact Jacobian of ``f`` at ``x`` (forward-mode autodiff)."""
    return jnp.asarray(jax.jacfwd(f)(jnp.asarray(x, float)))


def spectrum(f, x):
    """Jacobian eigenvalues at ``x``, ordered by descending real part."""
    ev = np.linalg.eigvals(np.asarray(jacobian(f, x)))
    return ev[np.argsort(-ev.real)]


def leading_complex_pair_re(f, x):
    """Real part of the complex-conjugate eigenvalue pair with the largest
    real part; ``nan`` if the spectrum at ``x`` is entirely real. This is
    the quantity whose zero-crossing marks a Hopf bifurcation."""
    ev = np.linalg.eigvals(np.asarray(jacobian(f, x)))
    cx = ev[np.abs(ev.imag) > _IMAG_TOL]
    return float(cx[np.argmax(cx.real)].real) if len(cx) else np.nan


def equilibrium(f, x0, *, tol=1e-11, maxiter=100):
    """Damped Newton to a fixed point ``f(x)=0`` from seed ``x0``; returns
    the equilibrium as a numpy array, or ``None`` if it fails to converge."""
    x = jnp.asarray(x0, float)
    for _ in range(maxiter):
        fx = f(x)
        if not bool(jnp.all(jnp.isfinite(fx))):
            return None
        nrm = float(jnp.linalg.norm(fx))
        if nrm < tol:
            return np.asarray(x)
        try:
            step = jnp.linalg.solve(jax.jacfwd(f)(x), fx)
        except Exception:
            return None
        a = 1.0
        for _ in range(20):  # backtrack until the residual decreases
            if float(jnp.linalg.norm(f(x - a * step))) < nrm:
                break
            a *= 0.5
        x = x - a * step
    return np.asarray(x) if float(jnp.linalg.norm(f(x))) < 1e-7 else None


def first_lyapunov_coefficient(f, x0):
    """First Lyapunov coefficient ``l1`` of a Hopf point at equilibrium
    ``x0`` (Kuznetsov projection). ``l1 < 0`` supercritical, ``l1 > 0``
    subcritical. Requires the Jacobian at ``x0`` to have a complex pair
    near the imaginary axis; raises ``ValueError`` otherwise."""
    x0 = jnp.asarray(x0, float)
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

    hess = np.asarray(jax.jacfwd(jax.jacfwd(f))(x0))  # d2 f_i / dx_j dx_k
    third = np.asarray(jax.jacfwd(jax.jacfwd(jax.jacfwd(f)))(x0))

    def bl(u, v):
        return np.einsum("ijk,j,k->i", hess, u, v)

    def tl(u, v, w):
        return np.einsum("ijkl,j,k,l->i", third, u, v, w)

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


def _refine_re0(field_of_param, a, b, seed, iters=40):
    """Bisect ``param in [a, b]`` toward the exact Re=0 crossing."""

    def re(p):
        f = field_of_param(p)
        eqp = equilibrium(f, seed)
        return leading_complex_pair_re(f, eqp) if eqp is not None else np.nan

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


def hopf_scan(
    field_of_param,
    params,
    x0_guess,
    *,
    lyapunov=True,
    refine=True,
):
    """Locate Hopf bifurcations as ``param`` sweeps ``params``.

    ``field_of_param(p)`` returns the vector field ``f: R^n->R^n`` at
    parameter ``p`` (e.g. via :func:`field_from_composite` on a composite
    rebuilt at ``p``). A Hopf is flagged wherever the equilibrium's leading
    complex eigenvalue pair crosses ``Re=0``; each crossing is refined by
    bisection (``refine``) and classified by the first Lyapunov coefficient
    (``lyapunov``). Equilibria are tracked by continuation — each seed is
    the previous converged fixed point — so a single ``x0_guess`` at the
    first ``param`` suffices. Returns ``list[HopfPoint]`` ordered by param.
    """
    params = np.asarray(params, float)
    guess = np.asarray(x0_guess, float)
    xs, re = [], []
    for p in params:
        f = field_of_param(float(p))
        eqp = equilibrium(f, guess)
        if eqp is None:
            xs.append(None)
            re.append(np.nan)
            continue
        guess = eqp  # continuation
        xs.append(eqp)
        re.append(leading_complex_pair_re(f, eqp))
    re = np.asarray(re)

    out: list[HopfPoint] = []
    for i in range(1, len(params)):
        if not (np.isfinite(re[i - 1]) and np.isfinite(re[i])):
            continue
        if re[i - 1] * re[i] >= 0:
            continue
        seed = xs[i - 1] if xs[i - 1] is not None else guess
        pc = float(params[i])
        if refine:
            pc = _refine_re0(
                field_of_param, float(params[i - 1]), float(params[i]), seed
            )
        f = field_of_param(pc)
        eqp = equilibrium(f, seed)
        if eqp is None:
            continue
        ev = np.linalg.eigvals(np.asarray(jacobian(f, eqp)))
        cx = ev[np.abs(ev.imag) > _IMAG_TOL]
        w = (
            float(abs(cx[np.argmax(cx.real)].imag))
            if len(cx)
            else float("nan")
        )
        l1 = None
        if lyapunov:
            try:
                l1 = first_lyapunov_coefficient(f, eqp)
            except Exception:
                l1 = None
        out.append(HopfPoint(param=pc, x=eqp, omega=w, lyapunov1=l1))
    return out
