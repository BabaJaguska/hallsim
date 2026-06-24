"""Per-group stiffness analysis for automatic solver selection.

Stiffness is a property of a system's local Jacobian spectrum, not of a
model's name. A group is stiff *for an explicit solver* when its fastest
**dissipative** mode — a large negative-real eigenvalue, a fast decay to
equilibrium — is far faster than the dynamics actually being resolved.
The decayed mode leaves the solution smooth, so accuracy would permit
large steps, but an explicit method's step is bounded by numerical
*stability* (``Δt ≲ 2/|λ|``) and is forced tiny anyway. That mismatch is
stiffness.

Crucially, **oscillation is not stiffness.** A fast oscillator has
large-*imaginary* eigenvalues (``λ ≈ ±iω``); resolving the oscillation
already demands ``Δt ~ 1/ω``, so an explicit solver is accuracy-limited,
not stability-limited, and handles it fine. The discriminator this module
uses is therefore the fastest mode being *decay-dominated*
(``|Re λ| ≳ |Im λ|``) **and** a large spread between the fastest and
slowest dissipative rates — not raw eigenvalue magnitude.

This is what lets the Scheduler pick an implicit (A-stable) solver for a
stiff group and a cheaper explicit one elsewhere with no human in the
loop. It is also directly inspectable: call :func:`analyze_groups` on any
composite to see why a group was classified the way it was.

The spectrum is measured at a concrete state ``y0`` (the Jacobian is
state-dependent), so this must run **eagerly** — outside ``jax.grad`` /
``jax.jvp`` / ``jax.vmap`` tracing, where eigenvalues would be abstract
tracers. :func:`analyze_groups` raises a clear error if handed a traced
state; the Scheduler resolves stiffness once, eagerly, and reuses the
verdict under tracing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np

from hallsim.config import DEFAULT_MAX_EXPLICIT_SUBSTEPS

if TYPE_CHECKING:
    from hallsim.composite import Composite

# A dissipative eigenvalue is "active" (counts toward the stiffness ratio)
# only if its decay rate is a non-negligible fraction of the fastest one;
# this drops the numerically-zero / conserved-quantity modes (Re λ ≈ 0)
# that would otherwise blow the ratio up for any system with a
# conservation law or an oscillator sitting near the imaginary axis.
_ACTIVE_FLOOR_FRAC = 1e-9


@dataclass
class GroupStiffness:
    """Jacobian-spectrum verdict for one continuous group at a state.

    Attributes
    ----------
    name:
        Group label.
    dim:
        Number of evolving state components in the group.
    spectral_abscissa:
        ``max(-Re λ)`` over decaying modes — the fastest dissipation
        rate (in the composite's canonical time unit). 0 if no mode
        decays. This, not ``|λ|``, is what makes an explicit solver
        stability-limited: a fast *oscillation* (large ``|Im λ|``)
        demands small steps for accuracy anyway and is not stiff.
    max_abs_im:
        ``max|Im λ|`` — the fastest oscillation frequency, for context.
    min_active_decay:
        Slowest dissipation rate among non-negligible decaying modes.
    stiffness_ratio:
        ``spectral_abscissa / min_active_decay`` — the spread of
        dissipative timescales. Reported as a diagnostic; note a large
        spread alone does *not* imply stiffness if the fastest rate is
        itself slow (ERiQ).
    dt:
        Solve interval the verdict was computed against (the explicit
        step ceiling).
    stiffness_index:
        ``spectral_abscissa × dt`` — stability-limited substeps per solve
        interval. The quantity the verdict thresholds.
    stiff:
        Final verdict: ``stiffness_index`` exceeds the substep budget.
    jacobian_cond:
        2-norm condition number of the restricted Jacobian. An implicit
        solver's per-step Newton iteration solves a linear system in this
        matrix; a large condition number (≫ 1e6) means that solve is
        ill-conditioned and Newton struggles — the implicit step is then
        forced tiny *independently of the error tolerance*, the failure an
        explicit solver (which solves no linear system) is immune to.
    state_scale_spread:
        ``max|y| / min nonzero |y|`` over the group's states at ``y0`` —
        the dynamic range the solver must handle in one vector. A wide
        spread (a 1e6-magnitude species beside a unit one) is the usual
        source of a large ``jacobian_cond`` and argues for
        non-dimensionalising the model before implicit integration.
    eigenvalues:
        The raw restricted-Jacobian spectrum (complex), for inspection.
    """

    name: str
    dim: int
    spectral_abscissa: float
    max_abs_im: float
    min_active_decay: float
    stiffness_ratio: float
    dt: float
    stiffness_index: float
    stiff: bool
    jacobian_cond: float = float("nan")
    state_scale_spread: float = float("nan")
    eigenvalues: np.ndarray = field(default=None, repr=False)

    def __str__(self) -> str:
        verdict = "STIFF → implicit" if self.stiff else "non-stiff → explicit"
        return (
            f"{self.name:>10}: {verdict:<22} "
            f"dim={self.dim:<3} "
            f"max|Re λ|={self.spectral_abscissa:.3g} "
            f"max|Im λ|={self.max_abs_im:.3g} "
            f"index={self.stiffness_index:.3g} "
            f"cond={self.jacobian_cond:.2g} "
            f"scale_spread={self.state_scale_spread:.2g}"
        )


def _restricted_jacobian(rhs, y0: jnp.ndarray, idxs: np.ndarray, t0: float):
    """Jacobian ``∂rhs/∂y`` at ``(t0, y0)`` restricted to ``idxs``.

    Returns a concrete ``np.ndarray``. Raises if ``rhs``/``y0`` carry JAX
    tracers (i.e. this was called inside a transform) — the caller is
    expected to run eagerly.
    """
    jac = jax.jacfwd(lambda y: rhs(t0, y))(y0)
    try:
        jac_np = np.asarray(jac)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ) as exc:  # pragma: no cover - defensive
        raise RuntimeError(
            "stiffness analysis needs a concrete Jacobian but got JAX "
            "tracers — run it eagerly, outside grad/jvp/vmap."
        ) from exc
    return jac_np[np.ix_(idxs, idxs)]


def classify_spectrum(
    name: str,
    dim: int,
    eigenvalues: np.ndarray,
    *,
    dt: float = 1.0,
    max_explicit_substeps: float = DEFAULT_MAX_EXPLICIT_SUBSTEPS,
    jacobian_cond: float = float("nan"),
    state_scale_spread: float = float("nan"),
) -> GroupStiffness:
    """Build a :class:`GroupStiffness` verdict from an eigenvalue spectrum.

    Stiff ⇔ the **stiffness index** ``spectral_abscissa × dt`` — the
    stability-limited substeps an explicit method is forced into over one
    solve interval — exceeds ``max_explicit_substeps``. Keying on the
    spectral abscissa (fastest *decay* rate) rather than ``|λ|`` is what
    excludes fast oscillators: their large eigenvalue is imaginary, so
    ``spectral_abscissa`` stays small and the verdict stays explicit, as
    it should (an explicit solver resolves an oscillation by accuracy, not
    stability).
    """
    re = eigenvalues.real
    abs_im = np.abs(eigenvalues.imag)
    decay = -re  # positive for decaying modes
    decaying = decay > 0
    spectral_abscissa = float(decay[decaying].max()) if decaying.any() else 0.0

    # "Active" decaying modes: non-negligible relative to the fastest.
    floor = _ACTIVE_FLOOR_FRAC * max(spectral_abscissa, 1e-300)
    active = decay[decaying & (decay > floor)]
    if active.size:
        min_active = float(active.min())
        ratio = spectral_abscissa / min_active
    else:
        min_active = spectral_abscissa
        ratio = 1.0

    stiffness_index = spectral_abscissa * dt
    stiff = stiffness_index > max_explicit_substeps
    return GroupStiffness(
        name=name,
        dim=dim,
        jacobian_cond=jacobian_cond,
        state_scale_spread=state_scale_spread,
        spectral_abscissa=spectral_abscissa,
        max_abs_im=float(abs_im.max()) if abs_im.size else 0.0,
        min_active_decay=min_active,
        stiffness_ratio=ratio,
        dt=dt,
        stiffness_index=stiffness_index,
        stiff=stiff,
        eigenvalues=eigenvalues,
    )


def analyze_groups(
    composite: "Composite",
    *,
    y0: jnp.ndarray | None = None,
    groups: dict[str, list[str]] | None = None,
    t0: float = 0.0,
    dt: float = 1.0,
    max_explicit_substeps: float = DEFAULT_MAX_EXPLICIT_SUBSTEPS,
) -> dict[str, GroupStiffness]:
    """Classify the stiffness of each continuous group of a composite.

    For each group, restricts the composite RHS Jacobian to the group's
    own evolving states (so off-group, Lie-frozen variables don't pollute
    the spectrum), measures the eigenvalues at ``y0``, and classifies.

    Parameters
    ----------
    composite:
        The wired composite.
    y0:
        State to linearize about. ``None`` uses
        ``composite.initial_state_vec()``. Must be concrete (eager).
    groups:
        Group map ``{name: [proc, ...]}``. ``None`` uses
        ``composite.auto_groups()``.
    t0:
        Time at which to evaluate the Jacobian.
    dt:
        Solve interval the explicit-step budget is measured against
        (typically the Scheduler's ``macro_dt``).
    max_explicit_substeps:
        Stiffness-index threshold (stability substeps per ``dt``) above
        which a group is called stiff.

    Returns
    -------
    ``{group_name: GroupStiffness}``.
    """
    if isinstance(y0, jax.core.Tracer):
        raise RuntimeError(
            "analyze_groups needs a concrete y0 — it was given a JAX "
            "tracer. Run stiffness analysis eagerly, outside "
            "grad/jvp/vmap."
        )
    keys = composite.store_keys()
    state = (
        composite.initial_state_vec(keys) if y0 is None else jnp.asarray(y0)
    )
    groups = groups if groups is not None else composite.auto_groups()

    # A batched (population) y0 has trailing axis n_vars and one leading row
    # per individual. The solver choice is a structural property of the shared
    # rate laws — essentially constant across a perturbation population — so
    # linearize about one representative member. This keeps the Jacobian 2-D;
    # jacfwd on a batched state would yield a (batch, n_vars, batch, n_vars)
    # tensor that eigvals cannot consume. Heterogeneous-parameter populations
    # spanning a stiff/non-stiff boundary are out of this assumption's scope —
    # warm_up on the stiffest member if a population straddles it.
    if state.ndim > 1:
        state = state.reshape(-1, state.shape[-1])[0]

    out: dict[str, GroupStiffness] = {}
    for gname, proc_names in groups.items():
        idxs = np.asarray(composite.evolved_indices(proc_names, keys))
        if idxs.size == 0:
            out[gname] = GroupStiffness(
                name=gname,
                dim=0,
                spectral_abscissa=0.0,
                max_abs_im=0.0,
                min_active_decay=0.0,
                stiffness_ratio=1.0,
                dt=dt,
                stiffness_index=0.0,
                stiff=False,
                eigenvalues=np.array([], dtype=complex),
            )
            continue
        rhs, _ = composite.build_rhs(proc_names)
        jac = _restricted_jacobian(rhs, state, idxs, t0)
        eig = np.linalg.eigvals(jac)
        try:
            cond = float(np.linalg.cond(jac))
        except np.linalg.LinAlgError:
            cond = float("inf")
        mags = np.abs(np.asarray(state)[idxs])
        nz = mags[mags > 0]
        spread = float(mags.max() / nz.min()) if nz.size else float("inf")
        out[gname] = classify_spectrum(
            gname,
            int(idxs.size),
            eig,
            dt=dt,
            max_explicit_substeps=max_explicit_substeps,
            jacobian_cond=cond,
            state_scale_spread=spread,
        )
    return out
