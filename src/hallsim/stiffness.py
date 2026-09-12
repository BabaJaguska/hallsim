"""Per-group stiffness analysis for automatic solver selection.

Stiffness is a property of the local Jacobian spectrum, not of a model's name.
A group is stiff *for an explicit solver* when its fastest **dissipative** mode
is far faster than the dynamics being resolved: the decayed mode leaves the
solution smooth, so accuracy would permit large steps, but an explicit method's
step is bounded by stability (``Δt ≲ 2/|λ|``) and is forced tiny anyway.

**Oscillation is not stiffness.** A fast oscillator has large-*imaginary*
eigenvalues, and resolving it already demands ``Δt ~ 1/ω``, so an explicit
solver is accuracy-limited rather than stability-limited and handles it fine.
The discriminator here is therefore the spectral abscissa — the fastest *decay*
rate — not raw eigenvalue magnitude.

The spectrum is state-dependent, so this runs **eagerly**: under grad/jvp/vmap
the eigenvalues would be tracers, and :func:`analyze_groups` raises on a traced
state. The Scheduler resolves the verdict once and reuses it under tracing.
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


class StiffnessInconclusive(RuntimeError):
    """The matrix-free spectrum estimate did not converge, even to the single
    dominant eigenvalue. A property of the group at this state, not of the
    trace; the caller routes the group by a safe default."""


class StiffnessNotConcrete(RuntimeError):
    """Reached under a trace, so there is no spectrum to analyse.

    Distinct from a device or resource failure, which is also a
    ``RuntimeError`` (``jax.errors.JaxRuntimeError``) and must not be
    mistaken for a cold trace.
    """


# A dissipative eigenvalue is "active" (counts toward the stiffness ratio)
# only if its decay rate is a non-negligible fraction of the fastest one;
# this drops the numerically-zero / conserved-quantity modes (Re λ ≈ 0)
# that would otherwise blow the ratio up for any system with a
# conservation law or an oscillator sitting near the imaginary axis.
_ACTIVE_FLOOR_FRAC = 1e-9

#: Group dimension up to which the Jacobian is formed densely. Above it the
#: spectrum's extremes are estimated matrix-free, and ``eigenvalues`` on the
#: verdict is that subset rather than the full spectrum.
DENSE_JACOBIAN_MAX_DIM = 512

#: Extremal eigenvalues requested from the iterative solver.
ITERATIVE_EIGS_K = 32

#: Relative accuracy asked of the iterative solver. The verdict reads the
#: spectral abscissa to a factor; at machine precision a clustered extreme (a
#: diffusion chain's top eigenvalues differ in the fourth digit) never
#: converges.
ITERATIVE_EIGS_TOL = 1e-3

#: Krylov subspace the iterative solver works in. The spectrum's extreme
#: converges with the subspace, not the tolerance: a 1024-ring's top 32
#: eigenvalues (a gapless cluster) take 1 600 products at 256 vectors and
#: never converge at ARPACK's default of 65.
ITERATIVE_EIGS_NCV = 256

#: When the iterative estimate does not converge, the Jacobian is formed
#: densely up to this size (about 20 s at 4096, once per plan, cached).
DENSE_FALLBACK_MAX_DIM = 4096

#: Colours the sparse Jacobian may need before the Gershgorin certificate
#: switches from the coloured pass to a column-chunked one.
GERSHGORIN_MAX_COLOURS = 64

#: Columns per chunk of the dense Gershgorin pass: one batched directional
#: derivative per chunk, memory ``chunk × n`` rather than ``n × n``.
GERSHGORIN_CHUNK = 256

#: Entries the sparsity pattern may hold before it is not built at all: a
#: block port declares every element against every other, and materialising
#: that as a pattern is an ``n²`` Python list before any colouring.
GERSHGORIN_MAX_ENTRIES = 2_000_000


@dataclass
class GroupStiffness:
    """Jacobian-spectrum verdict for one continuous group at a state.

    Attributes
    ----------
    name, dim:
        Group label and its number of evolving states.
    spectral_abscissa:
        ``max(-Re λ)`` over decaying modes — the fastest dissipation rate, 0 if
        nothing decays. This, not ``|λ|``, is what makes an explicit solver
        stability-limited.
    max_abs_im:
        ``max|Im λ|`` — the fastest oscillation frequency.
    min_active_decay, stiffness_ratio:
        Slowest non-negligible decay rate, and the spread
        ``spectral_abscissa / min_active_decay``. Diagnostic only: a wide
        spread doesn't imply stiffness if the fastest rate is itself slow.
    dt, stiffness_index, stiff:
        The solve interval the verdict was computed against,
        ``spectral_abscissa × dt`` (stability-limited substeps per interval),
        and whether that exceeds the substep budget.
    jacobian_cond:
        Condition number of the restricted Jacobian. An implicit solver's
        Newton step solves a linear system in this matrix, so ``≫ 1e6`` forces
        tiny steps *independently of the error tolerance* — the one failure an
        explicit solver is immune to.
    state_scale_spread:
        ``max|y| / min nonzero |y|`` at ``y0``. A wide spread is the usual
        source of a large ``jacobian_cond``, and argues for
        non-dimensionalising before implicit integration.
    eigenvalues:
        Raw restricted-Jacobian spectrum, for inspection.
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
    #: True when ``spectral_abscissa`` is a Gershgorin upper bound from the
    #: sparse Jacobian rather than a measured eigenvalue — enough to certify
    #: "not stiff" without an eigenvalue estimate.
    bounded: bool = False

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


def _concrete(value):
    """``value`` as a NumPy array, or raise if it is still a tracer."""
    try:
        return np.asarray(value)
    except (
        jax.errors.TracerArrayConversionError,
        jax.errors.ConcretizationTypeError,
    ) as exc:  # pragma: no cover - defensive
        raise StiffnessNotConcrete(
            "stiffness analysis needs concrete values but got JAX tracers — "
            "call Scheduler.warm_up(y0) once eagerly before differentiating."
        ) from exc


def _restricted_fn(rhs, y0: jnp.ndarray, idxs: np.ndarray, t0: float):
    """``rhs`` as a function of the group's own evolving states alone."""

    def g(v):
        return rhs(t0, y0.at[idxs].set(v))[idxs]

    return g


def _restricted_jacobian(rhs, y0: jnp.ndarray, idxs: np.ndarray, t0: float):
    """Dense ``(len(idxs), len(idxs))`` Jacobian at ``(t0, y0)``.

    Differentiates the restricted function, so the cost scales with the
    group's dimension rather than the whole store's.
    """
    g = _restricted_fn(rhs, y0, idxs, t0)
    return _concrete(jax.jacfwd(g)(y0[idxs]))


def _extremal_eigenvalues(
    rhs, y0: jnp.ndarray, idxs: np.ndarray, t0: float, k: int
) -> np.ndarray:
    """The ``k`` largest-magnitude eigenvalues of the restricted Jacobian,
    without forming it.

    Arnoldi over a JVP operator: each matrix-vector product costs one
    directional derivative of the RHS, so the spectrum's extremes come out in
    a few thousand RHS evaluations and O(ncv·n) memory instead of n forward passes and
    an n×n matrix. The stiffness verdict reads only the spectral abscissa, and
    the fastest-decaying mode of a dissipative system is a largest-magnitude
    eigenvalue.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    g = _restricted_fn(rhs, y0, idxs, t0)
    v0 = y0[idxs]
    n = int(idxs.size)

    jvp = jax.jit(lambda v: jax.jvp(g, (v0,), (v,))[1])
    _concrete(jvp(jnp.zeros_like(v0)))  # surface a traced RHS before ARPACK

    def matvec(v):
        return np.asarray(jvp(jnp.asarray(v, dtype=v0.dtype)))

    op = LinearOperator((n, n), matvec=matvec, dtype=np.asarray(v0).dtype)
    k = max(1, min(k, n - 2))
    try:
        return np.asarray(
            eigs(
                op,
                k=k,
                which="LM",
                tol=ITERATIVE_EIGS_TOL,
                ncv=min(n, max(ITERATIVE_EIGS_NCV, 2 * k + 1)),
                return_eigenvectors=False,
            )
        )
    except ArpackNoConvergence as exc:
        found = np.asarray(exc.eigenvalues)
        if found.size:
            # What converged is the extreme, which is all the verdict reads.
            return found
        if n <= DENSE_FALLBACK_MAX_DIM:
            return np.linalg.eigvals(_restricted_jacobian(rhs, y0, idxs, t0))
        raise StiffnessInconclusive(
            f"the extremal spectrum of a {n}-state group did not converge "
            f"in a {ITERATIVE_EIGS_NCV}-vector Krylov space, and the group "
            "is too large to form its Jacobian densely"
        ) from exc


def _pattern_entries(composite) -> int:
    """How many entries the composite's Jacobian pattern would hold, from
    port widths alone — before any of them is materialised."""
    from hallsim.store import as_paths

    total = 0
    for name, proc in composite.continuous_processes().items():
        topo = composite.topology.get(name, {})
        for port, deps in proc.port_dependencies().items():
            width = len(as_paths(topo[port]))
            total += width * sum(len(as_paths(topo[d])) for d in deps)
    return total


def _gershgorin_abscissa(composite, rhs, y0, idxs, t0: float):
    """An upper bound on the group's spectral abscissa from Gershgorin's
    discs over its sparse Jacobian, formed in as many directional
    derivatives as the pattern has colours — three for a chain, tens for a
    reaction network. ``None`` when the pattern needs more colours than
    the estimate it would replace."""
    from hallsim.structure import colour_columns, jacobian_pattern

    idxs = np.asarray(idxs)
    n = int(idxs.size)
    g = _restricted_fn(rhs, y0, idxs, t0)
    v0 = y0[idxs]
    batched = jax.jit(jax.vmap(lambda sd: jax.jvp(g, (v0,), (sd,))[1]))
    diag = np.zeros(n)
    radius = np.zeros(n)
    n_colours = None
    if _pattern_entries(composite) <= GERSHGORIN_MAX_ENTRIES:
        pattern = jacobian_pattern(composite)
        local = {int(gi): i for i, gi in enumerate(idxs)}
        kept = [
            (local[int(r)], local[int(c)])
            for r, c in zip(pattern.rows, pattern.cols)
            if int(r) in local and int(c) in local
        ]
        if not kept:
            return 0.0
        rows = np.asarray([r for r, _ in kept], dtype=np.int32)
        cols = np.asarray([c for _, c in kept], dtype=np.int32)
        colour, n_colours = colour_columns(n, rows, cols)
    if n_colours is not None and n_colours <= GERSHGORIN_MAX_COLOURS:
        seeds = np.zeros((n_colours, n))
        seeds[colour, np.arange(n)] = 1.0
        tangents = np.asarray(
            _concrete(batched(jnp.asarray(seeds, dtype=v0.dtype)))
        )
        values = tangents[colour[cols], rows]
        on = rows == cols
        diag[rows[on]] = values[on]
        np.add.at(radius, rows[~on], np.abs(values[~on]))
        return float(np.max(radius - diag))
    # No usable sparsity (a block port declares none): walk the columns in
    # chunks, one batched derivative per chunk, never holding n × n.
    for start in range(0, n, GERSHGORIN_CHUNK):
        columns = np.arange(start, min(n, start + GERSHGORIN_CHUNK))
        seeds = np.zeros((GERSHGORIN_CHUNK, n))
        seeds[np.arange(columns.size), columns] = 1.0
        tangents = np.asarray(
            _concrete(batched(jnp.asarray(seeds, dtype=v0.dtype)))
        )[: columns.size]
        block = tangents.T  # J[:, columns]
        k = np.arange(columns.size)
        diag[columns] = block[columns, k]
        block = np.abs(block)
        block[columns, k] = 0.0
        radius += block.sum(axis=1)
    return float(np.max(radius - diag))


def classify_spectrum(
    name: str,
    dim: int,
    eigenvalues: np.ndarray,
    *,
    dt: float = 1.0,
    max_explicit_substeps: float = DEFAULT_MAX_EXPLICIT_SUBSTEPS,
    jacobian_cond: float = float("nan"),
    state_scale_spread: float = float("nan"),
    bounded: bool = False,
) -> GroupStiffness:
    """Build a :class:`GroupStiffness` verdict from an eigenvalue spectrum.

    Stiff ⇔ ``spectral_abscissa × dt`` exceeds ``max_explicit_substeps``.
    Keying on the spectral abscissa rather than ``|λ|`` is what excludes fast
    oscillators: their large eigenvalue is imaginary, so the verdict stays
    explicit — correctly, since an explicit solver resolves an oscillation by
    accuracy, not stability.
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
        bounded=bounded,
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
    """``{group_name: GroupStiffness}`` for each continuous group.

    Restricts the composite RHS Jacobian to each group's own evolving states,
    so off-group Lie-frozen variables don't pollute the spectrum, then
    classifies the eigenvalues at ``y0`` (concrete, eager — defaults to
    ``initial_state_vec()``). ``dt`` is the interval the explicit-step budget
    is measured against, typically the Scheduler's ``macro_dt``.
    """
    if isinstance(y0, jax.core.Tracer):
        raise StiffnessNotConcrete(
            "analyze_groups needs a concrete y0 — it was given a JAX "
            "tracer. Run stiffness analysis eagerly, outside "
            "grad/jvp/vmap."
        )
    keys = composite.store_keys()
    state = (
        composite.initial_state_vec(keys) if y0 is None else jnp.asarray(y0)
    )
    groups = groups if groups is not None else composite.auto_groups()

    # Linearize a batched y0 about one representative member: the verdict is a
    # property of the shared rate laws, and jacfwd on a batched state gives a
    # 4-D tensor eigvals cannot consume. A population straddling the
    # stiff/non-stiff boundary needs warm_up on its stiffest member.
    if state.ndim > 1:
        state = state.reshape(-1, state.shape[-1])[0]

    out: dict[str, GroupStiffness] = {}
    for gname, proc_names in groups.items():
        idxs = _concrete(composite.evolved_indices(proc_names, keys))
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
        bounded = False
        if idxs.size <= DENSE_JACOBIAN_MAX_DIM:
            jac = _restricted_jacobian(rhs, state, idxs, t0)
            eig = np.linalg.eigvals(jac)
            try:
                cond = float(np.linalg.cond(jac))
            except np.linalg.LinAlgError:
                cond = float("inf")
        else:
            # cond needs the smallest singular value, which no extremal
            # method gives cheaply; it is diagnostic, so it abstains.
            # A Gershgorin bound below the budget is a certificate of "not
            # stiff" in a handful of derivatives; only a group it cannot
            # clear pays for the eigenvalue estimate.
            bound = _gershgorin_abscissa(composite, rhs, state, idxs, t0)
            if (
                bound is not None
                and np.isfinite(bound)
                and bound * dt <= max_explicit_substeps
            ):
                eig = np.asarray([-bound + 0j])
                bounded = True
            else:
                eig = _extremal_eigenvalues(
                    rhs, state, idxs, t0, ITERATIVE_EIGS_K
                )
                bounded = False
            cond = float("nan")
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
            bounded=bounded,
        )
    return out
