"""Fisher-information identifiability analysis for a ``CalibrationProblem``.

Because HallSim composites are differentiable end-to-end, the local
identifiability of a fit is one Jacobian away: the sensitivity of every
reporter prediction to every fitted parameter. This is the autodiff-native
analogue of the multistart-ensemble identifiability screens in frameworks
like COPASI (and the MOTA analysis in Dalle Pezze 2014).

For each fitted parameter it reports one of:

* **structural** — the parameter moves no reporter (its Jacobian column is
  ~0). No data of this kind can ever constrain it; a parameter with no
  downstream observable (e.g. a ROS rate when nothing reads ROS) lands here.
* **practical** — the parameter is confounded with another (|correlation|
  near 1) or lies in a sloppy near-null eigenmode of the Fisher information,
  so only some combination of parameters is determined, not it alone.
* **identifiable** — determined by the data, with a finite log-space
  uncertainty (reported in decades).

The recommended action is to fix the non-identifiable
parameters at their literature values and refit only the identifiable set.

Sensitivities are taken in ``log10`` parameter space (matching the loss's
log-normal prior and the optimizer's log treatment), so a rate constant and
an O(1) coefficient are compared on equal footing. Run under
``jax_enable_x64`` — the Fisher spectrum spans many orders of magnitude and
the eigendecomposition needs float64 to resolve the sloppy tail.
"""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np


def _prediction_fn(problem, base_params: dict, names: list[str]):
    """Map a log10-parameter vector (in ``names`` order) to the concatenated
    reporter predictions over every fit arm — the residual vector the loss
    squares, built from the same :meth:`CalibrationProblem.model_lfc`."""
    arms = list(problem.fit_arms)
    qts = {a: jnp.asarray(sorted(problem.data[a]), dtype=float) for a in arms}

    def preds(theta_log):
        p = dict(base_params)
        for i, n in enumerate(names):
            p[n] = 10.0 ** theta_log[i]
        return jnp.concatenate(
            [problem.model_lfc(p, a, qts[a]).reshape(-1) for a in arms]
        )

    return preds


def sensitivity_jacobian(problem, params: dict | None = None):
    """Jacobian ``∂(reporter predictions)/∂(log10 params)`` at ``params``
    (default: the problem's initial params), plus the parameter-name order.

    Shape ``(n_residuals, n_params)`` where ``n_residuals`` = Σ over fit arms
    of ``n_reporter × n_timepoint``. Forward-mode (``jacfwd``): parameters are
    few, residuals many."""
    params = dict(params if params is not None else problem.initial_params())
    names = list(problem.param_refs.keys())
    theta0 = jnp.asarray(
        [jnp.log10(jnp.asarray(float(params[n]))) for n in names]
    )
    fn = _prediction_fn(problem, params, names)
    # Eager warm-up: the first prediction populates the problem's cached
    # conservation laws (a concrete-only equilibration diagnostic that traces
    # badly), so jacfwd reuses them instead of re-deriving.
    fn(theta0)
    jac = jax.jacfwd(fn)(theta0)
    return np.asarray(jac, dtype=float), names


def residual_scale(problem, params: dict, n_fitted: int | None = None):
    """The size of a miss, in the reporters' own units, estimated from the
    fit's own leftovers: ``sqrt(SSR / (n_residuals - n_fitted))``.

    The Fisher information is ``JᵀJ / σ²``, so every uncertainty it yields
    scales with this number and a report that omits it has silently asserted
    ``σ = 1``. On a log2 fold-change readout that claims each measurement is
    good to a factor of two, which is far looser than a replicated assay, and
    it makes parameters the data does constrain look unconstrained. Estimated
    rather than declared so the verdicts need no input: this is the residual
    standard error every regression reports, and on a misspecified model it
    measures the model's own error, which is the honest scale to judge a
    parameter against.

    ``n_fitted`` defaults to the number of fitted references.
    """
    if n_fitted is None:
        n_fitted = len(problem.param_refs)
    res = []
    for arm in problem.fit_arms:
        times = sorted(problem.data[arm])
        sim = np.asarray(
            problem.model_lfc(params, arm, jnp.asarray(times, dtype=float))
        )
        for i, rep in enumerate(problem.reporters):
            for j, t in enumerate(times):
                res.append(
                    float(sim[i, j])
                    - float(problem.data[arm][t][rep.gene_symbol])
                )
    res = np.asarray(res, dtype=float)
    dof = max(res.size - int(n_fitted), 1)
    return float(np.sqrt(float(res @ res) / dof))


@dataclasses.dataclass
class IdentifiabilityReport:
    """Local identifiability of a fit, from the Fisher information ``JᵀJ``.

    ``fisher_diag`` is ``diag(JᵀJ)`` per parameter — the log10-space precision
    the data carries about it, and the scale a prior must reach to do anything.
    """

    names: list[str]
    verdict: dict[
        str, str
    ]  # name -> "identifiable" | "practical" | "structural"
    rel_sensitivity: dict[str, float]  # column norm, normalized to the max
    std_decades: dict[str, float]  # 1σ log10 uncertainty (inf if unbounded)
    eigenvalues: np.ndarray  # Fisher spectrum, ascending
    correlation: np.ndarray  # parameter correlation matrix (n × n)
    confounded: list[tuple[str, str, float]]  # (a, b, corr) with |corr|≥tol
    recommended_freeze: list[str]
    fisher_diag: dict[str, float] = dataclasses.field(default_factory=dict)
    sigma: float = 1.0  # residual scale the uncertainties are in

    @property
    def condition_number(self) -> float:
        """Fisher spectrum spread. Above ~1e16 the flat directions are below
        float64's noise floor and the problem is singular in practice."""
        return float(self.eigenvalues[-1] / max(self.eigenvalues[0], 1e-300))

    def __str__(self) -> str:
        order = {"structural": 0, "practical": 1, "identifiable": 2}
        rows = sorted(self.names, key=lambda n: (order[self.verdict[n]], n))
        w = max(len(n) for n in self.names)
        lines = [
            "Identifiability (Fisher information JᵀJ/σ², log10-param "
            f"space, σ = {self.sigma:.3g} in reporter units)",
            f"{'parameter':<{w}}  {'verdict':<12}{'rel.sens':>10}"
            f"{'σ (dec)':>10}",
            "-" * (w + 34),
        ]
        for n in rows:
            s = self.std_decades[n]
            s_str = "  ∞  " if not np.isfinite(s) else f"{s:.2f}"
            lines.append(
                f"{n:<{w}}  {self.verdict[n]:<12}"
                f"{self.rel_sensitivity[n]:>10.2e}{s_str:>10}"
            )
        cond = self.eigenvalues[-1] / max(self.eigenvalues[0], 1e-300)
        lines.append("")
        lines.append(
            f"Fisher spectrum: λ ∈ [{self.eigenvalues[0]:.2e}, "
            f"{self.eigenvalues[-1]:.2e}], condition number {cond:.1e}"
        )
        if self.confounded:
            lines.append("Confounded pairs (|corr| ≥ threshold):")
            for a, b, c in self.confounded:
                lines.append(f"  {a} ~ {b}   corr={c:+.3f}")
        practical = [n for n in self.names if self.verdict[n] == "practical"]
        if self.recommended_freeze:
            lines.append(
                "Recommended freeze (fix at literature value, refit rest): "
                + ", ".join(self.recommended_freeze)
            )
        elif practical:
            lines.append(
                f"No structural/confounded params; {len(practical)} weakly "
                "constrained (prior-dominated): " + ", ".join(practical)
            )
        else:
            lines.append("All fitted parameters are locally identifiable.")
        return "\n".join(lines)


def identifiability_report(
    problem,
    params: dict | None = None,
    *,
    struct_tol: float = 1e-6,
    corr_tol: float = 0.95,
    std_tol: float = 1.0,
    sigma: float | None = None,
) -> IdentifiabilityReport:
    """Fisher-information identifiability of ``problem`` at ``params``.

    A parameter is **structural**-non-identifiable when its relative
    sensitivity (Jacobian column norm / the largest column norm) is below
    ``struct_tol`` — it moves no reporter. Otherwise it is **practical**-non-
    identifiable when its 1σ log-space uncertainty exceeds ``std_tol`` decades
    (from the Fisher-information covariance ``pinv(JᵀJ)``) or it is confounded
    with another parameter at ``|correlation| ≥ corr_tol``. Everything else is
    **identifiable**.

    ``recommended_freeze`` lists the structural parameters plus, for each
    confounded pair, the less-sensitive member — the set to fix before
    refitting (the Dalle Pezze 2014 reduction step).

    ``sigma`` is the size of a miss in the reporters' units; every
    uncertainty here scales with it. It defaults to
    :func:`residual_scale`, measured from this fit's own leftovers, so a
    verdict never rests on an undeclared assumption about the data."""
    params = dict(params if params is not None else problem.initial_params())
    jac, names = sensitivity_jacobian(problem, params)
    if sigma is None:
        sigma = residual_scale(problem, params, n_fitted=len(names))
    return report_from_jacobian(
        jac,
        names,
        struct_tol=struct_tol,
        corr_tol=corr_tol,
        std_tol=std_tol,
        sigma=sigma,
    )


def report_from_jacobian(
    jac: np.ndarray,
    names: list[str],
    *,
    struct_tol: float = 1e-6,
    corr_tol: float = 0.95,
    std_tol: float = 1.0,
    sigma: float = 1.0,
) -> IdentifiabilityReport:
    """Identifiability verdicts from a sensitivity Jacobian ``∂preds/∂θ``
    (shape ``(n_residuals, n_params)``) and its parameter names — the pure
    linear-algebra core of :func:`identifiability_report`, separated so it can
    be exercised on a synthetic Jacobian without a model solve."""
    jac = np.asarray(jac, dtype=float)
    n = len(names)
    col = np.linalg.norm(jac, axis=0)
    rel = col / max(col.max(), 1e-300)

    fim = (jac.T @ jac) / float(sigma) ** 2
    eigval = np.linalg.eigvalsh(fim)
    cov = np.linalg.pinv(fim, rcond=1e-12)
    var = np.clip(np.diag(cov), 0.0, None)
    std = np.sqrt(var)
    # A structurally-flat direction has ~0 sensitivity → pinv drops it →
    # var≈0, which would masquerade as "tight". Force those to unbounded.
    std = np.where(rel < struct_tol, np.inf, std)

    denom = np.outer(std, std)
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.where(denom > 0, cov / denom, 0.0)
    np.fill_diagonal(corr, 1.0)

    confounded: list[tuple[str, str, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rel[i] >= struct_tol and rel[j] >= struct_tol:
                if abs(corr[i, j]) >= corr_tol:
                    confounded.append((names[i], names[j], float(corr[i, j])))

    confounded_names = {a for a, _, _ in confounded} | {
        b for _, b, _ in confounded
    }
    verdict: dict[str, str] = {}
    for i, name in enumerate(names):
        if rel[i] < struct_tol:
            verdict[name] = "structural"
        elif std[i] > std_tol or name in confounded_names:
            verdict[name] = "practical"
        else:
            verdict[name] = "identifiable"

    freeze = [nm for nm in names if verdict[nm] == "structural"]
    for a, b, _ in confounded:
        ia, ib = names.index(a), names.index(b)
        weaker = a if rel[ia] <= rel[ib] else b
        if weaker not in freeze:
            freeze.append(weaker)

    return IdentifiabilityReport(
        names=names,
        verdict=verdict,
        rel_sensitivity={names[i]: float(rel[i]) for i in range(n)},
        std_decades={names[i]: float(std[i]) for i in range(n)},
        eigenvalues=eigval,
        correlation=corr,
        confounded=confounded,
        recommended_freeze=freeze,
        fisher_diag={names[i]: float(fim[i, i]) for i in range(n)},
        sigma=float(sigma),
    )


@dataclasses.dataclass
class FitSetChoice:
    """The largest subset of a candidate pool that can be fitted together,
    from :func:`choose_fit_set`.

    ``keep`` is the chosen set in the order it was admitted; ``drop`` maps
    each rejected candidate to the reason, naming the parameter it
    duplicates where that is the reason. ``std_decades`` is each kept
    parameter's 1σ *within the kept set*, which is the number that will hold
    after the refit — a parameter's uncertainty depends on what else is
    being fitted, so a per-parameter screen cannot report it.
    """

    keep: list[str]
    drop: dict[str, str]
    std_decades: dict[str, float]
    rel_sensitivity: dict[str, float]
    sigma: float

    def __str__(self) -> str:
        w = max((len(n) for n in list(self.keep) + list(self.drop)), default=1)
        lines = [
            f"Fit set: {len(self.keep)} of {len(self.keep) + len(self.drop)} "
            f"candidates (σ = {self.sigma:.3g} in reporter units)",
            f"{'parameter':<{w}}  {'σ (dec)':>9}  reason",
            "-" * (w + 30),
        ]
        for n in self.keep:
            lines.append(f"{n:<{w}}  {self.std_decades[n]:>9.2f}  keep")
        for n, why in self.drop.items():
            lines.append(f"{n:<{w}}  {'—':>9}  drop: {why}")
        return "\n".join(lines)


def choose_fit_set(
    problem,
    params: dict | None = None,
    *,
    struct_tol: float = 1e-6,
    corr_tol: float = 0.95,
    std_tol: float = 1.0,
    sigma: float | None = None,
) -> FitSetChoice:
    """The largest subset of the candidate pool the data can fit together.

    Answers "which parameters should I fit?", which a list of per-parameter
    verdicts cannot: whether a parameter is identifiable depends on what else
    is in the fit, so the verdicts have to be recomputed as the set grows.
    Candidates are admitted in order of how much they move the reporters, and
    one is rejected when it moves no reporter, when it duplicates a parameter
    already admitted (``|correlation| ≥ corr_tol`` *within the admitted set*),
    or when admitting it would leave its own 1σ above ``std_tol`` decades.

    The pool is the problem's own fitted references, so a wider screen is a
    problem built with the wider pool (``build_problem(parameters=...)``).
    ``sigma`` defaults to :func:`residual_scale`.
    """
    params = dict(params if params is not None else problem.initial_params())
    jac, names = sensitivity_jacobian(problem, params)
    if sigma is None:
        sigma = residual_scale(problem, params, n_fitted=len(names))
    jac = np.asarray(jac, dtype=float) / float(sigma)

    col = np.linalg.norm(jac, axis=0)
    rel = col / max(col.max(), 1e-300)
    order = sorted(range(len(names)), key=lambda i: -col[i])

    keep: list[int] = []
    drop: dict[str, str] = {}
    std: dict[str, float] = {}
    for i in order:
        if rel[i] < struct_tol:
            drop[names[i]] = "moves no reporter"
            continue
        trial = keep + [i]
        sub = jac[:, trial]
        cov = np.linalg.pinv(sub.T @ sub, rcond=1e-12)
        sd = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        denom = np.outer(sd, sd)
        with np.errstate(invalid="ignore", divide="ignore"):
            corr = np.where(denom > 0, cov / denom, 0.0)
        worst_j, worst_c = None, 0.0
        for k in range(len(keep)):
            c = abs(float(corr[-1, k]))
            if c > worst_c:
                worst_j, worst_c = keep[k], c
        if worst_c >= corr_tol:
            signed = float(corr[-1, keep.index(worst_j)])
            drop[names[i]] = (
                f"duplicates {names[worst_j]} (corr {signed:+.2f})"
            )
            continue
        if sd[-1] > std_tol:
            drop[names[i]] = f"1σ = {sd[-1]:.2f} decades, above {std_tol:g}"
            continue
        keep = trial
        for k, idx in enumerate(keep):
            std[names[idx]] = float(sd[k])
    return FitSetChoice(
        keep=[names[i] for i in keep],
        drop=drop,
        std_decades=std,
        rel_sensitivity={names[i]: float(rel[i]) for i in range(len(names))},
        sigma=float(sigma),
    )


def log_summary(report: IdentifiabilityReport, logger) -> None:
    """Emit a one-line identifiability summary, escalating to a warning only
    when a parameter is *structurally* non-identifiable (moves no reporter — a
    warning-worthy sign it should be fixed, not fit). Sloppy / prior-dominated
    parameters are expected in systems-biology fits, so they stay at INFO."""
    structural = [n for n in report.names if report.verdict[n] == "structural"]
    n_ident = sum(
        1 for n in report.names if report.verdict[n] == "identifiable"
    )
    if structural:
        logger.warning(
            "identifiability: %d parameter(s) move no reporter and cannot be "
            "fit — %s; fix at literature values (see history.identifiability).",
            len(structural),
            ", ".join(structural),
        )
    logger.info(
        "identifiability: %d/%d parameters data-identifiable, %d confounded "
        "pair(s); see history.identifiability for the full report.",
        n_ident,
        len(report.names),
        len(report.confounded),
    )


# ── structural redundancy, from the symbolic forms alone ─────────────────


@dataclasses.dataclass(frozen=True)
class RedundancyGroup:
    """Parameters the dynamics cannot separate.

    Each member's field sensitivity ``∂f/∂θ`` is the first member's times
    the matching entry of ``ratios``, a factor free of state and time — so
    only one combination of them enters the dynamics, no data of any kind
    can resolve the split, and a fit places it arbitrarily. A ratio of ``1``
    means the sum is what is identifiable.
    """

    parameters: tuple
    ratios: tuple

    def describe(self) -> str:
        first = self.parameters[0]
        parts = [
            f"∂f/∂{p} = {r} · ∂f/∂{first}"
            for p, r in zip(self.parameters[1:], self.ratios[1:])
        ]
        return "; ".join(parts)


@dataclasses.dataclass(frozen=True)
class StructuralReport:
    """What :func:`structural_redundancy` found: ``groups`` of redundant
    parameters, ``inert`` ones the field never reads, and ``unassessed``
    ones whose effect passes through a process with no symbolic form."""

    groups: tuple
    inert: tuple
    unassessed: tuple
    assessed: tuple

    def __str__(self) -> str:
        lines = ["Structural redundancy (from declared symbolic forms)"]
        for g in self.groups:
            lines.append(
                "  redundant: "
                + ", ".join(g.parameters)
                + "  ("
                + g.describe()
                + ")"
            )
        if self.inert:
            lines.append(
                "  inert (never read by the dynamics): "
                + ", ".join(self.inert)
            )
        if self.unassessed:
            lines.append(
                "  unassessed (reach a process with no symbolic form): "
                + ", ".join(self.unassessed)
            )
        if not self.groups and not self.inert:
            lines.append(
                "  no structural redundancy among the assessed parameters"
            )
        return "\n".join(lines)


def _address(param) -> str:
    if isinstance(param, str):
        return param
    return f"{param.process_name}.{param.field}"


def _proportional(sa: dict, sb: dict, forbidden: set):
    """``ρ`` with ``sb = ρ·sa`` on their shared support and ``ρ`` free of
    ``forbidden`` symbols, else ``None``."""
    import sympy

    support = sorted(sa)
    first = support[0]
    ratio = sympy.nsimplify(sympy.cancel(sb[first] / sa[first]), rational=True)
    if ratio.free_symbols & forbidden:
        return None
    for path in support:
        diff = sympy.cancel(sb[path] - ratio * sa[path])
        if diff != 0 and sympy.simplify(diff) != 0:
            return None
    return ratio


def structural_redundancy(composite, params=None) -> StructuralReport:
    """Parameters that are redundant *structurally* — from the declared
    rate laws and stoichiometry, before any data or fit.

    For each parameter ``θ`` the field sensitivity ``∂f/∂θ`` is formed
    symbolically over the composite's store paths
    (:func:`hallsim.structure.symbolic_field`). Two parameters with
    sensitivities proportional by a factor free of state and time enter the
    dynamics only as one combination; such parameters are grouped. A
    parameter with zero sensitivity is inert. One whose symbol reaches a
    process with no symbolic form is unassessed, since that process may do
    anything with it — as is one no form reads while such a process exists.

    ``params`` restricts the analysis to those parameters, each a
    ``"<process>.<field>"`` address or a
    :class:`~hallsim.calibration.ParameterRef`; by default every parameter
    some symbolic form reads is assessed. Distinct from
    :func:`identifiability_report`, which needs a fit and finds *practical*
    confounding in the data.
    """
    import sympy
    from sympy.core.function import AppliedUndef

    from hallsim.structure import symbolic_field

    field = symbolic_field(composite)
    if params is None:
        names = sorted(field.parameters)
    else:
        names = [_address(p) for p in params]
    symbols = {n: sympy.Symbol(n) for n in names}
    from hallsim.sbml_math import TIME

    forbidden = (
        {sympy.Symbol(p) for p in field.derivatives}
        | {sympy.Symbol(p) for p in field.assigned}
        | {TIME}
    )

    opaque_args: set = set()
    for expr in list(field.derivatives.values()) + list(
        field.assigned.values()
    ):
        for applied in sympy.sympify(expr).atoms(AppliedUndef):
            for arg in applied.args:
                opaque_args |= arg.free_symbols
    unassessed = [
        n
        for n in names
        if symbols[n] in opaque_args
        or (n not in field.parameters and field.opaque)
    ]

    sens: dict[str, dict] = {}
    for n in names:
        if n in unassessed:
            continue
        s = {}
        for path, expr in field.derivatives.items():
            expr = sympy.sympify(expr)
            if symbols[n] in expr.free_symbols:
                d = sympy.diff(expr, symbols[n])
                if d != 0:
                    s[path] = d
        sens[n] = s
    inert = [n for n, s in sens.items() if not s]

    buckets: dict = {}
    for n, s in sens.items():
        if s:
            buckets.setdefault(frozenset(s), []).append(n)
    parent = {n: n for n in sens}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for members in buckets.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                if find(a) == find(b):
                    continue
                if _proportional(sens[a], sens[b], forbidden) is not None:
                    parent[find(b)] = find(a)
    grouped: dict = {}
    for n in sens:
        if sens[n]:
            grouped.setdefault(find(n), []).append(n)
    groups = []
    for members in grouped.values():
        if len(members) < 2:
            continue
        members = sorted(members)
        first = members[0]
        ratios = [sympy.Integer(1)] + [
            _proportional(sens[first], sens[m], forbidden) for m in members[1:]
        ]
        groups.append(
            RedundancyGroup(tuple(members), tuple(str(r) for r in ratios))
        )
    groups.sort(key=lambda g: g.parameters)
    return StructuralReport(
        groups=tuple(groups),
        inert=tuple(inert),
        unassessed=tuple(unassessed),
        assessed=tuple(n for n in names if n not in unassessed),
    )
