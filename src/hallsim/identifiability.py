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
    qts = {
        a: jnp.asarray(sorted(problem.data[a]), dtype=float) for a in arms
    }

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
    # badly), so the jacfwd below reuses them instead of re-deriving under trace.
    fn(theta0)
    jac = jax.jacfwd(fn)(theta0)
    return np.asarray(jac, dtype=float), names


@dataclasses.dataclass
class IdentifiabilityReport:
    """Local identifiability of a fit, from the Fisher information ``JᵀJ``."""

    names: list[str]
    verdict: dict[str, str]  # name -> "identifiable" | "practical" | "structural"
    rel_sensitivity: dict[str, float]  # column norm, normalized to the max
    std_decades: dict[str, float]  # 1σ log10 uncertainty (inf if unbounded)
    eigenvalues: np.ndarray  # Fisher spectrum, ascending
    correlation: np.ndarray  # parameter correlation matrix (n × n)
    confounded: list[tuple[str, str, float]]  # (a, b, corr) with |corr|≥tol
    recommended_freeze: list[str]

    def __str__(self) -> str:
        order = {"structural": 0, "practical": 1, "identifiable": 2}
        rows = sorted(
            self.names, key=lambda n: (order[self.verdict[n]], n)
        )
        w = max(len(n) for n in self.names)
        lines = [
            "Identifiability (Fisher information JᵀJ, log10-param space)",
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
    refitting (the Dalle Pezze 2014 reduction step)."""
    jac, names = sensitivity_jacobian(problem, params)
    return report_from_jacobian(
        jac,
        names,
        struct_tol=struct_tol,
        corr_tol=corr_tol,
        std_tol=std_tol,
    )


def report_from_jacobian(
    jac: np.ndarray,
    names: list[str],
    *,
    struct_tol: float = 1e-6,
    corr_tol: float = 0.95,
    std_tol: float = 1.0,
) -> IdentifiabilityReport:
    """Identifiability verdicts from a sensitivity Jacobian ``∂preds/∂θ``
    (shape ``(n_residuals, n_params)``) and its parameter names — the pure
    linear-algebra core of :func:`identifiability_report`, separated so it can
    be exercised on a synthetic Jacobian without a model solve."""
    jac = np.asarray(jac, dtype=float)
    n = len(names)
    col = np.linalg.norm(jac, axis=0)
    rel = col / max(col.max(), 1e-300)

    fim = jac.T @ jac
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
