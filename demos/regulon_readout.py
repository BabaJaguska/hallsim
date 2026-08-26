"""Score the regulon readout head genome-wide against GSE248823.

The canonical reporters read six curated transcripts off the multi-hallmark
composite. This asks the same composite a much larger question: expand its
three modelled TF activities through the CollecTRI prior and see how far the
predicted transcriptome-wide log2 fold change agrees with the measured one.

Two numbers matter and they are not the same:

* **Sign agreement** is a zero-parameter prediction — gains are positive, so
  direction comes from the prior and the modelled activity deltas alone.
* **Correlation** after :func:`fit_gains` adds one gain per TF (three
  parameters for ~10^3 genes).

Both are compared against a null that permutes the prior's gene assignment,
which preserves each TF's marginal sign balance and destroys only *which*
gene each edge points at.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

import jax.numpy as jnp

from demos.multi_hallmark_calibrate import (
    ARM_CONDITIONS,
    ARMS,
    PLATFORM,
    SAMPLE_POSITION_GROUPS,
    SERIES_MATRIX,
    build_problem,
)
from hallsim.gene_reporters import (
    GeneExpressionDataset,
    log2_fold_change,
    zerophase_mean,
    zerophase_rms_raw,
)
from hallsim.regulon import (
    ActivityBinding,
    Regulon,
    RegulonHead,
    fit_gains,
    score_predictions,
)

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent

# Summaries match the canonical reporters reading the same store paths, so the
# activity delta is collapsed the same way the validated readout collapses it.
TF_BINDINGS = [
    ActivityBinding(
        observable="gz06/x",
        tf="TP53",
        summary=zerophase_rms_raw(tau=0.75),
        description="p53 (Geva-Zatorsky x) — oscillates, read as envelope.",
        reference="Geva-Zatorsky et al. 2006, Mol Syst Biol 2:2006.0033",
    ),
    ActivityBinding(
        observable="dp14/FoxO3a",
        tf="FOXO3",
        summary=zerophase_mean(tau=2.0),
        description="Unphosphorylated FoxO3a — the transcriptionally active pool.",
        reference="Dalle Pezze et al. 2014, PLoS Comput Biol 10:e1003728",
    ),
    ActivityBinding(
        observable="nfkb/NFkBn",
        tf="RELA",
        summary=zerophase_mean(tau=0.75),
        description="Nuclear NF-κB — the transcriptionally competent pool.",
        reference="Ihekwaba et al. 2004, Syst Biol 1:93–103",
    ),
]

QUERY_DAYS = (7.0, 14.0)

# Arm → {day: sample group}, and the shared day-0 reference. Mirrors the
# calibration demo's "baseline" normalization so model and data contrast match.
DATA_ARMS = {
    "DDIS_vs_ctrl": {7.0: "ETOPOSIDE_D07", 14.0: "ETOPOSIDE_D14"},
    "RAPA_vs_ctrl": {7.0: "ETOPOSIDE_RAPA_D07", 14.0: "ETOPOSIDE_RAPA_D14"},
}
BASELINE_GROUP = "ETOPOSIDE_D00"


def model_activity_deltas(problem, arms, normalization: str = "paired"):
    """``(n_cond, n_tfs)`` modelled Δ activity, conditions in arm × day order.

    ``paired`` references the untreated arm at the matched time, ``baseline``
    the arm's own t=0. The two disagree on p53's sign; see
    ``docs/diary.md``.
    """
    params = problem.initial_params()
    if normalization == "baseline":
        rows = [
            np.asarray(problem.model_lfc(params, arm, list(QUERY_DAYS))).T
            for arm in arms
        ]
        return np.concatenate(rows, axis=0)

    qt = jnp.asarray(QUERY_DAYS)
    summaries = {}
    for cond in ("ctrl", *(ARM_CONDITIONS[a] for a in arms)):
        ts, trajs = problem.simulate_reporters(params, cond)
        summaries[cond] = np.asarray(
            problem._reporter_summaries(ts, trajs, qt)
        )
    return np.vstack(
        [
            np.log2(summaries[ARM_CONDITIONS[a]] / summaries["ctrl"]).T
            for a in arms
        ]
    )


def measured_deltas(dataset, arms, genes):
    """``(n_cond, n_genes)`` measured log2FC on the head's gene axis."""
    groups = dataset.sample_groups
    rows = []
    for arm in arms:
        for day in QUERY_DAYS:
            lfc = log2_fold_change(
                dataset.gene_expr,
                groups[DATA_ARMS[arm][day]],
                groups[BASELINE_GROUP],
            )
            rows.append(lfc.reindex(genes).to_numpy())
    return np.vstack(rows)


def permutation_null(head, activity, observed, n: int = 200, seed: int = 0):
    """Rescore with the prior's rows shuffled — same sign balance per TF,
    wrong gene assignment. Returns (mean, sd) of sign agreement."""
    rng = np.random.default_rng(seed)
    signs = np.asarray(head.signs)
    out = []
    for _ in range(n):
        permuted = signs[rng.permutation(signs.shape[0])]
        pred = np.asarray(activity) @ (permuted * np.exp(head.log_gain)).T
        out.append(score_predictions(pred, observed).sign_agreement)
    return float(np.mean(out)), float(np.std(out))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    problem = build_problem(reporters=TF_BINDINGS)
    dataset = GeneExpressionDataset.from_series_matrix(
        SERIES_MATRIX,
        PLATFORM,
        sample_groups={},
        sample_position_groups=SAMPLE_POSITION_GROUPS,
    )
    measured_genes = list(dataset.gene_expr.index)

    tfs = [b.tf for b in TF_BINDINGS]
    regulon = Regulon.from_collectri(tfs, restrict_to=measured_genes)
    log.info("Regulon coverage: %s", regulon.coverage)

    observed = measured_deltas(dataset, ARMS, regulon.genes)

    for normalization in ("paired", "baseline"):
        activity = model_activity_deltas(problem, ARMS, normalization)
        log.info("── %s normalization ──", normalization)
        for arm, row in zip(
            [f"{a} d{int(d)}" for a in ARMS for d in QUERY_DAYS], activity
        ):
            log.info("  Δactivity %-22s %s", arm, np.round(row, 4))

        head = RegulonHead(regulon)
        log.info(
            "  zero-parameter: %s",
            score_predictions(np.asarray(head(activity)), observed),
        )
        log.info(
            "  permuted-prior null sign: %.3f ± %.3f",
            *permutation_null(head, activity, observed),
        )

        fitted, loss = fit_gains(head, activity, observed)
        predicted = np.asarray(fitted(activity))
        log.info("  fitted gains: %s", np.round(np.exp(fitted.log_gain), 4))
        log.info(
            "  fitted (%d params): %s  mse=%.4f",
            len(regulon.tfs),
            score_predictions(predicted, observed),
            loss,
        )
        log.info(
            "  |measured| >= 0.5: %s",
            score_predictions(predicted, observed, min_abs_observed=0.5),
        )
        for j, tf in enumerate(regulon.tfs):
            own = np.asarray(regulon.signs[:, j]) != 0
            solo = np.outer(activity[:, j], np.asarray(regulon.signs[own, j]))
            log.info(
                "    %-6s targets=%4d  %s",
                tf,
                int(own.sum()),
                score_predictions(
                    solo, observed[:, own], min_abs_observed=0.5
                ),
            )


if __name__ == "__main__":
    main()
