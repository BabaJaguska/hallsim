"""Data validation — compare simulated trajectories against experimental data.

This module provides tools to validate HallSim simulations against
real biological measurements, particularly ssGSEA pathway scores from
scRNA-seq or bulk RNA-seq experiments.

The core idea (from the architecture diagram):

    Simulated cell state  -->  Map to pathway scores  -->  Compare
    (mTOR, AMPK, ROS...)      (Reactome, KEGG, GO)       with measured
                                                          ssGSEA scores

Workflow
--------
1. Define a mapping from simulation state variables to pathway gene sets
2. Run a simulation (baseline + perturbation)
3. Compute "simulated pathway scores" from the trajectory
4. Compare against measured ssGSEA scores (e.g., rapamycin vs control)
5. Report directional concordance and effect size correlation

This enables a concrete validation loop: does the model predict the
right direction of pathway activity changes seen in real data?

Because everything is JAX, you can also *calibrate* model parameters
by minimizing the distance between simulated and measured scores::

    def loss(params):
        trajectory = simulate(model_with(params), perturbation)
        sim_scores = map_to_pathways(trajectory)
        return distance(sim_scores, measured_scores)

    optimized = jax.grad(loss)(initial_params)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import jax.numpy as jnp


@dataclass
class PathwayMapping:
    """Maps a simulation state variable to a biological pathway.

    Attributes
    ----------
    state_var:
        Store path in the simulation (e.g., "eriq/mTOR_activity").
    pathway_name:
        Name of the pathway/gene set (e.g., "REACTOME_MTOR_SIGNALLING").
    direction:
        Expected relationship: +1 if state_var and pathway score
        should move in the same direction, -1 if inverse.
    transform:
        Optional function to transform the state variable before
        comparison (e.g., ``lambda x: jnp.log1p(x)``).
    description:
        Human-readable description of why this mapping makes sense.
    """

    state_var: str
    pathway_name: str
    direction: float = 1.0
    transform: Callable[[jnp.ndarray], jnp.ndarray] | None = None
    description: str = ""


@dataclass
class MeasuredScores:
    """Container for experimentally measured pathway scores.

    Attributes
    ----------
    condition:
        Name of the experimental condition (e.g., "rapamycin_100nM").
    control:
        Name of the control condition (e.g., "DMSO").
    pathway_scores:
        ``{pathway_name: score}`` — ssGSEA enrichment scores.
        Positive = upregulated vs control, negative = downregulated.
    metadata:
        Any additional info (cell type, time point, dataset ID, etc.).
    """

    condition: str
    control: str = "untreated"
    pathway_scores: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of comparing simulated vs measured pathway scores.

    Attributes
    ----------
    n_pathways:
        Number of pathways compared.
    concordance:
        Fraction of pathways where simulated and measured changes
        agree in direction (0.0 to 1.0).
    per_pathway:
        Detailed per-pathway comparison results.
    summary:
        Human-readable summary string.
    """

    n_pathways: int
    concordance: float
    per_pathway: list[dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def __str__(self):
        lines = [
            f"Validation: {self.n_pathways} pathways, "
            f"concordance = {self.concordance:.1%}",
            "",
        ]
        for p in self.per_pathway:
            arrow_sim = "+" if p["sim_delta"] > 0 else "-"
            arrow_meas = "+" if p["meas_delta"] > 0 else "-"
            match = "OK" if p["concordant"] else "MISMATCH"
            lines.append(
                f"  {p['pathway']:40s}  sim={arrow_sim}  meas={arrow_meas}  [{match}]"
            )
        return "\n".join(lines)


def validate_against_data(
    sim_baseline: dict[str, jnp.ndarray],
    sim_perturbed: dict[str, jnp.ndarray],
    measured: MeasuredScores,
    mappings: list[PathwayMapping],
) -> ValidationResult:
    """Compare simulated perturbation response against measured data.

    Parameters
    ----------
    sim_baseline:
        Final state from baseline simulation ``{store_path: value}``.
    sim_perturbed:
        Final state from perturbed simulation ``{store_path: value}``.
    measured:
        Experimentally measured pathway scores (treatment vs control).
    mappings:
        List of PathwayMapping defining state_var -> pathway correspondences.

    Returns
    -------
    ValidationResult with directional concordance analysis.
    """
    per_pathway = []
    concordant_count = 0

    for mapping in mappings:
        if mapping.state_var not in sim_baseline:
            continue
        if mapping.pathway_name not in measured.pathway_scores:
            continue

        # Simulated delta
        base_val = float(sim_baseline[mapping.state_var])
        pert_val = float(sim_perturbed[mapping.state_var])
        if mapping.transform is not None:
            base_val = float(mapping.transform(jnp.array(base_val)))
            pert_val = float(mapping.transform(jnp.array(pert_val)))
        sim_delta = (pert_val - base_val) * mapping.direction

        # Measured delta (already relative to control)
        meas_delta = measured.pathway_scores[mapping.pathway_name]

        # Concordance: same sign?
        concordant = (sim_delta > 0 and meas_delta > 0) or (sim_delta < 0 and meas_delta < 0)
        if concordant:
            concordant_count += 1

        per_pathway.append({
            "pathway": mapping.pathway_name,
            "state_var": mapping.state_var,
            "sim_delta": sim_delta,
            "meas_delta": meas_delta,
            "concordant": concordant,
        })

    n = len(per_pathway)
    concordance = concordant_count / n if n > 0 else 0.0

    return ValidationResult(
        n_pathways=n,
        concordance=concordance,
        per_pathway=per_pathway,
    )


# ── Pre-built pathway mappings for ERiQ ─────────────────────────────────

ERIQ_PATHWAY_MAPPINGS = [
    PathwayMapping(
        state_var="eriq/mTOR_activity",
        pathway_name="REACTOME_MTOR_SIGNALLING",
        direction=1.0,
        description="mTOR activity maps directly to mTOR signaling pathway",
    ),
    PathwayMapping(
        state_var="eriq/ROS_activity",
        pathway_name="REACTOME_DETOXIFICATION_OF_REACTIVE_OXYGEN_SPECIES",
        direction=1.0,
        description="ROS activity drives ROS detoxification response",
    ),
    PathwayMapping(
        state_var="eriq/mito_function",
        pathway_name="REACTOME_RESPIRATORY_ELECTRON_TRANSPORT",
        direction=1.0,
        description="Mitochondrial function correlates with ETC activity",
    ),
    PathwayMapping(
        state_var="eriq/glycolysis",
        pathway_name="REACTOME_GLYCOLYSIS",
        direction=1.0,
        description="Glycolytic flux maps to glycolysis pathway",
    ),
    PathwayMapping(
        state_var="eriq/mito_damage",
        pathway_name="REACTOME_MITOPHAGY",
        direction=1.0,
        description="Mitochondrial damage activates mitophagy",
    ),
    PathwayMapping(
        state_var="eriq/p53_activity",
        pathway_name="REACTOME_TRANSCRIPTIONAL_REGULATION_BY_TP53",
        direction=1.0,
        description="p53 activity maps to TP53 transcriptional program",
    ),
]
