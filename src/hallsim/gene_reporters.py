"""Gene-reporter validation: one-to-one mapping from a mechanistic
observable to a canonical reporter transcript.

Each entry in :data:`CANONICAL_REPORTERS` pairs one HallSim mechanistic
quantity (a state variable or an algebraic intermediate) with the
single canonical gene whose expression is the textbook readout of that
quantity, plus an expected sign and a literature anchor. The mapping has
no tunable parameters.

Validation pipeline:

1. Run the simulator under control and perturbed conditions.
2. Compute Δ_observable via :func:`derive_observables` applied to
   late-time (or trajectory-sampled) state snapshots.
3. Compute Δ_gene from a bulk or pseudo-bulk expression matrix via
   :func:`log2_fold_change`.
4. Aggregate per-reporter sign agreement and Spearman correlation via
   :func:`compute_concordance`.

Calibration belongs to the mechanism, not the readout layer: parameters
of the composite (e.g. ``alpha``, ``k_sasp``, ``MDAMAGE_SA``) are
differentiable end-to-end and can be fit against the gene-level Δ_data
using ``jax.grad`` + ``optax``, with a held-out split across conditions
(e.g. fit on DDIS, evaluate on OIS).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from hallsim.models.eriq import _compute_algebraic


# ── Reporter table ─────────────────────────────────────────────────


@dataclass(frozen=True)
class GeneReporter:
    """One mechanistic observable ↔ one canonical reporter gene.

    Attributes
    ----------
    observable:
        Key from :func:`derive_observables` output (e.g. ``"p53_activity"``).
    gene_symbol:
        HGNC gene symbol of the canonical transcript reporter.
    sign:
        ``+1`` if (observable ↑ ⇒ gene ↑); ``-1`` for inverse.
    description:
        Short mechanistic rationale for the mapping.
    reference:
        Primary literature anchor.
    """

    observable: str
    gene_symbol: str
    sign: int = 1
    description: str = ""
    reference: str = ""


CANONICAL_REPORTERS: list[GeneReporter] = [
    GeneReporter(
        observable="p53_activity",
        gene_symbol="CDKN1A",
        sign=+1,
        description=(
            "p21/CIP1/WAF1 — direct p53 transcriptional target via a "
            "well-characterized response element. Standard readout of "
            "p53 transcriptional activity in DDR and senescence."
        ),
        reference="el-Deiry et al. 1993, Cell 75:817–825",
    ),
    GeneReporter(
        observable="mito_damage",
        gene_symbol="DDB2",
        sign=+1,
        description=(
            "Damage-specific DNA Binding Protein 2 — direct p53 target "
            "induced by DDR signaling. Transcriptional readout of "
            "accumulated DNA damage."
        ),
        reference="Hwang et al. 1999, Nature 401:430–432",
    ),
    GeneReporter(
        observable="ROS_algebraic",
        gene_symbol="HMOX1",
        sign=+1,
        description=(
            "Heme oxygenase 1 — canonical Nrf2/ARE-driven antioxidant "
            "response gene; standard transcriptional reporter of "
            "oxidative stress."
        ),
        reference="Alam & Cook 2007, Antioxid Redox Signal 9:2499–2511",
    ),
    GeneReporter(
        observable="NFKB_algebraic",
        gene_symbol="NFKBIA",
        sign=+1,
        description=(
            "IκBα — direct NF-κB target via the autoregulatory negative "
            "feedback loop. Among the cleanest transcriptional reporters "
            "of NF-κB activity."
        ),
        reference="Sun et al. 1993, Science 259:1912–1915",
    ),
    GeneReporter(
        observable="mito_function",
        gene_symbol="CYCS",
        sign=+1,
        description=(
            "Cytochrome c — nuclear-encoded OXPHOS component whose "
            "transcript level tracks PGC-1α-driven mitochondrial "
            "biogenesis and OXPHOS capacity."
        ),
        reference="Scarpulla 2008, Physiol Rev 88:611–638",
    ),
    GeneReporter(
        observable="mTOR_activity_algebraic",
        gene_symbol="EIF4EBP1",
        sign=+1,
        description=(
            "4E-BP1 — mTORC1 substrate and mTOR-target gene. Gene-level "
            "readout is intentionally noisier than kinase-level activity "
            "in chronic stress; included so the mTOR axis is represented."
        ),
        reference="Brunn et al. 1997, Science 277:99–101",
    ),
]


# ── Mechanistic observable derivation ──────────────────────────────


def derive_observables(state: dict) -> dict[str, Any]:
    """Compute the named observables consumed by the reporter table.

    The set blends raw state variables and algebraic intermediates: NF-κB
    and the algebraic MTOR / ROS values come from
    :func:`hallsim.models.eriq._compute_algebraic` so the mapping
    matches what ERiQ downstream processes use.

    Parameters
    ----------
    state:
        Dict keyed by ``eriq/<name>`` store paths.

    Returns
    -------
    Dict mapping observable name → scalar (or batched array, since
    ``_compute_algebraic`` is shape-polymorphic).
    """
    sub = {
        "mito_function": state["eriq/mito_function"],
        "glycolysis": state["eriq/glycolysis"],
        "mito_damage": state["eriq/mito_damage"],
        "mTOR_activity": state["eriq/mTOR_activity"],
        "p53_activity": state["eriq/p53_activity"],
        "ROS_activity": state["eriq/ROS_activity"],
        "ROS_integrator_c": state["eriq/ROS_integrator_c"],
    }
    obs = _compute_algebraic(sub)
    return {
        "p53_activity": state["eriq/p53_activity"],
        "mito_damage": state["eriq/mito_damage"],
        "mito_function": state["eriq/mito_function"],
        "mTOR_activity_algebraic": obs["MTOR"],
        "NFKB_algebraic": obs["NFKB"],
        "ROS_algebraic": obs["ROS"],
    }


# ── Gene-expression parsing (Affymetrix-style series matrix) ───────


def load_gene_expression(
    series_matrix_path: Path,
    platform_path: Path,
) -> pd.DataFrame:
    """Parse GEO series-matrix expression + Affymetrix platform annotation
    into a ``(gene_symbol × sample)`` DataFrame.

    - Series-matrix expression values are typically log2-RMA normalized;
      no further transform is applied.
    - Probes are mapped to the first gene symbol in the
      ``gene_assignment`` column of the platform file.
    - Multi-probe-per-gene values are collapsed by mean.
    """
    with open(series_matrix_path) as f:
        lines = f.readlines()
    start = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("!series_matrix_table_begin")
    )
    end = next(
        i
        for i, ln in enumerate(lines)
        if ln.startswith("!series_matrix_table_end")
    )
    expr = pd.read_csv(
        StringIO("".join(lines[start + 1 : end])),
        sep="\t",
        header=0,
        index_col=0,
        quotechar='"',
    ).dropna(how="all")

    plat = pd.read_csv(platform_path, sep="\t", comment="#", low_memory=False)
    probe_to_gene: dict[str, str] = {}
    for probe, raw in zip(plat["ID"], plat["gene_assignment"].fillna("")):
        if not raw or raw == "---":
            continue
        first = raw.split("///", 1)[0]
        parts = [p.strip() for p in first.split("//")]
        if (
            len(parts) < 2
            or not parts[1]
            or parts[1] == "---"
            or not parts[1][0].isalpha()
        ):
            continue
        probe_to_gene[probe] = parts[1].strip()

    common = expr.index.intersection(probe_to_gene.keys())
    expr = expr.loc[common].copy()
    expr["__gene__"] = [probe_to_gene[p] for p in expr.index]
    return expr.groupby("__gene__").mean(numeric_only=True)


def log2_fold_change(
    gene_expr: pd.DataFrame,
    group_cols: list,
    baseline_cols: list,
) -> pd.Series:
    """Per-gene log2 fold change between two sample groups.

    Microarray series-matrix values are already log2-scaled, so this is a
    mean-difference computation.
    """
    return gene_expr[group_cols].mean(axis=1) - gene_expr[baseline_cols].mean(
        axis=1
    )


# ── Concordance computation ────────────────────────────────────────


@dataclass
class ReporterRow:
    """Per-reporter outcome for one condition comparison."""

    reporter: GeneReporter
    delta_sim: float
    delta_data: float
    sign_match: bool


@dataclass
class ConcordanceResult:
    """Aggregate sign-agreement + Spearman across reporters for one
    condition."""

    condition_name: str
    rows: list[ReporterRow] = field(default_factory=list)
    sign_agreement: float = 0.0
    spearman_r: float = 0.0
    n_compared: int = 0

    def __str__(self) -> str:
        lines = [
            f"{self.condition_name}: "
            f"sign agreement = {self.sign_agreement*100:.1f}% "
            f"({sum(r.sign_match for r in self.rows)}/{self.n_compared}), "
            f"Spearman r = {self.spearman_r:+.3f}",
            f"  {'observable':<28}  {'gene':<10}  "
            f"{'Δ_sim·sign':>12}  {'Δ_data':>10}  match",
        ]
        for r in self.rows:
            mk = "OK" if r.sign_match else "X"
            lines.append(
                f"  {r.reporter.observable:<28}  "
                f"{r.reporter.gene_symbol:<10}  "
                f"{r.delta_sim:>+12.4f}  {r.delta_data:>+10.4f}  {mk}"
            )
        return "\n".join(lines)


def compute_concordance(
    *,
    delta_observables: dict[str, float],
    delta_gene_expression: pd.Series,
    condition_name: str = "",
    reporters: list[GeneReporter] | None = None,
) -> ConcordanceResult:
    """Compare simulated observable changes to measured gene-expression
    changes one reporter at a time.

    Parameters
    ----------
    delta_observables:
        ``{observable_name: Δ_sim}``.
    delta_gene_expression:
        ``pd.Series`` indexed by gene symbol, values are Δ_data
        (log2 fold change for microarray).
    condition_name:
        Label for the report (e.g. ``"DDIS_D14_vs_D00"``).
    reporters:
        Defaults to :data:`CANONICAL_REPORTERS`.
    """
    if reporters is None:
        reporters = CANONICAL_REPORTERS

    rows: list[ReporterRow] = []
    sims: list[float] = []
    datas: list[float] = []
    for rep in reporters:
        if rep.observable not in delta_observables:
            continue
        if rep.gene_symbol not in delta_gene_expression.index:
            continue
        ds_raw = float(delta_observables[rep.observable])
        ds_signed = float(rep.sign) * ds_raw
        dd = float(delta_gene_expression[rep.gene_symbol])
        sign_match = (ds_signed * dd > 0) or (
            abs(ds_signed) < 1e-12 and abs(dd) < 1e-12
        )
        rows.append(
            ReporterRow(
                reporter=rep,
                delta_sim=ds_signed,
                delta_data=dd,
                sign_match=sign_match,
            )
        )
        sims.append(ds_signed)
        datas.append(dd)

    n = len(rows)
    if n == 0:
        return ConcordanceResult(condition_name=condition_name)
    from scipy.stats import spearmanr

    sa = float(np.mean([r.sign_match for r in rows]))
    rho, _ = spearmanr(sims, datas)
    return ConcordanceResult(
        condition_name=condition_name,
        rows=rows,
        sign_agreement=sa,
        spearman_r=float(rho) if np.isfinite(rho) else 0.0,
        n_compared=n,
    )
