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
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from hallsim.models.eriq import _compute_algebraic


# ── Trajectory summaries ───────────────────────────────────────────
#
# The trajectory-native contract: a summary maps a saved trajectory to a
# readout **at one or more query times**, so the calibration loss can ask
# each reporter for its value at every measured timepoint. The endpoint is
# just the degenerate single-time query — ``summary(ts, y)`` with
# ``query_times=None`` returns the final-time scalar (and stays
# shape-polymorphic for batched validation). Pass an array of query times
# and the summary returns one value per time, grid-independently (the
# running-integral windows are interpolated at the requested times, not
# snapped to the save grid).


def last_value(ts, y, query_times=None):
    """Instantaneous observable value at each query time (endpoint default).

    ``query_times=None`` returns the final-time value (shape-polymorphic —
    works on batched ``(n_time, ...)`` trajectories). Given query times it
    linear-interpolates ``y`` onto them. The right readout for a
    non-oscillating state, whose value *at* time ``t`` is the biological
    quantity of interest (CDKN1A accumulation, a kinase level, a pool size).
    """
    if query_times is None:
        return (
            y[-1] if hasattr(y, "__getitem__") and hasattr(y, "shape") else y
        )
    return jnp.interp(jnp.atleast_1d(jnp.asarray(query_times)), ts, y)


def window_mean(window: float = 0.5):
    """Exact mean of an observable over a trailing fixed-duration window.

    For an observable routed through a
    :class:`hallsim.models.running_integral.RunningIntegral` — whose path
    holds the cumulative integral ``A = ∫₀ᵗ source`` — the mean of the
    source over ``[t - window, t]`` is, by the fundamental theorem of
    calculus, ``(A(t) - A(t - window)) / window``. Early (``t < window``)
    the window clamps to ``[0, t]``.

    ``window`` is a **fixed duration in the trajectory's time unit** (days
    on the composite axis, so ``0.5`` is 12 h), sized to average over a few
    of the species' oscillation periods — a transcript readout reflects its
    recent transcription, not a fraction of the whole experiment. The
    readout is grid-independent: the integral is interpolated at the window
    edges (smooth, taken at the solver's fine steps). ``query_times=None``
    collapses to the exact mean over the last save interval (batch-safe).
    """
    if window <= 0:
        raise ValueError(f"window must be > 0; got {window!r}")

    def summarize(ts, y, query_times=None):
        if query_times is None:
            return (y[-1] - y[-2]) / (ts[-1] - ts[-2])
        qt = jnp.atleast_1d(jnp.asarray(query_times))
        t_lo = jnp.clip(qt - window, 0.0, None)
        a_hi = jnp.interp(qt, ts, y)
        a_lo = jnp.interp(t_lo, ts, y)
        return (a_hi - a_lo) / jnp.clip(qt - t_lo, 1e-12, None)

    return summarize


def window_rms(window: float = 0.5):
    """Root-mean-square of an observable over the trailing window.

    For a source routed through a ``RunningIntegral(power=2)`` (path holds
    ∫source²), this returns ``√⟨source²⟩`` at each query time. Unlike the
    plain mean it rises with oscillation amplitude, so it reads a
    pulsatile species' damage-encoded pulsing rather than its buffered
    mean; unlike bare amplitude it keeps the mean as a floor, so a
    quiescent baseline gives a finite fold-change instead of diverging.
    Same fixed-duration ``window`` and query-time contract as
    :func:`window_mean`. See docs/gz06-basal-p53.md.
    """
    mean_square = window_mean(window)

    def summarize(ts, y, query_times=None):
        return jnp.sqrt(jnp.clip(mean_square(ts, y, query_times), 0.0, None))

    return summarize


def leaky_rms():
    """RMS envelope from a *leaky* ``RunningIntegral`` (its ``tau`` set).

    The leaky integral path holds ``A = ∫₀ᵗ sourceᵖ·e^{-(t-s)/τ} ds`` — the
    exponentially-weighted (τ-memory) accumulation of ``sourceᵖ``, a smooth
    low-pass envelope with no window edge or oscillation phase to alias
    (unlike the boxcar :func:`window_rms`). With ``power=2``,
    ``A = τ·⟨source²⟩``, so ``√A ∝ RMS``; the constant ``√τ`` cancels in the
    log2 fold-change, so τ is not needed here. The matching integral must be
    leaky — over a flat (``tau=None``) integral this reads a growing
    cumulative sum, not an amplitude.
    """

    def summarize(ts, y, query_times=None):
        a = (
            y[-1]
            if query_times is None
            else jnp.interp(jnp.atleast_1d(jnp.asarray(query_times)), ts, y)
        )
        return jnp.sqrt(jnp.clip(a, 0.0, None))

    return summarize


def zerophase_rms(tau: float):
    """Zero-phase (forward-backward EMA) RMS amplitude envelope — smooth, and
    with *no phase lag*, so a query at day 7 reads the amplitude *at* day 7
    rather than a trailing average biased low by half a window.

    Reads a ``RunningIntegral(power=2)`` path (∫x²), recovers the per-interval
    ⟨x²⟩ by differencing, then low-passes it with an exponential moving average
    run **forward and backward** — the two passes' phase shifts cancel exactly.
    Each pass is warm-started at its boundary sample (an infinite edge-pad),
    which removes the endpoint start-up transient a cold-started pass leaves at
    the last timepoint. ``√`` gives the amplitude.

    This is an *acausal* readout — legitimate here because the integral is an
    output only (never read back into any derivative), so the summary is a
    post-hoc transform of the fully-solved trajectory; there is no way for the
    future to leak into the ODE. ``tau`` is the smoothing memory (trajectory
    time unit), sized to a few oscillation periods. Assumes uniform save
    spacing (the Scheduler's ``save_dt`` grid).
    """

    def summarize(ts, y, query_times=None):
        tmid = 0.5 * (ts[:-1] + ts[1:])
        ms = jnp.diff(y) / jnp.diff(ts)  # ⟨x²⟩ over each save interval
        a = jnp.exp(-(tmid[1] - tmid[0]) / tau)

        def ema(seq, init):
            def step(c, v):
                nc = a * c + (1.0 - a) * v
                return nc, nc

            return jax.lax.scan(step, init, seq)[1]

        fwd = ema(ms, ms[0])
        sm = ema(fwd[::-1], fwd[-1])[::-1]  # zero-phase, edge-padded
        rms = jnp.sqrt(jnp.clip(sm, 0.0, None))
        if query_times is None:
            return rms[-1]
        return jnp.interp(jnp.atleast_1d(jnp.asarray(query_times)), tmid, rms)

    return summarize


def cycle_average(fraction: float = 0.25):
    """Mean over the last ``fraction`` of the *saved* trajectory points.

    Endpoint-only (grid-based): phase-insensitive when the saves resolve
    the oscillation, but with a coarse grid the points alias and it
    degrades to ~the endpoint, and it cannot be evaluated at an arbitrary
    query time. Prefer :func:`window_mean` over a
    :class:`~hallsim.models.running_integral.RunningIntegral`, which is
    exact and query-time-aware. Kept for the legacy ERiQ reporter path.

    Parameters
    ----------
    fraction:
        Fraction of the saved points to average over (default 0.25).
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1]; got {fraction!r}")

    def summarize(ts, y, query_times=None):
        if query_times is not None:
            raise NotImplementedError(
                "cycle_average is grid-based and endpoint-only; use "
                "window_mean (over a RunningIntegral) for trajectory "
                "queries at arbitrary times."
            )
        n = int(y.shape[0])
        k = max(1, int(round(n * fraction)))
        return y[-k:].mean(axis=0)

    return summarize


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
    summary:
        Callable ``(ts, y, query_times=None) -> value(s)`` reading a
        trajectory ``y`` (shape ``(n_time, ...)``) at the query times —
        one value per time, or the endpoint scalar when
        ``query_times is None``. Defaults to :func:`last_value` (the
        instantaneous value). Use :func:`window_mean` / :func:`window_rms`
        over a :class:`~hallsim.models.running_integral.RunningIntegral`
        for observables routed through an oscillating state.
    """

    observable: str
    gene_symbol: str
    sign: int = 1
    description: str = ""
    reference: str = ""
    summary: Callable[[Any], Any] = field(default=last_value)


def oscillating_reporter(
    observable: str,
    gene_symbol: str,
    *,
    sign: int = 1,
    readout: str = "rms",
    window: float = 0.5,
    tau: float | None = None,
    description: str = "",
    reference: str = "",
) -> GeneReporter:
    """Phase-insensitive reporter for an OSCILLATING model species.

    ``readout="rms"`` (default) reads the pulse amplitude √⟨x²⟩ over a trailing
    boxcar ``window``; ``"leaky"`` reads the same amplitude from an exponential
    (τ-memory) envelope instead — smooth, no window edge to alias the
    oscillation, requires the source's ``RunningIntegral`` to be leaky
    (``tau`` set); ``"mean"`` reads the DC level. RMS is the default because a
    *buffered* mean — one
    the drive cannot move (e.g. p53, whose mean is analytically damage-blind
    under the Mdm2 feedback) — leaves a mean readout signal-blind, while RMS
    still sees the amplitude. Use ``"mean"`` only when the species' DC level
    is known to move with the drive (then mean ≈ rms anyway).
    ``"zerophase"`` reads √⟨x²⟩ through a forward-backward EMA (``tau`` memory):
    smooth *and* lag-free, so a timepoint query reads the amplitude *at* that
    time — the readout of choice when the onset timing matters. ``window`` is
    the trailing averaging duration (time unit of the trajectory), sized to
    a few oscillation periods. ``observable`` must be the matching
    :class:`~hallsim.models.running_integral.RunningIntegral` path — ∫x² (its
    default ``power=2``) for rms/zerophase, ∫x (``power=1``) for mean.
    """
    if readout not in ("rms", "mean", "leaky", "zerophase"):
        raise ValueError(
            "readout must be 'rms', 'mean', 'leaky', or 'zerophase', "
            f"got {readout!r}"
        )
    if readout == "zerophase":
        if tau is None:
            raise ValueError("zerophase readout requires tau")
        summary = zerophase_rms(tau)
    elif readout == "leaky":
        summary = leaky_rms()  # source's RunningIntegral must have tau set
    elif readout == "rms":
        summary = window_rms(window)
    else:
        summary = window_mean(window)
    return GeneReporter(
        observable=observable,
        gene_symbol=gene_symbol,
        sign=sign,
        description=description,
        reference=reference,
        summary=summary,
    )


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
            "accumulated DNA damage. Routed through a p53 oscillator "
            "(e.g. Geva-Zatorsky 2006) in the multi-hallmark composite, "
            "so the summary is a cycle-average rather than the endpoint: "
            "the final-time phase of the oscillator is arbitrary, but "
            "the cycle-mean is phase-insensitive and matches bulk "
            "transcriptomics' implicit population averaging."
        ),
        reference="Hwang et al. 1999, Nature 401:430–432",
        summary=cycle_average(0.25),
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


# ── Multi-hallmark composite reporters ─────────────────────────────
#
# These map directly to store paths in the DP14 + NFKB + GZ06 composite,
# unlike CANONICAL_REPORTERS which routes through ERiQ algebraic helpers.

MULTI_HALLMARK_REPORTERS: list[GeneReporter] = [
    GeneReporter(
        observable="dp14/CDKN1A",
        gene_symbol="CDKN1A",
        sign=+1,
        description=(
            "p21/CIP1/WAF1 — senescence and cell-cycle arrest marker. "
            "DallePezze 2014 models CDKN1A as transcribed by FoxO3a in "
            "the presence of DNA damage and degraded by phospho-Akt; "
            "DP14 has no explicit p53 species, so CDKN1A here is a "
            "senescence-state readout rather than a direct p53 readout."
        ),
        reference="el-Deiry et al. 1993, Cell 75:817–825",
    ),
    oscillating_reporter(
        observable="gz06/x2_integral",
        gene_symbol="DDB2",
        sign=+1,
        readout="zerophase",  # smooth, lag-free amplitude; onset timing matters
        tau=2.0,  # forward-backward EMA memory (~7 p53 periods)
        description=(
            "Damage-specific DNA Binding Protein 2 — direct p53 "
            "transcription target, mapped to GZ06's p53 (x). Read as the "
            "zero-phase RMS √⟨x²⟩ envelope, not the mean: GZ06 encodes damage "
            "in p53 *pulsing*, and its mean p53 is analytically buffered "
            "against the damage signal (psi cancels in the steady state), "
            "so the mean is damage-blind. A RunningIntegral (power=2, the "
            "default) writes ∫x² to `gz06/x2_integral`. "
            "See docs/gz06-basal-p53.md."
        ),
        reference="Geva-Zatorsky 2006; Lahav 2004; Purvis 2012",
    ),
    GeneReporter(
        observable="dp14/ROS",
        gene_symbol="HMOX1",
        sign=+1,
        description=(
            "Heme oxygenase 1 — Nrf2/ARE-driven oxidative-stress "
            "reporter. Mapped to DP14's ROS state as the closest "
            "available proxy until a curated Nrf2 module is added."
        ),
        reference="Alam & Cook 2007, Antioxid Redox Signal 9:2499–2511",
    ),
    oscillating_reporter(
        observable="nfkb/IkBat_integral",
        gene_symbol="NFKBIA",
        # Zero-phase RMS like DDB2. Unlike p53, the IκBα-transcript DC level
        # also moves with integrated NF-κB activity, so mean and rms coincide
        # here (6/6 timepoints, scratchpad/nfkbia_mean_vs_rms.py); the
        # zero-phase envelope is used for uniformity with DDB2 — smooth and
        # lag-free.
        sign=+1,
        readout="zerophase",
        tau=2.0,  # forward-backward EMA memory
        description=(
            "IκBα transcript — direct NF-κB target via the autoregulatory "
            "negative feedback loop. Maps to Ihekwaba 2004's IκBα mRNA "
            "species (IkBat), which rises with NF-κB transcriptional "
            "activity. Transcriptomic NFKBIA measures the transcript, not "
            "the cytoplasmic protein (IkBa), whose abundance moves "
            "inversely to activity through IKK-driven degradation. A "
            "RunningIntegral(power=2) writes ∫IkBat² to `nfkb/IkBat_integral` "
            "and window_rms differences it — phase-insensitive amplitude with "
            "a bounded calibration gradient."
        ),
        reference="Sun et al. 1993, Science 259:1912–1915",
    ),
    GeneReporter(
        observable="dp14/Mito_mass_new",
        gene_symbol="CYCS",
        sign=+1,
        description=(
            "Cytochrome c — OXPHOS / mitochondrial biogenesis readout. "
            "Mapped to DP14's newly-synthesized mitochondrial mass."
        ),
        reference="Scarpulla 2008, Physiol Rev 88:611–638",
    ),
    GeneReporter(
        observable="dp14/mTORC1_pS2448",
        gene_symbol="EIF4EBP1",
        sign=+1,
        description=(
            "4E-BP1 — mTORC1 substrate and mTOR-target gene. Mapped to "
            "DP14's phospho-mTORC1 (kinase-level proxy)."
        ),
        reference="Brunn et al. 1997, Science 277:99–101",
    ),
]


def derive_multi_hallmark_summaries(
    ts,
    state_trajectory: dict,
    reporters: list[GeneReporter] | None = None,
) -> dict[str, Any]:
    """Apply each reporter's ``summary`` callable to its store-path trajectory.

    Parameters
    ----------
    ts:
        Save times, shape ``(n_time,)`` — the axis of each trajectory.
        Passed to the summary so time-aware summaries (e.g.
        :func:`window_mean`) can compute a per-time mean.
    state_trajectory:
        ``{store_path: jnp.ndarray}`` with each array shaped
        ``(n_time, ...)``. Typically the result of
        ``Composite.unflatten(scheduler_result.ys)``.
    reporters:
        Defaults to :data:`MULTI_HALLMARK_REPORTERS`.

    Returns
    -------
    ``{reporter.observable: scalar (or batched array)}``, ready to feed
    :func:`compute_concordance`.
    """
    if reporters is None:
        reporters = MULTI_HALLMARK_REPORTERS
    return {
        rep.observable: rep.summary(ts, state_trajectory[rep.observable])
        for rep in reporters
        if rep.observable in state_trajectory
    }


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


def derive_observable_summaries(
    ts,
    state_trajectory: dict,
    reporters: list[GeneReporter] | None = None,
) -> dict[str, Any]:
    """Apply each reporter's ``summary`` callable to its observable trajectory.

    Use this when the loss / concordance routes through an oscillating
    state. ``derive_observables`` is shape-polymorphic (via
    :func:`_compute_algebraic`), so passing a state dict whose values
    are time-axis-leading trajectory arrays produces per-observable
    trajectories. Each reporter's ``summary`` then collapses its own
    observable to a scalar; defaults to :func:`last_value` (endpoint).

    Parameters
    ----------
    state_trajectory:
        ``{store_path: jnp.ndarray}`` with each array shaped
        ``(n_time, ...)`` — typically the output of
        ``Composite.unflatten(scheduler_result.ys)``.
    reporters:
        Defaults to :data:`CANONICAL_REPORTERS`.

    Returns
    -------
    ``{observable_name: scalar (or batched array)}`` ready to be
    consumed by :func:`compute_concordance`.
    """
    if reporters is None:
        reporters = CANONICAL_REPORTERS
    obs_traj = derive_observables(state_trajectory)
    return {
        rep.observable: rep.summary(ts, obs_traj[rep.observable])
        for rep in reporters
        if rep.observable in obs_traj
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

    Assumes ``gene_expr`` is on a log scale (microarray RMA, log-CPM and
    log-TPM all are), so the log2 fold change is the difference of group
    means. Linear-scale inputs (raw counts, TPM) must be log-transformed by
    the loader first.
    """
    return gene_expr[group_cols].mean(axis=1) - gene_expr[baseline_cols].mean(
        axis=1
    )


@dataclass
class GeneExpressionDataset:
    """A gene-expression dataset + named sample groups, with a uniform
    ``.delta(condition, baseline)`` interface for calibration.

    Calibration code consumes datasets through ``.delta(...)``; future
    datasets (Tabula Muris Senis, Ma 2020) implement the same interface
    by subclassing or providing a compatible class.

    Attributes
    ----------
    gene_expr:
        ``gene × sample`` DataFrame indexed by HGNC gene symbol.
    sample_groups:
        ``{group_name: [sample_column_name, ...]}``. Looked up by
        ``delta``.
    """

    gene_expr: pd.DataFrame
    sample_groups: dict[str, list]

    @classmethod
    def from_series_matrix(
        cls,
        series_matrix_path,
        platform_path,
        sample_groups: dict[str, list],
        sample_position_groups: dict[str, list[int]] | None = None,
    ) -> "GeneExpressionDataset":
        """Build from a GEO series matrix + Affymetrix platform pair.

        Use ``sample_groups`` for explicit column-name lists, or
        ``sample_position_groups`` for position-based selection that
        gets resolved against the loaded matrix's columns.
        """
        gene_expr = load_gene_expression(series_matrix_path, platform_path)
        if sample_position_groups is not None:
            samples = list(gene_expr.columns)
            sample_groups = {
                k: [samples[i] for i in idxs]
                for k, idxs in sample_position_groups.items()
            }
        return cls(gene_expr=gene_expr, sample_groups=sample_groups)

    @classmethod
    def from_dataframe(
        cls, gene_expr: pd.DataFrame, sample_groups: dict[str, list]
    ) -> "GeneExpressionDataset":
        """Any log-scale ``gene × sample`` matrix (gene-symbol indexed).

        The platform-agnostic constructor: microarray log-RMA, log-CPM /
        log-TPM from any RNA-seq pipeline, or pseudobulk logs from single
        cell — anything already on a log scale, where a fold change is a
        difference of group means. ``.delta`` / ``.variance`` then work
        unchanged.
        """
        return cls(gene_expr=gene_expr, sample_groups=sample_groups)

    @classmethod
    def from_counts_matrix(
        cls,
        counts: pd.DataFrame,
        sample_groups: dict[str, list],
        *,
        prior_count: float = 1.0,
    ) -> "GeneExpressionDataset":
        """RNA-seq raw counts (``gene × sample``) → log2-CPM.

        Library-size (CPM) normalized, then log2 with a prior count so zeros
        are finite and low counts are damped (edgeR-style log-CPM). A simple
        default — for a rigorous analysis run DESeq2/edgeR upstream (TMM /
        median-of-ratios + moderated dispersion) and pass their log-CPM via
        :meth:`from_dataframe`, with per-gene SEs as calibration weights.
        """
        lib = counts.sum(axis=0)
        cpm = counts.div(lib, axis=1) * 1e6
        return cls(
            gene_expr=np.log2(cpm + prior_count), sample_groups=sample_groups
        )

    def delta(self, condition: str, baseline: str) -> pd.Series:
        """Δ_data = log2 fold change between two named groups."""
        return log2_fold_change(
            self.gene_expr,
            self.sample_groups[condition],
            self.sample_groups[baseline],
        )

    def variance(self, condition: str, baseline: str) -> pd.Series:
        """Per-gene sampling variance of the log2 fold change.

        ``Var(mean_cond − mean_base) = s²_cond/n_cond + s²_base/n_base`` from
        replicate spread. Feed ``1/variance`` as ``weights`` to
        :class:`~hallsim.calibration.CalibrationProblem` to down-weight noisy
        genes. With few replicates this estimate is itself noisy — DESeq2 /
        edgeR moderated SEs (computed upstream, passed via
        :meth:`from_dataframe`) are steadier.
        """
        c = self.gene_expr[self.sample_groups[condition]]
        b = self.gene_expr[self.sample_groups[baseline]]
        return (
            c.var(axis=1, ddof=1) / c.shape[1]
            + b.var(axis=1, ddof=1) / b.shape[1]
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
    mean_abs_error: float = 0.0

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
        (a log2 fold change).
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

    sims, datas = np.asarray(sims), np.asarray(datas)
    sa = float(np.mean([r.sign_match for r in rows]))
    rho, _ = spearmanr(sims, datas)
    return ConcordanceResult(
        condition_name=condition_name,
        rows=rows,
        sign_agreement=sa,
        spearman_r=float(rho) if np.isfinite(rho) else 0.0,
        n_compared=n,
        mean_abs_error=float(np.mean(np.abs(sims - datas))),
    )
