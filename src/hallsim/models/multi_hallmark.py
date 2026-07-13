"""DP14-anchored multi-hallmark composite — three publications stitched.

This composite spans multiple Hallmarks of Aging in one validation
substrate:

- **Cellular Senescence** — DallePezze2014's CDKN1A and SA_beta_gal
  states are senescence-marker readouts.
- **Deregulated Nutrient Sensing** — DallePezze2014's mTORC1 / AMPK /
  Akt / FoxO3a signaling network is the canonical nutrient-sensing
  axis; rapamycin enters through this hallmark's severity (see below).
- **Genomic Instability** — DallePezze2014's DNA_damage state, with
  Geva-Zatorsky 2006 providing the p53/Mdm2 oscillator downstream.
- **Altered Intercellular Communication / inflammaging** — Ihekwaba 2004
  provides the NF-κB / IκBα regulatory module, the central node of
  cytokine-driven inflammatory signaling.

The three constituent SBML models:

- **DallePezze 2014** (BIOMD0000000582) is the base. Provides mTORC1 /
  AMPK / Akt / FoxO3a signaling, mitophagy and Mito_mass_* turnover
  dynamics, ROS, DNA_damage, CDKN1A (p21), and SA_beta_gal as direct
  state variables.
- **Geva-Zatorsky 2006** (BIOMD0000000157) supplies a p53–Mdm2
  oscillator. Its ``psi`` damage-input parameter is **driven live by
  DP14's ``DNA_damage`` state** through the DP14 → GZ06 edge below,
  Hill-interpolated from the fitted basal ψ (control) to ``GZ06_PSI_FULL``
  (DDIS). So the p53 oscillator's damage input is *caused* by DP14's
  accumulated damage rather than set independently by the hallmark: the
  Genomic Instability severity drives DP14's irradiation rate, DP14
  integrates DNA_damage, and the edge carries it to GZ06.
- **Ihekwaba 2004** (BIOMD0000000230) supplies NF-κB / IκBα dynamics.
  Its ``IkBa`` species is the NFKBIA reporter. NF-κB is driven into the
  module through two DP14 channels (see below), so both rapamycin
  (via mTOR) and genotoxic stress (via DNA damage) reach the NFKBIA
  readout rather than leaving the module inert.

The three cross-publication mechanistic edges:

- **DNA damage → p53** (DP14 ``DNA_damage`` → GZ06 ``psi``, via
  :meth:`SBMLProcess.with_param_driver`) Hill-interpolates GZ06's ψ
  damage-input between its fitted basal value and ``GZ06_PSI_FULL`` on
  DP14's accumulated DNA_damage. Activating per the canonical ATM/ATR →
  p53 DNA-damage-response axis. ψ's range is anchored by GZ06's own
  calibration, so this edge is structural (no free strength), unlike the
  two phenomenological IKK edges below.
- **mTORC1 → IKK** (:class:`hallsim.models.mtor_nfkb.MtorNFkBActivator`)
  reads DP14's ``mTORC1_pS2448`` and Hill-gates a positive derivative
  onto ``IKK``. Activating sign per Dan 2008 / Laberge 2015 — the
  nutrient-sensing / rapamycin channel.
- **DNA damage → IKK**
  (:class:`hallsim.models.damage_nfkb.DamageNFkBActivator`) reads DP14's
  ``DNA_damage`` and Hill-gates a positive derivative onto ``IKK``.
  Activating sign per the ATM → NEMO → IKK genotoxic pathway (Wu 2006 /
  Miyamoto 2011) — the genomic-instability channel through which the
  Genomic Instability hallmark reaches NF-κB.

Geva-Zatorsky 2006 ships vendored under ``models/zatorsky2006/`` and
loads from disk. DallePezze 2014 and Ihekwaba 2004 are not vendored in
the repo: on a clean checkout they download from BioModels on first
import (by their BIOMD ids) and cache under
``~/.cache/hallsim/biomodels/``, so the first build needs network
access.

Experimental conditions and pharmacological interventions both enter
through the hallmark layer. The DDIS+rapa experiment is a two-hallmark
severity matrix::

    from hallsim import apply_hallmarks
    from hallsim.models.multi_hallmark import build_multi_hallmark_composite

    comp = build_multi_hallmark_composite()
    # Untreated DDIS: full exogenous damage, full nutrient dysregulation
    ddis = apply_hallmarks(comp.processes, {
        "Genomic Instability": 1.0,
        "Deregulated Nutrient Sensing": 1.0,
    })
    # DDIS + rapamycin
    rapa = apply_hallmarks(comp.processes, {
        "Genomic Instability": 1.0,
        "Deregulated Nutrient Sensing": 0.3,
    })
    # Healthy control: no exogenous damage source
    ctrl = apply_hallmarks(comp.processes, {
        "Genomic Instability": 0.0,
        "Deregulated Nutrient Sensing": 0.5,
    })

For **Deregulated Nutrient Sensing**, severity 0.0 corresponds to
suppressed mTORC1 (rapamycin-rescued) and severity 1.0 to full
dysregulation; the mapping targets DP14's
``mTORC1_S2448_phos_by_AA_n_Akt_pS473`` rate constant.

For **Genomic Instability**, severity 0.0 corresponds to no exogenous
damage and severity 1.0 to DallePezze 2014's published irradiation
dose; the mapping targets DP14's ``DNA_damaged_by_irradiation`` rate
constant.

Both mappings hit the ``parameters`` field on
:class:`SBMLProcess`.

Gene-reporter mapping targets (the variables ``hallsim.gene_reporters``
compares against transcriptomic data):

- CDKN1A → ``dp14/CDKN1A`` (direct state)
- EIF4EBP1 → ``dp14/mTORC1_pS2448`` (kinase-level proxy)
- CYCS → ``dp14/Mito_mass_new`` (biogenesis proxy)
- DDB2 → ``gz06/x`` (p53 level proxy; DDB2 is a direct p53 target)
- NFKBIA → ``nfkb/IkBat`` (IκBα transcript). Transcriptomic NFKBIA
  measures the IκBα *transcript*, an NF-κB target that rises *with*
  activity — not the cytoplasmic IκBα *protein* (``nfkb/IkBa``), whose
  abundance moves inversely to activity through IKK-driven degradation.
- HMOX1 → ``dp14/ROS`` (oxidative-stress proxy; curated Nrf2/HMOX1
  models on BioModels are sparse so this is the best available until
  a curated Nrf2 module is identified or implemented)
"""

from __future__ import annotations

from pathlib import Path

from hallsim.composite import Composite
from hallsim.models.damage_nfkb import DamageNFkBActivator
from hallsim.models.mtor_nfkb import MtorNFkBActivator
from hallsim.models.running_integral import RunningIntegral
from hallsim.sbml_import import process_from_sbml

_MODELS_DIR = Path(__file__).parent.parent.parent.parent / "models"

DP14_SBML_PATH = (
    _MODELS_DIR / "dallepezze2014" / "dallepezze2014_BIOMD0000000582.xml"
)
GZ06_SBML_PATH = (
    _MODELS_DIR / "zatorsky2006" / "zatorsky2006_BIOMD0000000157.xml"
)
NFKB_SBML_PATH = (
    _MODELS_DIR / "ihekwaba2004" / "ihekwaba2004_BIOMD0000000230.xml"
)

# SBML default for the DP14 mTORC1-phosphorylation rate constant we
# expose for hallmark control. Registered with the SBML default value
# so behaviour is unchanged until a hallmark twists it; the constant is
# also documented at the model level (DallePezze 2014 supplementary
# materials Table S2). Kept as a module-level name so the hallmark
# mapping in `hallsim.hallmarks` can reference the same default.
DP14_MTOR_PHOS_RATE_DEFAULT = 162.471039450073
DP14_MTOR_PHOS_RATE_NAME = "mTORC1_S2448_phos_by_AA_n_Akt_pS473"

# DP14's exogenous-damage rate constant. In the kinetic law
#   d(DNA_damage)/dt = DNA_damaged_by_irradiation · Irradiation
#                      + DNA_damaged_by_ROS · ROS
#                      − DNA_repaired_by_DDR · DNA_damage
# this rate dominates the endogenous ROS contribution by ~3700:1 at
# DP14's published calibration. `Irradiation` itself is governed by a
# time-piecewise assignmentRule and isn't a settable knob, so we drive
# the experimental dose via this rate constant instead — the Genomic
# Instability hallmark severity scales it.
#
# The SBML default (9237.72) is DallePezze 2014's γ-irradiation
# calibration. The effective dose for DDIS-induced senescence (a
# different experimental setup) is fit from data via
# :func:`hallsim.calibration.CalibrationProblem`. This default is a
# starting point only.
DP14_IRRADIATION_RATE_DEFAULT = 9237.72311545872
DP14_IRRADIATION_RATE_NAME = "DNA_damaged_by_irradiation"

# GZ06's damage-signal parameter. Geva-Zatorsky 2006 calibrated their
# p53-Mdm2 oscillator with ψ representing the damage stimulus; ψ ≈ 1 is
# the "full irradiation" reference.
#
# In GZ06 p53 production is entirely ψ-driven (rate = beta_x·psi,
# alpha_x=0), so ψ=0 forces p53→0 — biologically wrong: unstressed cells
# keep a low basal p53 from ~50 spontaneous DSBs per cell cycle
# (docs/gz06-basal-p53.md). So ψ is driven live from DP14's DNA_damage
# state (with_param_driver below), Hill-interpolated from a nonzero basal
# floor (control) to GZ06_PSI_FULL (DDIS). The basal floor is the fitted
# mechanism parameter, exposed as the ordinary `parameters.psi` entry
# (the driver reads it as the Hill's lower bound).
GZ06_PSI_NAME = "psi"
GZ06_PSI_DEFAULT = 1.0  # full-damage reference (standalone screening)
GZ06_PSI_BASAL_DEFAULT = 0.3  # control basal ψ; fitted in the composite
GZ06_PSI_FULL = 1.0  # saturated ψ at full DDIS (the driver's `hi`)
# Hill gate mapping DP14 DNA_damage → GZ06 psi. K is the geometric midpoint
# of DP14's DNA_damage operating range *in the calibrated etoposide regime*
# (~9.6 control → ~37 DDIS at the fitted etoposide potency ~10), NOT the
# γ-irradiation default (~2.8e4). Control sits near basal ψ; DDIS drives it
# up toward GZ06_PSI_FULL. n=2 matches the damage→IKK edge's cooperativity.
# See docs/gz06-basal-p53.md.
GZ06_PSI_DRIVE_K = 19.0
GZ06_PSI_DRIVE_N = 2.0

# Canonical clock for the composite: one t_span unit = one day. DP14
# (the senescence spine) is natively in days, so it runs unchanged;
# GZ06 (hours) and NFKB (seconds) are chain-rule-rescaled onto this axis
# by SBMLProcess.reconciled_to so dt advances every sub-model by the same
# real-world duration. The day scale matches the validation experiment
# (GSE248823 is a D00–D14 time course) and the senescence dynamics we
# read out; the faster modules settle to their attractor / cycle-average
# on this axis, which is what bulk transcriptomics sampled per-day sees.
CANONICAL_TIME_SECONDS = 86400.0


def build_multi_hallmark_composite(*, validate: bool = True):
    """Compose DP14 + GZ06 + Ihekwaba into one multi-hallmark composite.

    Parameters
    ----------
    validate:
        Topology validation. Semantic validation is configured per
        sub-composite and the outer merge.

    Returns
    -------
    Composite with the three sub-models wired through one ``dp14/``,
    ``gz06/``, and ``nfkb/`` namespace each. Apply hallmarks via
    :func:`hallsim.apply_hallmarks` to get treated / control variants
    (see module docstring).
    """
    nfkb = process_from_sbml(str(NFKB_SBML_PATH), name="nfkb").reconciled_to(
        CANONICAL_TIME_SECONDS
    )
    # GZ06's psi is driven by the Genomic Instability hallmark, which
    # interpolates from this basal value (control) up to GZ06_PSI_FULL at
    # DDIS — not by a topology edge from DP14. Seeded at the basal level
    # so control keeps a nonzero p53; the basal value is fittable.
    # psi is driven live by DP14's DNA_damage state (edge below), Hill-
    # interpolated from the fitted basal ψ (control) to GZ06_PSI_FULL
    # (DDIS). The `parameters.psi` basal stays the fittable control floor.
    gz06 = (
        process_from_sbml(
            str(GZ06_SBML_PATH),
            name="gz06",
            parameters={GZ06_PSI_NAME: GZ06_PSI_BASAL_DEFAULT},
        )
        .reconciled_to(CANONICAL_TIME_SECONDS)
        .with_param_driver(
            GZ06_PSI_NAME,
            "psi_source",
            hi=GZ06_PSI_FULL,
            K=GZ06_PSI_DRIVE_K,
            n=GZ06_PSI_DRIVE_N,
        )
    )
    processes: dict = {
        "dp14": process_from_sbml(
            str(DP14_SBML_PATH),
            name="dp14",
            parameters={
                DP14_MTOR_PHOS_RATE_NAME: DP14_MTOR_PHOS_RATE_DEFAULT,
                DP14_IRRADIATION_RATE_NAME: DP14_IRRADIATION_RATE_DEFAULT,
            },
        ).reconciled_to(CANONICAL_TIME_SECONDS),
        "nfkb": nfkb,
        "gz06": gz06,
        # mTORC1 → IKK crosstalk edge: couples DP14's nutrient-sensing
        # network to the NF-κB module so rapamycin (mTOR ↓) suppresses
        # NF-κB and the NFKBIA reporter responds. Activating sign per
        # Dan 2008 / Laberge 2015. Shares NFKB's timescale so IKK's full
        # derivative (intrinsic + mTOR drive) is solved in one group; the
        # slow mTOR input is read frozen across the fast macro-step.
        "mtor_nfkb": MtorNFkBActivator(timescale=nfkb.timescale),
        # DNA damage → IKK crosstalk edge: couples DP14's genomic-
        # instability state to the NF-κB module so the Genomic Instability
        # hallmark (which drives DP14's DNA_damage) reaches the NFKBIA
        # readout. Activating sign per the ATM → NEMO → IKK genotoxic
        # pathway (Wu 2006 / Miyamoto 2011); required for the NF-κB-
        # dependent SASP in DDIS senescence (Salminen 2012). Shares NFKB's
        # timescale so both IKK drives solve in one group.
        "damage_nfkb": DamageNFkBActivator(timescale=nfkb.timescale),
        # Phase-insensitive RMS readouts for both oscillating reporters. A
        # RunningIntegral over each oscillator (in its source's group, so the
        # integral sees the real oscillation) accumulates ∫x² — the default
        # power=2 — and window_rms differences it → √⟨x²⟩. DDB2 needs RMS
        # because GZ06's mean p53 is damage-blind (buffered): the damage lives
        # in the pulse amplitude. NFKBIA's transcript mean is itself drive-
        # responsive, so mean ≈ rms there (6/6 timepoints); RMS is used for
        # both, uniformly — no per-reporter override.
        "gz06_x2_integral": RunningIntegral(timescale=gz06.timescale),
        "nfkb_ikbat_integral": RunningIntegral(timescale=nfkb.timescale),
    }
    # The mtor_nfkb edge reads DP14's active mTORC1 and writes additively
    # to the NF-κB module's IKK pool. The RunningIntegral observers read an
    # oscillator and write its cumulative integral to a new path. The SBML
    # processes carry no topology entries, so each auto-prefixes to its own
    # ``<name>/`` namespace; only these wires cross namespaces.
    topology: dict = {
        # DP14 DNA_damage → GZ06 psi: the genomic-instability → p53 edge.
        # GZ06's psi damage-input is now caused by DP14's accumulated
        # DNA_damage state (Hill-gated in with_param_driver), so the two
        # published models are mechanistically wired rather than both set
        # independently by the hallmark severity.
        "gz06": {"psi_source": "dp14/DNA_damage"},
        "mtor_nfkb": {
            "mTORC1_pS2448": "dp14/mTORC1_pS2448",
            "IKK": "nfkb/IKK",
        },
        "damage_nfkb": {
            "DNA_damage": "dp14/DNA_damage",
            "IKK": "nfkb/IKK",
        },
        "gz06_x2_integral": {
            "source": "gz06/x",
            "integral": "gz06/x2_integral",
        },
        "nfkb_ikbat_integral": {
            "source": "nfkb/IkBat",
            "integral": "nfkb/IkBat_integral",
        },
    }
    return Composite(
        processes=processes,
        topology=topology,
        validate=validate,
        # Cross-model ontology IDs at the coupling points carry
        # different GO terms by design — the semantic checker would
        # flag this as a conflict. Unit and graph checks remain active.
        semantic_validation={"check_semantics": False},
    )
