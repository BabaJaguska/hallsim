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
  oscillator. Its ``psi`` damage-input parameter is **driven by the
  Genomic Instability hallmark directly** at GZ06's own calibrated
  scale (~1.0 at full DDIS). There is no DP14↔GZ06 topology coupling:
  the hallmark severity sets both DP14's irradiation rate AND GZ06's
  psi simultaneously, each at its own model's regime. Biologically
  appropriate for sustained damage (etoposide DDIS is a constant
  insult — both models reflect the same experimental condition without
  needing dynamic coupling between them).
- **Ihekwaba 2004** (BIOMD0000000230) supplies NF-κB / IκBα dynamics.
  Its ``IkBa`` species is the NFKBIA reporter. NF-κB is driven into the
  module through two DP14 channels (see below), so both rapamycin
  (via mTOR) and genotoxic stress (via DNA damage) reach the NFKBIA
  readout rather than leaving the module inert.

The two cross-publication mechanistic edges, both additive into
Ihekwaba's ``IKK`` pool:

- **mTORC1 → IKK** (:class:`hallsim.models.mtor_nfkb.MtorNFkBActivator`)
  reads DP14's ``mTORC1_pS2448`` and Hill-gates a positive derivative
  onto ``IKK``. Activating sign per Dan 2008 / Laberge 2015 — the
  nutrient-sensing / rapamycin channel.
- **DNA damage → IKK**
  (:class:`hallsim.models.damage_nfkb.DamageNFkBActivator`) reads DP14's
  ``DNA_damage`` and Hill-gates a positive derivative onto ``IKK``.
  Activating sign per the ATM → NEMO → IKK genotoxic pathway (Wu 2006 /
  Miyamoto 2011) — the genomic-instability channel through which the
  Genomic Instability hallmark reaches NF-κB. DP14↔GZ06 remain coupled
  only through the shared hallmark severity.

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
# the "full irradiation" reference (GZ06_PSI_FULL in hallmarks).
#
# In GZ06 p53 production is entirely ψ-driven (rate = beta_x·psi,
# alpha_x=0), so ψ=0 forces p53→0 — biologically wrong: unstressed cells
# keep a low basal p53 from ~50 spontaneous DSBs per cell cycle
# (docs/gz06-basal-p53.md). So the composite seeds ψ at a nonzero BASAL
# level and the Genomic Instability hallmark interpolates it up to
# GZ06_PSI_FULL at DDIS. The basal level is a fitted mechanism parameter
# (calibrated against the modest measured DDB2 / p53-target fold-change).
GZ06_PSI_NAME = "psi"
GZ06_PSI_DEFAULT = 1.0  # full-damage reference (standalone screening)
GZ06_PSI_BASAL_DEFAULT = 0.3  # control basal ψ; fitted in the composite

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
    gz06 = process_from_sbml(
        str(GZ06_SBML_PATH),
        name="gz06",
        parameters={GZ06_PSI_NAME: GZ06_PSI_BASAL_DEFAULT},
    ).reconciled_to(CANONICAL_TIME_SECONDS)
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
        # Phase-insensitive readouts for the two oscillating reporters.
        # Each RunningIntegral integrates its oscillator (∫x, ∫IkBat)
        # *in its source's group* (matched timescale) so the integral is
        # taken at the oscillation's own resolution; the DDB2 / NFKBIA
        # reporters read the trailing-window mean off these via
        # `window_mean`. Replaces reading the phase-dependent endpoint.
        "gz06_x_integral": RunningIntegral(timescale=gz06.timescale),
        "nfkb_ikbat_integral": RunningIntegral(timescale=nfkb.timescale),
    }
    # The mtor_nfkb edge reads DP14's active mTORC1 and writes additively
    # to the NF-κB module's IKK pool. The RunningIntegral observers read an
    # oscillator and write its cumulative integral to a new path. The SBML
    # processes carry no topology entries, so each auto-prefixes to its own
    # ``<name>/`` namespace; only these wires cross namespaces.
    topology: dict = {
        "mtor_nfkb": {
            "mTORC1_pS2448": "dp14/mTORC1_pS2448",
            "IKK": "nfkb/IKK",
        },
        "damage_nfkb": {
            "DNA_damage": "dp14/DNA_damage",
            "IKK": "nfkb/IKK",
        },
        "gz06_x_integral": {
            "source": "gz06/x",
            "integral": "gz06/x_integral",
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
