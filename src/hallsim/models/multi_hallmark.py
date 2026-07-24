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

- **DNA damage → p53** (DP14 ``DNA_damage`` → GZ06 ``psi``, via the
  ``psi_bridge`` :class:`HillSignalEdge` that GZ06 reads with
  ``with_param_input``) Hill-interpolates GZ06's ψ damage-input between its
  fitted basal value and ``GZ06_PSI_FULL`` on DP14's accumulated DNA_damage.
  Activating per the canonical ATM/ATR →
  p53 DNA-damage-response axis. ψ's range is anchored by GZ06's own
  calibration, so this edge is structural (no free strength).
- **IKKβ → IKK** (a :class:`hallsim.models.hill_edge.HillActivationEdge`)
  reads DP14's ``IKKbeta`` and Hill-gates a positive derivative onto
  Ihekwaba's ``IKK``. DP14's IKKβ is the same kinase (IKBKB) as the
  signalosome pool, and its defining catalytic role is to activate NF-κB
  (Karin & Ben-Neriah 2000); the Genomic Instability hallmark reaches it
  because IKKβ is ROS-activated inside DP14 (DallePezze 2014's own
  reaction), not via a phenomenological damage→IKK gate. This also wires
  the two IKBKB pools together, so ``RedundancyChecker`` no longer flags
  them as unwired duplicates.
- **mTORC1 → IKK** (another ``HillActivationEdge``) reads DP14's
  ``mTORC1_pS2448`` and Hill-gates a positive derivative onto ``IKK``.
  Activating sign per Dan 2008 / Laberge 2015 — the nutrient-sensing /
  rapamycin channel, distinct from the ROS→IKKβ path above (mTOR→SASP is
  not captured by DP14's ROS-driven IKKβ).

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
- DDB2 → ``gz06/x_integral`` (trailing mean of p53; DDB2 is a direct p53 target)
- MDM2 → ``gz06/y_integral`` (trailing mean of Mdm2; ⟨y⟩∝ψ, the damage-graded
  canonical p53 target)
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
from hallsim.models.hill_edge import HillActivationEdge, HillSignalEdge
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

# Etoposide dose window (days), composite time. GSE248823 protocol:
# "etoposide treatment at 20µM for two 2 days; cells were then washed and
# incubated with fresh medium without drug." So the DDIS damage input is a
# 2-day dose pulse, not a sustained 14-day exposure — delivered via
# `SBMLProcess.with_pulse_window` on the `Irradiation` boundary input.
# After washout DNA_damage decays (DNA_repair) but the senescence phenotype
# persists (ROS-sustained damage + downstream network).
#
# Dose at [0, 2] assumes reporter days count from experiment start (day 0 =
# etoposide applied) — the common convention; the fold-change data has only
# d7/d14 (no d0), so nothing pins it further. If the source paper counts
# reporter days from washout instead, shift the *read* timepoints +2 (to
# 9/16); the dose window stays [0, 2].
DDIS_ETOPOSIDE_DOSE_WINDOW = (0.0, 2.0)
DP14_IRRADIATION_INPUT_NAME = "Irradiation"

# Rapamycin is added to fresh medium at washout (GSE248823: "rapamycin was
# added to fresh medium ... just before use"), i.e. the end of the etoposide
# dose window — so the mTOR-suppression intervention starts at day 2, not t=0.
# The rapamycin arm is identical to DDIS until this day; a calibration
# ParamStep on the mTOR rate constant delivers the timed drop.
RAPA_INTERVENTION_DAY = DDIS_ETOPOSIDE_DOSE_WINDOW[1]

# GZ06's damage-signal parameter. Geva-Zatorsky 2006 calibrated their
# p53-Mdm2 oscillator with ψ representing the damage stimulus; ψ ≈ 1 is
# the "full irradiation" reference.
#
# In GZ06 p53 production is entirely ψ-driven (rate = beta_x·psi,
# alpha_x=0), so ψ=0 forces p53→0 — biologically wrong: unstressed cells
# keep a low basal p53 from ~50 spontaneous DSBs per cell cycle
# (docs/gz06-basal-p53.md). So ψ is driven live from DP14's DNA_damage
# state (psi_bridge HillSignalEdge → with_param_input below), Hill-interpolated
# from a nonzero basal floor (control) to GZ06_PSI_FULL (DDIS). The basal floor
# is the fitted mechanism parameter, exposed as the ordinary `parameters.psi`
# (the driver reads it as the Hill's lower bound).
GZ06_PSI_NAME = "psi"
GZ06_PSI_DEFAULT = 1.0  # full-damage reference (standalone screening)
GZ06_PSI_BASAL_DEFAULT = 0.3  # control basal ψ; fitted in the composite
GZ06_PSI_FULL = 1.0  # saturated ψ at full DDIS (the driver's `hi`)
# Hill gate mapping DP14 DNA_damage → GZ06 psi. K=19 sits below the DDIS
# damage operating point so the damaged arms lift ψ well into the Hill's
# responsive band (DDB2's p53-amplitude readout tracks it) while control
# stays near basal. n=2 matches the damage→IKK edge. See docs/gz06-basal-p53.md.
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


def build_multi_hallmark_composite(
    *, validate: bool = True, dose_window=DDIS_ETOPOSIDE_DOSE_WINDOW
):
    """Compose DP14 + GZ06 + Ihekwaba into one multi-hallmark composite.

    Parameters
    ----------
    validate:
        Topology validation. Semantic validation is configured per
        sub-composite and the outer merge.
    dose_window:
        ``(t_start, t_end)`` damage-input pulse window (default the 2-day
        etoposide dose). ``None`` delivers **sustained** damage — the
        ``Irradiation`` input holds its severity value for the whole run
        instead of washing out.

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
    # GZ06's psi driven live by DP14's DNA_damage (edge below), Hill-
    # interpolated from the fittable basal ψ (control floor, nonzero p53) to
    # GZ06_PSI_FULL (DDIS).
    gz06 = (
        process_from_sbml(
            str(GZ06_SBML_PATH),
            name="gz06",
            parameters={GZ06_PSI_NAME: GZ06_PSI_BASAL_DEFAULT},
        )
        .reconciled_to(CANONICAL_TIME_SECONDS)
        .with_param_input(GZ06_PSI_NAME, "psi_in")
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
        # DNA damage → ψ: GZ06's p53 damage-input, assigned as an algebraic
        # Hill of DP14's DNA_damage between the fittable basal (control p53)
        # and GZ06_PSI_FULL (DDIS). GZ06 reads ψ via with_param_input; the
        # transform lives here as a composable edge on gz06/psi_signal.
        "psi_bridge": HillSignalEdge(
            timescale=gz06.timescale,
            basal=GZ06_PSI_BASAL_DEFAULT,
            hi=GZ06_PSI_FULL,
            K=GZ06_PSI_DRIVE_K,
            n=GZ06_PSI_DRIVE_N,
            source_ontology={"go": "GO:0006974"},
            source_description="DP14 accumulated DNA damage",
            hallmark="Genomic Instability",
            reference="Banin et al. 1998",
            description="DNA damage → ψ (Geva-Zatorsky p53 input).",
        ),
        # mTORC1 → IKK (activating; Dan 2008, Laberge 2015): rapamycin ↓mTOR
        # suppresses NF-κB. K=4.0 = DP14 mTORC1 midpoint (rapa→DDIS).
        "mtor_nfkb": HillActivationEdge(
            timescale=nfkb.timescale,
            k_act=0.02,
            K=(4.0,),
            n=(2.0,),
            target_default=0.1,
            target_ontology={"go": "GO:0008384"},
            target_description="Ihekwaba IKK, receives mTORC1 activation",
            source_ontology=({"go": "GO:0031931"},),
            source_descriptions=("DP14 active mTORC1 (pS2448)",),
            hallmark="Deregulated Nutrient Sensing",
            reference="Dan et al. 2008; Laberge et al. 2015",
            description="mTORC1 → IKK edge (DallePezze 2014 → Ihekwaba 2004).",
        ),
        # IKKβ → IKK (activating): DP14's IKKβ and Ihekwaba's signalosome IKK
        # are the same kinase (IKBKB); this edge lets DP14's mechanistically
        # computed IKKβ drive the NF-κB cascade, which is IKKβ's defining
        # catalytic role (Karin & Ben-Neriah 2000). The genomic-instability
        # drive reaches it because IKKβ is ROS-activated *inside* DP14
        # (DallePezze 2014's own reaction), not via a phenomenological
        # damage→IKK gate. IKKβ's homeostatic band is narrow (ctrl 11.9 → DDIS
        # 16.5), so K=25/n=4 puts the gate in its low-occupancy foot: near-
        # silent at baseline (H≈0.05, matching NF-κB homeostasis) and rising
        # super-linearly with the IKKβ increase (H≈0.16 at DDIS). A gate
        # centered in the band (K≈14) leaves NF-κB half-driven at rest and
        # blocks equilibration. Also wires the two IKBKB pools together,
        # resolving the cross-model redundancy.
        "ikkbeta_nfkb": HillActivationEdge(
            timescale=nfkb.timescale,
            k_act=0.1,
            K=(25.0,),
            n=(4.0,),
            target_default=0.1,
            target_ontology={"go": "GO:0008384"},
            target_description="DP14 IKKβ activity summed into the NF-κB IKK pool",
            source_ontology=({"uniprot": "O14920"},),
            source_descriptions=("DP14 active IKKβ",),
            hallmark="Genomic Instability",
            reference="Karin & Ben-Neriah 2000",
            description="IKKβ → IKK edge (DallePezze 2014 → Ihekwaba 2004).",
        ),
        # ∫ observers for the oscillating reporters (read in the source's group
        # so the integral sees the oscillation), differenced to a grid-
        # independent readout. p53 (x) uses power=2 → ∫x² feeds DDB2's RMS
        # amplitude √⟨x²⟩: GZ06's ψ-cancellation makes ⟨x⟩ damage-blind, but the
        # oscillation amplitude grows with damage. Mdm2 (y) and IκBα-transcript
        # use power=1: their DC level ⟨·⟩ is already damage-responsive.
        "gz06_x_integral": RunningIntegral(
            timescale=gz06.timescale, power=2.0
        ),
        "gz06_y_integral": RunningIntegral(
            timescale=gz06.timescale, power=1.0
        ),
        # NFKBIA reads the IκBα *transcript* (nfkb/IkBat) directly, not a TF
        # activity, so power=1 (mean): its DC level is damage-responsive (no
        # mean-cancellation), unlike the p53 axis. power=2 over-predicts it.
        "nfkb_ikbat_integral": RunningIntegral(
            timescale=nfkb.timescale, power=1.0
        ),
        # p53 → CDKN1A (p21): canonical p53 target (el-Deiry 1993; Shi 2021
        # Hill n=1.8). GZ06's group to read p53 live; Hill-gated flux
        # integrated by DP14's CDKN1A turnover.
        "p53_cdkn1a": HillActivationEdge(
            timescale=gz06.timescale,
            k_act=10.0,
            K=(0.3,),
            n=(1.8,),
            target_default=0.0,
            target_ontology={"go": "GO:0006357"},
            target_description="p53-driven transcription summed into CDKN1A",
            source_ontology=({"go": "GO:0006977"},),
            source_descriptions=("GZ06 p53 level",),
            hallmark="Genomic Instability",
            reference="el-Deiry et al. 1993; Purvis et al. 2012; Shi et al. 2021",
            description="p53 → CDKN1A (p21) edge (Geva-Zatorsky 2006 → DallePezze).",
        ),
    }
    if dose_window is not None:
        processes["dp14"] = processes["dp14"].with_pulse_window(
            DP14_IRRADIATION_INPUT_NAME, *dose_window
        )
    # SBML processes carry no topology entries (each auto-prefixes to its own
    # ``<name>/`` namespace); only these edges cross namespaces.
    topology: dict = {
        # DP14 DNA_damage → ψ (algebraic Hill edge) → GZ06 reads ψ as an input.
        "psi_bridge": {
            "source": "dp14/DNA_damage",
            "signal": "gz06/psi_signal",
        },
        "gz06": {"psi_in": "gz06/psi_signal"},
        "mtor_nfkb": {
            "source": "dp14/mTORC1_pS2448",
            "target": "nfkb/IKK",
        },
        "ikkbeta_nfkb": {
            "source": "dp14/IKKbeta",
            "target": "nfkb/IKK",
        },
        "gz06_x_integral": {
            "source": "gz06/x",
            "integral": "gz06/x_integral",
        },
        "gz06_y_integral": {
            "source": "gz06/y",
            "integral": "gz06/y_integral",
        },
        "nfkb_ikbat_integral": {
            "source": "nfkb/IkBat",
            "integral": "nfkb/IkBat_integral",
        },
        # p53 → CDKN1A: read GZ06 p53, add transcription flux to DP14's p21.
        "p53_cdkn1a": {"source": "gz06/x", "target": "dp14/CDKN1A"},
    }
    return Composite(
        processes=processes,
        topology=topology,
        validate=validate,
        semantic_validation=True,
    )
