"""DP14-anchored multi-hallmark composite — three publications stitched.

Spans four Hallmarks of Aging in one validation substrate: Cellular Senescence
and Deregulated Nutrient Sensing (DallePezze 2014's CDKN1A / SA_beta_gal and
mTORC1–AMPK–Akt–FoxO3a axes), Genomic Instability (DP14's DNA_damage feeding
the Geva-Zatorsky 2006 p53–Mdm2 oscillator), and Altered Intercellular
Communication (Ihekwaba 2004's NF-κB / IκBα module).

Constituents — DallePezze 2014 (BIOMD0000000582), Geva-Zatorsky 2006
(BIOMD0000000157), Ihekwaba 2004 (BIOMD0000000230) — ship vendored under
``hallsim/models/sbml/``; a missing file falls back to the BioModels id and
downloads on first import.

Cross-publication edges:

- **DNA damage → p53**: ``psi_bridge`` Hill-interpolates GZ06's ψ input
  between its fitted basal value and ``GZ06_PSI_FULL`` on DP14's accumulated
  DNA_damage, so p53's damage input is *caused* by DP14 rather than set
  independently. ATM/ATR → p53; ψ's range is anchored by GZ06's own
  calibration, so the edge carries no free strength.
- **IKKβ → IKK**: DP14's IKKβ is the same kinase (IKBKB) as Ihekwaba's
  signalosome pool, and activating NF-κB is its defining role (Karin &
  Ben-Neriah 2000). Genomic Instability reaches it because IKKβ is
  ROS-activated inside DP14, not through a phenomenological damage→IKK gate.
- **mTORC1 → IKK**: the nutrient-sensing / rapamycin channel (Dan 2008,
  Laberge 2015), distinct from the ROS→IKKβ path above.

Conditions and drugs both enter through the hallmark layer::

    comp = build_multi_hallmark_composite()
    ddis = apply_hallmarks(comp.processes, {"Genomic Instability": 1.0})
    rapa = apply_hallmarks(comp.processes, {
        "Genomic Instability": 1.0,
        "Deregulated Nutrient Sensing": -1.0,   # mTORC1 suppressed
    })

Severity 0 is homeostasis for both; Nutrient Sensing runs -1 (rapamycin) to +1
(hyperactivation) on DP14's mTORC1 phosphorylation rate, Genomic Instability 0
to 1 (DallePezze's published irradiation dose) on its damage rate.

Gene reporters (see :mod:`hallsim.gene_reporters`): CDKN1A → ``dp14/CDKN1A``,
GLB1 → ``dp14/SA_beta_gal``, BNIP3 → ``dp14/FoxO3a``, DDB2 → ``gz06/x``
(RMS amplitude), MDM2 → ``gz06/y``, and NFKBIA → ``nfkb/IkBat`` — the IκBα
*transcript*, which rises with NF-κB activity, not the cytoplasmic protein,
which moves inversely to it.

``test_gene_reporters.py`` checks this list against
``MULTI_HALLMARK_REPORTERS``, so it fails rather than drifts.
"""

from __future__ import annotations

from hallsim.composite import Composite
from hallsim.models.forcing import drive_pulse
from hallsim.models.hill_edge import HillActivationEdge, HillSignalEdge
from hallsim.models.sbml import sbml_source
from hallsim.sbml_import import process_from_sbml

DP14_SBML_PATH = sbml_source(
    "dallepezze2014",
    "dallepezze2014_BIOMD0000000582.xml",
    "BIOMD0000000582",
)
GZ06_SBML_PATH = sbml_source(
    "zatorsky2006", "zatorsky2006_BIOMD0000000157.xml", "BIOMD0000000157"
)
NFKB_SBML_PATH = sbml_source(
    "ihekwaba2004", "ihekwaba2004_BIOMD0000000230.xml", "BIOMD0000000230"
)

# SBML defaults, named at module level so hallsim.hallmarks can target the
# same constants. DallePezze 2014 supplementary Table S2.
DP14_MTOR_PHOS_RATE_DEFAULT = 162.471039450073
DP14_MTOR_PHOS_RATE_NAME = "mTORC1_S2448_phos_by_AA_n_Akt_pS473"

# DP14's `Irradiation` input is a time-piecewise assignmentRule, not a settable
# knob, so the experimental dose is driven through this rate constant instead.
# The SBML value is DallePezze's γ-irradiation calibration; the DDIS-equivalent
# dose is fit by CalibrationProblem, so this is a starting point only.
DP14_IRRADIATION_RATE_DEFAULT = 9237.72311545872
DP14_IRRADIATION_RATE_NAME = "DNA_damaged_by_irradiation"

# GSE248823: etoposide 20µM for 2 days, then washout — a dose pulse, not a
# sustained 14-day exposure. Days count from experiment start; if the source
# paper counts from washout instead, shift the *read* timepoints +2 and leave
# this window alone.
DDIS_ETOPOSIDE_DOSE_WINDOW = (0.0, 2.0)
DP14_IRRADIATION_INPUT_NAME = "Irradiation"

# Rapamycin enters the fresh medium at washout, so the rapa arm is identical to
# DDIS until this day; a calibration ParamStep delivers the timed mTOR drop.
RAPA_INTERVENTION_DAY = DDIS_ETOPOSIDE_DOSE_WINDOW[1]

# GZ06 p53 production is entirely ψ-driven (alpha_x=0), so ψ=0 forces p53→0 —
# wrong for unstressed cells, which hold a low basal p53 (docs/gz06-basal-p53.md).
# ψ is therefore driven from DP14's DNA_damage via psi_bridge, Hill-interpolated
# from a nonzero fitted basal floor up to GZ06_PSI_FULL.
GZ06_PSI_NAME = "psi"
GZ06_PSI_DEFAULT = 1.0  # full-damage reference (standalone screening)
GZ06_PSI_BASAL_DEFAULT = 0.3  # control basal ψ; fitted in the composite
GZ06_PSI_FULL = 1.0  # saturated ψ at full DDIS (the driver's `hi`)
# K sits at the damage operating point separating control from DDIS (placed via
# suggest_hill_gate) and is fitted, so the data decides where p53 fires.
GZ06_PSI_DRIVE_K = 52.0
GZ06_PSI_DRIVE_N = 2.0

# One t_span unit = one day, matching GSE248823's D00–D14 course. DP14 is
# natively in days and runs unchanged; GZ06 (hours) and NFKB (seconds) are
# rescaled onto this axis by reconciled_to, and settle to their cycle-average
# on it — which is what per-day bulk transcriptomics samples.
CANONICAL_TIME_SECONDS = 86400.0


def build_multi_hallmark_composite(
    *, validate: bool = True, dose_window=DDIS_ETOPOSIDE_DOSE_WINDOW
):
    """Compose DP14 + GZ06 + Ihekwaba into one composite, namespaced
    ``dp14/``, ``gz06/`` and ``nfkb/``; apply hallmarks for the treated and
    control variants.

    ``dose_window`` is the ``(t_start, t_end)`` damage pulse; ``None`` holds
    ``Irradiation`` at its severity for the whole run instead of washing out.
    ``validate`` covers topology only — semantic validation is configured per
    sub-composite and at the merge.
    """
    nfkb = process_from_sbml(str(NFKB_SBML_PATH), name="nfkb").reconciled_to(
        CANONICAL_TIME_SECONDS
    )
    gz06 = (
        process_from_sbml(
            str(GZ06_SBML_PATH),
            name="gz06",
            parameters={GZ06_PSI_NAME: GZ06_PSI_BASAL_DEFAULT},
        )
        .reconciled_to(CANONICAL_TIME_SECONDS)
        .with_param_input(GZ06_PSI_NAME, "psi_in")
    )
    dp14 = process_from_sbml(
        str(DP14_SBML_PATH),
        name="dp14",
        parameters={
            DP14_MTOR_PHOS_RATE_NAME: DP14_MTOR_PHOS_RATE_DEFAULT,
            DP14_IRRADIATION_RATE_NAME: DP14_IRRADIATION_RATE_DEFAULT,
        },
    ).reconciled_to(CANONICAL_TIME_SECONDS)
    processes: dict = {
        "dp14": dp14,
        "nfkb": nfkb,
        "gz06": gz06,
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
        # K=4.0 is the DP14 mTORC1 midpoint across the rapa→DDIS band.
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
        # IKKβ's homeostatic band is narrow (ctrl 11.9 → DDIS 16.5), so K=25/n=4
        # sits in its low-occupancy foot: near-silent at baseline (H≈0.05) and
        # super-linear at DDIS (H≈0.16). A gate centred in the band (K≈14)
        # leaves NF-κB half-driven at rest and blocks equilibration.
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
        # Oscillating reporters read their raw species and summarize post-hoc,
        # so no integral observer accumulates, lags, or stiffens the solve.
        # n=1.8 per Shi 2021.
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
        # p53 → CDKN1A: read GZ06 p53, add transcription flux to DP14's p21.
        "p53_cdkn1a": {"source": "gz06/x", "target": "dp14/CDKN1A"},
    }
    # Etoposide exposure: a PulseSource ("irradiation_pulse") drives DP14's
    # Irradiation input over the dose window — composed from the general
    # port-coupling path, not a special-cased pulse. Its amplitude is the
    # Genomic Instability exposure level (set per condition via the hallmark).
    if dose_window is not None:
        drive_pulse(
            processes,
            topology,
            target="dp14",
            input_name=DP14_IRRADIATION_INPUT_NAME,
            t_start=dose_window[0],
            t_end=dose_window[1],
            amplitude=1.0,
            source_name="irradiation_pulse",
            hallmark="Genomic Instability",
        )
    return Composite(
        processes=processes,
        topology=topology,
        validate=validate,
        semantic_validation=True,
    )
