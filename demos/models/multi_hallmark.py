"""DP14-anchored multi-hallmark composite — three publications stitched.

Spans four Hallmarks of Aging in one validation substrate: Cellular Senescence
and Deregulated Nutrient Sensing (DallePezze 2014's CDKN1A / SA_beta_gal and
mTORC1–AMPK–Akt–FoxO3a axes), Genomic Instability (DP14's DNA_damage feeding
the Geva-Zatorsky 2006 p53–Mdm2 oscillator), and Altered Intercellular
Communication (Kallenberger 2014's CD95L, an extracellular ligand delivered by
another cell).

Constituents — DallePezze 2014 (BIOMD0000000582), Geva-Zatorsky 2006
(BIOMD0000000157) and Kallenberger 2014 (BIOMD0000000524) — ship vendored under
``demos/models/sbml/``; a missing file falls back to the BioModels id and
downloads on first import.

Apoptosis is not itself a hallmark, and Kallenberger is not here as one. It is
the **alternative fate**: without it the composite has a single destination and
"the cell senesces" is an assumption of the model set rather than a computed
result. With it, one damage signal reaches two competing outputs. Ihekwaba 2004
held the intercellular slot until 2026-08-31 and was removed — refereed three
ways, it contributed nothing (19/24 with both its edges ablated), the edges
supplied 100% of its IKK rather than perturbing it, and its only NF-κB-inducible
transcript was its own inhibitor. See docs/review-ihekwaba2004-wetlab.md.

Cross-publication edges:

- **DNA damage ⊣ p53 degradation**: ``damage_bridge`` Hill-interpolates GZ06's
  ``alpha_x`` *downward* from a quiescent control value to the deposit's own
  ``alpha_x = 0`` on DP14's accumulated DNA_damage. ATM phosphorylates p53
  Ser15 and blocks its degradation (Banin 1998), and crossing the p53 Hopf at
  ``alpha_x = 0.1662`` is what starts the pulses — so a damaged arm oscillates
  and a control arm does not. GZ06 fitted *irradiated* cells, so its published
  parameter set is the damaged end of this edge and the range carries no free
  strength. ``psi`` is the paper's ξ, a production-noise gain, and is not a
  damage variable; it stays at its published 1.0.

- **p53 → CD95 receptor**: ``fas_induction`` Hill-interpolates Kallenberger's
  ``CD95`` up from the deposit's own 12 toward 3.5x that on GZ06's p53, and
  ``fas_receptor`` gives it first-order receptor turnover so the level
  integrates the p53 pulse train rather than following it. Wild-type p53
  raises surface Fas 3-4x (Owen-Schaub 1995) through a response element in the
  FAS first intron, and ``CD95act`` is cubic in ``CD95``. Both halves are
  needed: GZ06's mean p53 is damage-blind under ψ-cancellation, so only the
  oscillation amplitude carries the signal and an edge reading a *level* out
  of it has to rectify. It under-delivers at the cooperativity p53's
  tetrameric binding supports — see P0.45.

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
(RMS amplitude), MDM2 → ``gz06/y0`` — the Mdm2 *precursor*, which GZ06's
Table I defines as the transcript, not the protein ``y`` — and FAS →
``k14/CD95_level``. FAS is the only transcript readout the apoptosis arm
admits: everything below the receptor is post-translational, and the model
conserves total Bid and total caspase-8, so BID and CASP8 have no
transcript-level meaning in it however well they are measured.

``test_gene_reporters.py`` checks this list against
``MULTI_HALLMARK_REPORTERS``, so it fails rather than drifts.
"""

from __future__ import annotations

from hallsim.composite import Composite
from hallsim.models.forcing import drive_pulse, drive_step
from hallsim.models.clamp_edge import ClampEdge
from hallsim.models.hill_edge import (
    HillActivationEdge,
    HillSignalEdge,
    place_hill_gate,
    place_hill_gate_for_crossing,
)
from demos.models.sbml import sbml_source
from hallsim.sbml_import import process_from_sbml

DP14_SBML_PATH = sbml_source(
    "dallepezze2014",
    "dallepezze2014_BIOMD0000000582.xml",
    "BIOMD0000000582",
)
GZ06_SBML_PATH = sbml_source(
    "zatorsky2006", "zatorsky2006_BIOMD0000000157.xml", "BIOMD0000000157"
)
K14_SBML_PATH = sbml_source(
    "kallenberger2014",
    "kallenberger2014_BIOMD0000000524.xml",
    "BIOMD0000000524",
)
# SBML defaults, named at module level so hallsim.hallmarks can target the
# same constants. DallePezze 2014 supplementary Table S2.
DP14_MTOR_PHOS_RATE_DEFAULT = 162.471039450073
DP14_MTOR_PHOS_RATE_NAME = "mTORC1_S2448_phos_by_AA_n_Akt_pS473"

# GSE248823: etoposide 20µM for 2 days, then washout — a dose pulse, not a
# sustained 14-day exposure. Days count from experiment start; if the source
# paper counts from washout instead, shift the *read* timepoints +2 and leave
# this window alone.
DDIS_ETOPOSIDE_DOSE_WINDOW = (0.0, 2.0)
DP14_IRRADIATION_INPUT_NAME = "Irradiation"

# DP14's `Irradiation` input is a time-piecewise assignmentRule, not a settable
# knob, so the experimental dose is driven through this rate constant instead.
# The SBML value is calibrated to a 5-minute pulse, so it is rescaled to the
# dose window's exposure — without this the composite delivers 593x the dose.
DP14_SBML_IRRADIATION_RATE = 9237.72311545872
DP14_NATIVE_IRRADIATION_EXPOSURE = 0.003375
DP14_IRRADIATION_RATE_DEFAULT = (
    DP14_SBML_IRRADIATION_RATE
    * DP14_NATIVE_IRRADIATION_EXPOSURE
    / (DDIS_ETOPOSIDE_DOSE_WINDOW[1] - DDIS_ETOPOSIDE_DOSE_WINDOW[0])
)
DP14_IRRADIATION_RATE_NAME = "DNA_damaged_by_irradiation"

# Rapamycin enters the fresh medium at washout, so the rapa arm is identical to
# DDIS until this day; the nutrient drive's StepSource switches level there.
RAPA_INTERVENTION_DAY = DDIS_ETOPOSIDE_DOSE_WINDOW[1]

# DP14 pins `Amino_Acids` (and `Insulin`, not driven here) at 1 for all time,
# so no arm can lower mTOR drive without driving the input.
DP14_NUTRIENT_INPUT_NAME = "Amino_Acids"
DP14_NUTRIENT_BASAL = 1.0

# GZ06's `psi` is the paper's ξ, a noise gain on protein production, and stays
# at its published 1.0. Damage enters on `alpha_x`, the Mdm2-independent p53
# degradation ATM blocks (Banin 1998) — the channel `simulate gz06-damage-scan`
# picks: its Hopf is at 0.1662 and damage crosses it, where alpha_k's and
# alpha_y's damage directions move away. GZ06 fitted irradiated cells, so the
# published alpha_x = 0 IS the damaged state and the edge runs down to it.
GZ06_PSI_NAME = "psi"
GZ06_PSI_PUBLISHED = 1.0
GZ06_ALPHA_X_NAME = "alpha_x"
GZ06_ALPHA_X_HOPF = 0.1662
GZ06_ALPHA_X_CONTROL = 4 * GZ06_ALPHA_X_HOPF  # decay tau 5.5 h; fitted
GZ06_ALPHA_X_DAMAGED = 0.0  # the deposit's own value
GZ06_DAMAGE_DRIVE_N = 2.0
# Measured DNA_damage: control ceiling and DDIS mean (operating_ranges).
GZ06_DAMAGE_OFF_LEVEL = 9.59
GZ06_DAMAGE_ON_LEVEL = 12.13
# K placed so the signal crosses the Hopf between them — derived, not a
# literal, so the placement moves with the levels it was placed from.
GZ06_DAMAGE_GATE = place_hill_gate_for_crossing(
    off_level=GZ06_DAMAGE_OFF_LEVEL,
    on_level=GZ06_DAMAGE_ON_LEVEL,
    basal=GZ06_ALPHA_X_CONTROL,
    hi=GZ06_ALPHA_X_DAMAGED,
    critical=GZ06_ALPHA_X_HOPF,
    n=GZ06_DAMAGE_DRIVE_N,
)
GZ06_DAMAGE_DRIVE_K = GZ06_DAMAGE_GATE.K

# Declares no unitDefinition; its rate laws are in minutes.
K14_NATIVE_TIME_SECONDS = 60.0
K14_CD95_NAME = "CD95"
K14_CD95L_NAME = "CD95L"
# Deposit ICs. Neither takes part in a reaction — both enter only the
# `CD95act` rule — so the importer exposes them as constants, not species.
K14_CD95_BASAL = 12.0
K14_CD95L_DEPOSIT_DOSE = 16.6  # 500 ng/ml T4-CD95L, per the curator
# Placed at the maximum receptor-level contrast, measured on Kallenberger
# alone; the deposit's own dose saturates and discriminates almost nothing.
K14_CD95L_DOSE = 2.0
# Kallenberger's own window. Bid is never resynthesised, so a sustained
# ligand runs to completion at any dose — this must be a window, not a bath.
K14_CD95L_CHALLENGE_MINUTES = 240.0
K14_CD95L_CHALLENGE_DAY = 7.0
# Owen-Schaub 1995: wild-type p53 raises surface Fas 3-4x. Range midpoint.
K14_CD95_INDUCED = 3.5 * K14_CD95_BASAL
# p53 binds as a tetramer, so n is capped at 4 whatever the levels want.
K14_FAS_HILL_N = 4.0
K14_P53_OFF_LEVEL = 0.3519  # ctrl mean of gz06/x, days 7-14
K14_P53_ON_LEVEL = 0.7048  # DDIS peak of gz06/x, days 7-14
K14_FAS_GATE = place_hill_gate(
    off_level=K14_P53_OFF_LEVEL, on_level=K14_P53_ON_LEVEL
)
# Receptor turnover: the lag rectifying p53's duty cycle into a level. Not
# fitted — it shifts when FAS peaks, and both sampled days are long past the
# peak, so it is unidentifiable against this dataset (sigma 12 decades).
K14_FAS_TURNOVER_RATE = 8.0  # 1/day; tau = 3 h

# One t_span unit = one day, matching GSE248823's D00–D14 course. DP14 is
# natively in days and runs unchanged; GZ06 (hours) and K14 (minutes) are
# rescaled onto this axis by reconciled_to, and settle to their cycle-average
# on it — which is what per-day bulk transcriptomics samples.
CANONICAL_TIME_SECONDS = 86400.0


K14_CD95L_CHALLENGE_WINDOW = (
    K14_CD95L_CHALLENGE_DAY,
    K14_CD95L_CHALLENGE_DAY + K14_CD95L_CHALLENGE_MINUTES / 1440.0,
)


def build_multi_hallmark_composite(
    *,
    validate: bool = True,
    dose_window=DDIS_ETOPOSIDE_DOSE_WINDOW,
    ligand_challenge=K14_CD95L_CHALLENGE_WINDOW,
):
    """Compose DP14 + GZ06 + K14 into one composite, namespaced ``dp14/``,
    ``gz06/`` and ``k14/``; apply hallmarks for the treated and control
    variants.

    ``dose_window`` is the ``(t_start, t_end)`` damage pulse; ``None`` holds
    ``Irradiation`` at its severity for the whole run instead of washing out.
    ``ligand_challenge`` is the ``(t_start, t_end)`` CD95L window; ``None``
    leaves the apoptosis arm unchallenged and exactly at rest.
    ``validate`` covers topology only — semantic validation is configured per
    sub-composite and at the merge.
    """
    gz06 = (
        process_from_sbml(
            str(GZ06_SBML_PATH),
            name="gz06",
            parameters={
                GZ06_PSI_NAME: GZ06_PSI_PUBLISHED,
                GZ06_ALPHA_X_NAME: GZ06_ALPHA_X_CONTROL,
            },
        )
        .reconciled_to(CANONICAL_TIME_SECONDS)
        .with_param_input(GZ06_ALPHA_X_NAME, "alpha_x_in")
    )
    dp14 = process_from_sbml(
        str(DP14_SBML_PATH),
        name="dp14",
        parameters={
            DP14_MTOR_PHOS_RATE_NAME: DP14_MTOR_PHOS_RATE_DEFAULT,
            DP14_IRRADIATION_RATE_NAME: DP14_IRRADIATION_RATE_DEFAULT,
        },
    ).reconciled_to(CANONICAL_TIME_SECONDS)
    # Import freezes these as inert sinks — nothing inside the model reads
    # them back — but they are its readouts (P0.42).
    k14 = (
        process_from_sbml(
            str(K14_SBML_PATH),
            name="k14",
            native_time_seconds=K14_NATIVE_TIME_SECONDS,
            # An undriven constant keeps the deposit's value, which here is
            # a saturating ligand dose — so zero it (P0.46).
            parameters={K14_CD95L_NAME: 0.0},
        )
        .reconciled_to(CANONICAL_TIME_SECONDS)
        .with_unfrozen("tBid", "mCherry", "mGFP", "PrNES", "PrER")
        .with_param_input(K14_CD95_NAME, "cd95_in")
    )
    processes: dict = {
        "dp14": dp14,
        "gz06": gz06,
        "k14": k14,
        "fas_induction": HillSignalEdge(
            timescale=gz06.timescale,
            basal=K14_CD95_BASAL,
            hi=K14_CD95_INDUCED,
            K=K14_FAS_GATE.K,
            n=K14_FAS_HILL_N,
            source_ontology={"go": "GO:0006977"},
            source_description="GZ06 p53 level",
            hallmark="Genomic Instability",
            reference="Owen-Schaub et al. 1995, Mol Cell Biol 15:3032–3040",
            description="p53 → CD95/Fas receptor induction (setpoint).",
        ),
        "fas_receptor": ClampEdge(
            timescale=gz06.timescale,
            k_clamp=K14_FAS_TURNOVER_RATE,
            target_default=K14_CD95_BASAL,
            target_ontology={"uniprot": "P25445"},
            target_description="CD95/Fas surface receptor level",
            hallmark="Altered Intercellular Communication",
        ),
        "damage_bridge": HillSignalEdge(
            timescale=gz06.timescale,
            basal=GZ06_ALPHA_X_CONTROL,
            hi=GZ06_ALPHA_X_DAMAGED,
            K=GZ06_DAMAGE_DRIVE_K,
            n=GZ06_DAMAGE_DRIVE_N,
            source_ontology={"go": "GO:0006974"},
            source_description="DP14 accumulated DNA damage",
            hallmark="Genomic Instability",
            reference="Banin et al. 1998, Science 281:1674–1677",
            description="DNA damage ⊣ p53 degradation (GZ06 alpha_x).",
        ),
        # Oscillating reporters read their raw species and summarize post-hoc,
        # so no integral observer accumulates, lags, or stiffens the solve.
        # n=1.8 per Shi 2021.
        "p53_cdkn1a": HillActivationEdge(
            timescale=gz06.timescale,
            k_act=10.0,
            K=(0.3,),
            n=(1.8,),
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
        # DP14 DNA_damage ⊣ alpha_x (algebraic Hill edge) → GZ06 reads it as an
        # input; crossing the Hopf at 0.1662 starts the p53 pulses.
        "damage_bridge": {
            "source": "dp14/DNA_damage",
            "signal": "gz06/alpha_x_signal",
        },
        "gz06": {"alpha_x_in": "gz06/alpha_x_signal"},
        # p53 → CDKN1A: read GZ06 p53, add transcription flux to DP14's p21.
        "p53_cdkn1a": {"source": "gz06/x", "target": "dp14/CDKN1A"},
        # p53 → CD95: Hill sets the setpoint, the clamp gives it turnover.
        "fas_induction": {"source": "gz06/x", "signal": "k14/CD95_setpoint"},
        "fas_receptor": {
            "target": "k14/CD95_level",
            "setpoint": "k14/CD95_setpoint",
        },
        "k14": {"cd95_in": "k14/CD95_level"},
    }
    # Extracellular ligand from another cell — the composite's only input
    # that is not the cell's own state.
    if ligand_challenge is not None:
        drive_pulse(
            processes,
            topology,
            target="k14",
            input_name=K14_CD95L_NAME,
            t_start=ligand_challenge[0],
            t_end=ligand_challenge[1],
            amplitude=K14_CD95L_DOSE,
            source_name="cd95l_challenge",
            hallmark="Altered Intercellular Communication",
        )
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
            driven_rate=(
                DP14_IRRADIATION_RATE_NAME,
                DP14_SBML_IRRADIATION_RATE,
            ),
        )
    # severity=0 leaves before == after, so ctrl/DDIS keep the deposit's drive.
    drive_step(
        processes,
        topology,
        target="dp14",
        input_name=DP14_NUTRIENT_INPUT_NAME,
        t_step=RAPA_INTERVENTION_DAY,
        before=DP14_NUTRIENT_BASAL,
        after=DP14_NUTRIENT_BASAL,
        source_name="nutrient_drive",
        hallmark="Deregulated Nutrient Sensing",
    )
    return Composite(
        processes=processes,
        topology=topology,
        validate=validate,
        semantic_validation=True,
    )
