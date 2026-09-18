"""The hallmarks of aging as perturbation handles.

A hallmark of aging (Lopez-Otin et al., 2023) is a signed severity in
[-1, 1] modulating parameters across one or more Processes: -1 is the full
opposite perturbation (mTOR suppression), 0 homeostasis, +1 severely impaired.
A hallmark with no meaningful opposite — there is no negative DNA damage —
uses the [0, 1] half. :data:`HALLMARK_REGISTRY` maps 5 of the 12 today; each
new one is a single :class:`hallsim.handles.Handle` entry. The machinery,
and how to apply a registry, is :mod:`hallsim.handles`.
"""

from __future__ import annotations

from hallsim.handles import FittableCoeff, Handle, ParameterMapping

# ── Registry ────────────────────────────────────────────────────────────

# Hallmark definitions for ERiQ-based processes.
# Process names match those in build_eriq_composite().

HALLMARK_REGISTRY: dict[str, Handle] = {
    "Loss of Proteostasis": Handle(
        name="Loss of Proteostasis",
        description=(
            "Reduced proteasomal degradation capacity in Proctor 2007. "
            "Severity 0 preserves the current activity, 1 inhibits it "
            "completely. This [0, 1] intervention models one mechanism "
            "of proteostasis loss, not all protein quality control."
        ),
        category="Primary",
        references=["Proctor et al. 2007 (BIOMD0000000105), Figures 2–3"],
        mappings=[
            ParameterMapping(
                process_name="p07",
                param_name="parameters.k69",
                floor=1.0,
                slope=-1.0,
                description=(
                    "Proteasome activity: k69 = base * (1 - severity). "
                    "Linear interpolation is a modeling convention between "
                    "normal activity and the paper's complete inhibition."
                ),
            ),
        ],
    ),
    "Stem Cell Exhaustion": Handle(
        name="Stem Cell Exhaustion",
        description=(
            "Age-dependent decline in stem cell niche signaling. "
            "Wnt, EGF, Shh, and Notch pathways deteriorate, reducing "
            "self-renewal capacity and regenerative potential."
        ),
        category="Integrative",
        references=[
            "Lopez-Otin et al. 2023",
            "Sivakumar et al. 2011 (BIOMD0000000398)",
        ],
        mappings=[
            # Stem-cell niche severity is the direct knob (no calibrated
            # base behind it) — base is ignored.
            ParameterMapping(
                process_name="niche",
                param_name="severity",
                transform=lambda h, base: h,
                description="Niche deterioration severity — scales decay of all ligands",
            ),
        ],
    ),
    "Mitochondrial Dysfunction": Handle(
        name="Mitochondrial Dysfunction",
        description=(
            "Impairment in mitochondrial function leading to reduced ATP "
            "production, increased ROS generation, and accumulation of "
            "mitochondrial damage."
        ),
        category="Primary",
        references=["Lopez-Otin et al. 2023", "Alfego & Kriete 2017"],
        mappings=[
            # severity=0 → base (no perturbation); severity=1 → 3*base
            # (the published "severely impaired" factor).
            ParameterMapping(
                process_name="oxidative_stress",
                param_name="MDAMAGE_SA",
                floor=1.0,
                slope=2.0,
                description="Damage accumulation rate scales 1x→3x with dysfunction",
            ),
        ],
    ),
    "Deregulated Nutrient Sensing": Handle(
        name="Deregulated Nutrient Sensing",
        description=(
            "Imbalance in nutrient-sensing pathways (mTOR, AMPK, sirtuins). "
            "Chronic mTOR activation, impaired AMPK response, declining NAD+. "
            "Pharmacological mTORC1 inhibitors (rapamycin and analogs) map "
            "to this hallmark as a downward severity shift."
        ),
        category="Primary",
        references=[
            "Lopez-Otin et al. 2023",
            "Alfego & Kriete 2017",
            "DallePezze 2014 (BIOMD0000000582)",
        ],
        mappings=[
            # ERiQ-based composites: severity=0 → base; severity=1 → 1.5*base.
            ParameterMapping(
                process_name="energy",
                param_name="GLYCOL_SA",
                floor=1.0,
                slope=0.5,
                description="Glycolytic flux scales 1x→1.5x with nutrient dysregulation (ERiQ-based composites)",
            ),
            # DP14-based composites: severity is the level a StepSource
            # ("rapamycin_drive", `forcing.drive_step`) holds DP14's mTORC1
            # S2448 phosphorylation rate at after the step — the kinase
            # rapamycin inhibits, not the amino-acid input, which rapamycin
            # leaves alone. Arms still differ only in u(t).
            # Skipped for composites without the source.
            ParameterMapping(
                process_name="rapamycin_drive",
                param_name="after",
                floor=1.0,
                slope=FittableCoeff(
                    init=0.7,
                    clamp=(0.05, 0.95),
                    prior=0.7,
                    prior_sigma=0.3,
                    description="mTORC1 inhibition gain (severity=-1 → (1-gain)x the published S2448 phosphorylation rate under rapamycin)",
                ),
                description=(
                    "DP14 mTORC1 S2448 phosphorylation rate after the step: "
                    "published at severity=0, (1-gain)x at severity=-1 "
                    "(rapamycin), (1+gain)x at severity=+1"
                ),
            ),
        ],
    ),
    "Genomic Instability": Handle(
        name="Genomic Instability",
        description=(
            "Exogenous DNA damage exposure. Drives ERiQ's damage_repair "
            "(eta) and DP14's Irradiation exposure input — severity is the "
            "normalized exposure level (0=none, 1=full). GZ06's psi is no "
            "longer set here: it is driven by DP14's DNA_damage state "
            "through a topology edge (see multi_hallmark). The per-exposure "
            "damage potency is a mechanism parameter fit separately, not "
            "part of this dial."
        ),
        category="Primary",
        references=[
            "Lopez-Otin et al. 2023",
            "DallePezze 2014 (BIOMD0000000582)",
            "Geva-Zatorsky 2006 (BIOMD0000000157)",
        ],
        mappings=[
            # ERiQ-based composites: severity=0 → base; severity=1 → 5*base.
            ParameterMapping(
                process_name="damage_repair",
                param_name="eta",
                floor=1.0,
                slope=4.0,
                description="Damage production rate scales 1x→5x with instability (ERiQ-based composites)",
            ),
            # DP14-based composites: severity IS the exogenous-exposure
            # level — an identity dial (0 = no exposure, 1 = full DDIS dose).
            # It sets the amplitude of the forcing source (`forcing.drive_pulse`
            # adds a PulseSource named "irradiation_pulse" driving DP14's
            # Irradiation input over the dose window). The damage *potency* per
            # unit exposure (`DNA_damaged_by_irradiation`) is a separate
            # mechanism parameter Calibrator fits; severity never touches it.
            # Skipped for composites without the pulse source (apply() ignores
            # mappings whose process is absent).
            ParameterMapping(
                process_name="irradiation_pulse",
                param_name="amplitude",
                transform=lambda h, base: h,
                description=(
                    "Exogenous-exposure level (irradiation PulseSource "
                    "amplitude): 0 at severity=0 (no exposure), full at "
                    "severity=1 (full DDIS dose)"
                ),
            ),
            # GZ06's psi is not mapped here — it is driven by DP14's
            # DNA_damage via a topology edge (see multi_hallmark), so GI
            # severity reaches GZ06 through Irradiation → DNA_damage → psi.
        ],
    ),
}
