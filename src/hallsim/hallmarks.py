"""The hallmarks of aging as perturbation handles.

A hallmark of aging (Lopez-Otin et al., 2023) is a signed severity in
[-1, 1] modulating parameters across one or more Processes: -1 is the full
opposite perturbation (mTOR suppression), 0 homeostasis, +1 severely impaired.
A hallmark with no meaningful opposite — there is no negative DNA damage —
uses the [0, 1] half. :data:`HALLMARK_REGISTRY` carries all 12; five are
grounded in a calibrated composite, the other seven point at plausible
rates of the same models and say so in their description. Each is a single
:class:`hallsim.handles.Handle` entry. The machinery, and how to apply a
registry, is :mod:`hallsim.handles`.
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
            # DP14-based composites: the mitochondrial dysfunction rate the
            # model already carries, 1x -> 3x; placeholder gain, not calibrated.
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.mito_dysfunction",
                floor=1.0,
                slope=2.0,
                description="Mitochondrial dysfunction rate scales 1x->3x (DP14-based composites; uncalibrated placeholder)",
            ),
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
    # The seven below complete the Lopez-Otin 2023 inventory. Each points at
    # a rate the multi-hallmark composite already carries and none is
    # calibrated against data: they are plausible placeholders, chosen off
    # the parameters the demo fits so the calibration surface is unchanged.
    "Telomere Attrition": Handle(
        name="Telomere Attrition",
        description=(
            "Persistent telomere-initiated DNA damage response: the basal "
            "level of the damage signal into p53 (GZ06 alpha_x at zero "
            "acute damage) falls, so p53 sits closer to its pulsing regime. "
            "Uncalibrated placeholder."
        ),
        category="Primary",
        references=[
            "Lopez-Otin et al. 2023",
            "d'Adda di Fagagna et al. 2003 (telomere-initiated DDR)",
        ],
        mappings=[
            ParameterMapping(
                process_name="damage_bridge",
                param_name="basal",
                floor=1.0,
                slope=-0.5,
                description="Damage-free p53 degradation drive 1x->0.5x with attrition",
            ),
        ],
    ),
    "Epigenetic Alterations": Handle(
        name="Epigenetic Alterations",
        description=(
            "Derepression of CDK-inhibitor loci with heterochromatin loss: "
            "CDKN1B transcription rises. Uncalibrated placeholder."
        ),
        category="Primary",
        references=["Lopez-Otin et al. 2023"],
        mappings=[
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.CDKN1B_transcr_by_FoxO3a_n_DNA_damage",
                floor=1.0,
                slope=1.0,
                description="CDKN1B transcription 1x->2x (DP14-based composites)",
            ),
        ],
    ),
    "Disabled Macroautophagy": Handle(
        name="Disabled Macroautophagy",
        description=(
            "Autophagic flux falls: DP14's mitophagy of new and old "
            "mitochondria slows. Uncalibrated placeholder."
        ),
        category="Primary",
        references=["Lopez-Otin et al. 2023"],
        mappings=[
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.mitophagy_new",
                floor=1.0,
                slope=-0.8,
                description="Mitophagy of new mitochondria 1x->0.2x",
            ),
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.mitophagy_old",
                floor=1.0,
                slope=-0.8,
                description="Mitophagy of old mitochondria 1x->0.2x",
            ),
        ],
    ),
    "Cellular Senescence": Handle(
        name="Cellular Senescence",
        description=(
            "Senescence as a lever rather than an outcome: p53-driven "
            "CDKN1A transcription and ROS-driven SA-beta-gal accumulation "
            "gain. Uncalibrated placeholder."
        ),
        category="Antagonistic",
        references=["Lopez-Otin et al. 2023"],
        mappings=[
            ParameterMapping(
                process_name="p53_cdkn1a",
                param_name="hi",
                floor=1.0,
                slope=1.0,
                description="p53 -> CDKN1A transcription gain 1x->2x",
            ),
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.sen_ass_beta_gal_inc_by_ROS",
                floor=1.0,
                slope=1.0,
                description="ROS-driven SA-beta-gal accumulation 1x->2x",
            ),
        ],
    ),
    "Altered Intercellular Communication": Handle(
        name="Altered Intercellular Communication",
        description=(
            "Endocrine drift: the insulin/IGF-1 input DP14 reads falls. "
            "Uncalibrated placeholder."
        ),
        category="Integrative",
        references=["Lopez-Otin et al. 2023"],
        mappings=[
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.Insulin",
                floor=1.0,
                slope=-0.5,
                description="Insulin input 1x->0.5x",
            ),
        ],
    ),
    "Chronic Inflammation": Handle(
        name="Chronic Inflammation",
        description=(
            "Inflammaging: ROS-driven IKKbeta and JNK activation gain. "
            "Uncalibrated placeholder."
        ),
        category="Integrative",
        references=[
            "Lopez-Otin et al. 2023",
            "Franceschi et al. 2018 (inflammaging)",
        ],
        mappings=[
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.IKKbeta_activ_by_ROS",
                floor=1.0,
                slope=2.0,
                description="IKKbeta activation by ROS 1x->3x",
            ),
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.JNK_activ_by_ROS",
                floor=1.0,
                slope=1.0,
                description="JNK activation by ROS 1x->2x",
            ),
        ],
    ),
    "Dysbiosis": Handle(
        name="Dysbiosis",
        description=(
            "Microbial products feed the inflammatory axis: a weaker "
            "IKKbeta activation gain than Chronic Inflammation. "
            "Uncalibrated placeholder."
        ),
        category="Integrative",
        references=["Lopez-Otin et al. 2023"],
        mappings=[
            ParameterMapping(
                process_name="dp14",
                param_name="parameters.IKKbeta_activ_by_ROS",
                floor=1.0,
                slope=0.5,
                description="IKKbeta activation by ROS 1x->1.5x",
            ),
        ],
    ),
}
