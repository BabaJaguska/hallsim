"""ERiQ — Energy Restriction in Quiescence model, decomposed into composable Processes.

Ported from the MATLAB implementation by Alfego & Kriete (2017):
    Alfego, D., & Kriete, A. (2017). Simulation of cellular energy
    restriction in quiescence (ERiQ) — a theoretical model for aging.
    Biology, 6(4), 44.

The original monolithic ODE system (11 state variables, ~20 algebraic
nodes) is decomposed into three composable Processes:

1. **ERiQEnergyMetabolism** — mitochondrial function, enzymes,
   glycolysis, glycolytic enzymes (4 ODEs)
2. **ERiQOxidativeStress** — mitochondrial damage, ROS feedback
   integrators (3 ODEs)
3. **ERiQSignaling** — mTOR and p53 regulatory feedback loops (4 ODEs)

Each Process computes its needed algebraic intermediates internally
from shared state variables.  The algebraic nodes (AKT, AMPK, SIRT,
mTOR, p53, FOXO, NFkB, etc.) are dense and tightly coupled, so some
computation is duplicated across modules — this is cheap arithmetic
and keeps the modules independently testable and replaceable.

Homeostatic initial conditions from the original paper::

    Y0 = [0.0724, 3.6239, -1.3358, 2.4010, -2.1968,
          -0.0000, -0.1936, -0.0000, 0.8734, -0.7944, 0.0794]

    Order: [MDAMAGE, MFUNCT, MENZY, GLYCOL, GLYENZ,
            Cy, Ay, Cx, Ax, Cz, Az]
"""

from __future__ import annotations

import jax.numpy as jnp

from hallsim.process import Port, PortRole, Process


# ── Original algebraic computation (Alfego & Kriete 2017) ──────────────


def _compute_algebraic_original(state: dict) -> dict:
    """Original ERiQ algebraic equations with raw reciprocal terms.

    Preserved for comparison and for demonstrating the ModelAnalyzer.
    These equations have numerical singularities (1/x terms) that cause
    blow-up when state variables approach zero.  Use ``_compute_algebraic``
    for the revised, numerically stable version.
    """
    eps = 1e-6

    mfunct = state.get("mito_function", 3.6239)
    glycol = state.get("glycolysis", 2.401)
    mdamage = state.get("mito_damage", 0.0724)
    mtor_act = state.get("mTOR_activity", -0.1936)
    p53_act = state.get("p53_activity", 0.8734)
    ros_act = state.get("ROS_activity", 0.0794)
    ros_int = state.get("ROS_integrator_c", -0.7944)

    p = state.get("_params", {})
    ROS_SA = p.get("ROS_SA", 1.0)
    PTEN_SA = p.get("PTEN_SA", 1.0)
    AKT_SA = p.get("AKT_SA", 1.0)
    AMPK_SA = p.get("AMPK_SA", 1.0)
    NADr_SA = p.get("NADr_SA", 1.0)
    SIRT_SA = p.get("SIRT_SA", 1.0)
    PGC1a_SA = p.get("PGC1a_SA", 1.0)
    MTOR_SA = p.get("MTOR_SA", 1.0)
    NFKB_SA = p.get("NFKB_SA", 1.0)
    P53_SA = p.get("P53_SA", 1.0)
    P53_Base = p.get("P53_Base", 4.0)
    P53_Act = p.get("P53_Act", 1.0)
    FOXO_SA = p.get("FOXO_SA", 1.0)
    FREERAD_SA = p.get("FREERAD_SA", 1.0)
    AUTO_SA = p.get("AUTO_SA", 1.0)
    HIF_SA = p.get("HIF_SA", 1.0)
    PYR_SA = p.get("PYR_SA", 1.0)
    GLU_SA = p.get("GLU_SA", 1.0)
    MDR_SA = p.get("MDR_SA", 1.0)
    MDR = p.get("MDR", 1.8e-3)

    ATPm = mfunct
    ATPg = glycol
    ATPr = ATPm + ATPg
    ROS = 10.0 * ROS_SA * ros_act
    PTEN = PTEN_SA * (1.0 / (mfunct + eps))  # SINGULARITY
    GF = 0.1
    AKT = AKT_SA * (GF + PTEN + ROS / 5.0)  # SIGN ERROR: PTEN adds
    AMPK = AMPK_SA * (1.0 / (ATPr + eps))  # SINGULARITY
    NADr = NADr_SA * mfunct
    SIRT = SIRT_SA * NADr  # No saturation
    PGC1a = PGC1a_SA * (AMPK + 0.1 * SIRT)
    MTORs = AKT - 4.0 * AMPK
    MTORa = mtor_act - 1.5 * MTORs
    MTOR = MTOR_SA * (1.0 + MTORs + MTORa)
    NFKB = NFKB_SA * (AKT + 0.25 * ROS + 0.25 * MTOR)
    P53s = 0.3 * (P53_Base - AKT - NFKB + 0.5 * ROS) * P53_Act
    P53a = p53_act - P53s
    P53 = P53_SA * (P53s + P53a)
    FOXO = FOXO_SA * (1.0 / (AKT + eps))  # SINGULARITY
    Uz = FREERAD_SA * (P53 + 0.2 * mdamage + ros_int - 0.05 * FOXO)
    AUTOPHAGY = (
        AUTO_SA * 0.001 * (1.0 / (MTOR + eps) + 0.5 * FOXO + ROS + P53)
    )  # SINGULARITY
    HIF = HIF_SA * AKT
    PYR = PYR_SA * glycol * 0.7
    GLU = GLU_SA * (1.0 / (NFKB + eps))  # SINGULARITY
    MD = MDR_SA * (jnp.abs(mfunct + ROS) * MDR + (ROS - 0.8) * 0.0001)

    return {
        "ATPm": ATPm,
        "ATPg": ATPg,
        "ATPr": ATPr,
        "ROS": ROS,
        "PTEN": PTEN,
        "AKT": AKT,
        "AMPK": AMPK,
        "NADr": NADr,
        "SIRT": SIRT,
        "PGC1a": PGC1a,
        "MTORs": MTORs,
        "MTORa": MTORa,
        "MTOR": MTOR,
        "NFKB": NFKB,
        "P53s": P53s,
        "P53a": P53a,
        "P53": P53,
        "FOXO": FOXO,
        "Uz": Uz,
        "AUTOPHAGY": AUTOPHAGY,
        "HIF": HIF,
        "PYR": PYR,
        "GLU": GLU,
        "MD": MD,
        # Use 1/SIRT for original glycolysis equation
        "SIRT_gly_inhibition": 1.0 / (SIRT + eps),
    }


# ── Revised algebraic computation ─────────────────────────────────────



def _compute_algebraic(state: dict) -> dict:
    """Compute all algebraic (non-ODE) nodes from the 11 state variables.

    Parameters are taken from a `params` dict if present in state,
    otherwise default _SA=1.0 values are used.  This function is called
    internally by each Process to get the intermediates it needs.

    Equation revisions vs. the original ERiQ (Alfego & Kriete 2017):
    All raw reciprocal ``1/x`` terms have been replaced with biophysically
    grounded Hill or Michaelis-Menten functions to eliminate singularities
    and incorporate saturation / ultrasensitivity.
    See ``docs/eriq-equation-revisions.md`` for detailed rationale and
    literature references for each change.
    """
    # State variables (with defaults for safety)
    mfunct = state.get("mito_function", 3.6239)
    glycol = state.get("glycolysis", 2.401)
    mdamage = state.get("mito_damage", 0.0724)
    mtor_act = state.get("mTOR_activity", -0.1936)  # Ay
    p53_act = state.get("p53_activity", 0.8734)  # Ax
    ros_act = state.get("ROS_activity", 0.0794)  # Az
    ros_int = state.get("ROS_integrator_c", -0.7944)  # Cz

    # Scaling factors (SA parameters) — read from params sub-dict
    p = state.get("_params", {})
    ROS_SA = p.get("ROS_SA", 1.0)
    PTEN_SA = p.get("PTEN_SA", 1.0)
    AKT_SA = p.get("AKT_SA", 1.0)
    AMPK_SA = p.get("AMPK_SA", 1.0)
    NADr_SA = p.get("NADr_SA", 1.0)
    SIRT_SA = p.get("SIRT_SA", 1.0)
    PGC1a_SA = p.get("PGC1a_SA", 1.0)
    MTOR_SA = p.get("MTOR_SA", 1.0)
    NFKB_SA = p.get("NFKB_SA", 1.0)
    P53_SA = p.get("P53_SA", 1.0)
    P53_Base = p.get("P53_Base", 4.0)
    P53_Act = p.get("P53_Act", 1.0)
    FOXO_SA = p.get("FOXO_SA", 1.0)
    FREERAD_SA = p.get("FREERAD_SA", 1.0)
    AUTO_SA = p.get("AUTO_SA", 1.0)
    HIF_SA = p.get("HIF_SA", 1.0)
    PYR_SA = p.get("PYR_SA", 1.0)
    GLU_SA = p.get("GLU_SA", 1.0)
    MDR_SA = p.get("MDR_SA", 1.0)
    MDR = p.get("MDR", 1.8e-3)

    # Hill / Michaelis-Menten parameters (see docs/eriq-equation-revisions.md)
    K_AMPK = p.get("K_AMPK", 3.0)  # ATP half-inhibition for AMPK
    n_AMPK = p.get("n_AMPK", 2.0)  # AMPK Hill coefficient
    K_PTEN = p.get("K_PTEN", 0.5)  # ROS half-inhibition for PTEN
    n_PTEN = p.get("n_PTEN", 1.0)  # PTEN Hill coefficient
    K_AKT_PTEN = p.get("K_AKT_PTEN", 0.5)  # PTEN half-inhibition of AKT
    K_FOXO = p.get("K_FOXO", 0.5)  # AKT half-inhibition of FOXO
    n_FOXO = p.get("n_FOXO", 2.0)  # FOXO Hill coefficient
    K_AUTO = p.get("K_AUTO", 1.0)  # mTOR half-inhibition of autophagy
    n_AUTO = p.get("n_AUTO", 2.0)  # autophagy Hill coefficient
    Km_SIRT = p.get("Km_SIRT", 1.5)  # NAD+ Km for sirtuin (MM)
    K_SIRT_gly = p.get("K_SIRT_gly", 0.5)  # SIRT half-inhibition of glycolysis
    K_GLU = p.get("K_GLU", 1.0)  # NFkB half-inhibition of glucose uptake

    # ATP
    ATPm = mfunct
    ATPg = glycol
    ATPr = ATPm + ATPg

    # ROS
    ROS = 10.0 * ROS_SA * ros_act

    # AMPK — activated by low ATP (inhibitory Hill function of ATPr)
    # Replaces: AMPK = 1/ATPr.  See Hardie et al. (2012), n=2 for
    # ultrasensitive switch from cooperative AMP binding + adenylate kinase.
    AMPK = AMPK_SA * K_AMPK**n_AMPK / (K_AMPK**n_AMPK + ATPr**n_AMPK)

    # PTEN — inactivated by ROS (inhibitory Hill function of ROS)
    # Replaces: PTEN = 1/mfunct.  ROS oxidizes catalytic Cys124, forming
    # disulfide bond → inactive PTEN.  See Lee et al. (2002).
    PTEN = PTEN_SA * K_PTEN**n_PTEN / (K_PTEN**n_PTEN + ROS**n_PTEN)

    # AKT — activated by growth factors, inhibited by PTEN
    # Replaces: AKT = GF + PTEN + ROS/5.  PTEN degrades PIP3 required for
    # AKT membrane recruitment → PTEN *inhibits* AKT (sign correction).
    GF = 0.1
    ROS_boost = ROS / 5.0
    AKT = AKT_SA * (GF + ROS_boost) * K_AKT_PTEN / (K_AKT_PTEN + PTEN)

    # NAD+/NADH ratio (unchanged — linear proxy)
    NADr = NADr_SA * mfunct

    # Sirtuin — Michaelis-Menten dependence on NAD+
    # Replaces: SIRT = NADr (linear).  SIRT1 Km(NAD+) ≈ 100–170 μM, near
    # physiological levels → operates in responsive region of MM curve.
    # See Borra et al. (2004), Cantó et al. (2015).
    SIRT = SIRT_SA * NADr / (Km_SIRT + NADr)

    # PGC1alpha (unchanged)
    PGC1a = PGC1a_SA * (AMPK + 0.1 * SIRT)

    # mTOR (unchanged — already bounded algebraic form)
    MTORs = AKT - 4.0 * AMPK
    MTORa = mtor_act - 1.5 * MTORs
    MTOR = MTOR_SA * (1.0 + MTORs + MTORa)

    # NFkB (unchanged — linear combination)
    NFKB = NFKB_SA * (AKT + 0.25 * ROS + 0.25 * MTOR)

    # p53 (unchanged — algebraic feedback)
    P53s = 0.3 * (P53_Base - AKT - NFKB + 0.5 * ROS) * P53_Act
    P53a = p53_act - P53s
    P53 = P53_SA * (P53s + P53a)

    # FOXO — inhibited by AKT (inhibitory Hill function)
    # Replaces: FOXO = 1/AKT.  AKT phosphorylates FOXO at 3 sites →
    # 14-3-3 binding → cytoplasmic sequestration.  Multi-site phosphorylation
    # gives n≈2.  See Calzone et al. (2010).
    FOXO = FOXO_SA * K_FOXO**n_FOXO / (K_FOXO**n_FOXO + AKT**n_FOXO)

    # Free radical driver (Uz) — unchanged
    Uz = FREERAD_SA * (P53 + 0.2 * mdamage + ros_int - 0.05 * FOXO)

    # Autophagy — mTOR inhibition via Hill function
    # Replaces: 0.001 * (1/MTOR + ...).  mTORC1 phosphorylates ULK1 at
    # Ser757 → switch-like suppression.  See Kapuy et al. (2014).
    # The 0.001 prefactor from the original model is retained — it scales
    # autophagy to the correct magnitude relative to damage accumulation.
    mtor_inhibition = K_AUTO**n_AUTO / (K_AUTO**n_AUTO + MTOR**n_AUTO)
    AUTOPHAGY = AUTO_SA * 0.001 * (mtor_inhibition + 0.5 * FOXO + ROS + P53)

    # HIF (unchanged)
    HIF = HIF_SA * AKT

    # Pyruvate (unchanged)
    PYR = PYR_SA * glycol * 0.7

    # Glucose uptake — NFkB inhibition via MM form
    # Replaces: GLU = 1/NFKB.  Chronic NFkB → insulin resistance → impaired
    # glucose uptake.  See Shoelson et al. (2006).
    GLU = GLU_SA * K_GLU / (K_GLU + NFKB)

    # Mitochondrial damage rate — mass-action (ROS × undamaged mass)
    # Replaces: |mfunct + ROS| * MDR + (ROS - 0.8) * 0.0001.
    # Damage ∝ ROS × functional mitochondria (targets available).
    # See Kowald & Kirkwood (2000).
    MD = MDR_SA * MDR * ROS * jnp.maximum(mfunct, 0.0)

    return {
        "ATPm": ATPm,
        "ATPg": ATPg,
        "ATPr": ATPr,
        "ROS": ROS,
        "PTEN": PTEN,
        "AKT": AKT,
        "AMPK": AMPK,
        "NADr": NADr,
        "SIRT": SIRT,
        "PGC1a": PGC1a,
        "MTORs": MTORs,
        "MTORa": MTORa,
        "MTOR": MTOR,
        "NFKB": NFKB,
        "P53s": P53s,
        "P53a": P53a,
        "P53": P53,
        "FOXO": FOXO,
        "Uz": Uz,
        "AUTOPHAGY": AUTOPHAGY,
        "HIF": HIF,
        "PYR": PYR,
        "GLU": GLU,
        "MD": MD,
        # Export SIRT inhibition term for glycolysis derivative
        "SIRT_gly_inhibition": K_SIRT_gly / (K_SIRT_gly + SIRT),
    }


# ── Homeostatic initial conditions ──────────────────────────────────────

ERIQ_HOMEOSTATIC_IC = {
    "mito_damage": 0.0724,
    "mito_function": 3.6239,
    "mito_enzymes": -1.3358,
    "glycolysis": 2.4010,
    "glycolytic_enzymes": -2.1968,
    "mTOR_integrator_c": -0.0000,
    "mTOR_activity": -0.1936,
    "p53_integrator_c": -0.0000,
    "p53_activity": 0.8734,
    "ROS_integrator_c": -0.7944,
    "ROS_activity": 0.0794,
}

# Default sensitivity-analysis scaling factors (all 1.0 = baseline)
# + Hill / Michaelis-Menten parameters (see docs/eriq-equation-revisions.md)
ERIQ_DEFAULT_PARAMS = {
    # Original SA scaling factors
    "ROS_SA": 1.0,
    "PTEN_SA": 1.0,
    "AKT_SA": 1.0,
    "AMPK_SA": 1.0,
    "NADr_SA": 1.0,
    "SIRT_SA": 1.0,
    "PGC1a_SA": 1.0,
    "MTOR_SA": 1.0,
    "NFKB_SA": 1.0,
    "P53_SA": 1.0,
    "P53_Base": 4.0,
    "P53_Act": 1.0,
    "FOXO_SA": 1.0,
    "FREERAD_SA": 1.0,
    "AUTO_SA": 1.0,
    "HIF_SA": 1.0,
    "PYR_SA": 1.0,
    "GLU_SA": 1.0,
    "MDR_SA": 1.0,
    "MDR": 1.8e-3,
    "MDAMAGE_SA": 1.0,
    "GLYCOL_SA": 1.0,
    # Hill / MM parameters for revised equations
    "K_AMPK": 3.0,
    "n_AMPK": 2.0,  # AMPK: inhibitory Hill of ATP
    "K_PTEN": 0.5,
    "n_PTEN": 1.0,  # PTEN: inhibitory Hill of ROS
    "K_AKT_PTEN": 0.5,  # AKT: PTEN half-inhibition
    "K_FOXO": 0.5,
    "n_FOXO": 2.0,  # FOXO: inhibitory Hill of AKT
    "K_AUTO": 1.0,
    "n_AUTO": 2.0,  # Autophagy: inhibitory Hill of mTOR
    "Km_SIRT": 1.5,  # SIRT: Michaelis-Menten for NAD+
    "K_SIRT_gly": 0.5,  # Glycolysis: SIRT inhibition half-max
    "K_GLU": 1.0,  # Glucose uptake: NFkB inhibition
}

# Store path prefix used by default topology
ERIQ_PREFIX = "eriq"


# ── Process 1: Energy Metabolism ────────────────────────────────────────


class ERiQEnergyMetabolism(Process):
    """Mitochondrial oxidative phosphorylation + glycolysis.

    4 state variables:
    - mito_function (MFUNCT): mitochondrial output efficiency
    - mito_enzymes (MENZY): mitochondrial enzyme levels
    - glycolysis (GLYCOL): glycolytic flux
    - glycolytic_enzymes (GLYENZ): glycolytic enzyme levels

    Reads from other modules: mito_damage, mTOR_activity, p53_activity,
    ROS_activity, ROS_integrator_c (for algebraic intermediates).
    """

    GLYCOL_SA: float = 1.0

    def ports_schema(self):
        return {
            # EXCLUSIVE: this process owns these derivatives
            "mito_function": Port(
                role=PortRole.EXCLUSIVE,
                default=3.6239,
                units="dimensionless",
                description="Mitochondrial output efficiency",
            ),
            "mito_enzymes": Port(
                role=PortRole.EXCLUSIVE,
                default=-1.3358,
                units="dimensionless",
                description="Mitochondrial enzyme levels",
            ),
            "glycolysis": Port(
                role=PortRole.EXCLUSIVE,
                default=2.4010,
                units="dimensionless",
                description="Glycolytic flux",
            ),
            "glycolytic_enzymes": Port(
                role=PortRole.EXCLUSIVE,
                default=-2.1968,
                units="dimensionless",
                description="Glycolytic enzyme levels",
            ),
            # INPUT: reads from other modules
            "mito_damage": Port(role=PortRole.INPUT, default=0.0724),
            "mTOR_activity": Port(role=PortRole.INPUT, default=-0.1936),
            "p53_activity": Port(role=PortRole.INPUT, default=0.8734),
            "ROS_activity": Port(role=PortRole.INPUT, default=0.0794),
            "ROS_integrator_c": Port(role=PortRole.INPUT, default=-0.7944),
        }

    def derivative(self, t, state):
        obs = _compute_algebraic(state)

        menzy = state["mito_enzymes"]
        glyenz = state["glycolytic_enzymes"]
        mdamage = state["mito_damage"]

        # Mitochondrial function
        # r2 = PGC1a + PYR + P53 - 0.2*HIF - 0.2*NFkB
        r2 = (
            obs["PGC1a"]
            + obs["PYR"]
            + obs["P53"]
            - 0.2 * obs["HIF"]
            - 0.2 * obs["NFKB"]
        )
        gain2 = 0.05  # original MATLAB value
        u2 = r2 + menzy
        dMito_function = gain2 * u2 - obs["SIRT"] * 0.02

        # Mitochondrial enzymes
        k3 = 1.0
        k4 = r2 - mdamage
        dMito_enzymes = -(k3 * obs["ATPm"]) - (k4 * menzy)

        # Glycolysis — SIRT inhibition via saturating Hill form
        # Replaces: 1/SIRT (singularity).  SIRT1 inhibits glycolysis by
        # deacetylating HIF-1α and glycolytic enzymes; effect saturates.
        # See docs/eriq-equation-revisions.md, Revision 6.
        r3 = obs["GLU"] + 0.01 * obs["HIF"] + 0.01 * obs["NADr"]
        gain3 = 0.25
        u3 = r3 + glyenz
        dGlycolysis = self.GLYCOL_SA * (
            gain3 * u3 + obs["SIRT_gly_inhibition"]
        )

        # Glycolytic enzymes
        k5 = -1.0
        k6 = r3
        dGlycolytic_enzymes = k5 * obs["ATPg"] - k6 * glyenz

        return {
            "mito_function": dMito_function,
            "mito_enzymes": dMito_enzymes,
            "glycolysis": dGlycolysis,
            "glycolytic_enzymes": dGlycolytic_enzymes,
        }


# ── Process 2: Oxidative Stress & Damage ────────────────────────────────


class ERiQOxidativeStress(Process):
    """Mitochondrial damage accumulation + ROS feedback regulation.

    3 state variables:
    - mito_damage (MDAMAGE): structural mitochondrial damage
    - ROS_integrator_c (Cz): ROS feedback integrator
    - ROS_activity (Az): ROS activity (ROS = 10 * ROS_SA * Az)

    Reads from other modules: mito_function, glycolysis, mTOR_activity,
    p53_activity.
    """

    MDAMAGE_SA: float = 1.0

    def ports_schema(self):
        return {
            # EXCLUSIVE
            "mito_damage": Port(
                role=PortRole.EXCLUSIVE,
                default=0.0724,
                units="dimensionless",
                description="Mitochondrial structural damage",
                ontology={"go": "GO:0000422"},  # mitochondrion degradation
            ),
            "ROS_integrator_c": Port(
                role=PortRole.EXCLUSIVE,
                default=-0.7944,
                units="dimensionless",
                description="ROS regulatory feedback integrator (Cz)",
            ),
            "ROS_activity": Port(
                role=PortRole.EXCLUSIVE,
                default=0.0794,
                units="dimensionless",
                description="ROS activity level (Az); ROS = 10 * ROS_SA * Az",
                ontology={"chebi": "CHEBI:26523"},
            ),
            # INPUT
            "mito_function": Port(role=PortRole.INPUT, default=3.6239),
            "glycolysis": Port(role=PortRole.INPUT, default=2.4010),
            "mTOR_activity": Port(role=PortRole.INPUT, default=-0.1936),
            "p53_activity": Port(role=PortRole.INPUT, default=0.8734),
        }

    def derivative(self, t, state):
        obs = _compute_algebraic(state)

        # Mitochondrial damage: dMDAMAGE = MDAMAGE_SA * (MD - AUTOPHAGY)
        dMito_damage = self.MDAMAGE_SA * (obs["MD"] - obs["AUTOPHAGY"])

        # ROS feedback (from MATLAB f_ROS_feedback):
        # dCz = -ROS - Cz
        # dAz = gz * Uz  where gz = 0.01
        ros_int = state["ROS_integrator_c"]
        dROS_integrator_c = -obs["ROS"] - ros_int
        dROS_activity = 0.01 * obs["Uz"]

        return {
            "mito_damage": dMito_damage,
            "ROS_integrator_c": dROS_integrator_c,
            "ROS_activity": dROS_activity,
        }


# ── Process 3: Signaling Feedback ───────────────────────────────────────


class ERiQSignaling(Process):
    """mTOR and p53 regulatory feedback loops.

    4 state variables:
    - mTOR_integrator_c (Cy): mTOR feedback integrator
    - mTOR_activity (Ay): mTOR activity level
    - p53_integrator_c (Cx): p53 feedback integrator
    - p53_activity (Ax): p53 activity level

    Reads from other modules: mito_function, glycolysis, ROS_activity,
    ROS_integrator_c, mito_damage.
    """

    def ports_schema(self):
        return {
            # EXCLUSIVE
            "mTOR_integrator_c": Port(
                role=PortRole.EXCLUSIVE,
                default=-0.0000,
                units="dimensionless",
                description="mTOR regulatory feedback integrator (Cy)",
            ),
            "mTOR_activity": Port(
                role=PortRole.EXCLUSIVE,
                default=-0.1936,
                units="dimensionless",
                description="mTOR activity level (Ay)",
                ontology={"go": "GO:0031929"},  # TOR signaling
            ),
            "p53_integrator_c": Port(
                role=PortRole.EXCLUSIVE,
                default=-0.0000,
                units="dimensionless",
                description="p53 regulatory feedback integrator (Cx)",
            ),
            "p53_activity": Port(
                role=PortRole.EXCLUSIVE,
                default=0.8734,
                units="dimensionless",
                description="p53 activity level (Ax)",
                ontology={"go": "GO:0030330"},  # DNA damage response, p53
            ),
            # INPUT
            "mito_function": Port(role=PortRole.INPUT, default=3.6239),
            "glycolysis": Port(role=PortRole.INPUT, default=2.4010),
            "ROS_activity": Port(role=PortRole.INPUT, default=0.0794),
            "ROS_integrator_c": Port(role=PortRole.INPUT, default=-0.7944),
            "mito_damage": Port(role=PortRole.INPUT, default=0.0724),
        }

    def derivative(self, t, state):
        obs = _compute_algebraic(state)

        mtor_int = state["mTOR_integrator_c"]
        p53_int = state["p53_integrator_c"]

        # mTOR feedback (from MATLAB f_MTOR_feedback):
        # Uy = ry + Cy  where ry = 0
        # dCy = -MTORa - Cy
        # dAy = gy * Uy  where gy = 0.1
        ry = 0.0
        gy = 0.1
        Uy = ry + mtor_int
        dMTOR_integrator_c = -obs["MTORa"] - mtor_int
        dMTOR_activity = gy * Uy

        # p53 feedback (from MATLAB f_P53_feedback):
        # Ux = rx + Cx  where rx = 0
        # dCx = -P53a - Cx
        # dAx = gx * Ux  where gx = 0.1
        rx = 0.0
        gx = 0.1
        Ux = rx + p53_int
        dp53_integrator_c = -obs["P53a"] - p53_int
        dp53_activity = gx * Ux

        return {
            "mTOR_integrator_c": dMTOR_integrator_c,
            "mTOR_activity": dMTOR_activity,
            "p53_integrator_c": dp53_integrator_c,
            "p53_activity": dp53_activity,
        }


# ── Convenience: build a full ERiQ composite ────────────────────────────


def build_eriq_composite(
    *,
    GLYCOL_SA: float = 1.0,
    MDAMAGE_SA: float = 1.0,
    prefix: str = ERIQ_PREFIX,
    validate: bool = True,
    semantic_validation: bool = False,
):
    """Build a Composite wiring the 3 ERiQ Processes together.

    All 11 state variables are placed under ``{prefix}/`` store paths.
    Shared variables (read by multiple processes) are wired to the same
    store path so they see each other's updates at each macro step.

    Parameters
    ----------
    GLYCOL_SA, MDAMAGE_SA:
        Scaling factors exposed at the top level for quick sensitivity
        analysis.  All other _SA parameters can be modulated via
        hallmark handles.
    prefix:
        Store path prefix (default: ``"eriq"``).
    """
    from hallsim.composite import Composite

    processes = {
        "energy": ERiQEnergyMetabolism(GLYCOL_SA=GLYCOL_SA),
        "oxidative_stress": ERiQOxidativeStress(MDAMAGE_SA=MDAMAGE_SA),
        "signaling": ERiQSignaling(),
    }

    p = prefix

    # Topology: map port names to store paths.
    # Shared state variables get the same store path across processes.
    topology = {
        "energy": {
            "mito_function": f"{p}/mito_function",
            "mito_enzymes": f"{p}/mito_enzymes",
            "glycolysis": f"{p}/glycolysis",
            "glycolytic_enzymes": f"{p}/glycolytic_enzymes",
            "mito_damage": f"{p}/mito_damage",
            "mTOR_activity": f"{p}/mTOR_activity",
            "p53_activity": f"{p}/p53_activity",
            "ROS_activity": f"{p}/ROS_activity",
            "ROS_integrator_c": f"{p}/ROS_integrator_c",
        },
        "oxidative_stress": {
            "mito_damage": f"{p}/mito_damage",
            "ROS_integrator_c": f"{p}/ROS_integrator_c",
            "ROS_activity": f"{p}/ROS_activity",
            "mito_function": f"{p}/mito_function",
            "glycolysis": f"{p}/glycolysis",
            "mTOR_activity": f"{p}/mTOR_activity",
            "p53_activity": f"{p}/p53_activity",
        },
        "signaling": {
            "mTOR_integrator_c": f"{p}/mTOR_integrator_c",
            "mTOR_activity": f"{p}/mTOR_activity",
            "p53_integrator_c": f"{p}/p53_integrator_c",
            "p53_activity": f"{p}/p53_activity",
            "mito_function": f"{p}/mito_function",
            "glycolysis": f"{p}/glycolysis",
            "ROS_activity": f"{p}/ROS_activity",
            "ROS_integrator_c": f"{p}/ROS_integrator_c",
            "mito_damage": f"{p}/mito_damage",
        },
    }

    return Composite(
        processes,
        topology,
        validate=validate,
        semantic_validation=semantic_validation,
    )


# ── Original-equation variants (for ModelAnalyzer demonstration) ───────


class _OriginalERiQEnergyMetabolism(ERiQEnergyMetabolism):
    """ERiQEnergyMetabolism using original (pre-revision) equations."""

    def derivative(self, t, state):
        obs = _compute_algebraic_original(state)

        menzy = state["mito_enzymes"]
        glyenz = state["glycolytic_enzymes"]
        mdamage = state["mito_damage"]

        r2 = (
            obs["PGC1a"]
            + obs["PYR"]
            + obs["P53"]
            - 0.2 * obs["HIF"]
            - 0.2 * obs["NFKB"]
        )
        gain2 = 0.05
        u2 = r2 + menzy
        dMito_function = gain2 * u2 - obs["SIRT"] * 0.02

        k3 = 1.0
        k4 = r2 - mdamage
        dMito_enzymes = -(k3 * obs["ATPm"]) - (k4 * menzy)

        r3 = obs["GLU"] + 0.01 * obs["HIF"] + 0.01 * obs["NADr"]
        gain3 = 0.25
        u3 = r3 + glyenz
        # Original: 1/SIRT singularity
        dGlycolysis = self.GLYCOL_SA * (
            gain3 * u3 + obs["SIRT_gly_inhibition"]
        )

        k5 = -1.0
        k6 = r3
        dGlycolytic_enzymes = k5 * obs["ATPg"] - k6 * glyenz

        return {
            "mito_function": dMito_function,
            "mito_enzymes": dMito_enzymes,
            "glycolysis": dGlycolysis,
            "glycolytic_enzymes": dGlycolytic_enzymes,
        }


class _OriginalERiQOxidativeStress(ERiQOxidativeStress):
    """ERiQOxidativeStress using original (pre-revision) equations."""

    def derivative(self, t, state):
        obs = _compute_algebraic_original(state)
        ros_int = state["ROS_integrator_c"]
        dMito_damage = self.MDAMAGE_SA * (obs["MD"] - obs["AUTOPHAGY"])
        dROS_integrator_c = -obs["ROS"] - ros_int
        dROS_activity = 0.01 * obs["Uz"]
        return {
            "mito_damage": dMito_damage,
            "ROS_integrator_c": dROS_integrator_c,
            "ROS_activity": dROS_activity,
        }


class _OriginalERiQSignaling(ERiQSignaling):
    """ERiQSignaling using original (pre-revision) equations."""

    def derivative(self, t, state):
        obs = _compute_algebraic_original(state)
        mtor_int = state["mTOR_integrator_c"]
        p53_int = state["p53_integrator_c"]
        gy = 0.1
        Uy = mtor_int
        dMTOR_integrator_c = -obs["MTORa"] - mtor_int
        dMTOR_activity = gy * Uy
        gx = 0.1
        Ux = p53_int
        dp53_integrator_c = -obs["P53a"] - p53_int
        dp53_activity = gx * Ux
        return {
            "mTOR_integrator_c": dMTOR_integrator_c,
            "mTOR_activity": dMTOR_activity,
            "p53_integrator_c": dp53_integrator_c,
            "p53_activity": dp53_activity,
        }


def build_eriq_composite_original(
    *,
    prefix: str = ERIQ_PREFIX,
):
    """Build ERiQ composite with original (pre-revision) equations.

    This uses the raw reciprocal equations from Alfego & Kriete (2017)
    that have numerical singularities.  Useful for:

    - Demonstrating the ModelAnalyzer catching issues
    - Comparing original vs revised dynamics
    - Reproducing the original paper's results (requires implicit solver)
    """
    from hallsim.composite import Composite

    processes = {
        "energy": _OriginalERiQEnergyMetabolism(),
        "oxidative_stress": _OriginalERiQOxidativeStress(),
        "signaling": _OriginalERiQSignaling(),
    }

    p = prefix
    topology = {
        "energy": {
            "mito_function": f"{p}/mito_function",
            "mito_enzymes": f"{p}/mito_enzymes",
            "glycolysis": f"{p}/glycolysis",
            "glycolytic_enzymes": f"{p}/glycolytic_enzymes",
            "mito_damage": f"{p}/mito_damage",
            "mTOR_activity": f"{p}/mTOR_activity",
            "p53_activity": f"{p}/p53_activity",
            "ROS_activity": f"{p}/ROS_activity",
            "ROS_integrator_c": f"{p}/ROS_integrator_c",
        },
        "oxidative_stress": {
            "mito_damage": f"{p}/mito_damage",
            "ROS_integrator_c": f"{p}/ROS_integrator_c",
            "ROS_activity": f"{p}/ROS_activity",
            "mito_function": f"{p}/mito_function",
            "glycolysis": f"{p}/glycolysis",
            "mTOR_activity": f"{p}/mTOR_activity",
            "p53_activity": f"{p}/p53_activity",
        },
        "signaling": {
            "mTOR_integrator_c": f"{p}/mTOR_integrator_c",
            "mTOR_activity": f"{p}/mTOR_activity",
            "p53_integrator_c": f"{p}/p53_integrator_c",
            "p53_activity": f"{p}/p53_activity",
            "mito_function": f"{p}/mito_function",
            "glycolysis": f"{p}/glycolysis",
            "ROS_activity": f"{p}/ROS_activity",
            "ROS_integrator_c": f"{p}/ROS_integrator_c",
            "mito_damage": f"{p}/mito_damage",
        },
    }

    return Composite(processes, topology, validate=False)
