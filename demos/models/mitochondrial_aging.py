"""Mitochondrial decline with age — mtDNA genetics, bioenergetics, quality control.

A composite spanning the mitochondrial arm of ageing on one day-resolved clock,
built from three independently published models plus the mechanisms none of
them carry.

Reused, imported from SBML
--------------------------
``dp14`` — **DallePezze et al. 2014**, *Dynamic modelling of pathways to
cellular senescence reveals strategies for targeted interventions*, PLoS Comput
Biol 10(8):e1003728 (BIOMD0000000582). Native clock: days. Supplies the
senescence programme and the organelle-level mitochondrial pools: new/old
mitochondrial mass and membrane potential, mitophagy, ROS, DNA damage,
AMPK/Akt/mTORC1/FoxO3a/JNK, CDKN1A, SA-β-gal. This is the anchor and the
coupling to Cellular Senescence, Deregulated Nutrient Sensing and Genomic
Instability.

``bioe`` — **Nazaret et al. 2009**, *Mitochondrial energetic metabolism: a
simplified model of TCA cycle with ATP production*, J Theor Biol 258:455–464
(BIOMD0000000232). Native clock: seconds. Supplies the inner-membrane potential
as a biophysical state (mV), matrix ATP/ADP, NAD⁺/NADH, respiratory flux and
proton leak. It enters the day-axis composite through its **quasi-steady
state** (:class:`MitoBioenergetics`), because on a day clock the bioenergetics
are adiabatic — and because reconciled onto that clock its stiffness index is
~1e9 and the implicit solver does not finish. The reduction is derived, fitted
with ``Calibrator`` and validated held-out by
``demos/mito_bioenergetics_reduction.py``.

``redox`` — **Kowald, Lehrach & Klipp 2006**, *Alternative pathways as a
mechanism for the negative effects associated with overexpression of superoxide
dismutase*, J Theor Biol 238:828–840 (BIOMD0000000108). Native clock: seconds
(``k = 1.6e9`` M⁻¹s⁻¹ for the SOD reaction fixes it). O₂•⁻ / Cu,ZnSOD / H₂O₂ /
catalase / lipid-peroxidation chain. Same adiabatic argument: it is screened
and reported on its native clock and sets the *form* of :class:`MitoRedox`'s
scavenging term, but its microsecond radical chemistry does not enter the
day-axis ODE.

Written here
------------
:class:`MtDNAPopulation` — Kowald & Kirkwood 2014 (PNAS 111:2972–2977) Eqs 1–3
for competing wild-type and deletion-mutant mtDNA under ATP-controlled
replication priming, with two documented extensions: replication-error mutant
generation (Elson et al. 2001) and selective mitophagy of mutants (Twig et al.
2008), both zero-by-default so the published model is recovered exactly.

:class:`MtDNAThreshold` — heteroplasmy and the phenotypic-threshold map from
mutant load to respiratory competence (Boulet 1992; Chomyn 1992; Rocha 2018).

:class:`MitoBioenergetics` — the Nazaret 2009 quasi-steady state as algebraic
ASSIGNED outputs: Δψm, matrix ATP, NAD⁺ fraction and the NAD⁺/NADH ratio, as
functions of respiratory competence and proton-leak conductance.

:class:`MitoRedox` — mitochondrial superoxide/H₂O₂ production as a function of
Δψm (Korshunov 1997) and respiratory-chain block (Murphy 2009), against an
inducible SOD2/catalase capacity (Kops 2002; Nemoto & Finkel 2002).

:class:`MitoQualityControl` — PINK1 stabilisation on depolarised membrane and
Parkin-dependent selective mitophagy (Narendra 2010; Jin 2010), PGC-1α
biogenesis under AMPK and NAD⁺/sirtuin control and mTORC1 inhibition (Jäger
2007; Cantó 2009; Kim 2011), with fusion state gating how selective mitophagy
can be (Twig 2008; Figge 2012).

The vicious cycle
-----------------
The self-amplifying ROS → mtDNA-mutation → more-ROS loop is **not** the default
driver here. mtDNA mutator mice carry 5–10× the point-mutation load with no
rise in H₂O₂, superoxide, aconitase inactivation or protein carbonyls
(Trifunovic 2005; Kujoth 2005; Hiona 2010), mutations accumulate linearly
rather than exponentially, and the aged mutation spectrum shows no rise in the
G→T signature of 8-oxo-guanine (Ameur 2011; Kennedy 2013). ``mu_ros`` is
therefore 0 by default and mutants arise from replication error; the channel
exists so the hypothesis can be *switched on and tested* (``simulate mito-aging
vicious-cycle``) rather than assumed.

>>> comp = build_mitochondrial_aging_composite()
>>> res = Scheduler().run(comp, t_span=(0.0, 30.0), macro_dt=1.0)
"""

from __future__ import annotations

import equinox as eqx
import jax.numpy as jnp

from hallsim.kinetics import hill_gate, hill_inhibition
from demos.models.sbml import sbml_source
from hallsim.process import Port, PortRole, Process, calibratable

# ── Constituent sources ─────────────────────────────────────────────────

SENESCENCE_SBML_PATH = sbml_source(
    "dallepezze2014",
    "dallepezze2014_BIOMD0000000582.xml",
    "BIOMD0000000582",
)
# The deposited BIOMD0000000232 leaves its rule-target parameters and its two
# assignment-rule boundary species without values, which sbmltoodejax cannot
# evaluate at t0. The sibling `_initialised` file supplies exactly those nine
# attributes; see the directory README for the diff.
BIOENERGETICS_SBML_PATH = sbml_source(
    "nazaret2009",
    "nazaret2009_BIOMD0000000232_initialised.xml",
    "BIOMD0000000232",
)
REDOX_SBML_PATH = sbml_source(
    "kowald2006", "kowald2006_BIOMD0000000108.xml", "BIOMD0000000108"
)

# One t_span unit = one day: DallePezze 2014's native clock, and the sampling
# interval of the senescence time courses this composite is scored against.
CANONICAL_TIME_SECONDS = 86400.0

# Kowald 2006 ships unnamed species ids; these are the `name` attributes.
REDOX_SUPEROXIDE = "species_0000001"
REDOX_SOD = "species_0000002"
REDOX_H2O2 = "species_0000006"
REDOX_LIPID_PEROXYL = "species_0000007"
REDOX_HYDROXYL = "species_0000008"
REDOX_LIPID_HYDROPEROXIDE = "species_0000009"
REDOX_LIPID_RADICAL = "species_0000011"
REDOX_SUPEROXIDE_SOURCE = "k1"  # zero-order O2•− production, 6.6e-7 M/s

# Nazaret 2009 published values (Table 1 / SBML listOfParameters).
BIOE_KRESP_NOMINAL = 2.5
BIOE_KLEAK_NOMINAL = 0.000426
BIOE_NAD_TOTAL = 1.07  # mM, NAD+ + NADH
BIOE_ATP_TOTAL = 4.16  # mM, ATP + ADP
BIOE_DPSI_REFERENCE = 150.0  # mV, ΔΨm at the published operating point

# DallePezze 2014 species this composite reads or writes.
DP14_ROS = "dp14/ROS"
DP14_MASS_NEW = "dp14/Mito_mass_new"
DP14_MASS_OLD = "dp14/Mito_mass_old"
DP14_MITOPHAGY = "dp14/Mitophagy"
DP14_POT_NEW = "dp14/Mito_membr_pot_new"
DP14_POT_OLD = "dp14/Mito_membr_pot_old"
DP14_AMPK = "dp14/AMPK_pT172"
DP14_MTOR = "dp14/mTORC1_pS2448"
DP14_FOXO3A = "dp14/FoxO3a"
DP14_SA_BETA_GAL = "dp14/SA_beta_gal"
DP14_IRRADIATION_RATE_NAME = "DNA_damaged_by_irradiation"
DP14_IRRADIATION_INPUT_NAME = "Irradiation"

# Kowald & Kirkwood 2014 Table 1, converted from h⁻¹ to d⁻¹ (×24). Their `c`
# is dimensionless in that conversion: it is fixed by requiring the
# mutant-free steady state to sit at 1000 wild-type copies, and every rate in
# that balance scales by the same factor.
MTDNA_S_WT_PER_DAY = 24.0  # s_wt = 1 h⁻¹
MTDNA_MUT_ADVANTAGE = 1.5  # s_mt / s_wt
MTDNA_HALF_LIFE_DAYS = 10.0  # halfL = 240 h
MTDNA_ATP_HALF_SAT = 1.3034  # c, set so W* = 1000 with no mutant
MTDNA_ATP_PER_WT = 2.4  # f = 0.1 h⁻¹
MTDNA_ATP_TURNOVER = 4.8  # v1 = 0.2 h⁻¹
MTDNA_ATP_UPKEEP = 0.24  # v2 = 0.01 h⁻¹
MTDNA_WT_INITIAL = 1000.0
MTDNA_ATP_INITIAL = 450.0  # the mutant-free fixed point of Eq 3

# Surrogate coefficients for the Nazaret 2009 quasi-steady state, fitted to its
# exact fixed points by demos/mito_bioenergetics_reduction.py (held-out split
# over respiratory competence). Re-run that demo to regenerate them.
BIOE_SURROGATE_DEFAULTS: dict[str, float] = {
    "dpsi_max": 155.022,
    "dpsi_min": 100.144,
    "dpsi_K": 0.0588236,
    "dpsi_n": 0.919706,
    "dpsi_leak": 0.0139378,
    "atp_max": 3.49111,
    "atp_min": 0.588842,
    "atp_K": 0.0959981,
    "atp_n": 1.2794,
    "atp_leak": 0.0381393,
    "nad_max": 0.886059,
    "nad_min": 0.0546932,
    "nad_K": 0.150834,
    "nad_n": 1.76494,
    "nad_leak": 0.001,
}


def bioenergetic_surrogate(resp, leak, params):
    """``(n, 3)`` array of ``(ΔΨm, matrix ATP, NAD⁺ fraction)``.

    Each observable is a Hill in respiratory competence ``resp`` (1 = intact
    chain) times a linear proton-leak penalty in ``leak`` (1 = published
    conductance)::

        y(resp, leak) = [y_min + (y_max − y_min)·resp^n/(K^n + resp^n)]
                        · (1 − y_leak·(leak − 1))

    Shared by the fit (``demos/mito_bioenergetics_reduction.py``) and by
    :class:`MitoBioenergetics`, so the composite evaluates exactly what was
    fitted.
    """
    resp = jnp.atleast_1d(jnp.asarray(resp))
    leak = jnp.atleast_1d(jnp.asarray(leak))
    out = []
    for tag in ("dpsi", "atp", "nad"):
        lo = params[f"{tag}_min"]
        hi = params[f"{tag}_max"]
        gate = hill_gate(resp, params[f"{tag}_K"], params[f"{tag}_n"])
        penalty = 1.0 - params[f"{tag}_leak"] * (leak - 1.0)
        out.append((lo + (hi - lo) * gate) * jnp.maximum(penalty, 0.0))
    return jnp.stack(out, axis=-1)


# ── mtDNA population genetics ───────────────────────────────────────────


class MtDNAPopulation(Process):
    """Competing wild-type and deletion-mutant mtDNA under ATP-controlled
    replication priming — Kowald & Kirkwood 2014, PNAS 111:2972–2977, Eqs 1–3.

    In their model replication is primed by processing an mtDNA transcript, and
    product inhibition down-regulates transcription once enough respiratory
    subunits exist. A deletion that removes a feedback gene (ND4/ND5) escapes
    that inhibition and primes replication faster, which is the selection
    advantage that drives clonal expansion::

        d(W)/dt = s_wt · B · W · c/(c + A) − λ·W − m·s_wt·B·W·c/(c + A)
        d(M)/dt = s_mt · B · M · c/(c + A) − λ·(1 + σ)·M
                  + m·s_wt·B·W·c/(c + A)
        d(A)/dt = f·R·W − v₂·(W + M) − v₁·A

    with ``λ = ln2 / half_life``. Rates are the published hourly values ×24
    (day clock). ``A`` is Kowald & Kirkwood's cell-level ATP variable — the
    energy signal that gates replication — and is deliberately *not* merged
    with Nazaret 2009's matrix ATP pool, which is a different quantity in mM.

    Three inputs extend the published equations, all neutral at their defaults
    so ``B = R = 1``, ``σ = m = 0`` reproduces it exactly:

    ``biogenesis`` (``B``)
        PGC-1α-driven replication drive (Jäger 2007; Cantó 2009).
    ``resp_competence`` (``R``)
        the fraction of wild-type ATP output the chain can still deliver, from
        :class:`MtDNAThreshold`.
    ``mitophagy_selectivity`` (``σ``)
        extra removal of mutant genomes by Parkin-dependent mitophagy — Kowald
        & Kirkwood assume equal half-lives, but dysfunctional mitochondria are
        degraded *preferentially* (Twig 2008; Narendra 2010), which is the
        counter-force to clonal expansion this composite exists to weigh.

    Mutants are generated at ``mutation_rate`` per wild-type replication
    (Elson et al. 2001, Am J Hum Genet 68:802–806, estimate 1e-5–5e-5), plus an
    optional ``ros_mutagenesis`` term per unit relative ROS. That second term is
    the mitochondrial-free-radical vicious cycle and is **0 by default**: it is
    not supported by the mutator-mouse or mutation-spectrum evidence
    (Trifunovic 2005; Ameur 2011; Kennedy 2013), and is exposed so the model can
    test it rather than assume it.
    """

    hallmark = "Mitochondrial Dysfunction"
    reference = "Kowald & Kirkwood 2014, PNAS 111:2972-2977"
    description = (
        "Wild-type vs deletion-mutant mtDNA competition under "
        "transcription-primed, ATP-controlled replication."
    )

    timescale: float | None = 3600.0

    s_wt: float = MTDNA_S_WT_PER_DAY
    mut_advantage: float = calibratable(
        MTDNA_MUT_ADVANTAGE,
        clamp=(1.0, 4.0),
        description="s_mt/s_wt — replication-priming advantage of a "
        "feedback-gene deletion (Kowald & Kirkwood 2014 scan 1.1–2.0).",
    )
    half_life: float = MTDNA_HALF_LIFE_DAYS
    atp_half_sat: float = MTDNA_ATP_HALF_SAT
    atp_per_wt: float = MTDNA_ATP_PER_WT
    atp_turnover: float = MTDNA_ATP_TURNOVER
    atp_upkeep: float = MTDNA_ATP_UPKEEP
    mutation_rate: float = calibratable(
        2.0e-5,
        clamp=(1e-6, 1e-3),
        description="deletion probability per wild-type replication "
        "(Elson 2001: 1e-5 to 5e-5).",
    )
    ros_mutagenesis: float = 0.0

    def ports_schema(self):
        return {
            "wt": Port(
                role=PortRole.EVOLVED,
                default=MTDNA_WT_INITIAL,
                units="count",
                description="wild-type mtDNA copies per cell",
                ontology={"so": "SO:0001032"},
            ),
            "mut": Port(
                role=PortRole.EVOLVED,
                default=1.0,
                units="count",
                description="deletion-mutant mtDNA copies per cell",
                ontology={"so": "SO:0000159"},
            ),
            "atp": Port(
                role=PortRole.EVOLVED,
                default=MTDNA_ATP_INITIAL,
                units="dimensionless",
                description="cell-level ATP signal gating mtDNA replication "
                "(Kowald & Kirkwood 2014 Eq 3); not the matrix ATP pool",
            ),
            "resp_competence": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="dimensionless",
                description="fraction of wild-type respiratory output",
            ),
            "biogenesis": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="dimensionless",
                description="PGC-1α replication drive, 1 = basal",
            ),
            "mitophagy_selectivity": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="extra fractional removal rate of mutant genomes",
            ),
            "ros": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="dimensionless",
                description="mitochondrial ROS relative to basal",
            ),
        }

    def derivative(self, t, state):
        wt = jnp.maximum(state["wt"], 0.0)
        mut = jnp.maximum(state["mut"], 0.0)
        atp = state["atp"]
        priming = self.atp_half_sat / (self.atp_half_sat + jnp.abs(atp))
        drive = self.s_wt * state["biogenesis"] * priming
        rep_wt = drive * wt
        rep_mut = drive * self.mut_advantage * mut
        decay = jnp.log(2.0) / self.half_life
        new_mut = (
            self.mutation_rate + self.ros_mutagenesis * state["ros"]
        ) * rep_wt
        return {
            "wt": rep_wt - new_mut - decay * wt,
            "mut": rep_mut
            + new_mut
            - decay * (1.0 + state["mitophagy_selectivity"]) * mut,
            "atp": self.atp_per_wt * state["resp_competence"] * wt
            - self.atp_upkeep * (wt + mut)
            - self.atp_turnover * atp,
        }


class RespiratoryCompetence(Process):
    """Heteroplasmy, the phenotypic threshold, and the respiratory competence
    the rest of the composite reads.

    Two independent things cost a cell respiratory capacity, and the model
    needs both because they dominate in different regimes.

    *Genetic.* Respiratory output is buffered against mutant mtDNA until a
    sharp threshold: ~15% residual wild-type restores near-normal COX activity
    for MERRF (Boulet et al. 1992, Am J Hum Genet 51:1187), ~6% for MELAS
    (Chomyn et al. 1992, PNAS 89:4221), and single human muscle fibres carrying
    large deletions lose complex I at 65–80% and complex IV at 72–91% mutant
    load (Rocha et al. 2018, Ann Neurol 83:115).

    *Organellar.* Senescent cells lose complex activity without any change in
    mtDNA: aged human dermal fibroblasts show 40–50% lower complex I/II/IV/V
    with a bulk deletion load that never leaves ~0.3% (Mapuskar et al. 2017,
    Cancer Res 77:5054; Gerhard et al. 2002, Mech Ageing Dev 123:155). That is
    DallePezze's dysfunctional mass pool, which retains a fraction ``φ`` of the
    functional pool's per-unit capacity::

        h = M / (W + M)
        R_gen  = 1 − h^n / (K^n + h^n)
        R_mass = (m_new + φ·m_old) / (m_new + m_old)
        R = R_gen · R_mass

    Both channels are multiplicative because they are independent losses. In a
    dividing fibroblast ``R_gen ≈ 1`` and ``R_mass`` carries everything; in a
    post-mitotic cell over years the reverse.
    """

    hallmark = "Mitochondrial Dysfunction"
    reference = "Boulet 1992; Chomyn 1992; Rocha 2018; Mapuskar 2017"
    description = (
        "Respiratory competence from the mtDNA phenotypic threshold and the "
        "functional/dysfunctional organelle split."
    )

    timescale: float | None = 3600.0
    threshold: float = calibratable(
        0.8,
        clamp=(0.4, 0.98),
        description="mutant fraction at half-maximal respiratory loss.",
    )
    steepness: float = 8.0
    dysfunctional_capacity: float = calibratable(
        0.5,
        clamp=(0.05, 1.0),
        description="respiratory capacity per unit dysfunctional mass, "
        "relative to functional; 40-50% complex loss in aged fibroblasts "
        "(Mapuskar 2017).",
    )

    def ports_schema(self):
        return {
            "heteroplasmy": Port(
                role=PortRole.ASSIGNED,
                default=0.0,
                units="dimensionless",
                description="mutant fraction of total mtDNA",
            ),
            "resp_competence": Port(
                role=PortRole.ASSIGNED,
                default=1.0,
                units="dimensionless",
                description="fraction of wild-type respiratory output",
            ),
            "wt": Port(
                role=PortRole.INPUT,
                default=MTDNA_WT_INITIAL,
                units="count",
                description="wild-type mtDNA copies per cell",
            ),
            "mut": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="count",
                description="deletion-mutant mtDNA copies per cell",
            ),
            "mass_new": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="dimensionless",
                description="DallePezze functional mitochondrial mass",
            ),
            "mass_old": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="DallePezze dysfunctional mitochondrial mass",
            ),
        }

    def assign(self, t, state):
        wt = jnp.maximum(state["wt"], 0.0)
        mut = jnp.maximum(state["mut"], 0.0)
        h = mut / (wt + mut + 1e-9)
        m_new = jnp.maximum(state["mass_new"], 0.0)
        m_old = jnp.maximum(state["mass_old"], 0.0)
        by_mass = (m_new + self.dysfunctional_capacity * m_old) / (
            m_new + m_old + 1e-9
        )
        return {
            "heteroplasmy": h,
            "resp_competence": hill_inhibition(
                h, self.threshold, self.steepness
            )
            * by_mass,
        }


# ── Bioenergetics (adiabatic reduction of Nazaret 2009) ─────────────────


class MitoBioenergetics(Process):
    """Δψm, matrix ATP and NAD⁺/NADH as the quasi-steady state of Nazaret 2009.

    On a day clock the TCA / respiratory-chain / ATP-synthase system is
    adiabatic: it relaxes in seconds, so its slow-manifold value — not its
    transient — is what the senescence programme sees. These four outputs are
    ASSIGNED, evaluated from :func:`bioenergetic_surrogate` on respiratory
    competence and relative proton-leak conductance, with coefficients fitted
    to the published model's exact fixed points (``Calibrator``, held-out over
    competence; ``demos/mito_bioenergetics_reduction.py``).

    Proton leak is an input, not a constant: oligomycin-resistant respiration
    roughly doubles in irradiation-induced senescent fibroblasts (Passos et al.
    2010, Mol Syst Biol 6:347, Fig 1C), and leak is what converts a maintained
    respiratory rate into a lower Δψm and a lower ATP yield.
    """

    hallmark = "Mitochondrial Dysfunction"
    reference = "Nazaret et al. 2009, J Theor Biol 258:455-464"
    description = (
        "Quasi-steady-state inner-membrane potential, matrix ATP and "
        "NAD+/NADH from respiratory competence and proton leak."
    )

    timescale: float | None = 3600.0
    coefficients: dict = eqx.field(
        default_factory=lambda: dict(BIOE_SURROGATE_DEFAULTS)
    )

    def ports_schema(self):
        return {
            "dpsi": Port(
                role=PortRole.ASSIGNED,
                default=BIOE_DPSI_REFERENCE,
                units="mV",
                description="inner-membrane potential",
                ontology={"go": "GO:0051881"},
            ),
            "atp_matrix": Port(
                role=PortRole.ASSIGNED,
                default=3.35,
                units="mM",
                description="matrix ATP concentration",
                ontology={"chebi": "CHEBI:15422"},
            ),
            "nad_fraction": Port(
                role=PortRole.ASSIGNED,
                default=0.85,
                units="dimensionless",
                description="NAD+ / (NAD+ + NADH) in the matrix",
            ),
            "nad_redox": Port(
                role=PortRole.ASSIGNED,
                default=5.5,
                units="dimensionless",
                description="matrix NAD+/NADH ratio",
            ),
            "resp_competence": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="dimensionless",
                description="fraction of wild-type respiratory output",
            ),
            "leak": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="dimensionless",
                description="proton-leak conductance relative to healthy",
            ),
        }

    def assign(self, t, state):
        vals = bioenergetic_surrogate(
            state["resp_competence"], state["leak"], self.coefficients
        )
        dpsi, atp, nad = vals[..., 0], vals[..., 1], vals[..., 2]
        nad = jnp.clip(nad, 1e-3, 1.0 - 1e-3)
        return {
            "dpsi": jnp.squeeze(dpsi),
            "atp_matrix": jnp.squeeze(atp),
            "nad_fraction": jnp.squeeze(nad),
            "nad_redox": jnp.squeeze(nad / (1.0 - nad)),
        }


# ── ROS production and antioxidant defence ──────────────────────────────


class MitoRedox(Process):
    """Mitochondrial ROS against an inducible antioxidant capacity.

    Two production channels, because the two organelle pools are in opposite
    regimes. A **polarised, functional** mitochondrion makes superoxide as a
    steep function of the protonmotive force — a ~10 mV fall roughly halves
    H₂O₂ output in isolated mitochondria (Korshunov, Skulachev & Starkov 1997,
    FEBS Lett 416:15) — and makes more again when electron flow is blocked
    (Murphy 2009, Biochem J 417:1). A **dysfunctional** one is depolarised, so
    the protonmotive term would predict it is quiet; in fact it is the main
    source, at a yield ``ρ`` per unit mass. Both are per unit mass, because
    senescent fibroblasts raise ROS *and* mass and the measured increase
    survives normalisation (Passos et al. 2007, PLoS Biol 5:e110, Fig 1C:
    MitoSOX +863%, DHR +545%)::

        d(ROS)/dt = k_prod·[ m_new·e^{γ(ΔΨ−ΔΨ₀)/ΔΨ₀}·(1 + κ(1−R)) + ρ·m_old ]
                    − k_scav · antiox · ROS
        d(antiox)/dt = k_basal + k_foxo · h(FoxO3a) − k_deg · antiox

    ``ΔΨ`` here is the *functional* pool's intrinsic potential from
    :class:`MitoBioenergetics`, not the whole-cell mean — applying a
    protonmotive law to a population average that the dysfunctional pool
    dominates gets the sign of the senescent ROS rise backwards.

    The scavenging term is first order in capacity because Kowald 2006's
    explicit chemistry is dominated by the SOD and catalase reactions, both
    first order in enzyme at the H₂O₂ levels reached there; ``antiox`` lumps
    SOD2 and catalase, which are FoxO3a transcriptional targets (Kops et al.
    2002, Nature 419:316; Nemoto & Finkel 2002, Science 295:2450). ROS is in
    units of the healthy steady state, so ``antiox = ROS = 1`` at rest.
    """

    hallmark = "Mitochondrial Dysfunction"
    reference = "Korshunov 1997; Murphy 2009; Kowald 2006 (BIOMD0000000108)"
    description = (
        "Δψm- and block-dependent mitochondrial ROS production against "
        "FoxO3a-inducible SOD2/catalase capacity."
    )

    timescale: float | None = 3600.0

    k_prod: float = calibratable(
        1.0,
        clamp=(0.05, 20.0),
        description="basal ROS production per unit mitochondrial mass per day.",
    )
    k_scav: float = calibratable(
        1.0,
        clamp=(0.05, 20.0),
        description="scavenging rate per unit antioxidant capacity per day.",
    )
    dpsi_gain: float = 6.0
    block_gain: float = calibratable(
        3.0,
        clamp=(0.0, 20.0),
        description="fold rise in superoxide at complete respiratory block.",
    )
    dysfunctional_yield: float = calibratable(
        0.6,
        clamp=(0.0, 5.0),
        description="ROS produced per unit dysfunctional mass, relative to a "
        "polarised functional unit; anchored to Passos 2007 (MitoSOX +863%, "
        "DHR +545% in senescent MRC5).",
    )
    antiox_basal: float = 1.0
    antiox_foxo: float = calibratable(
        1.0,
        clamp=(0.0, 10.0),
        description="FoxO3a-driven SOD2/catalase induction gain.",
    )
    antiox_decay: float = 2.0
    foxo_half: float = 8.0
    mass_reference: float = 1.0

    def ports_schema(self):
        return {
            "ros": Port(
                role=PortRole.EVOLVED,
                default=1.0,
                units="dimensionless",
                description="mitochondrial ROS, healthy steady state = 1",
                ontology={"chebi": "CHEBI:26523"},
            ),
            "antiox": Port(
                role=PortRole.EVOLVED,
                default=1.0,
                units="dimensionless",
                description="SOD2 + catalase capacity, healthy = 1",
                ontology={"go": "GO:0016209"},
            ),
            "dpsi": Port(
                role=PortRole.INPUT,
                default=BIOE_DPSI_REFERENCE,
                units="mV",
                description="inner-membrane potential",
            ),
            "resp_competence": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="dimensionless",
                description="fraction of wild-type respiratory output",
            ),
            "mass_new": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="dimensionless",
                description="DallePezze functional mitochondrial mass",
            ),
            "mass_old": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="DallePezze dysfunctional mitochondrial mass",
            ),
            "foxo": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="active FoxO3a driving antioxidant transcription",
            ),
        }

    def derivative(self, t, state):
        pmf = jnp.exp(
            self.dpsi_gain
            * (state["dpsi"] - BIOE_DPSI_REFERENCE)
            / BIOE_DPSI_REFERENCE
        )
        block = 1.0 + self.block_gain * (1.0 - state["resp_competence"])
        functional = jnp.maximum(state["mass_new"], 0.0) * pmf * block
        dysfunctional = self.dysfunctional_yield * jnp.maximum(
            state["mass_old"], 0.0
        )
        antiox = jnp.maximum(state["antiox"], 0.0)
        return {
            "ros": self.k_prod
            * (functional + dysfunctional)
            / self.mass_reference
            - self.k_scav * antiox * jnp.maximum(state["ros"], 0.0),
            "antiox": self.antiox_basal
            + self.antiox_foxo * hill_gate(state["foxo"], self.foxo_half, 2.0)
            - self.antiox_decay * antiox,
        }


# ── Quality control: PINK1-Parkin mitophagy and PGC-1α biogenesis ───────


class MitoQualityControl(Process):
    """PINK1–Parkin selective mitophagy and PGC-1α biogenesis.

    PINK1 is imported and cleaved in polarised mitochondria and is degraded as
    fast as it arrives; loss of Δψm blocks import, PINK1 accumulates on the
    outer membrane and recruits Parkin, tagging that organelle for autophagy
    (Narendra et al. 2010, PLoS Biol 8:e1000298; Jin et al. 2010, J Cell Biol
    191:933). Selectivity requires prior fission: a damaged unit must be split
    off before it can be removed, and fusion instead dilutes the defect through
    the network (Twig et al. 2008, EMBO J 27:433; Figge et al. 2012, PLoS
    Comput Biol 8:e1002576). Senescent mitochondria are hyperfused — ~8-fold
    longer in senescent IMR90 (Kim et al. 2023, Life Sci Alliance 6:e202302127)
    with Drp1 and Fis1 strongly down (Mai et al. 2010, J Cell Sci 123:917) —
    so ``fused_fraction`` is the mechanism by which the ageing network loses
    the ability to *select*::

        d(PINK1)/dt = k_on · [1 − h(ΔΨ; K, n)] − k_off · PINK1
        d(PGC1α)/dt = (a·h(AMPK) + s·h(NAD⁺/NADH)) · [1 − h(mTORC1)]
                      − k_deg · PGC1α

        selectivity = k_sel · PINK1 · (1 − fused_fraction) · mitophagy_flux
        biogenesis  = 1 + k_bio · PGC1α

    PGC-1α integrates AMPK phosphorylation and SIRT1 deacetylation, the latter
    NAD⁺-dependent (Jäger et al. 2007, PNAS 104:12017; Cantó et al. 2009,
    Nature 458:1056), and is opposed by mTORC1, which also blocks autophagy
    initiation through ULK1 (Kim et al. 2011, Nat Cell Biol 13:132).
    """

    hallmark = "Mitochondrial Dysfunction"
    reference = "Narendra 2010; Jin 2010; Twig 2008; Jager 2007; Canto 2009"
    description = (
        "Depolarisation-gated PINK1/Parkin mitophagy selectivity and "
        "AMPK/NAD+-driven, mTORC1-opposed PGC-1α biogenesis."
    )

    timescale: float | None = 3600.0

    pink1_on: float = 4.0
    pink1_off: float = 4.0
    dpsi_half: float = 135.0
    dpsi_steepness: float = 8.0
    selectivity_gain: float = calibratable(
        0.5,
        clamp=(0.0, 20.0),
        description="mutant-genome removal per unit PINK1 x mitophagy flux.",
    )
    fused_fraction: float = 0.3
    mitophagy_reference: float = 10.0

    pgc1a_ampk: float = 1.0
    pgc1a_sirt: float = calibratable(
        1.0,
        clamp=(0.0, 10.0),
        description="NAD+/sirtuin drive on PGC-1α.",
    )
    pgc1a_decay: float = 2.0
    ampk_half: float = 8.0
    nad_redox_half: float = 4.0
    mtor_half: float = 10.0
    biogenesis_gain: float = calibratable(
        0.5,
        clamp=(0.0, 10.0),
        description="mtDNA replication drive per unit PGC-1α.",
    )

    def ports_schema(self):
        return {
            "pink1": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="dimensionless",
                description="PINK1 stabilised on depolarised mitochondria",
                ontology={"uniprot": "Q9BXM7"},
            ),
            "pgc1a": Port(
                role=PortRole.EVOLVED,
                default=0.5,
                units="dimensionless",
                description="active PGC-1α",
                ontology={"uniprot": "Q9UBK2"},
            ),
            "mitophagy_selectivity": Port(
                role=PortRole.ASSIGNED,
                default=0.0,
                units="dimensionless",
                description="extra fractional removal rate of mutant genomes",
            ),
            "biogenesis": Port(
                role=PortRole.ASSIGNED,
                default=1.0,
                units="dimensionless",
                description="mtDNA replication drive, 1 = basal",
            ),
            "dpsi": Port(
                role=PortRole.INPUT,
                default=BIOE_DPSI_REFERENCE,
                units="mV",
                description="inner-membrane potential",
            ),
            "mitophagy_flux": Port(
                role=PortRole.INPUT,
                default=10.0,
                units="dimensionless",
                description="DallePezze bulk mitophagy activity",
            ),
            "ampk": Port(
                role=PortRole.INPUT,
                default=10.0,
                units="dimensionless",
                description="AMPK pT172",
            ),
            "mtor": Port(
                role=PortRole.INPUT,
                default=10.0,
                units="dimensionless",
                description="mTORC1 pS2448",
            ),
            "nad_redox": Port(
                role=PortRole.INPUT,
                default=5.5,
                units="dimensionless",
                description="matrix NAD+/NADH ratio",
            ),
        }

    def derivative(self, t, state):
        depolarised = hill_inhibition(
            state["dpsi"], self.dpsi_half, self.dpsi_steepness
        )
        sirt = hill_gate(state["nad_redox"], self.nad_redox_half, 2.0)
        ampk = hill_gate(state["ampk"], self.ampk_half, 2.0)
        mtor_block = hill_inhibition(state["mtor"], self.mtor_half, 2.0)
        return {
            "pink1": self.pink1_on * depolarised
            - self.pink1_off * jnp.maximum(state["pink1"], 0.0),
            "pgc1a": (self.pgc1a_ampk * ampk + self.pgc1a_sirt * sirt)
            * mtor_block
            - self.pgc1a_decay * jnp.maximum(state["pgc1a"], 0.0),
        }

    def assign(self, t, state):
        flux = state["mitophagy_flux"] / self.mitophagy_reference
        return {
            "mitophagy_selectivity": self.selectivity_gain
            * jnp.maximum(state["pink1"], 0.0)
            * (1.0 - self.fused_fraction)
            * flux,
            "biogenesis": 1.0
            + self.biogenesis_gain * jnp.maximum(state["pgc1a"], 0.0),
        }


class MitoMembranePotential(Process):
    """Whole-cell Δψm: the biophysical potential scaled by the organelle
    population's polarisation state.

    Two independent things move Δψm with age and neither alone is the
    measurement. Nazaret 2009 gives the potential a *competent, coupled*
    mitochondrion holds, as a function of respiratory capacity and proton leak.
    DallePezze 2014 gives how the population redistributes: functional mass is
    converted to a dysfunctional pool whose potential per unit mass collapses.
    The observable is the product::

        ΔΨ_cell = ΔΨ_bioe(R, leak) · [(Ψ_new + Ψ_old)/(m_new + m_old)]
                                     / [(Ψ_new + Ψ_old)/(m_new + m_old)]|_{t=0}

    i.e. an absolute mV from the biophysics times the population's
    potential-per-unit-mass relative to the young cell. Multiplicative, because
    the two effects are independent; routed through its own store path and read
    as an INPUT, which is how this framework expresses multiplication (EVOLVED
    ports sum).

    Reported per unit mitochondrial mass on purpose: senescent fibroblasts
    raise mass 2–8× while each organelle gets worse, so per-cell and
    per-mitochondrion readouts move in opposite directions (Hutter et al. 2004,
    Biochem J 380:919; Kim et al. 2023, Life Sci Alliance 6:e202302127). The
    JC-1 measurement this is compared against — 54.6 ± 4.1% of proliferating in
    senescent MRC5 (Passos et al. 2007, PLoS Biol 5:e110, Fig 1B) — is the
    per-mitochondrion quantity.
    """

    hallmark = "Mitochondrial Dysfunction"
    reference = "Passos 2007; Hutter 2004; DallePezze 2014"
    description = (
        "Whole-cell Δψm as the biophysical potential times the organelle "
        "population's potential-per-unit-mass relative to young."
    )

    timescale: float | None = 3600.0
    # DallePezze 2014 initial state: Mito_membr_pot_new / Mito_mass_new.
    reference_ratio: float = 12.12
    # Potential retained by the dysfunctional pool, as a fraction of the
    # functional pool's. DallePezze's own Mito_membr_pot_old state sits near
    # zero, which would put the whole-cell JC-1 signal at ~4% of young; the
    # measurement is 54.6 +/- 4.1% (Passos 2007, Fig 1B). 0.7 is the value that
    # reproduces it, i.e. dysfunctional mitochondria stay substantially
    # polarised — consistent with their not being cleared immediately.
    depolarised_pool_potential: float = calibratable(
        0.7,
        clamp=(0.05, 1.0),
        description="dysfunctional-pool potential as a fraction of the "
        "functional pool's; anchored to Passos 2007 JC-1 = 54.6% of young.",
    )

    def ports_schema(self):
        return {
            "dpsi_cell": Port(
                role=PortRole.ASSIGNED,
                default=BIOE_DPSI_REFERENCE,
                units="mV",
                description="whole-cell mean inner-membrane potential",
                ontology={"go": "GO:0051881"},
            ),
            "dpsi_intrinsic": Port(
                role=PortRole.INPUT,
                default=BIOE_DPSI_REFERENCE,
                units="mV",
                description="potential of a competent, coupled mitochondrion",
            ),
            "pot_new": Port(
                role=PortRole.INPUT,
                default=12.12,
                units="dimensionless",
                description="DallePezze functional-pool membrane potential",
            ),
            "pot_old": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="DallePezze dysfunctional-pool membrane potential",
            ),
            "mass_new": Port(
                role=PortRole.INPUT,
                default=1.0,
                units="dimensionless",
                description="DallePezze functional mitochondrial mass",
            ),
            "mass_old": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="dimensionless",
                description="DallePezze dysfunctional mitochondrial mass",
            ),
        }

    def assign(self, t, state):
        mass_new = jnp.maximum(state["mass_new"], 0.0)
        mass_old = jnp.maximum(state["mass_old"], 0.0)
        rho_new = state["pot_new"] / (mass_new + 1e-9)
        rho_cell = (
            mass_new * rho_new
            + mass_old * rho_new * self.depolarised_pool_potential
        ) / (mass_new + mass_old + 1e-9)
        return {
            "dpsi_cell": state["dpsi_intrinsic"]
            * rho_cell
            / self.reference_ratio
        }


class SenescentProtonLeak(Process):
    """Proton-leak conductance as a function of senescence burden.

    Oligomycin-resistant (leak-driven) respiration roughly doubles in
    irradiation-induced senescent MRC5 fibroblasts, matching deep replicative
    senescence (Passos et al. 2010, Mol Syst Biol 6:347, Fig 1C). The leak is
    read off DallePezze's SA-β-gal, the model's own senescence-burden state,
    Hill-interpolated between the healthy and senescent conductances — the
    algebraic sibling of a coupling edge, so the bioenergetic module reads a
    plain input rather than reaching into the senescence model.
    """

    hallmark = "Cellular Senescence"
    reference = "Passos et al. 2010, Mol Syst Biol 6:347"
    description = "SA-β-gal → proton-leak conductance (senescent uncoupling)."

    timescale: float | None = 3600.0
    basal: float = 1.0
    senescent: float = calibratable(
        2.0,
        clamp=(1.0, 5.0),
        description="leak conductance at full senescence, relative to healthy.",
    )
    K: float = 4.0
    n: float = 2.0

    def ports_schema(self):
        return {
            "leak": Port(
                role=PortRole.ASSIGNED,
                default=1.0,
                units="dimensionless",
                description="proton-leak conductance relative to healthy",
            ),
            "burden": Port(
                role=PortRole.INPUT,
                default=0.81,
                units="dimensionless",
                description="DallePezze SA-β-gal senescence burden",
                ontology={"go": "GO:0090398"},
            ),
        }

    def assign(self, t, state):
        gate = hill_gate(state["burden"], self.K, self.n)
        return {"leak": self.basal + (self.senescent - self.basal) * gate}


# ── Hallmark registry ───────────────────────────────────────────────────


def mito_hallmark_registry(registry: dict | None = None) -> dict:
    """``HALLMARK_REGISTRY`` with Genomic Instability wired to this composite.

    The global handle drives exogenous damage through ERiQ's ``damage_repair``
    or through the demo's ``irradiation_pulse`` amplitude; neither exists
    here, so severity would be silently inert. This composite carries the dose
    on DallePezze's ``DNA_damaged_by_irradiation`` rate constant instead
    (``floor=0`` — severity 0 really is unirradiated). Model-local so the
    multi-hallmark demo, which holds that constant fixed and doses through
    the pulse, is
    untouched: pass it as ``with_hallmarks(..., registry=...)`` or
    ``CalibrationProblem(hallmark_registry=...)``.
    """
    import copy

    from hallsim.hallmarks import HALLMARK_REGISTRY, ParameterMapping

    base = copy.deepcopy(HALLMARK_REGISTRY if registry is None else registry)
    base["Genomic Instability"].mappings.append(
        ParameterMapping(
            process_name="dp14",
            param_name=f"parameters.{DP14_IRRADIATION_RATE_NAME}",
            floor=0.0,
            slope=1.0,
            description=(
                "Exogenous DNA-damage dose: 0 at severity 0 (unirradiated), "
                "the published DallePezze rate at severity 1"
            ),
        )
    )
    return base


# ── Composite factory ───────────────────────────────────────────────────

MITO_PREFIX = "mito"

# Prior scale for the mitochondrial-ROS → DallePezze-ROS edge. DallePezze's ROS
# pool sits at 10 and turns over per day, so an edge able to move it by ~10% of
# its own pool per day is the host-module scale — the Occam anchor
# docs/coupling-edge-priors.md uses for the demo's IKK edges. The data sets
# the value within that scale.
MITO_ROS_EDGE_STRENGTH = 1.0


def build_mitochondrial_aging_composite(
    *,
    prefix: str = MITO_PREFIX,
    surrogate: dict | None = None,
    ros_mutagenesis: float = 0.0,
    fused_fraction: float = 0.3,
    irradiation_rate: float | None = None,
    validate: bool = True,
    semantic_validation: bool | dict = True,
):
    """Compose DallePezze 2014 with the mtDNA, bioenergetic, redox and
    quality-control modules on a shared day axis.

    ``ros_mutagenesis`` opens the vicious-cycle channel (0 = off, the default
    and the position the mutator-mouse evidence supports).
    ``fused_fraction`` is the share of the network that is fused and therefore
    shielded from selective mitophagy — the fission/fusion handle.
    ``surrogate`` overrides the fitted Nazaret 2009 quasi-steady-state
    coefficients (see ``demos/mito_bioenergetics_reduction.py``).

    ``irradiation_rate`` sets DallePezze's ``DNA_damaged_by_irradiation``
    constant — the exogenous-damage dose. ``None`` keeps the published value;
    ``0.0`` gives the unirradiated arm. The ``Irradiation`` species itself is
    not the knob: it is consumed within the first hours, and the model's
    time-piecewise exposure rule does not survive SBML import, so the dose
    lives in the rate constant (the same conclusion ``multi_hallmark`` reached).
    """
    from hallsim.composite import Composite
    from hallsim.models.hill_edge import HillActivationEdge
    from hallsim.sbml_import import process_from_sbml

    p = prefix
    dp14 = process_from_sbml(
        str(SENESCENCE_SBML_PATH),
        name="dp14",
        parameters=(
            {}
            if irradiation_rate is None
            else {DP14_IRRADIATION_RATE_NAME: irradiation_rate}
        ),
    )

    processes = {
        "dp14": dp14,
        "mtdna": MtDNAPopulation(ros_mutagenesis=ros_mutagenesis),
        "competence": RespiratoryCompetence(),
        "leak": SenescentProtonLeak(),
        "bioe": MitoBioenergetics(
            **({} if surrogate is None else {"coefficients": surrogate})
        ),
        "redox": MitoRedox(),
        "potential": MitoMembranePotential(),
        "qc": MitoQualityControl(fused_fraction=fused_fraction),
        # Mitochondrial H2O2 feeds the senescence model's ROS pool. DallePezze
        # already produces ROS from its own dysfunctional-mass pool, so this is
        # an additive channel carrying the mtDNA/bioenergetic contribution, not
        # a replacement — K sits at the healthy operating level so it is
        # half-open at rest and rises with mitochondrial ROS.
        "mito_ros": HillActivationEdge(
            timescale=dp14.timescale,
            k_act=MITO_ROS_EDGE_STRENGTH,
            K=(1.0,),
            n=(2.0,),
            target_default=10.0,
            target_ontology={"chebi": "CHEBI:26523"},
            target_description="DallePezze ROS pool, receiving mitochondrial "
            "superoxide/H2O2",
            source_ontology=({"chebi": "CHEBI:16240"},),
            source_descriptions=("mitochondrial ROS relative to basal",),
            hallmark="Mitochondrial Dysfunction",
            reference="Passos et al. 2007, PLoS Biol 5:e110",
            description="mitochondrial ROS → DallePezze ROS pool.",
        ),
    }

    topology = {
        "mtdna": {
            "wt": f"{p}/mtDNA_wt",
            "mut": f"{p}/mtDNA_mut",
            "atp": f"{p}/ATP_signal",
            "resp_competence": f"{p}/resp_competence",
            "biogenesis": f"{p}/biogenesis",
            "mitophagy_selectivity": f"{p}/mitophagy_selectivity",
            "ros": f"{p}/ROS",
        },
        "competence": {
            "wt": f"{p}/mtDNA_wt",
            "mut": f"{p}/mtDNA_mut",
            "mass_new": DP14_MASS_NEW,
            "mass_old": DP14_MASS_OLD,
            "heteroplasmy": f"{p}/heteroplasmy",
            "resp_competence": f"{p}/resp_competence",
        },
        "leak": {"leak": f"{p}/leak", "burden": DP14_SA_BETA_GAL},
        "potential": {
            "dpsi_cell": f"{p}/dPsi_cell",
            "dpsi_intrinsic": f"{p}/dPsi",
            "pot_new": DP14_POT_NEW,
            "pot_old": DP14_POT_OLD,
            "mass_new": DP14_MASS_NEW,
            "mass_old": DP14_MASS_OLD,
        },
        "bioe": {
            "dpsi": f"{p}/dPsi",
            "atp_matrix": f"{p}/ATP_matrix",
            "nad_fraction": f"{p}/NAD_fraction",
            "nad_redox": f"{p}/NAD_redox",
            "resp_competence": f"{p}/resp_competence",
            "leak": f"{p}/leak",
        },
        "redox": {
            "ros": f"{p}/ROS",
            "antiox": f"{p}/antiox",
            "dpsi": f"{p}/dPsi",
            "resp_competence": f"{p}/resp_competence",
            "mass_new": DP14_MASS_NEW,
            "mass_old": DP14_MASS_OLD,
            "foxo": DP14_FOXO3A,
        },
        "qc": {
            "pink1": f"{p}/PINK1",
            "pgc1a": f"{p}/PGC1a",
            "mitophagy_selectivity": f"{p}/mitophagy_selectivity",
            "biogenesis": f"{p}/biogenesis",
            "dpsi": f"{p}/dPsi_cell",
            "mitophagy_flux": DP14_MITOPHAGY,
            "ampk": DP14_AMPK,
            "mtor": DP14_MTOR,
            "nad_redox": f"{p}/NAD_redox",
        },
        "mito_ros": {"source": f"{p}/ROS", "target": DP14_ROS},
    }
    return Composite(
        processes=processes,
        topology=topology,
        validate=validate,
        semantic_validation=semantic_validation,
    )
