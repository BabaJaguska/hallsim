"""Composed system: DamageRepair → SBML p53 oscillator → ERiQ.

A six-process composite (1 damage + 1 SBML oscillator + 1 bridge + 3
ERiQ sub-processes) that demonstrates HallSim's central composability
claim: an upstream hand-built damage-accumulation Process drives an
SBML-imported p53-Mdm2 oscillator from a different publication, whose
dynamic p53 output replaces ERiQ's intrinsic algebraic p53, thereby
propagating dose-dependent genomic instability through to ERiQ's
downstream metabolic and signaling readouts.

Composition chain
-----------------

    SaturatingRemoval (DDR mode, alpha=0.01..0.05)
        damage  ────────────►  ▼
                              p53/psi   (GZ06 tunable INPUT)
                              │
    Geva-Zatorsky 2006 (BIOMD0000000157)
        x  (p53 protein)  ────────────►  ▼
                                        p53/x  (Bridge INPUT)
                                        │
    P53Bridge
        p53_activity  ────────────►  ▼
                                    eriq/p53_activity  (ERiQ INPUT)
                                    │
    ERiQ (energy + oxidative_stress + signaling-without-p53)
        mTOR, NF-κB, autophagy, ATP, ROS, mito_damage  ── observables

Three independent literature sources are composed via topology alone:

1. Karin & Alon 2019 / Reinhardt & Yaffe 2009 — saturating-removal motif
   for the upstream damage pool.
2. Geva-Zatorsky et al. 2006 (Mol Sys Biol) — imported as SBML
   BIOMD0000000157, exposing ``psi`` (damage input) as a tunable INPUT
   port via ``process_from_sbml(..., tunable_params=("psi",))``.
3. Alfego & Kriete 2017 — ERiQ metabolic-signaling core, with the p53
   compartment replaced by the bridge.

This is the preprint composability figure: showing that downstream ERiQ
readouts (NF-κB, autophagy, mTOR) reflect upstream dose-dependent damage
oscillations propagated through a dynamic p53-Mdm2 oscillator from a
different group's published model.

Usage
-----
>>> from hallsim.models.damage_p53_eriq import build_damage_p53_eriq_composite
>>> from hallsim.scheduler import Scheduler
>>> comp = build_damage_p53_eriq_composite(alpha=0.03)  # moderate damage
>>> result = Scheduler().run(
...     comp, t_span=(0.0, 100.0), macro_dt=5.0, save_dt=5.0,
... )
>>> # Observe how ERiQ's p53/NF-κB/autophagy track upstream damage:
>>> result.get("damage_repair/damage"), result.get("eriq/p53_activity")

Notes
-----
The GZ06 SBML file is bundled at ``models/zatorsky2006/...xml``.
The composite uses :class:`ERiQSignalingNoP53` so the bridge can own
``eriq/p53_activity`` without a topology conflict; ERiQ's intrinsic mTOR
dynamics are unchanged.
"""

from __future__ import annotations

from pathlib import Path

from hallsim.composite import Composite
from hallsim.models.eriq import (
    ERIQ_PREFIX,
    ERiQEnergyMetabolism,
    ERiQOxidativeStress,
    ERiQSignalingNoP53,
)
from hallsim.models.p53_bridge import P53Bridge
from hallsim.models.sasp_mtor import SASPmTORActivator, mTORInhibitor
from hallsim.models.saturating_removal import SaturatingRemoval
from hallsim.sbml_import import process_from_sbml

GZ06_SBML_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "models"
    / "zatorsky2006"
    / "zatorsky2006_BIOMD0000000157.xml"
)

DP14_SBML_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "models"
    / "dallepezze2014"
    / "dallepezze2014_BIOMD0000000582.xml"
)


def build_damage_p53_eriq_composite(
    *,
    alpha: float = 0.01,
    beta: float = 0.10,
    K_repair: float = 9.0,
    bridge_gain: float = 1.0,
    bridge_offset: float = 0.0,
    bridge_k_track: float = 5.0,
    GLYCOL_SA: float = 1.0,
    MDAMAGE_SA: float = 1.0,
    with_sasp_mtor: bool = False,
    sasp_k: float = 0.05,
    rapamycin_strength: float = 0.0,
    eriq_prefix: str = ERIQ_PREFIX,
    sbml_path: str | None = None,
    validate: bool = False,
):
    """Build the DamageRepair → GZ06 → Bridge → ERiQ composite.

    Six processes by default. With ``with_sasp_mtor=True``, adds a
    seventh — :class:`SASPmTORActivator` — that captures the
    paradoxical high-mTOR phenotype of senescent cells (Carroll 2017,
    Laberge 2015, Herranz 2015, Houssaini 2018, Fielder 2017). When
    enabled, chronic damage + chronic p53 push mTOR_activity upward in
    DDIS regimes, overriding AMPK's inhibitory pressure.

    Parameters
    ----------
    alpha:
        DSB induction rate. The Genomic Instability hallmark lever.
        ``0.01`` = baseline, ``0.03`` = moderate, ``0.05`` = high.
    beta, K_repair:
        Repair kinetics for SaturatingRemoval. Defaults give D_ss = 1.0
        at alpha=0.01, matching GZ06's psi baseline.
    bridge_gain, bridge_offset, bridge_k_track:
        Linear rescaling and tracking rate from GZ06's ``x`` (p53 protein)
        to ERiQ's ``p53_activity``. Default gain=1, offset=0 maps the
        oscillator's [0,1]-ish range onto ERiQ's p53_activity range.
    GLYCOL_SA, MDAMAGE_SA:
        Pass-throughs to ERiQ for sensitivity analysis.
    with_sasp_mtor:
        If True, add :class:`SASPmTORActivator` to the composite. This
        is what enables capturing the DDIS mTORC1-paradox during
        validation against ssGSEA pathway data — without it, the
        canonical AMPK→mTOR axis dominates and mTORC1 trends down with
        damage.
    sasp_k:
        Rate constant for the SASP→mTOR term. Active only when
        ``with_sasp_mtor=True``. Default 0.05 matches the order of
        ERiQSignaling's gy=0.1 integrator gain.
    rapamycin_strength:
        Strength of the rapamycin perturbation (additive negative
        driver on ``mTOR_activity``). Default 0.0 disables; values
        ~0.05 model moderate pharmacological inhibition. Corresponds
        to the ``DDIS+rapamycin`` rescue arm in the GSE248823
        validation.
    eriq_prefix:
        Store path prefix for ERiQ state variables.
    sbml_path:
        Path to the GZ06 SBML file. Defaults to the bundled local copy.
    validate:
        Topology validation flag. Defaults to False because the SBML
        process exposes many auto-generated EVOLVED ports that don't
        need cross-process validation.

    Returns
    -------
    Composite ready to simulate.
    """
    sbml_path = str(sbml_path or GZ06_SBML_PATH)

    # Build processes with timescale annotations so the multi-rate
    # Scheduler can split them into appropriate groups. Timescale ratios
    # in this composite span ~250x (bridge fast → damage slow), so a
    # bare single-group Diffrax solve is forced into very small steps
    # by the bridge's fast tracking. With timescales set, the Scheduler
    # auto-groups within ~100x, splitting the bridge into its own group.
    damage_proc = SaturatingRemoval(
        alpha=alpha, beta=beta, K=K_repair, eta=0.0, timescale=50.0
    )
    gz06_proc = process_from_sbml(
        sbml_path, name="gz06_p53", tunable_params=("psi",), timescale=5.0
    )
    bridge_proc = P53Bridge(
        gain=bridge_gain,
        offset=bridge_offset,
        k_track=bridge_k_track,
        timescale=0.2,  # 1 / k_track
    )
    eriq_energy = ERiQEnergyMetabolism(GLYCOL_SA=GLYCOL_SA)
    eriq_oxstress = ERiQOxidativeStress(MDAMAGE_SA=MDAMAGE_SA)
    eriq_signaling = ERiQSignalingNoP53()

    p = eriq_prefix

    # Topology — the wiring that demonstrates composition.
    # damage → p53/psi → p53/x → eriq/p53_activity → ERiQ downstream.
    topology = {
        "damage_repair": {
            # SaturatingRemoval owns 'damage' EXCLUSIVE; route to
            # p53/psi so GZ06 reads it as the damage-input parameter.
            "damage": "p53/psi",
        },
        "gz06_p53": {
            # GZ06 species (EVOLVED) get their own store paths under p53/.
            "x": "p53/x",
            "y0": "p53/y0",
            "y": "p53/y",
            # psi is INPUT (tunable param) — same store path as
            # damage_repair's damage output. Effectively: psi := damage.
            "psi": "p53/psi",
        },
        "p53_bridge": {
            # Bridge reads GZ06's x as INPUT, owns eriq/p53_activity.
            "p53_x": "p53/x",
            "p53_activity": f"{p}/p53_activity",
        },
        "eriq_energy": {
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
        "eriq_oxstress": {
            "mito_damage": f"{p}/mito_damage",
            "ROS_integrator_c": f"{p}/ROS_integrator_c",
            "ROS_activity": f"{p}/ROS_activity",
            "mito_function": f"{p}/mito_function",
            "glycolysis": f"{p}/glycolysis",
            "mTOR_activity": f"{p}/mTOR_activity",
            "p53_activity": f"{p}/p53_activity",
        },
        "eriq_signaling": {
            "mTOR_integrator_c": f"{p}/mTOR_integrator_c",
            "mTOR_activity": f"{p}/mTOR_activity",
            "p53_activity": f"{p}/p53_activity",
            "mito_function": f"{p}/mito_function",
            "glycolysis": f"{p}/glycolysis",
            "ROS_activity": f"{p}/ROS_activity",
            "ROS_integrator_c": f"{p}/ROS_integrator_c",
            "mito_damage": f"{p}/mito_damage",
        },
    }

    processes = {
        "damage_repair": damage_proc,
        "gz06_p53": gz06_proc,
        "p53_bridge": bridge_proc,
        "eriq_energy": eriq_energy,
        "eriq_oxstress": eriq_oxstress,
        "eriq_signaling": eriq_signaling,
    }

    if with_sasp_mtor:
        # Reads damage from p53/psi (the SaturatingRemoval output) and
        # p53_activity from the bridge-owned eriq/p53_activity. Writes
        # additively to eriq/mTOR_activity alongside ERiQSignalingNoP53.
        processes["sasp_mtor"] = SASPmTORActivator(k_sasp=sasp_k)
        topology["sasp_mtor"] = {
            "mTOR_activity": f"{p}/mTOR_activity",
            "mito_damage": "p53/psi",
            "p53_activity": f"{p}/p53_activity",
        }

    if rapamycin_strength > 0.0:
        # Pharmacological mTOR inhibition; pulls mTOR_activity downward.
        # Composes additively with ERiQSignalingNoP53 and (if present)
        # SASPmTORActivator via EVOLVED port-role semantics.
        processes["rapamycin"] = mTORInhibitor(strength=rapamycin_strength)
        topology["rapamycin"] = {
            "mTOR_activity": f"{p}/mTOR_activity",
        }

    return Composite(
        processes=processes,
        topology=topology,
        validate=validate,
    )


def build_damage_p53_eriq_dp14_composite(
    *,
    alpha: float = 0.05,
    MDAMAGE_SA: float = 1.0,
    with_sasp_mtor: bool = True,
    sasp_k: float = 0.05,
    rapamycin_strength: float = 0.0,
    validate: bool = True,
):
    """``damage_p53_eriq`` composed with the DallePezze2014 senescence model.

    Merges the existing ERiQ + GZ06 + SASPmTOR composite with
    BioModels BIOMD0000000582 (DallePezze2014) by sharing three
    canonical store paths:

    - ``eriq/mTOR_activity`` is the canonical mTOR pool. DP14's
      ``mTORC1_pS2448`` shares this path so the rapamycin perturbation
      (which writes to ``eriq/mTOR_activity``) propagates into DP14's
      downstream signaling, mitophagy regulation, and CDKN1A readout.
    - ``eriq/ROS_activity`` is the canonical reactive-oxygen pool;
      DP14's ``ROS`` shares it.
    - ``eriq/mito_damage`` is the canonical damage pool; DP14's
      ``DNA_damage`` shares it so DP14's mitophagy-driven turnover can
      reach the same accumulator ERiQ's oxidative stress writes to.

    All other DP14 state variables (CDKN1A, CDKN1B, Mitophagy,
    Mito_mass_*, Akt, AMPK, FoxO3a, JNK, IKKbeta, SA_beta_gal, ...)
    live under the ``dp14/`` prefix and are read by gene-reporter
    validation as state-level observables.

    Parameters
    ----------
    alpha, MDAMAGE_SA, with_sasp_mtor, sasp_k, rapamycin_strength:
        Forwarded to :func:`build_damage_p53_eriq_composite`.
    validate:
        Topology validation. Semantic validation is set explicitly per
        composite below; the merged composite enables it.
    """
    from hallsim.sbml_import import process_from_sbml

    eriq_full = build_damage_p53_eriq_composite(
        alpha=alpha,
        MDAMAGE_SA=MDAMAGE_SA,
        with_sasp_mtor=with_sasp_mtor,
        sasp_k=sasp_k,
        rapamycin_strength=rapamycin_strength,
        validate=validate,
    )

    dp14_proc = process_from_sbml(
        str(DP14_SBML_PATH), name="dp14_dallepezze2014"
    )
    dp14_ports = list(dp14_proc.ports_schema().keys())
    dp14 = Composite(
        processes={"main": dp14_proc},
        topology={"main": {p: p for p in dp14_ports}},
        validate=validate,
        semantic_validation=False,
    )

    rewire = {
        "dp14/mTORC1_pS2448": "eriq/mTOR_activity",
        "dp14/ROS": "eriq/ROS_activity",
        "dp14/DNA_damage": "eriq/mito_damage",
    }

    return Composite(
        processes={"eriq": eriq_full, "dp14": dp14},
        rewire=rewire,
        validate=validate,
    )
