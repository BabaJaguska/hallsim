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
from hallsim.models.saturating_removal import SaturatingRemoval
from hallsim.sbml_import import process_from_sbml

GZ06_SBML_PATH = (
    Path(__file__).parent.parent.parent.parent
    / "models"
    / "zatorsky2006"
    / "zatorsky2006_BIOMD0000000157.xml"
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
    eriq_prefix: str = ERIQ_PREFIX,
    sbml_path: str | None = None,
    validate: bool = False,
):
    """Build the six-process DamageRepair → GZ06 → Bridge → ERiQ composite.

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

    return Composite(
        processes={
            "damage_repair": damage_proc,
            "gz06_p53": gz06_proc,
            "p53_bridge": bridge_proc,
            "eriq_energy": eriq_energy,
            "eriq_oxstress": eriq_oxstress,
            "eriq_signaling": eriq_signaling,
        },
        topology=topology,
        validate=validate,
    )
