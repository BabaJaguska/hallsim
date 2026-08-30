"""Chronic vs transient exposure: what a ClampEdge is for.

A ligand the model *integrates* — consumed by receptor uptake — drains away
on its own, so a composite built from that model can only ever show an acute
response. Aging phenotypes are the other regime: chronic SASP factors, a
sustained inflammatory tone, a drug held at trough concentration for months.
:func:`hallsim.models.clamp_edge.clamp_species` holds the species there.

Three panels:

1. **Ligand.** Unclamped drains to zero; clamped holds at the setpoint.
2. **Downstream exposure.** A leaky response integrator — the acute run
   relaxes back to baseline, the chronic runs plateau. Same model, same
   parameters; only the boundary condition differs.
3. **Residual vs clamp rate.** Measured endpoint offset against the
   documented ``flux / k_clamp`` line, with the rate
   :func:`~hallsim.models.clamp_edge.place_clamp_rate` picks for a 1%
   tolerance marked. The relation is the reason the rate is placed rather
   than guessed.

    python demos/clamp_setpoint.py [config.json]
"""

import jax

jax.config.update("jax_enable_x64", True)

import json
import logging
import sys

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hallsim.composite import Composite
from hallsim.io import outdir
from hallsim.models.clamp_edge import (
    clamp_species,
    measure_unclamped_flux,
    place_clamp_rate,
)
from hallsim.process import Port, PortRole, Process
from hallsim.scheduler import Scheduler

log = logging.getLogger("clamp_setpoint")

DEFAULTS = {
    "level": 2.0,
    "t1": 60.0,
    "macro_dt": 1.0,
    "rel_error": 0.01,
    "k_clamps": [0.05, 0.2, 1.0],
    "outdir": "clamp_setpoint",
    "plot": "clamp_setpoint.png",
}

LIGAND = "medium/ligand"
RESPONSE = "cell/response"


class LigandUptake(Process):
    """Receptor-mediated uptake of a ligand the model integrates.

    Michaelis-Menten consumption, no resupply: the species is neither an SBML
    constant (``with_param_input``) nor a boundary input (``drive_pulse``), so
    a clamp is the only way to sustain it.
    """

    timescale: float | None = 1.0
    v_max: float = 0.5
    K_m: float = 1.0

    def ports_schema(self):
        return {
            "ligand": Port(
                role=PortRole.EVOLVED,
                default=2.0,
                units="uM",
                description="free ligand in the medium",
                ontology={"chebi": "CHEBI:26523"},
            )
        }

    def derivative(self, t, state):
        L = jnp.maximum(state["ligand"], 0.0)
        return {"ligand": -self.v_max * L / (self.K_m + L)}


class ReceptorResponse(Process):
    """Leaky downstream readout: ``dR/dt = k_on·L − R/tau``. Plateaus under a
    held ligand, relaxes to baseline once the ligand is gone."""

    timescale: float | None = 1.0
    k_on: float = 1.0
    tau: float = 10.0

    def ports_schema(self):
        return {
            "response": Port(
                role=PortRole.EVOLVED,
                default=0.0,
                units="uM",
                description="downstream signalling response",
            ),
            "ligand": Port(
                role=PortRole.INPUT,
                default=0.0,
                units="uM",
                description="free ligand driving the response",
            ),
        }

    def derivative(self, t, state):
        return {
            "response": self.k_on * state["ligand"]
            - state["response"] / self.tau
        }


def _build(level=None, k_clamp=None):
    """The two-process composite, optionally with ``ligand`` clamped."""
    processes = {"cell": LigandUptake(), "response": ReceptorResponse()}
    topology = {
        "cell": {"ligand": LIGAND},
        "response": {"response": RESPONSE, "ligand": LIGAND},
    }
    if k_clamp is not None:
        clamp_species(
            processes,
            topology,
            target="cell",
            species="ligand",
            level=level,
            k_clamp=k_clamp,
        )
    return Composite(processes=processes, topology=topology)


def run(cfg):
    sched = Scheduler()
    t_span = (0.0, cfg["t1"])
    kw = dict(t_span=t_span, macro_dt=cfg["macro_dt"], save_dt=cfg["macro_dt"])

    level = cfg["level"]
    baseline = _build()
    flux = float(measure_unclamped_flux(baseline, LIGAND, level))
    placed = place_clamp_rate(
        flux,
        level,
        rel_error=cfg["rel_error"],
        tau_model=LigandUptake().timescale,
    )
    log.info(
        "flux at setpoint %.4g uM: %.4g uM/t -> k_clamp %.4g (%s)",
        level,
        flux,
        placed.k_clamp,
        placed.note,
    )

    runs = {"unclamped": sched.run(baseline, **kw)}
    for k in cfg["k_clamps"]:
        runs[f"k={k:g}"] = sched.run(_build(level, k), **kw)
    runs[f"placed k={placed.k_clamp:.3g}"] = sched.run(
        _build(level, placed.k_clamp), **kw
    )

    sweep_k = np.geomspace(
        min(cfg["k_clamps"]) / 2.0,
        max(placed.k_clamp, *cfg["k_clamps"]) * 2,
        12,
    )
    residuals = [
        level - float(sched.run(_build(level, float(k)), **kw).get(LIGAND)[-1])
        for k in sweep_k
    ]
    return runs, sweep_k, np.asarray(residuals), flux, placed


def plot(runs, sweep_k, residuals, flux, placed, cfg, path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    for label, res in runs.items():
        ts = np.asarray(res.ts)
        axes[0].plot(ts, np.asarray(res.get(LIGAND)), lw=1.6, label=label)
        axes[1].plot(ts, np.asarray(res.get(RESPONSE)), lw=1.6, label=label)
    axes[0].axhline(
        cfg["level"], color="0.4", ls=":", lw=1, label="setpoint", zorder=0
    )
    axes[0].set_title("Ligand — drains, or is held")
    axes[1].set_title("Downstream response — acute vs chronic")
    for ax, ylab in zip(axes[:2], ["ligand (uM)", "response (uM)"]):
        ax.set_xlabel("time")
        ax.set_ylabel(ylab)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="best")

    axes[2].loglog(sweep_k, residuals, "o-", lw=1.4, label="measured")
    axes[2].loglog(
        sweep_k,
        abs(flux) / sweep_k,
        "--",
        color="0.4",
        lw=1.2,
        label="flux / k_clamp",
    )
    axes[2].axhline(
        cfg["rel_error"] * cfg["level"],
        color="C3",
        ls=":",
        lw=1,
        label=f"{cfg['rel_error']:.0%} of setpoint",
    )
    axes[2].axvline(placed.k_clamp, color="C2", ls=":", lw=1, label="placed k")
    axes[2].set_title("Residual offset vs clamp rate")
    axes[2].set_xlabel("k_clamp")
    axes[2].set_ylabel("setpoint − held level (uM)")
    axes[2].grid(alpha=0.3, which="both")
    axes[2].legend(fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    log.info("wrote %s", path)
    return fig


def run_demo(argv=(), **overrides):
    """Solve, sweep and plot; the entry point `simulate clamp` calls."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = dict(DEFAULTS)
    if argv:
        with open(argv[0]) as f:
            cfg.update(json.load(f))
    cfg.update(overrides)
    runs, sweep_k, residuals, flux, placed = run(cfg)
    path = outdir(cfg["outdir"]) / cfg["plot"]
    plot(runs, sweep_k, residuals, flux, placed, cfg, path)
    return path


if __name__ == "__main__":
    run_demo(sys.argv[1:])
