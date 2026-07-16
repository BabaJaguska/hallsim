"""Load Gerard et al. 2014 (PLoS Comput Biol 10:e1003455) from its published
XPPAUT .ode file and simulate the inflammatory switch on its own.

Demonstrates the XPP intake path (`hallsim.xpp_import.process_from_xpp`): a
14-ODE model written for XPP, never for HallSim, imported and integrated as a
plain Composite with no hand-translation. The loop is
NF-kB -> LIN28 -| let-7 -> IL-6 -> STAT3 -> miR-21 -| PTEN -> NF-kB; a Src
input switches the system from the low (untransformed) to the high
(transformed / sustained-inflammation) state.
"""

from __future__ import annotations

import diffrax as dfx
import jax.numpy as jnp
import matplotlib.pyplot as plt

from hallsim.composite import Composite
from hallsim.scheduler import Scheduler
from hallsim.xpp_import import process_from_xpp
from outdir import outdir

ODE_PATH = "data/xpp/gerard2014_inflammation.ode"
T_END = 4500.0  # the model's own @ Total
KEY_SPECIES = ["nfkb", "il6", "stat3", "ras", "let7", "pten"]


def simulate(parameters=None):
    proc = process_from_xpp(ODE_PATH, parameters=parameters)
    comp = Composite(
        processes={proc._name: proc},
        topology={proc._name: {s: s for s in proc._state_names}},
    )
    # The .ode declares meth=stiff; the transformed branch is stiff enough to
    # exhaust an explicit solver's step budget, so pin the A-stable Kvaerno5.
    sched = Scheduler(solver=dfx.Kvaerno5())
    res = sched.run(comp, t_span=(0.0, T_END), macro_dt=5.0, save_dt=10.0)
    return proc, res


def main():
    out = outdir("gerard2014_xpp")

    # Shipped defaults hold Src at 0; raising Srcmax delivers the Src pulse
    # that triggers the switch (the .ode's forcing machinery, params only).
    runs = [
        ("Src = 0 (resting)", None),
        ("Src = 2 (sustained)", {"Srcmax": 2.0}),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for ax, (label, params) in zip(axes, runs):
        proc, res = simulate(params)
        finite = bool(jnp.all(jnp.isfinite(res.ys)))
        print(f"{label}: {len(proc._state_names)} states, finite={finite}")
        for s in KEY_SPECIES:
            v = res.get(s)
            ax.plot(res.ts, v, label=s.upper())
            print(f"   {s:6s} final={float(v[-1]):.3g}")
        ax.set_title(label)
        ax.set_xlabel("time")
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel("concentration")
    fig.suptitle("Gerard 2014 inflammatory switch — imported from XPP .ode")
    fig.tight_layout()
    path = out / "gerard2014_switch.png"
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
