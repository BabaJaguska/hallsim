"""The canonical Geva-Zatorsky 2006 output: the p53-Mdm2 oscillator.

Runs the bare GZ06 SBML model (BIOMD157) on its native hourly clock and
plots p53 and Mdm2 over time — the undamped, out-of-phase pulses (period
~5.5 h) that are the model's signature. Shown at the basal and full-damage
psi the flagship interpolates between.

    .venv_hallsim/bin/python demos/gz06_oscillator.py
"""
from __future__ import annotations

from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)
import numpy as np  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from hallsim.composite import Composite  # noqa: E402
from hallsim.scheduler import Scheduler  # noqa: E402
from hallsim.sbml_import import process_from_sbml  # noqa: E402

plt.rcParams.update({"font.family": "monospace",
                     "font.monospace": ["DejaVu Sans Mono"]})

GZ = str(Path(__file__).resolve().parent.parent / "models" /
         "zatorsky2006" / "zatorsky2006_BIOMD0000000157.xml")
C_P53 = "#6d28d9"   # violet
C_MDM2 = "#b45309"  # amber
GRID = "#e6e6e2"
T_END = 45.0        # hours


def run(psi):
    gz = process_from_sbml(GZ, name="gz06", parameters={"psi": psi})
    comp = Composite(processes={"gz06": gz}, topology={},
                     semantic_validation=False)
    r = Scheduler().run(comp, t_span=(0.0, T_END), macro_dt=T_END,
                        save_dt=0.05)
    return np.asarray(r.ts), np.asarray(r.get("gz06/x")), \
        np.asarray(r.get("gz06/y"))


def main():
    fig, axes = plt.subplots(2, 1, figsize=(8.4, 5.6), sharex=True)
    for ax, (psi, tag) in zip(axes, [(1.0, "psi = 1.0   (full damage / DDIS)"),
                                     (0.3, "psi = 0.3   (basal / control)")]):
        ts, x, y = run(psi)
        ax.plot(ts, x, color=C_P53, lw=2.0, label="p53  (x)")
        ax.plot(ts, y, color=C_MDM2, lw=2.0, label="Mdm2 (y)")
        ax.set_title(tag, fontsize=10.5, color="#334155", loc="left")
        ax.grid(True, color=GRID, lw=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.set_ylabel("level (a.u.)", fontsize=9.5)
    axes[0].legend(loc="upper right", fontsize=9, frameon=False, ncol=2)
    axes[1].set_xlabel("time (hours)", fontsize=9.5)
    fig.suptitle("Geva-Zatorsky 2006 — p53-Mdm2 oscillator  [BIOMD157]",
                 fontsize=12, x=0.02, ha="left", fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    from outdir import outdir
    out = outdir("gz06_oscillator")
    for ext in ("png", "pdf"):
        fig.savefig(out / f"gz06_oscillator.{ext}", dpi=160,
                    bbox_inches="tight", facecolor="white")
    print(f"wrote gz06_oscillator.png/.pdf -> {out}", flush=True)


if __name__ == "__main__":
    main()
