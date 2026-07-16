#!/usr/bin/env python
"""Reproduce all 5 Sivakumar2011 models from BioModels (394-398)."""
import sys, warnings
import xml.etree.ElementTree as ET
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import diffrax as dfx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from hallsim.sbml_import import process_from_sbml
from hallsim.composite import Composite
from hallsim.scheduler import Scheduler

MODEL_DIR = Path(__file__).parent.parent / "models" / "sivakumar2011"

models = {
    "EGF (394)":       MODEL_DIR / "egf_BIOMD0000000394.xml",
    "Shh (395)":       MODEL_DIR / "shh_BIOMD0000000395.xml",
    "Notch (396)":     MODEL_DIR / "notch_BIOMD0000000396.xml",
    "Wnt (397)":       MODEL_DIR / "wnt_BIOMD0000000397.xml",
    "Crosstalk (398)": MODEL_DIR / "crosstalk_BIOMD0000000398.xml",
}


def get_species_names(sbml_path):
    """Extract species id -> human name mapping from SBML XML."""
    tree = ET.parse(sbml_path)
    names = {}
    for ns in [
        "http://www.sbml.org/sbml/level2",
        "http://www.sbml.org/sbml/level2/version1",
    ]:
        for sp in tree.findall(f".//{{{ns}}}species"):
            sid = sp.get("id", "")
            name = sp.get("name", sid)
            # Clean up sbml naming quirks
            name = name.replace("_minus_", "-").replace("_space_", " ")
            name = name.replace("_br_", "").replace("_sub_", "")
            name = name.replace("_Beta_", "B-").replace("_beta_", "b-")
            name = name.replace("Complex(", "").replace("Complex", "")
            name = name.strip("_()/")
            if len(name) > 30:
                name = name[:27] + "..."
            names[sid] = name
    return names


sim = Scheduler(solver=dfx.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=500_000, dt0=1e-4)

fig, axes = plt.subplots(len(models), 1, figsize=(14, 4 * len(models)))

for ax, (label, sbml_path) in zip(axes, models.items()):
    print(f"\n{'='*60}\n{label}: {sbml_path.name}")
    try:
        proc = process_from_sbml(str(sbml_path), name=label)
        species = list(proc.ports_schema().keys())
        names = get_species_names(sbml_path)
        print(f"  {len(species)} species")

        comp = Composite({label: proc}, {label: {s: s for s in species}})
        result = sim.run(comp, t_span=(0.0, 100.0), macro_dt=0.5, save_dt=0.5)
        print(f"  {len(result.ts)} timepoints, solver OK")

        for s in species:
            traj = result.get(s)
            if float(traj[-1]) != float(traj[0]):
                ax.plot(result.ts, traj, lw=1.2, alpha=0.8,
                        label=names.get(s, s))
        ax.set_title(label, fontsize=12, fontweight="bold")
        ax.set_ylabel("Concentration")
        ax.legend(fontsize=7, ncol=3, loc="upper right",
                  framealpha=0.9, borderpad=0.3, handlelength=1.2)
    except Exception as e:
        print(f"  FAILED: {e}")
        ax.set_title(f"{label} -- FAILED", color="red")
        ax.text(0.5, 0.5, str(e)[:100], transform=ax.transAxes, ha="center")

axes[-1].set_xlabel("Time (hours)")
fig.suptitle("Sivakumar2011 -- All 5 BioModels reproduced",
             fontsize=14, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
from outdir import outdir
out = outdir("run_all_sivakumar") / "all_models.png"
fig.savefig(str(out), dpi=150)
plt.close()
print(f"\nPlot saved to {out}")
