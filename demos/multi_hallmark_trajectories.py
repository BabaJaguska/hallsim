"""Plot the multi-hallmark composite's reporter trajectories across arms.

Runs the three experimental arms (ctrl / DDIS / DDIS+rapa) over the full
canonical day axis and plots each gene-reporter observable so the
directional response — especially NFKBIA (IκBα transcript) dropping with
rapamycin — is visible as a trajectory, not just an endpoint.

Figure written to ``outputs/subsystem_diagnostics/composite_trajectories.png``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from hallsim.composite import Composite  # noqa: E402
from hallsim.hallmarks import apply_hallmarks  # noqa: E402
from hallsim.models.multi_hallmark import (  # noqa: E402
    build_multi_hallmark_composite,
)
from hallsim.scheduler import Scheduler  # noqa: E402

ARMS = [
    (0.0, 0.5, "ctrl", "tab:green"),
    (1.0, 1.0, "DDIS", "tab:red"),
    (1.0, 0.3, "DDIS+rapa", "tab:blue"),
]

# (store path, gene reporter, title)
PANELS = [
    ("dp14/DNA_damage", "—", "DNA damage"),
    ("dp14/CDKN1A", "CDKN1A", "p21 / CDKN1A"),
    ("dp14/mTORC1_pS2448", "EIF4EBP1", "mTORC1 (active)"),
    ("nfkb/IkBat", "NFKBIA", "IκBα transcript (NFKBIA)"),
    ("gz06/x", "DDB2", "p53 (GZ06 x)"),
    ("dp14/ROS", "HMOX1", "ROS"),
]

OUT = Path(__file__).parent.parent / "outputs" / "subsystem_diagnostics"


def run(gi, dns):
    base = build_multi_hallmark_composite()
    procs = apply_hallmarks(
        base.processes,
        {"Genomic Instability": gi, "Deregulated Nutrient Sensing": dns},
    )
    comp = Composite(
        processes=procs,
        topology=base.topology,
        validate=False,
        semantic_validation={"check_semantics": False},
    )
    return Scheduler().run(
        comp,
        t_span=(0.0, 50.0),
        macro_dt=5.0,
        y0=comp.initial_state_vec(),
        save_dt=1.0,
    )


def main():
    runs = {label: run(gi, dns) for gi, dns, label, _ in ARMS}
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, (path, gene, title) in zip(axes.flat, PANELS):
        for gi, dns, label, color in ARMS:
            res = runs[label]
            ax.plot(
                np.asarray(res.ts),
                np.asarray(res.get(path)),
                label=label,
                color=color,
                lw=1.6,
            )
        gene_tag = f"  [{gene}]" if gene != "—" else ""
        ax.set_title(f"{title}{gene_tag}")
        ax.set_xlabel("time (days)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle(
        "Multi-hallmark composite — reporter trajectories across arms "
        "(canonical day axis, rtol=1e-6)",
        fontsize=13,
    )
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "composite_trajectories.png"
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
