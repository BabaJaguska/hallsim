"""Repeated standalone Proctor 2007 Gillespie trajectories.

Run via the CLI: simulate proctor2007-ssa
The bundled SBML describes proteasome inhibition (k69=0); by default this
uses its documented normal-condition value k69=1e-3. All other parameters
and initial counts are identical across runs; only the random seed changes.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from hallsim.io import outdir
from hallsim.sbml_import import process_from_sbml
from hallsim.stochastic import simulate_ssa

SBML_PATH = (
    Path(__file__).parent
    / "models"
    / "sbml"
    / "proctor2007"
    / "proctor2007_BIOMD0000000105.xml"
)
SPECIES = ("NatP", "MisP", "Ub", "Proteasome", "AggP", "SeqAggP")


def run(
    *,
    runs=6,
    hours=24.0,
    samples=289,
    seed=0,
    max_events=2_000_000,
    inhibited=False,
    output=None,
):
    """Save an overlay figure and the full sampled ensemble; return its path."""
    if runs < 2 or samples < 2 or not np.isfinite(hours) or hours <= 0:
        raise ValueError(
            "need runs >= 2, samples >= 2, and positive finite hours"
        )
    process = process_from_sbml(
        str(SBML_PATH),
        name="proctor2007",
        native_time_seconds=1.0,
        parameters={"k69": 0.0 if inhibited else 1e-3},
    ).as_stochastic()
    names = tuple(process._species_names)
    trajectories, counts = [], []
    for i in range(runs):
        result = simulate_ssa(
            process,
            t_span=(0.0, hours * 3600),
            save_dt=hours * 3600 / (samples - 1),
            seed=seed + i,
            max_events=max_events,
        )
        states = np.asarray(result.states)
        if not np.all(
            np.isfinite(states) & (states >= 0) & (states == np.floor(states))
        ):
            raise RuntimeError("trajectory contains invalid molecule counts")
        trajectories.append(states)
        counts.append(result.reaction_indices.size)
        print(
            f"Run {i + 1}/{runs}, seed={seed + i}: {counts[-1]:,} reactions",
            flush=True,
        )
    ensemble = np.stack(trajectories)
    times = np.asarray(result.times) / 3600
    output = outdir("proctor2007_ssa") if output is None else Path(output)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "trajectories.npz",
        times_hours=times,
        states=ensemble,
        species=np.asarray(names),
        seeds=np.arange(seed, seed + runs),
        event_counts=counts,
        k69=float(process.parameters["k69"]),
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    for ax, name in zip(axes.flat, SPECIES):
        for i, states in enumerate(ensemble):
            ax.step(
                times,
                states[:, names.index(name)],
                where="post",
                alpha=0.75,
                linewidth=1,
                label=f"Seed {seed + i}",
            )
        ax.set(title=name, ylabel="Molecule count")
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Time (hours)")
    axes[0, 0].legend(fontsize=8, ncol=2)
    condition = "inhibited" if inhibited else "normal"
    fig.suptitle(f"Proctor 2007 — {runs} independent SSA runs ({condition})")
    fig.tight_layout()
    path = output / "trajectories.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f"Saved {path}")
    return path
