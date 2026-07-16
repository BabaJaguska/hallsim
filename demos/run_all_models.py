#!/usr/bin/env python
"""Demo: run all HallSim models and generate plots.

Runs:
1. ERiQ (3-process decomposition) — aging cell trajectory
2. ERiQ + SaturatingRemoval composed — aging cell + generic damage
3. SBML MAPK cascade — signaling dynamics from BioModels
4. NeuralODE — learned dynamics on a dummy oscillator
5. ERiQ with hallmark perturbation — Mitochondrial Dysfunction severity sweep
6. Stem Cell Niche — Sivakumar2011 crosstalk + niche deterioration sweep

Saves plots to demos/plots/
"""

import os
import sys
import traceback

# Ensure hallsim is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from outdir import outdir
PLOT_DIR = str(outdir("run_all_models"))

saved_plots = []


def demo_eriq():
    """Run ERiQ composite and plot key state variables."""
    print("\n" + "=" * 60)
    print("1. ERiQ Model (decomposed: Energy + OxStress + Signaling)")
    print("=" * 60)

    from hallsim.models.eriq import build_eriq_composite
    from hallsim.scheduler import Scheduler
    from hallsim.plotting import plot_trajectories, plot_phase_portrait

    comp = build_eriq_composite()
    sim = Scheduler()

    print(f"   Processes: {list(comp.processes.keys())}")
    print(f"   Store paths: {len(comp.store_paths())} state variables")
    print("   Simulating t=[0, 5000]...")

    result = sim.run(comp, t_span=(0.0, 5000.0), macro_dt=5.0, save_dt=5.0)

    # Key variables
    key_paths = [
        "eriq/mito_function", "eriq/mito_damage", "eriq/glycolysis",
        "eriq/ROS_activity", "eriq/mTOR_activity", "eriq/p53_activity",
    ]

    path = os.path.join(PLOT_DIR, "eriq_trajectories.png")
    fig = plot_trajectories(
        result, paths=key_paths,
        title="ERiQ: Cellular Aging Trajectory",
        figsize=(14, 5),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved_plots.append(path)
    print(f"   Saved: {path}")

    # Subplots view
    path = os.path.join(PLOT_DIR, "eriq_subplots.png")
    fig = plot_trajectories(
        result, paths=key_paths,
        title="ERiQ: Individual State Variables",
        figsize=(14, 8), ncols=3,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved_plots.append(path)
    print(f"   Saved: {path}")

    # Phase portrait: mito_damage vs mito_function
    path = os.path.join(PLOT_DIR, "eriq_phase.png")
    fig = plot_phase_portrait(
        result, "eriq/mito_damage", "eriq/mito_function",
        title="ERiQ: Damage vs Function Phase Portrait",
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved_plots.append(path)
    print(f"   Saved: {path}")

    print(f"   Final mito_damage: {float(result.get('eriq/mito_damage')[-1]):.4f}")
    print(f"   Final mito_function: {float(result.get('eriq/mito_function')[-1]):.4f}")

    plt.close("all")
    return result


def demo_eriq_plus_damage():
    """Compose ERiQ with SaturatingRemoval — two models sharing a cell."""
    print("\n" + "=" * 60)
    print("2. ERiQ + SaturatingRemoval (composed)")
    print("=" * 60)

    from hallsim.models.eriq import (
        ERiQEnergyMetabolism, ERiQOxidativeStress, ERiQSignaling,
    )
    from hallsim.models.saturating_removal import SaturatingRemoval
    from hallsim.composite import Composite
    from hallsim.scheduler import Scheduler
    from hallsim.plotting import plot_trajectories

    # Build ERiQ processes + a SaturatingRemoval process
    processes = {
        "energy": ERiQEnergyMetabolism(),
        "oxidative_stress": ERiQOxidativeStress(),
        "signaling": ERiQSignaling(),
        "damage_repair": SaturatingRemoval(eta=0.5, beta=1.0, K=0.1, tau_scale=0.001),
    }

    p = "eriq"
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
        "damage_repair": {
            "damage": "cell/generic_damage",
        },
    }

    comp = Composite(processes, topology)
    sim = Scheduler()

    print(f"   Processes: {list(comp.processes.keys())}")
    print(f"   Store paths: {comp.store_paths()}")
    print("   Simulating t=[0, 3000]...")

    result = sim.run(comp, t_span=(0.0, 3000.0), macro_dt=5.0, save_dt=5.0)

    path = os.path.join(PLOT_DIR, "eriq_composed.png")
    fig = plot_trajectories(
        result,
        paths=["eriq/mito_damage", "eriq/mito_function", "eriq/ROS_activity",
               "cell/generic_damage"],
        title="ERiQ + SaturatingRemoval: Composed Aging + Damage Model",
        figsize=(14, 5),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved_plots.append(path)
    print(f"   Saved: {path}")

    plt.close("all")


def demo_mapk():
    """Run SBML MAPK cascade model through the composable framework."""
    print("\n" + "=" * 60)
    print("3. SBML MAPK Cascade (BioModels #10, Kholodenko2000)")
    print("=" * 60)

    try:
        from hallsim.sbml_import import process_from_sbml
    except ImportError as e:
        print(f"   SKIPPED: {e}")
        return

    try:
        proc = process_from_sbml(10, name="mapk_cascade")
    except Exception as e:
        print(f"   SKIPPED (load error): {e}")
        return

    from hallsim.composite import Composite
    from hallsim.scheduler import Scheduler
    from hallsim.plotting import plot_trajectories

    schema = proc.ports_schema()
    species = list(schema.keys())
    print(f"   Species ({len(species)}): {species}")

    topology = {"mapk": {name: f"mapk/{name}" for name in species}}

    comp = Composite(
        processes={"mapk": proc},
        topology=topology,
        validate=False,
    )

    # Scheduler's default Kvaerno5 (implicit ESDIRK) is well-suited to
    # the mildly stiff Kholodenko cascade. Moderate tolerances preserve
    # the limit cycle without the ppm-level local-error budget that
    # tight tolerances (e.g. atol=1e-12) require, which would push step
    # count above the default budget.
    n_secs = 150 * 60
    sim = Scheduler(rtol=1e-6, atol=1e-9, dt0=1e-3)
    print(f"   Simulating t=[0, {n_secs}] ({n_secs/60:.0f} min)...")
    result = sim.run(
        comp, t_span=(0.0, float(n_secs)), macro_dt=10.0, save_dt=5.0
    )

    # Plot key species
    all_paths = [f"mapk/{s}" for s in species]
    path = os.path.join(PLOT_DIR, "mapk_all.png")
    fig = plot_trajectories(
        result, paths=all_paths,
        title="MAPK Cascade (Kholodenko2000, BioModels #10)",
        figsize=(14, 6),
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved_plots.append(path)
    print(f"   Saved: {path}")

    # Subplots
    path = os.path.join(PLOT_DIR, "mapk_subplots.png")
    fig = plot_trajectories(
        result, paths=all_paths,
        title="MAPK Cascade: Individual Species",
        figsize=(16, 10), ncols=4,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved_plots.append(path)
    print(f"   Saved: {path}")

    plt.close("all")


def demo_neuralode():
    """Train a NeuralODE on a simple oscillator and plot predictions."""
    print("\n" + "=" * 60)
    print("4. NeuralODE: Learn Oscillator Dynamics")
    print("=" * 60)

    from hallsim.models.neuralode import (
        simulate_conditioned, fit_neuralode_shooting,
    )
    from hallsim.composite import Composite
    from hallsim.scheduler import Scheduler
    from hallsim.plotting import plot_trajectories, plot_phase_portrait

    # Ground truth: damped oscillator
    def oscillator(t, y, args=None):
        return jnp.stack([y[1], -y[0] - 0.1 * y[1]])

    ts_data = jnp.linspace(0, 10, 50)
    key = jax.random.PRNGKey(42)

    print("   Generating training data (damped oscillator)...")
    ys_data, _ = simulate_conditioned(
        lambda u: oscillator, ts_data, jnp.zeros((1, 1)), n_ics=64,
        y0_range=(0.1, 1.0), key=key,
    )
    print(f"   Data shape: {ys_data.shape}")

    print("   Training NeuralODE (multiple shooting, 1000 steps)...")
    proc = fit_neuralode_shooting(
        ts_data, ys_data,
        fields=("x", "y"),
        segments=4,
        width=64, depth=3,
        steps=1000, batch_size=32, seed=42,
    )

    # Run the trained model in a Composite
    comp = Composite(
        processes={"neural": proc},
        topology={"neural": {"x": "osc/x", "y": "osc/y"}},
        validate=False,
    )

    # Use one training IC for comparison
    keys = comp.store_keys()
    y0 = comp.initial_state_vec(keys)
    y0 = y0.at[keys.index("osc/x")].set(0.5).at[keys.index("osc/y")].set(0.5)
    sim = Scheduler()

    print("   Predicting trajectory t=[0, 20]...")
    result = sim.run(comp, t_span=(0.0, 20.0), macro_dt=0.2, save_dt=0.2, y0=y0)

    # Also generate ground truth for comparison
    import diffrax as dfx
    gt_sol = dfx.diffeqsolve(
        dfx.ODETerm(oscillator), dfx.Tsit5(),
        t0=0.0, t1=20.0, dt0=0.1,
        y0=jnp.array([0.5, 0.5]),
        saveat=dfx.SaveAt(ts=jnp.arange(0, 20.0, 0.2)),
        stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-6),
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time series
    ts_np = np.asarray(result.ts)
    axes[0].plot(ts_np, np.asarray(result.get("osc/x")), label="NeuralODE x", linewidth=2)
    axes[0].plot(ts_np, np.asarray(result.get("osc/y")), label="NeuralODE y", linewidth=2)
    gt_ts = np.asarray(gt_sol.ts)
    axes[0].plot(gt_ts, np.asarray(gt_sol.ys[:, 0]), "--", label="True x", alpha=0.7)
    axes[0].plot(gt_ts, np.asarray(gt_sol.ys[:, 1]), "--", label="True y", alpha=0.7)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    axes[0].set_title("NeuralODE vs Ground Truth")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Phase portrait
    axes[1].plot(np.asarray(result.get("osc/x")), np.asarray(result.get("osc/y")),
                 label="NeuralODE", linewidth=2)
    axes[1].plot(np.asarray(gt_sol.ys[:, 0]), np.asarray(gt_sol.ys[:, 1]),
                 "--", label="True", alpha=0.7)
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    axes[1].set_title("Phase Portrait")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("NeuralODE: Learned Damped Oscillator", fontsize=13)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "neuralode_oscillator.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved_plots.append(path)
    print(f"   Saved: {path}")

    plt.close("all")


def demo_hallmark_sweep():
    """Sweep Mitochondrial Dysfunction severity and overlay ERiQ trajectories."""
    print("\n" + "=" * 60)
    print("5. Hallmark Severity Sweep: Mitochondrial Dysfunction")
    print("=" * 60)

    from hallsim.models.eriq import build_eriq_composite
    from hallsim.hallmarks import HALLMARK_REGISTRY
    from hallsim.composite import Composite
    from hallsim.scheduler import Scheduler

    severities = [0.0, 0.3, 0.6, 1.0]
    handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(severities)))

    for sev, color in zip(severities, colors):
        comp = build_eriq_composite()
        modified_procs = handle.apply(comp.processes, severity=sev)
        comp_mod = Composite(modified_procs, comp.topology, validate=False)

        sim = Scheduler()
        result = sim.run(comp_mod, t_span=(0.0, 3000.0), macro_dt=5.0, save_dt=5.0)

        ts = np.asarray(result.ts)
        label = f"severity={sev:.1f}"

        axes[0].plot(ts, np.asarray(result.get("eriq/mito_damage")),
                     color=color, label=label, linewidth=1.5)
        axes[1].plot(ts, np.asarray(result.get("eriq/mito_function")),
                     color=color, label=label, linewidth=1.5)
        axes[2].plot(ts, np.asarray(result.get("eriq/ROS_activity")),
                     color=color, label=label, linewidth=1.5)

        print(f"   severity={sev:.1f}: final damage={float(result.get('eriq/mito_damage')[-1]):.4f}")

    for ax, title in zip(axes, ["Mitochondrial Damage", "Mitochondrial Function", "ROS Activity"]):
        ax.set_xlabel("Time")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hallmark Severity Sweep: Mitochondrial Dysfunction", fontsize=13)
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "hallmark_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved_plots.append(path)
    print(f"   Saved: {path}")

    plt.close("all")


def demo_stem_cell_niche():
    """Sweep niche deterioration severity on Sivakumar2011 crosstalk model."""
    print("\n" + "=" * 60)
    print("6. Stem Cell Niche: Sivakumar2011 Crosstalk + Niche Decay")
    print("=" * 60)

    try:
        from hallsim.models.stem_cell_niche import (
            build_niche_crosstalk,
            CROSSTALK_WNT, CROSSTALK_EGF, CROSSTALK_SHH, CROSSTALK_NOTCH,
        )
    except ImportError as e:
        print(f"   SKIPPED: {e}")
        return

    from hallsim.scheduler import Scheduler
    import diffrax as dfx

    severities = [0.0, 0.3, 0.6, 1.0]
    sim = Scheduler(solver=dfx.Tsit5(), rtol=1e-6, atol=1e-8, max_steps=500_000, dt0=1e-4)

    ligands = {
        CROSSTALK_WNT: "Wnt",
        CROSSTALK_EGF: "EGF",
        CROSSTALK_SHH: "Shh",
        CROSSTALK_NOTCH: "Notch",
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    colors = plt.cm.Blues(np.linspace(0.3, 1.0, len(severities)))

    for sev, color in zip(severities, colors):
        print(f"   severity={sev:.1f}...")
        try:
            comp = build_niche_crosstalk(severity=sev)
            result = sim.run(comp, t_span=(0.0, 100.0), macro_dt=0.5, save_dt=0.5)

            ts = np.asarray(result.ts)
            for ax, (species_id, name) in zip(axes, ligands.items()):
                vals = np.asarray(result.get(species_id))
                ax.plot(ts, vals, color=color, label=f"sev={sev:.1f}",
                        linewidth=1.5)
                ax.set_title(f"{name} ({species_id})", fontsize=11)
                ax.set_xlabel("Time")
                ax.set_ylabel("Concentration")
                ax.grid(True, alpha=0.3)

            print(f"   severity={sev:.1f}: final Wnt={float(result.get(CROSSTALK_WNT)[-1]):.3f}")
        except Exception as e:
            print(f"   severity={sev:.1f} FAILED: {e}")

    for ax in axes:
        ax.legend(fontsize=8)

    fig.suptitle(
        "Stem Cell Exhaustion: Niche Deterioration Severity Sweep\n"
        "(Sivakumar2011 crosstalk model + StemCellNiche process)",
        fontsize=13,
    )
    fig.tight_layout()
    path = os.path.join(PLOT_DIR, "stem_cell_niche_sweep.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    saved_plots.append(path)
    print(f"   Saved: {path}")

    plt.close("all")


if __name__ == "__main__":
    print("HallSim Model Demos")
    print(f"Plots will be saved to {os.path.abspath(PLOT_DIR)}")

    demos = [
        ("ERiQ", demo_eriq),
        ("ERiQ + Damage", demo_eriq_plus_damage),
        ("MAPK", demo_mapk),
        ("NeuralODE", demo_neuralode),
        ("Hallmark Sweep", demo_hallmark_sweep),
        ("Stem Cell Niche", demo_stem_cell_niche),
    ]

    for name, fn in demos:
        try:
            fn()
        except Exception:
            print(f"\n   *** {name} demo FAILED ***")
            traceback.print_exc()

    print("\n" + "=" * 60)
    if saved_plots:
        print(f"All demos complete! {len(saved_plots)} plots saved to:")
        for p in saved_plots:
            print(f"  {p}")
    else:
        print("WARNING: No plots were generated. Check errors above.")
    print("=" * 60)
