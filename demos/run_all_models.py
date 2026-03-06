#!/usr/bin/env python
"""Demo: run all HallSim models and generate plots.

Runs:
1. ERiQ (3-process decomposition) — aging cell trajectory
2. ERiQ + SaturatingRemoval composed — aging cell + generic damage
3. SBML MAPK cascade — signaling dynamics from BioModels
4. NeuralODE — learned dynamics on a dummy oscillator
5. ERiQ with hallmark perturbation — Mitochondrial Dysfunction severity sweep

Saves plots to demos/plots/
"""

import os
import sys

# Ensure hallsim is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)


def demo_eriq():
    """Run ERiQ composite and plot key state variables."""
    print("\n" + "=" * 60)
    print("1. ERiQ Model (decomposed: Energy + OxStress + Signaling)")
    print("=" * 60)

    from hallsim.models.eriq import build_eriq_composite
    from hallsim.simulator import Simulator
    from hallsim.plotting import plot_trajectories, plot_phase_portrait

    comp = build_eriq_composite()
    sim = Simulator()

    print(f"   Processes: {list(comp.processes.keys())}")
    print(f"   Store paths: {len(comp.store_paths())} state variables")
    print("   Simulating t=[0, 5000]...")

    result = sim.run(comp, t_span=(0.0, 5000.0), dt=5.0)

    # Key variables
    key_paths = [
        "eriq/mito_function", "eriq/mito_damage", "eriq/glycolysis",
        "eriq/ROS_activity", "eriq/mTOR_activity", "eriq/p53_activity",
    ]

    fig = plot_trajectories(
        result, paths=key_paths,
        title="ERiQ: Cellular Aging Trajectory",
        figsize=(14, 5),
    )
    fig.savefig(os.path.join(PLOT_DIR, "eriq_trajectories.png"), dpi=150, bbox_inches="tight")
    print(f"   Saved: eriq_trajectories.png")

    # Subplots view
    fig = plot_trajectories(
        result, paths=key_paths,
        title="ERiQ: Individual State Variables",
        figsize=(14, 8), ncols=3,
    )
    fig.savefig(os.path.join(PLOT_DIR, "eriq_subplots.png"), dpi=150, bbox_inches="tight")
    print(f"   Saved: eriq_subplots.png")

    # Phase portrait: mito_damage vs mito_function
    fig = plot_phase_portrait(
        result, "eriq/mito_damage", "eriq/mito_function",
        title="ERiQ: Damage vs Function Phase Portrait",
    )
    fig.savefig(os.path.join(PLOT_DIR, "eriq_phase.png"), dpi=150, bbox_inches="tight")
    print(f"   Saved: eriq_phase.png")

    print(f"   Final mito_damage: {float(result.ys['eriq/mito_damage'][-1]):.4f}")
    print(f"   Final mito_function: {float(result.ys['eriq/mito_function'][-1]):.4f}")

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
    from hallsim.simulator import Simulator
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
    sim = Simulator()

    print(f"   Processes: {list(comp.processes.keys())}")
    print(f"   Store paths: {comp.store_paths()}")
    print("   Simulating t=[0, 3000]...")

    result = sim.run(comp, t_span=(0.0, 3000.0), dt=5.0)

    fig = plot_trajectories(
        result,
        paths=["eriq/mito_damage", "eriq/mito_function", "eriq/ROS_activity",
               "cell/generic_damage"],
        title="ERiQ + SaturatingRemoval: Composed Aging + Damage Model",
        figsize=(14, 5),
    )
    fig.savefig(os.path.join(PLOT_DIR, "eriq_composed.png"), dpi=150, bbox_inches="tight")
    print(f"   Saved: eriq_composed.png")

    plt.close("all")


def demo_mapk():
    """Run SBML MAPK cascade model from BioModels."""
    print("\n" + "=" * 60)
    print("3. SBML MAPK Cascade (BioModels #10, via sbmltoodejax)")
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
    from hallsim.simulator import Simulator
    from hallsim.plotting import plot_trajectories

    schema = proc.ports_schema()
    species = list(schema.keys())
    print(f"   Species ({len(species)}): {species}")

    topology = {
        "mapk": {name: f"mapk/{name}" for name in species},
    }

    comp = Composite(
        processes={"mapk": proc},
        topology=topology,
        validate=False,
    )

    sim = Simulator()
    print("   Simulating t=[0, 1000]...")
    result = sim.run(comp, t_span=(0.0, 1000.0), dt=1.0)

    # Plot all species
    all_paths = [f"mapk/{s}" for s in species]
    fig = plot_trajectories(
        result, paths=all_paths,
        title="MAPK Cascade (BioModels #10): All Species",
        figsize=(14, 6),
    )
    fig.savefig(os.path.join(PLOT_DIR, "mapk_all.png"), dpi=150, bbox_inches="tight")
    print(f"   Saved: mapk_all.png")

    # Subplots
    fig = plot_trajectories(
        result, paths=all_paths,
        title="MAPK Cascade: Individual Species",
        figsize=(16, 10), ncols=3,
    )
    fig.savefig(os.path.join(PLOT_DIR, "mapk_subplots.png"), dpi=150, bbox_inches="tight")
    print(f"   Saved: mapk_subplots.png")

    plt.close("all")


def demo_neuralode():
    """Train a NeuralODE on a simple oscillator and plot predictions."""
    print("\n" + "=" * 60)
    print("4. NeuralODE: Learn Oscillator Dynamics")
    print("=" * 60)

    from hallsim.models.neuralode import (
        NeuralODEProcess, generate_training_data, train_neuralode,
    )
    from hallsim.composite import Composite
    from hallsim.simulator import Simulator
    from hallsim.plotting import plot_trajectories, plot_phase_portrait

    # Ground truth: damped oscillator
    def oscillator(t, y, args=None):
        return jnp.stack([y[1], -y[0] - 0.1 * y[1]])

    ts_train = jnp.linspace(0, 10, 50)
    key = jax.random.PRNGKey(42)

    print("   Generating training data (damped oscillator)...")
    ts_data, ys_data = generate_training_data(
        oscillator, ts_train, n_vars=2, dataset_size=64, key=key,
    )
    print(f"   Data shape: {ys_data.shape}")

    print("   Training NeuralODE (200 steps)...")
    proc = train_neuralode(
        ts_data, ys_data,
        fields=["x", "y"],
        width=32, depth=2,
        steps=200, batch_size=32, seed=42,
    )

    # Run the trained model in a Composite
    comp = Composite(
        processes={"neural": proc},
        topology={"neural": {"x": "osc/x", "y": "osc/y"}},
        validate=False,
    )

    # Use one training IC for comparison
    y0 = {"osc/x": jnp.array(0.5), "osc/y": jnp.array(0.5)}
    sim = Simulator()

    print("   Predicting trajectory t=[0, 20]...")
    result = sim.run(comp, t_span=(0.0, 20.0), dt=0.2, y0=y0)

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
    axes[0].plot(ts_np, np.asarray(result.ys["osc/x"]), label="NeuralODE x", linewidth=2)
    axes[0].plot(ts_np, np.asarray(result.ys["osc/y"]), label="NeuralODE y", linewidth=2)
    gt_ts = np.asarray(gt_sol.ts)
    axes[0].plot(gt_ts, np.asarray(gt_sol.ys[:, 0]), "--", label="True x", alpha=0.7)
    axes[0].plot(gt_ts, np.asarray(gt_sol.ys[:, 1]), "--", label="True y", alpha=0.7)
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Value")
    axes[0].set_title("NeuralODE vs Ground Truth")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # Phase portrait
    axes[1].plot(np.asarray(result.ys["osc/x"]), np.asarray(result.ys["osc/y"]),
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
    fig.savefig(os.path.join(PLOT_DIR, "neuralode_oscillator.png"), dpi=150, bbox_inches="tight")
    print(f"   Saved: neuralode_oscillator.png")

    plt.close("all")


def demo_hallmark_sweep():
    """Sweep Mitochondrial Dysfunction severity and overlay ERiQ trajectories."""
    print("\n" + "=" * 60)
    print("5. Hallmark Severity Sweep: Mitochondrial Dysfunction")
    print("=" * 60)

    from hallsim.models.eriq import build_eriq_composite
    from hallsim.hallmarks import HALLMARK_REGISTRY
    from hallsim.composite import Composite
    from hallsim.simulator import Simulator
    import numpy as np

    severities = [0.0, 0.3, 0.6, 1.0]
    handle = HALLMARK_REGISTRY["Mitochondrial Dysfunction"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    colors = plt.cm.Reds(np.linspace(0.3, 1.0, len(severities)))

    for sev, color in zip(severities, colors):
        comp = build_eriq_composite()
        modified_procs = handle.apply(comp.processes, severity=sev)
        comp_mod = Composite(modified_procs, comp.topology, validate=False)

        sim = Simulator()
        result = sim.run(comp_mod, t_span=(0.0, 3000.0), dt=5.0)

        ts = np.asarray(result.ts)
        label = f"severity={sev:.1f}"

        axes[0].plot(ts, np.asarray(result.ys["eriq/mito_damage"]),
                     color=color, label=label, linewidth=1.5)
        axes[1].plot(ts, np.asarray(result.ys["eriq/mito_function"]),
                     color=color, label=label, linewidth=1.5)
        axes[2].plot(ts, np.asarray(result.ys["eriq/ROS_activity"]),
                     color=color, label=label, linewidth=1.5)

        print(f"   severity={sev:.1f}: final damage={float(result.ys['eriq/mito_damage'][-1]):.4f}")

    for ax, title in zip(axes, ["Mitochondrial Damage", "Mitochondrial Function", "ROS Activity"]):
        ax.set_xlabel("Time")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Hallmark Severity Sweep: Mitochondrial Dysfunction", fontsize=13)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "hallmark_sweep.png"), dpi=150, bbox_inches="tight")
    print(f"   Saved: hallmark_sweep.png")

    plt.close("all")


if __name__ == "__main__":
    print("HallSim Model Demos")
    print("Plots will be saved to demos/plots/")

    demo_eriq()
    demo_eriq_plus_damage()
    demo_mapk()
    demo_neuralode()
    demo_hallmark_sweep()

    print("\n" + "=" * 60)
    print("All demos complete! Check demos/plots/ for figures.")
    print("=" * 60)
