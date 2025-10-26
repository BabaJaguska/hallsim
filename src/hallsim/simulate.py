from hallsim.agents import Cell
from math import isqrt
import logging
import click
import os
from matplotlib import pyplot as plt

logfile = "logs/hallsim.log"
if not os.path.exists("logs"):
    os.makedirs("logs")
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Also log to console


def simulate_basic(
    n_steps: int = 250, dt: float = 0.5, keep_trajectory: bool = True
):
    """Run a basic simulation for a number of steps."""
    logger.info(f"Starting basic simulation for {n_steps} steps at dt={dt}.")
    logger.info(f"Saving trajectory is set to {keep_trajectory}.")
    t0 = 0.0
    t1 = n_steps * dt
    cell = Cell(coords=(0, 0))
    logger.info(f"Initial cell state: {cell}")
    ts, ys = cell.integrate(t0, t1, dt, keep_trajectory=keep_trajectory)
    cell.evolve(ts, ys)
    # logger.info(f"Final cell state after {n_steps} steps: {cell}")
    if keep_trajectory:
        plot_trajectory_and_relations(cell, n_steps, dt)
    else:
        logger.info("Trajectory not saved; skipping plot.")


def simulate_with_kick(
    n_steps: int = 1000,
    dt: float = 0.5,
    kick_step: int = 500,
    keep_trajectory: bool = True,
):
    """Run a simulation with a perturbation (kick) at a specified step."""
    logger.info(
        f"Starting simulation with kick at step {kick_step} for {n_steps} steps at dt={dt}."
    )
    logger.info(f"Saving trajectory is set to {keep_trajectory}.")
    t0 = 0.0
    t1 = n_steps * dt
    tk = kick_step * dt
    cell = Cell(coords=(0, 0))
    kick_dict = {
        "mito_function": -0.3,
    }

    logger.info(f"Initial cell state: {cell}")
    ts, ys = cell.integrate_with_kick(
        t0, t1, tk, kick_dict, dt, keep_trajectory=keep_trajectory
    )
    cell.evolve(ts, ys)
    # logger.info(f"Final cell state after {n_steps} steps: {cell}")
    if keep_trajectory:
        plot_trajectory(cell, n_steps, dt)
    else:
        logger.info("Trajectory not saved; skipping plot.")


def plot_trajectory(cell: Cell, n_steps: int, dt: float):
    attributes_to_plot = [
        "mito_damage",
        "mito_function",
        # "p53_activity",
        "ROS_activity",
        "mito_enzymes",
        "glycolysis",
        # "mTOR_activity",
        # "test_field1",
        # "test_field2",
    ]

    plt.figure(figsize=(18, 9))
    time_points = [i * dt for i in range(n_steps + 1)]
    for attr in attributes_to_plot:
        if hasattr(cell.state, attr):
            values = getattr(cell.state, attr)
            plt.plot(time_points, values, label=attr)
        else:
            logger.warning(
                f"Cell state does not have attribute '{attr}' to plot."
            )
    plt.xlabel("Time (step units)")
    plt.ylabel("Levels")
    plt.title("Cell State Trajectories")
    plt.legend()
    plt.grid()
    plt.show()


def plot_trajectory_and_relations(cell: Cell, n_steps: int, dt: float):
    attributes_to_plot = [
        "mito_damage",
        "mito_function",
        # "p53_activity",
        "ROS_activity",
        "mito_enzymes",
        "glycolysis",
        # "mTOR_activity",
        "test_field1",
        "test_field2",
    ]
    x_to_plot, y_to_plot = "mito_damage", "mito_function"
    x1_to_plot, y1_to_plot = "p53_activity", "ROS_activity"

    plt.figure(figsize=(18, 9))
    plt.subplot(2, 3, 1)
    time_points = [i * dt for i in range(n_steps + 1)]
    for attr in attributes_to_plot:
        if hasattr(cell.state, attr):
            values = getattr(cell.state, attr)
            if attr == "test_field1" or attr == "test_field2":
                plt.plot(
                    time_points,
                    values + 1.2,
                    label=attr,
                    linestyle="--",
                )
            else:
                plt.plot(time_points, values, label=attr)
    plt.xlabel("Time (step units)")
    plt.ylabel("Levels")
    plt.title("Cell State Trajectories")
    plt.legend()
    plt.grid()
    plt.subplot(2, 3, 2)
    plt.plot(
        getattr(cell.state, x_to_plot),
        getattr(cell.state, y_to_plot),
        marker="o",
    )
    plt.xlabel(x_to_plot)
    plt.ylabel(y_to_plot)
    plt.title(f"{y_to_plot} vs {x_to_plot}")
    plt.grid()
    plt.subplot(2, 3, 3)
    plt.plot(
        getattr(cell.state, x1_to_plot),
        getattr(cell.state, y1_to_plot),
        marker="o",
        color="orange",
    )
    plt.xlabel(x1_to_plot)
    plt.ylabel(y1_to_plot)
    plt.title(f"{y1_to_plot} vs {x1_to_plot}")
    plt.grid()
    plt.subplot(2, 3, 4)
    plt.plot(
        getattr(cell.state, "mito_function"),
        getattr(cell.state, "glycolysis"),
        marker="o",
        color="green",
    )
    plt.xlabel("mito_function")
    plt.ylabel("glycolysis")
    plt.title("glycolysis vs mito_function")
    plt.grid()
    plt.subplot(2, 3, 5)
    plt.plot(
        getattr(cell.state, "mTOR_activity"),
        getattr(cell.state, "glycolysis"),
        marker="o",
        color="red",
    )
    plt.xlabel("mTOR_activity")
    plt.ylabel("glycolysis")
    plt.title("glycolysis vs mTOR_activity")
    plt.grid()
    plt.subplot(2, 3, 6)
    plt.plot(
        getattr(cell.state, "mito_function"),
        getattr(cell.state, "mito_enzymes"),
        marker="o",
        color="purple",
    )
    plt.xlabel("mito_function")
    plt.ylabel("mito_enzymes")
    plt.title("mito_enzymes vs mito_function")
    plt.grid()
    plt.savefig("cell_trajectory.png")
    plt.show()


def simulate_dummy(num_cells: int = 16):
    """Simulate the creation of cells and their AMPK levels."""
    logger.info(f"Simulating with {num_cells} cells.")
    remaining_cells = 0
    n_rows = isqrt(num_cells)
    n_cols = num_cells // n_rows if n_rows != 0 else num_cells

    if n_rows * n_cols < num_cells:
        remaining_cells = num_cells - (n_rows * n_cols)

    cells = [Cell(coords=(x, y)) for x in range(n_rows) for y in range(n_cols)]

    if remaining_cells:
        cells += [Cell(coords=(n_rows, y)) for y in range(remaining_cells)]
        n_rows += 1

    n_cells = len(cells)
    logger.info(
        f"Created {n_cells} cells in a grid of {n_rows} rows and {n_cols} columns."
    )

    click.echo(
        f"Grid representation of cells:\n{_ascii_grid(n_rows, n_cols, n_cells)}"
    )


def _ascii_grid(rows: int, cols: int, total: int) -> str:
    dots = ["." * min(cols, total - r * cols) for r in range(rows)]
    return "\n".join(dots)
