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
    n_steps: int = 50, dt: float = 0.5, keep_trajectory: bool = True
):
    """Run a basic simulation for a number of steps."""
    logger.info(f"Starting basic simulation for {n_steps} steps at dt={dt}.")
    logger.info(f"Saving trajectory is set to {keep_trajectory}.")
    t0 = 0.0
    t1 = n_steps * dt
    cell = Cell(coords=(0, 0))
    logger.info(f"Initial cell state: {cell}")
    cell.step(t0, t1, dt, keep_trajectory=keep_trajectory)

    # logger.info(f"Final cell state after {n_steps} steps: {cell}")

    attributes_to_plot = ["mito_damage", "mito_function",
                          "p53_activity", "ROS_activity",
                          "mito_enzymes", "glycolysis", "mTOR_activity"]
    x_to_plot, y_to_plot = "mito_damage", "mito_function"
    x1_to_plot, y1_to_plot = "p53_activity", "ROS_activity"
   
    if keep_trajectory:
        plt.figure(figsize=(18, 9))
        plt.subplot(2,3,1)
        time_points = [i * dt for i in range(n_steps + 1)]
        for attr in attributes_to_plot:
            if hasattr(cell.state, attr):
                values = getattr(cell.state, attr)
                plt.plot(time_points, values, label=attr)
        plt.xlabel("Time (step units)")
        plt.ylabel("Levels")
        plt.title("Cell State Trajectories")
        plt.legend()
        plt.grid()
        plt.subplot(2,3,2)
        plt.plot(getattr(cell.state, x_to_plot), getattr(cell.state, y_to_plot), marker='o')
        plt.xlabel(x_to_plot)
        plt.ylabel(y_to_plot)
        plt.title(f"{y_to_plot} vs {x_to_plot}")
        plt.grid()
        plt.subplot(2,3,3)
        plt.plot(getattr(cell.state, x1_to_plot), getattr(cell.state, y1_to_plot), marker='o', color='orange')
        plt.xlabel(x1_to_plot)
        plt.ylabel(y1_to_plot)
        plt.title(f"{y1_to_plot} vs {x1_to_plot}")
        plt.grid()
        plt.subplot(2,3,4)
        plt.plot(getattr(cell.state, "mito_function"), getattr(cell.state, "glycolysis"), marker='o', color='green')
        plt.xlabel("mito_function")
        plt.ylabel("glycolysis")
        plt.title("glycolysis vs mito_function")
        plt.grid()
        plt.subplot(2,3,5)
        plt.plot(getattr(cell.state, "mTOR_activity"), getattr(cell.state, "glycolysis"), marker='o', color='red')
        plt.xlabel("mTOR_activity")
        plt.ylabel("glycolysis")
        plt.title("glycolysis vs mTOR_activity")
        plt.grid()
        plt.subplot(2,3,6)
        plt.plot(getattr(cell.state, "mito_function"), getattr(cell.state, "mito_enzymes"), marker='o', color='purple')
        plt.xlabel("mito_function")
        plt.ylabel("mito_enzymes")
        plt.title("mito_enzymes vs mito_function")
        plt.grid()
        plt.savefig("cell_trajectory.png")
        plt.show()
    else:
        logger.warning("Trajectory not kept; no plots to display.")


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
