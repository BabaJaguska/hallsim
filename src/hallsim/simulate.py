from hallsim.agents import Cell
from math import isqrt
import logging
import click

logfile = "logs/hallsim.log"
logging.basicConfig(
    filename=logfile,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())  # Also log to console


def simulate_basic(n_steps: int = 5):
    """Run a basic simulation for a number of steps."""
    logger.info(f"Starting basic simulation for {n_steps} steps.")
    cell = Cell(coords=(0, 0))
    for step in range(n_steps):
        logger.info("Step {step + 1}:")
        logger.info(cell)
        cell.step(step)


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

    click.echo(f"Grid representation of cells:\n{_ascii_grid(n_rows, n_cols, n_cells)}")


def _ascii_grid(rows: int, cols: int, total: int) -> str:
    dots = ["." * min(cols, total - r * cols) for r in range(rows)]
    return "\n".join(dots)
