import click
from hallsim.agents import Cell
from math import isqrt


def ascii_grid(rows: int, cols: int, total: int) -> str:
    dots = ["." * min(cols, total - r * cols) for r in range(rows)]
    return "\n".join(dots)


@click.command()
@click.option(
    "--num-cells", "-n", default=16, help="Number of cells to create"
)
def simulate(num_cells):
    """Simulate the creation of cells and their AMPK levels."""
    print(f"Simulating with {num_cells} cells.")
    remaining_cells = 0
    n_rows = isqrt(num_cells)
    n_cols = num_cells // n_rows if n_rows != 0 else num_cells

    if n_rows * n_cols < num_cells:
        remaining_cells = num_cells - (n_rows * n_cols)

    cells = [Cell(x, y) for x in range(n_rows) for y in range(n_cols)]

    if remaining_cells:
        cells += [Cell(n_rows, y) for y in range(remaining_cells)]
        n_rows += 1

    n_cells = len(cells)
    print(
        f"Created {n_cells} cells in a grid of {n_rows} rows and {n_cols} columns."
    )

    print(ascii_grid(n_rows, n_cols, n_cells))
