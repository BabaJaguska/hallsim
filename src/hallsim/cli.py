import click
from hallsim.simulate import simulate_basic, simulate_dummy


@click.group()
def simulate():
    """Group for simulation commands."""
    pass


@simulate.command()
@click.option(
    "--num-cells", "-n", default=16, help="Number of cells to create"
)
def dummy(num_cells):
    simulate_dummy(num_cells)


@simulate.command()
@click.option(
    "--num_steps", "-n", default=5, help="Number of simulation steps"
)
def basic(num_steps):
    simulate_basic(num_steps)


@simulate.command()
def lol():
    """A placeholder command for future use."""
    print("LOL command executed. This is a placeholder.")
