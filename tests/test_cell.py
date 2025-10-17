from pytest import fixture, mark
from hallsim.agents import Cell
from copy import deepcopy


@fixture
def cell():
    return Cell(coords=(0, 0))


@mark.parametrize("coords", [(0, 0), (1, 2), (-1, -2), (3.5, 4.5)])
def test_cell_initialization(coords):
    cell = Cell(coords=coords)
    assert all(cell.coords == coords)
    assert cell.state is not None
    assert hasattr(cell, "coord_names")


def test_cell_repr(cell):
    assert repr(cell) != ""


def test_cell_integrate(cell):
    initial_state = deepcopy(cell.state)
    ts, ys = cell.integrate(t0=0.0, t1=10.0, keep_trajectory=False)
    cell.evolve(ts, ys)
    print(cell.state)  # so this is now arrays for each key
    assert hasattr(cell, "integrate")  # Ensure step method exists
    assert cell.state != initial_state  # State should change after step


def test_cell_step_trajectory(cell):
    ts, ys = cell.integrate(t0=0.0, t1=10.0, keep_trajectory=True)
    cell.evolve(ts, ys)
    cell_state = cell.state
    # assure the states are vectors of the same size
    lengths = [
        len(getattr(cell_state, key))
        for key in cell_state.__dataclass_fields__
    ]
    assert all(
        length == lengths[0] for length in lengths
    ), "All state attributes should have the same length"
