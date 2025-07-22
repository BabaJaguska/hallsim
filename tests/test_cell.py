from pytest import fixture, mark

from hallsim.agents import Cell


@fixture
def cell():
    return Cell(coords=(0, 0))


@mark.parametrize("coords", [(0, 0), (1, 2), (-1, -2), (3.5, 4.5)])
def test_cell_initialization(coords):
    cell = Cell(coords=coords)
    assert all(cell.coords == coords)
    assert cell.state is not None
    assert hasattr(cell, 'coord_names')


def test_cell_repr(cell):
    assert repr(cell) != ""
