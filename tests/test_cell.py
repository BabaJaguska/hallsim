from pytest import fixture, mark

from hallsim.agents import Cell


@fixture
def cell():
    return Cell(0, 0)


@mark.parametrize("x, y", [(1, 2), (3, 4), (5, 6)])
def test_cell_initialization(x, y):
    cell = Cell(x, y)
    assert cell.x == x
    assert cell.y == y
    assert cell.AMPK == 0


def test_cell_repr(cell):
    assert repr(cell) == "Cell(0, 0)"


def test_cell_increment_AMPK(cell):
    cell.increment_AMPK()
    assert cell.AMPK == 1
