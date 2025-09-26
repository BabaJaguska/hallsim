from hallsim.models import eriq
from hallsim.agents import Cell
from pytest import fixture


@fixture
def cell():
    """Fixture to create a Cell instance."""
    return Cell(coords=(0, 0))


@fixture
def eriq_model():
    """Fixture to create an Eriq model instance."""
    return eriq.ERiQ()


def test_eriq_init(eriq_model):
    """Test initialization of the ERiQ model."""
    assert eriq_model is not None


def test_eriq_call(eriq_model, cell):
    """Test calling the ERiQ model with a Cell instance."""
    t = 2.0  # Example time step
    state = cell.state.state_to_pytree()
    result = eriq_model(t, state)
    assert isinstance(
        result, dict
    )  # Assuming the model returns a dict of deltas
    assert result is not None
