from pytest import mark, fixture
from copy import deepcopy
from hallsim.hallmark_registry import (
    HALLMARK_REGISTRY,
    get_hallmark,
    list_hallmarks,
)
from hallsim.agents import Cell


@fixture
def sample_cell():
    """Fixture to create a sample cell for testing."""
    cell = Cell(
        hallmarks={
            name: get_hallmark(name, handle=0.9) for name in HALLMARK_REGISTRY
        }
    )
    return cell


def test_hallmark_retrieval():
    """Test retrieving hallmarks from the registry."""
    list_of_hallmark_names = list_hallmarks()
    assert len(list_of_hallmark_names) == len(HALLMARK_REGISTRY)
    for name in list_of_hallmark_names:
        hallmark = get_hallmark(name, handle=0.5)
        assert hallmark.name == name
        assert 0.0 <= hallmark.handle <= 1.0


@mark.parametrize("delta", [-0.3, 0.2, 0.5, -0.7])
def test_hallmark_handle_update(delta):
    """Test updating hallmark handle values."""
    hallmark = get_hallmark("Mitochondrial Dysfunction", handle=0.5)
    original_handle = hallmark.handle
    new_hallmark = hallmark.intervene(delta)
    assert 0.0 <= new_hallmark.handle <= 1.0
    expected_handle = max(0.0, min(1.0, original_handle + delta))
    assert (new_hallmark.handle - expected_handle) < 1e-6


def test_hallmark_within_cell(sample_cell):
    """Test that hallmarks are correctly associated with a cell."""
    cell = sample_cell
    for name in HALLMARK_REGISTRY:
        assert name in cell.hallmarks
        assert cell.hallmarks[name] is not None


def test_hallmark_effect_on_model_params(sample_cell):
    """Test hallmark effects on model parameters within a cell."""
    original_parameters = {}
    for model in sample_cell.models.values():
        original_parameters[model] = deepcopy(model.params)

    # Apply hallmark interventions
    sample_cell.apply_hallmarks()
    new_parameters = {}
    for model in sample_cell.models.values():
        new_parameters[model] = model.params

    # Check that parameters have changed according to hallmark effects
    print("Original Parameters:", original_parameters)
    print("New Parameters:", new_parameters)
    assert original_parameters != new_parameters
    # assert original_parameters == new_parameters  # This line is intended to fail for testing purposes
