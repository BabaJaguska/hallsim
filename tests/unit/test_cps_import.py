"""COPASI ``.cps`` import — conversion at the boundary.

A paper's supplement is often a `.cps`; COPASI exports SBML natively, so the
translation is borrowed rather than reimplemented and everything downstream
sees an ordinary SBML import.

Marked `network` only because the COPASI bindings are an optional extra, not
because anything here touches a network.
"""

import pytest

from hallsim.cps_import import cps_to_sbml, is_cps

COPASI = pytest.importorskip("COPASI", reason="needs the 'copasi' extra")


@pytest.fixture(scope="module")
def cps_file(tmp_path_factory):
    """A two-reaction model, written by COPASI itself."""
    dm = COPASI.CRootContainer.addDatamodel()
    m = dm.getModel()
    m.setObjectName("fixture")
    m.createCompartment("cell", 1.0)
    m.createMetabolite("A", "cell", 10.0)
    m.createMetabolite("B", "cell", 0.0)
    m.createReaction("prod").setReactionScheme("A -> B")
    m.createReaction("deg").setReactionScheme("B ->")
    m.compileIfNecessary()
    path = tmp_path_factory.mktemp("cps") / "fixture.cps"
    assert dm.saveModel(str(path), True)
    return path


def test_is_cps_is_by_extension():
    assert is_cps("model.cps") and is_cps("MODEL.CPS")
    assert not is_cps("model.xml") and not is_cps(42)


def test_converts_to_sbml_that_imports(cps_file):
    """COPASI's export carries an RDF creation history libsbml reads back
    as a warning; the importer takes the file as it is."""
    import libsbml

    from hallsim.sbml_import import process_from_sbml

    out = cps_to_sbml(cps_file)
    assert libsbml.readSBML(out).getModel().getNumReactions() == 2
    assert len(process_from_sbml(out).reaction_channels()) == 2


def test_conversion_is_cached_on_content(cps_file):
    first = cps_to_sbml(cps_file)
    assert cps_to_sbml(cps_file) == first, "same bytes, same cached path"


def test_a_missing_file_says_so(tmp_path):
    with pytest.raises(FileNotFoundError):
        cps_to_sbml(tmp_path / "absent.cps")


def test_process_from_sbml_routes_a_cps_path(cps_file):
    """The point of the whole module: downstream needs no special case."""
    from hallsim.sbml_import import process_from_sbml

    proc = process_from_sbml(str(cps_file), name="fixture")
    assert set(proc.ports_schema()) == {"A", "B"}


def test_a_converted_model_triages(cps_file):
    from hallsim.intake import triage_process
    from hallsim.sbml_import import process_from_sbml

    verdict = triage_process(
        process_from_sbml(str(cps_file), name="fixture"), 10.0, name="fixture"
    )
    assert verdict.status in ("pass", "flag")
    assert verdict.n_species == 2
