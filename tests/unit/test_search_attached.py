"""The data a model ships beside itself: PEtab problems, COPASI fitting
experiments, a BioModels deposit's own tables."""

import pandas as pd
import pytest

from hallsim.search import attached
from hallsim.search.datasets import Design


def test_a_petab_measurement_table_states_the_design():
    measurements = pd.DataFrame(
        {
            "observableId": ["pSTAT5A_rel"] * 4 + ["pSTAT5B_rel"] * 2,
            "simulationConditionId": ["c1", "c1", "c1", "c2", "c1", "c2"],
            "measurement": [7.9, 66.4, 81.2, 5.0, 3.0, 4.0],
            "time": [0.0, 2.5, 5.0, 0.0, 10.0, 5.0],
        }
    )
    conditions = pd.DataFrame(
        {"conditionId": ["c1", "c2"], "conditionName": ["control", "drug"]}
    )
    d = attached.petab_design(measurements, conditions)
    assert d.arms == ("control", "drug") and d.control == "control"
    assert dict(d.per_arm)["control"] == (0.0, 2.5, 5.0, 10.0)
    assert d.n_titles == 6 and d.time_course and d.time_unit == ""


SBML = """<?xml version="1.0" encoding="UTF-8"?>
<sbml xmlns="http://www.sbml.org/sbml/level3/version2/core" level="3" version="2">
<model id="m"><listOfCompartments><compartment id="c" constant="true"/></listOfCompartments>
<listOfSpecies>
<species id="p53" metaid="m_p53" compartment="c" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false">
<annotation><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
<rdf:Description rdf:about="#m_p53"><bqbiol:is><rdf:Bag>
<rdf:li rdf:resource="http://identifiers.org/uniprot/P04637"/>
</rdf:Bag></bqbiol:is></rdf:Description></rdf:RDF></annotation></species>
<species id="atp" metaid="m_atp" compartment="c" hasOnlySubstanceUnits="false" boundaryCondition="false" constant="false">
<annotation><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" xmlns:bqbiol="http://biomodels.net/biology-qualifiers/">
<rdf:Description rdf:about="#m_atp"><bqbiol:is><rdf:Bag>
<rdf:li rdf:resource="urn:miriam:chebi:CHEBI%3A15422"/>
<rdf:li rdf:resource="https://identifiers.org/CHEBI:16761"/>
</rdf:Bag></bqbiol:is></rdf:Description></rdf:RDF></annotation></species>
</listOfSpecies></model></sbml>
"""


def test_species_curies_are_read_from_cv_terms():
    import libsbml

    model = libsbml.readSBMLFromString(SBML).getModel()
    assert model is not None
    assert attached.sbml_species_curies(model) == (
        "uniprot:P04637",
        "chebi:15422",
        "chebi:16761",
    )


def test_copasi_fitting_experiments_are_read_by_copasi(tmp_path, monkeypatch):
    """A model with one time-course experiment, written by basico and read
    back through the same bindings."""
    basico = pytest.importorskip("basico")
    monkeypatch.chdir(tmp_path)
    model = basico.new_model(name="probe")
    basico.add_species("A", initial_concentration=1.0)
    basico.add_experiment(
        "course",
        pd.DataFrame({"Time": [0, 1, 2, 3], "[A]": [1.0, 0.5, 0.25, 0.125]}),
    )
    basico.save_model(str(tmp_path / "probe.cps"))
    basico.remove_datamodel(model)
    (exp,) = attached.copasi_experiments(tmp_path / "probe.cps")
    assert (exp.name, exp.basename, exp.kind) == (
        "course",
        "course.txt",
        "time course",
    )
    assert exp.n_rows == 4
    (tmp_path / "junk.cps").write_text("<not xml")
    assert attached.copasi_experiments(tmp_path / "junk.cps") == ()


def test_a_deposits_own_tables_become_a_candidate(tmp_path, monkeypatch):
    def fake_download(accession, timeout=60.0, names=None, **_):
        out = []
        for name in names or ():
            path = tmp_path / name
            if name == "data.csv":
                path.write_text(
                    "Time (h),Treatment,value\n0,ctrl,1\n6,ctrl,2\n24,ctrl,3\n"
                    "0,drug,1\n6,drug,4\n24,drug,9\n"
                )
            elif name == "model.cps":
                path.write_text("a copasi file")
            else:
                continue
            out.append(path)
        return out

    monkeypatch.setattr(attached, "download_biomodel_files", fake_download)
    monkeypatch.setattr(
        attached,
        "copasi_experiments",
        lambda path: (
            attached.CopasiExperiment(
                "Experiment_1", "../Downloads/data.csv", "time course", 6
            ),
            attached.CopasiExperiment(
                "Experiment_2", "steady.csv", "steady state"
            ),
        ),
    )
    cand = attached.biomodels_data(
        "BIOMD1",
        ["model.xml", "model.cps", "data.csv", "curation_notes.txt"],
        name="A model",
        organism="Homo sapiens",
        pubmed="123",
        ids=("uniprot:P04637",),
    )
    assert cand.source == "biomodels" and cand.accession == "BIOMD1"
    assert cand.files == ("data.csv",)
    assert cand.factors == (
        "COPASI time course: data.csv",
        "COPASI steady state: steady.csv",
    )
    assert cand.design.arms == ("ctrl", "drug") and cand.design.time_course
    assert cand.measured == attached.Measured(
        "targeted", False, ("uniprot:P04637",)
    )
    assert (
        attached.biomodels_data("BIOMD2", ["model.xml", "readme.txt"]) is None
    )


def test_a_deposit_without_copasi_bindings_still_reads_its_tables(
    tmp_path, monkeypatch
):
    def fake_download(accession, timeout=60.0, names=None, **_):
        path = tmp_path / "data.csv"
        path.write_text("t,x\n0,1\n1,2\n2,3\n")
        return [path]

    def no_bindings(path):
        raise ImportError("needs the copasi extra")

    monkeypatch.setattr(attached, "download_biomodel_files", fake_download)
    monkeypatch.setattr(attached, "copasi_experiments", no_bindings)
    cand = attached.biomodels_data("BIOMD3", ["model.cps", "data.csv"])
    assert cand.files == ("data.csv",) and cand.factors == ()
    assert cand.design.time_course


def test_the_petab_index_is_built_from_the_tree(monkeypatch):
    from types import SimpleNamespace

    tree = {
        "tree": [
            {"path": "README.md"},
            {
                "path": "Benchmark-Models/Boehm_JProteomeRes2014/Boehm_JProteomeRes2014.yaml"
            },
            {
                "path": "Benchmark-Models/Boehm_JProteomeRes2014/measurementData_Boehm.tsv"
            },
            {
                "path": "Benchmark-Models/Boehm_JProteomeRes2014/model_Boehm.xml"
            },
            {"path": "Benchmark-Models/Broken_2020/Broken_2020.yaml"},
        ]
    }
    monkeypatch.setattr(
        attached, "get_json", lambda url, params, timeout: tree
    )
    problem = SimpleNamespace(
        measurement_df=pd.DataFrame(
            {
                "observableId": ["pSTAT5A_rel"] * 3,
                "simulationConditionId": ["model1_data1"] * 3,
                "measurement": [1.0, 2.0, 3.0],
                "time": [0.0, 2.5, 5.0],
            }
        ),
        condition_df=pd.DataFrame(
            {"conditionName": ["condition1"]},
            index=pd.Index(["model1_data1"], name="conditionId"),
        ),
        observable_df=pd.DataFrame(
            index=pd.Index(["pSTAT5A_rel"], name="observableId")
        ),
        model=SimpleNamespace(sbml_model=None),
    )

    def load(url):
        if "Broken" in url:
            raise ValueError("no such problem")
        assert url.endswith(
            "/Boehm_JProteomeRes2014/Boehm_JProteomeRes2014.yaml"
        )
        return problem

    monkeypatch.setattr(attached, "load_problem", load)
    monkeypatch.setattr(
        attached, "cached_index", lambda name, build, refresh=False: build()
    )
    (cand,) = attached.petab_problems()  # the broken problem is skipped
    assert (
        cand.source == "petab" and cand.accession == "Boehm_JProteomeRes2014"
    )
    assert cand.design == Design(
        arms=("condition1",),
        per_arm=(("condition1", (0.0, 2.5, 5.0)),),
        n_titles=3,
    )
    assert cand.measured.ids == () and "pSTAT5A_rel" in cand.summary
    assert cand.n_samples == 3
    assert attached.search_petab("boehm") == [cand]
    assert attached.search_petab("pSTAT5A_rel") == [cand]
    assert attached.search_petab("nothing here") == []
