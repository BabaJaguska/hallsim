"""MetaboLights assignment + sample files as ChEBI-keyed contrasts.

Parsing is ``metabolights-utils``; what is tested here is the layer over
it — the ChEBI keying, the design the declared factors imply, and the
scale-free contrast.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from hallsim import metabolites
from hallsim.dataset_census import READABLE
from hallsim.metabolites import (
    MetaboliteDataset,
    isa_design,
    maf_quantities,
    read_isa_samples,
    read_maf,
)

MAF = """\
database_identifier\tchemical_formula\tmetabolite_identification\tmass_to_charge\tSample1\tSample2\tSample3\tSample4
CHEBI:17234\tC6H12O6\tglucose\t180.06\t100\t110\t200\t220
CHEBI:17234\tC6H12O6\tglucose [neg]\t179.05\t1000\t1100\t2000\t2200
CHEBI:16414\tC5H11NO2\tL-valine\t117.07\t50\t55\t25\t27.5
HMDB0000122\t\tan hmdb-only row\t180.06\t5\t5\t5\t5
KEGG:C00031\t\ta kegg-only row\t180.06\t5\t5\t5\t5
\t\tunassigned feature\t432.10\t7\t7\t7\t7
CHEBI:86472|CHEBI:71727\t\ttwo candidate compounds\t301.10\t9\t9\t9\t9
unknown|CHEBI:18089\t\tone candidate unknown\t302.10\t9\t9\t9\t9
CHEBI:15422\tC10H16N5O13P3\tATP\t507.00\t0\t-1\t10\t10
"""

SAMPLES = """\
Source Name\tCharacteristics[Organism]\tSample Name\tFactor Value[Treatment]\tFactor Value[Time]\tUnit
donor1\tHomo sapiens\tSample1\tcontrol\t24\thour
donor2\tHomo sapiens\tSample2\tcontrol\t24\thour
donor3\tHomo sapiens\tSample3\trapamycin\t24\thour
donor4\tHomo sapiens\tSample4\trapamycin\t24\thour
"""

COLUMNS = ["Sample1", "Sample2", "Sample3", "Sample4"]
GROUPS = {
    "control @ 24h": ["Sample1", "Sample2"],
    "rapamycin @ 24h": ["Sample3", "Sample4"],
}


@dataclass
class _Table:
    columns: list
    data: dict


@dataclass
class _File:
    table: _Table


def _study(maf_text=MAF, samples_text=SAMPLES):
    """A stand-in for a metabolights-utils study model: only the two
    attributes our layer reads."""

    def table(text):
        rows = [line.split("\t") for line in text.strip("\n").split("\n")]
        header = rows[0]
        return _Table(
            header,
            {c: [r[i] for r in rows[1:]] for i, c in enumerate(header)},
        )

    samples = table(samples_text)

    @dataclass
    class Model:
        samples: dict
        assays: dict
        metabolite_assignments: dict

    return Model(
        samples={"s_x.txt": _File(samples)},
        assays={
            "a_x.txt": _File(
                _Table(
                    ["Sample Name"],
                    {"Sample Name": samples.data["Sample Name"]},
                )
            )
        },
        metabolite_assignments={"m_x_maf.tsv": _File(table(maf_text))},
    )


@pytest.fixture
def study(tmp_path):
    (tmp_path / "m_study_v2_maf.tsv").write_text(MAF)
    (tmp_path / "s_study.txt").write_text(SAMPLES)
    return tmp_path


@pytest.fixture
def dataset():
    return MetaboliteDataset.from_study(_study())


def test_a_maf_keeps_chebi_rows_and_counts_what_it_drops(study):
    table = read_maf(study / "m_study_v2_maf.tsv", samples=COLUMNS)

    assert table.features == 9
    assert len(table.quantities) == 4
    assert table.compounds == 3
    assert table.unannotated == 1
    # A feature standing for either of two compounds belongs to neither.
    assert table.ambiguous == 2
    assert table.other_namespace == {"HMDB": 1, "KEGG": 1}
    assert table.coverage == pytest.approx(4 / 9)
    assert list(table.quantities.index) == [
        "chebi:17234",
        "chebi:17234",
        "chebi:16414",
        "chebi:15422",
    ]
    assert table.names["chebi:16414"] == "L-valine"
    assert list(table.quantities.columns) == COLUMNS


def test_below_detection_is_missing_not_zero(study):
    table = read_maf(study / "m_study_v2_maf.tsv", samples=COLUMNS)
    atp = table.quantities.loc["chebi:15422"]
    assert np.isnan(atp["Sample1"]) and np.isnan(atp["Sample2"])
    assert atp["Sample3"] == 10.0


def test_a_table_without_an_identifier_column_is_refused():
    frame = pd.DataFrame({"name": ["glucose"], "Sample1": ["1"]})
    with pytest.raises(ValueError, match="database_identifier"):
        maf_quantities(frame, ["Sample1"])


def test_samples_naming_no_column_is_refused(study):
    with pytest.raises(ValueError, match="none of the"):
        read_maf(study / "m_study_v2_maf.tsv", samples=["nope"])


def test_the_design_comes_from_declared_factors(study):
    design, groups = isa_design(read_isa_samples(study / "s_study.txt"))

    assert design.arms == ("control", "rapamycin")
    assert design.control == "control"
    assert design.time_unit == "h"
    assert design.per_arm == (("control", (24.0,)), ("rapamycin", (24.0,)))
    assert groups == GROUPS


def test_a_factor_that_identifies_one_sample_is_not_an_arm():
    """A declared factor can be a per-sample identifier — MTBLS92 ships a
    Series factor — and crossing it into the label makes every sample its
    own arm."""
    samples = (
        "Source Name\tSample Name\tFactor Value[Treatment]\t"
        "Factor Value[Series]\n"
        "d1\tSample1\tcontrol\tS-001\n"
        "d2\tSample2\tcontrol\tS-002\n"
        "d3\tSample3\trapamycin\tS-003\n"
        "d4\tSample4\trapamycin\tS-004\n"
    )
    design, groups = isa_design(
        pd.DataFrame(
            [r.split("\t") for r in samples.strip("\n").split("\n")[1:]],
            columns=samples.split("\n")[0].split("\t"),
        )
    )
    assert design.arms == ("control", "rapamycin")
    assert sorted(groups) == ["control", "rapamycin"]


def test_a_named_time_factor_that_is_absent_is_refused(study):
    frame = read_isa_samples(study / "s_study.txt")
    with pytest.raises(ValueError, match=r"Factor Value\[Day\]"):
        isa_design(frame, time_factor="Day")


def test_two_features_of_one_compound_average_their_ratios(dataset):
    """The two glucose rows sit an order of magnitude apart and carry the
    same 2-fold change, so a scale-free contrast returns exactly 1.0."""
    delta = dataset.delta("rapamycin @ 24h", "control @ 24h")

    assert dataset.features_per_compound()["chebi:17234"] == 2
    assert delta["chebi:17234"] == pytest.approx(1.0)
    assert delta["chebi:16414"] == pytest.approx(-1.0)
    # Nothing to divide by: ATP is below detection in the baseline group.
    assert "chebi:15422" not in delta.index


def test_a_contrast_is_keyed_by_curie_so_it_joins_model_species(dataset):
    delta = dataset.delta("rapamycin @ 24h", "control @ 24h")
    species = {"chebi:17234", "chebi:29033"}

    assert set(delta.index) & species == {"chebi:17234"}


def test_a_study_takes_its_sample_columns_from_the_assay_file(dataset):
    assert dataset.sample_groups == GROUPS
    assert dataset.design.arms == ("control", "rapamycin")
    assert list(dataset.quantities.columns) == COLUMNS


def test_a_detached_maf_takes_groups_directly(study):
    ds = MetaboliteDataset.from_maf(
        study / "m_study_v2_maf.tsv", sample_groups=GROUPS
    )
    assert ds.design is None
    assert ds.delta("rapamycin @ 24h", "control @ 24h")[
        "chebi:16414"
    ] == pytest.approx(-1.0)


def test_variance_counts_only_the_samples_that_carry_a_value(dataset):
    var = dataset.variance("rapamycin @ 24h", "control @ 24h")
    # Two replicates each, two features averaged: finite and positive.
    assert var["chebi:17234"] > 0 and np.isfinite(var["chebi:17234"])
    # One usable baseline replicate leaves no spread to estimate.
    assert np.isnan(var["chebi:15422"])


def test_an_unknown_group_names_the_ones_there_are(dataset):
    with pytest.raises(KeyError, match="control @ 24h"):
        dataset.delta("nope", "control @ 24h")


def test_the_census_reads_an_assignment_file():
    assert "maf" in READABLE


def test_arm_deltas_works_unchanged_on_metabolites():
    @dataclass
    class Arm:
        condition: str
        reference: str | None

    ds = MetaboliteDataset.from_study(
        _study(
            samples_text=(
                "Source Name\tSample Name\tFactor Value[Treatment]\t"
                "Factor Value[Time]\tUnit\n"
                "d1\tSample1\trapamycin\t0\thour\n"
                "d2\tSample2\trapamycin\t0\thour\n"
                "d3\tSample3\trapamycin\t24\thour\n"
                "d4\tSample4\trapamycin\t24\thour\n"
            )
        )
    )
    # One condition across every sample distinguishes no arm, so the
    # factor is stripped and the single arm is unnamed.
    assert ds.design.arms == ("all",)
    out = ds.arm_deltas(
        {"rapa": {0.0: "all @ 0h", 24.0: "all @ 24h"}},
        {"rapa": Arm("rapamycin", "t0")},
    )
    assert list(out["rapa"]) == [24.0]
    assert out["rapa"][24.0]["chebi:17234"] == pytest.approx(1.0)


def test_log_values_are_the_log2_of_the_intensities(dataset):
    assert isinstance(dataset.log_values, pd.DataFrame)
    assert dataset.log_values.loc["chebi:16414", "Sample1"] == pytest.approx(
        np.log2(50.0)
    )


# ── Metabolomics Workbench (mwTab) ──────────────────────────────────

MWTAB_TEXT = "#METABOLOMICS WORKBENCH STUDY_ID:ST000999 ANALYSIS_ID:AN000999\nVERSION             \t1\nCREATED_ON          \t2026-09-25\n#PROJECT\nPR:PROJECT_TITLE                \ta synthetic study\n#STUDY\nST:STUDY_TITLE                  \ta synthetic study\n#SUBJECT\nSU:SUBJECT_TYPE                 \tHuman\n#SUBJECT_SAMPLE_FACTORS:        \tSUBJECT(optional)[tab]SAMPLE[tab]FACTORS(NAME:VALUE pairs separated by |)[tab]Raw file names and additional sample data\nSUBJECT_SAMPLE_FACTORS          \t-\tS1\tTreatment:control | Time:24 h\t\nSUBJECT_SAMPLE_FACTORS          \t-\tS2\tTreatment:control | Time:24 h\t\nSUBJECT_SAMPLE_FACTORS          \t-\tS3\tTreatment:rapamycin | Time:24 h\t\nSUBJECT_SAMPLE_FACTORS          \t-\tS4\tTreatment:rapamycin | Time:24 h\t\n#COLLECTION\nCO:COLLECTION_SUMMARY           \tx\n#TREATMENT\nTR:TREATMENT_SUMMARY            \tx\n#SAMPLEPREP\nSP:SAMPLEPREP_SUMMARY           \tx\n#CHROMATOGRAPHY\nCH:CHROMATOGRAPHY_TYPE          \tGC\n#ANALYSIS\nAN:ANALYSIS_TYPE                \tMS\n#MS\nMS:INSTRUMENT_NAME              \tx\nMS:MS_TYPE                      \tEI\nMS:ION_MODE                     \tPOSITIVE\n#MS_METABOLITE_DATA\nMS_METABOLITE_DATA:UNITS        \tPeak height\nMS_METABOLITE_DATA_START\nSamples\tS1\tS2\tS3\tS4\nFactors\tTreatment:control | Time:24 h\tTreatment:control | Time:24 h\tTreatment:rapamycin | Time:24 h\tTreatment:rapamycin | Time:24 h\nglucose\t100\t110\t200\t220\nvaline\t50\t55\t25\t27.5\natp\t0\t-1\t10\t10\nmystery\t7\t7\t7\t7\nMS_METABOLITE_DATA_END\n#METABOLITES\nMETABOLITES_START\nmetabolite_name\tpubchem_id\tinchi_key\tkegg_id\nglucose\t5793\t\tC00031\nvaline\t6287\t\tC00183\natp\t5957\t\tC00002\nmystery\t\t\tC99999\nMETABOLITES_END\n#END\n"
#: What UniChem answers for this fixture, so the test makes no request.
FAKE_CHEBI = {
    ("5793", "pubchem"): ("chebi:17234",),
    ("6287", "pubchem"): ("chebi:16414", "chebi:57762"),
    ("5957", "pubchem"): ("chebi:15422",),
}


@pytest.fixture
def mwtab_file(tmp_path):
    path = tmp_path / "ST000999.mwtab.txt"
    path.write_text(MWTAB_TEXT)
    return path


@pytest.fixture
def offline_unichem(monkeypatch):
    def fake(query, source="", *, timeout=30.0):
        return FAKE_CHEBI.get((query, source), ())

    monkeypatch.setattr(metabolites, "chebi_for", fake)


def test_mwtab_keys_its_compounds_through_the_crosswalk(
    offline_unichem, mwtab_file
):
    """mwTab carries PubChem, InChIKey and KEGG but never ChEBI, so the
    join is structural rather than by name."""
    table = metabolites.read_mwtab(mwtab_file)

    assert table.features == 4
    assert table.keyed == 3
    # KEGG-only compounds do not resolve: UniChem does not index KEGG.
    assert table.unannotated == 1
    assert table.coverage == pytest.approx(3 / 4)
    assert table.names["chebi:17234"] == "glucose"


def test_several_chebi_ids_for_one_compound_are_all_kept(
    offline_unichem, mwtab_file
):
    """ChEBI models protonation states separately, so valine is two
    accessions for one molecule — granularity, not uncertainty."""
    table = metabolites.read_mwtab(mwtab_file)

    assert len(table.quantities) == 4
    assert sorted(table.quantities.index) == [
        "chebi:15422",
        "chebi:16414",
        "chebi:17234",
        "chebi:57762",
    ]


def test_the_workbench_design_comes_from_its_factor_dicts(
    offline_unichem, mwtab_file
):
    ds = metabolites.MetaboliteDataset.from_workbench(mwtab_file)

    assert ds.design.arms == ("control", "rapamycin")
    assert ds.design.control == "control"
    assert ds.design.time_unit == "h"
    assert ds.design.timepoints == (24.0,)
    assert ds.sample_groups == {
        "control @ 24h": ["S1", "S2"],
        "rapamycin @ 24h": ["S3", "S4"],
    }


def test_a_workbench_contrast_reaches_a_model_species(
    offline_unichem, mwtab_file
):
    ds = metabolites.MetaboliteDataset.from_workbench(mwtab_file)
    delta = ds.delta("rapamycin @ 24h", "control @ 24h")

    assert delta["chebi:17234"] == pytest.approx(1.0)
    # Both accessions of one molecule carry the same contrast.
    assert delta["chebi:16414"] == pytest.approx(-1.0)
    assert delta["chebi:57762"] == pytest.approx(-1.0)
    # Below detection throughout the baseline: no contrast to report.
    assert "chebi:15422" not in delta.index


def test_a_time_factor_value_is_read_with_the_title_parser():
    """``24 h`` inside a factor value needs no separate unit field."""
    design, groups = metabolites.factor_design(
        ["S1", "S2", "S3"],
        [
            {"Treatment": "ctrl", "Time": "0 h"},
            {"Treatment": "ctrl", "Time": "24 h"},
            {"Treatment": "ctrl", "Time": "48 h"},
        ],
    )
    assert design.time_unit == "h"
    assert design.timepoints == (0.0, 24.0, 48.0)
    assert design.time_course
    assert sorted(groups) == ["all @ 0h", "all @ 24h", "all @ 48h"]


def test_the_census_reads_a_workbench_study():
    from hallsim.dataset_census import LOADERS, READABLE

    assert LOADERS[("metabolomics_workbench", "metabolomics", False)] == (
        "mwtab"
    )
    assert "mwtab" in READABLE


def test_an_nmr_study_keeps_its_table_under_another_block(
    offline_unichem, tmp_path
):
    path = tmp_path / "ST000998.mwtab.txt"
    path.write_text(
        MWTAB_TEXT.replace("MS_METABOLITE_DATA", "NMR_METABOLITE_DATA")
    )
    table = metabolites.read_mwtab(path)
    assert table.features == 4 and table.keyed == 3


def test_a_failed_lookup_is_not_cached_as_no_compound(monkeypatch, tmp_path):
    """One timeout must not record a permanent false negative."""
    import urllib.request

    from hallsim import datasets

    monkeypatch.setattr(datasets, "_cache_dir", lambda: tmp_path)

    def down(*a, **k):
        raise OSError("unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", down)
    assert metabolites.chebi_for("602", "pubchem") == ()
    assert list(tmp_path.glob("*.json")) == []
